# This file is a part of Julia. License is MIT: https://julialang.org/license

# Low-overhead parallel loops on parked worker tasks.
#
# `parallel_for` and `@threads :parked` split the work into `nchunks` chunks and run
# them on a pool of persistent worker tasks, one sticky task per thread of the `:default`
# thread pool. A worker that finished a parallel region spins for a short while (by
# default 100 µs, `JULIA_PARALLEL_SPIN_NS`) waiting for the next one, then parks in the
# scheduler (`wait()`), where it costs nothing and its thread can run other tasks.
#
# A parallel region does not create, enqueue or switch to a `Task` per chunk:
#
# - The calling task (the "dispatcher") publishes a region descriptor (a C function
#   pointer specialized on the type of the loop function, the loop function itself, the
#   number of chunks, the dynamic scope) under a sequence lock, then writes the region's
#   generation number into the work slot of each worker it needs (one cache line per
#   worker). Spinning workers see the new generation, parked ones are `schedule`d.
# - Every participant first claims its "home" chunk. The dispatcher always participates,
#   claims chunk 1 and then steals every chunk that is still unclaimed; workers only take
#   the chunks beyond the first `nworkers` (which have no home participant). A chunk is
#   claimed with a CAS on a per-chunk cache line, so a chunk runs exactly once, and a
#   worker that cannot run (its thread is busy with another task, or it is still waking
#   up) does not delay the region by more than one chunk: the dispatcher takes its chunk.
# - The dispatcher waits until every chunk has finished (spinning, then parking), and
#   then throws a `CompositeException` of the captured exceptions, if any.
#
# Each chunk runs with the dispatcher's dynamic scope (`ScopedValue`s), an empty
# task-local storage and a task-local RNG state forked from the dispatcher's in the same
# way as for the tasks of `@threads`, so that `rand()` in the body gives the same results
# as with `@threads :dynamic`. `current_task()` is the worker task (or the dispatcher), not
# a fresh task.
#
# Only one region runs on the pool at a time. A parallel loop started from inside a region
# (from the loop body) runs serially; one started by another task while the pool is busy
# falls back to the task-based implementation of `@threads :dynamic`.

public parallel_for

const _PF_LINE = 16             # 16 x 8 bytes = 128 bytes: one slot per (pair of) cache line(s)
const _PF_MAX_CHUNKS_PER_THREAD = 8

# Per worker slot: [work generation, parked flag]
const _PF_W_WORK = 1
const _PF_W_PARKED = 2
# Per chunk slot: [claimed generation, finished generation]
const _PF_C_CLAIM = 1
const _PF_C_FIN = 2

mutable struct ParkedPool
    const nworkers::Int          # = threadpoolsize(:default), one worker per default thread
    const tid_offset::Int        # thread ids of the default pool are tid_offset .+ (1:nworkers)
    const maxchunks::Int
    const spin_ns::UInt64
    const workers::Vector{Task}
    const wslots::AtomicMemory{Int}
    const cslots::AtomicMemory{Int}
    const errs::Vector{Any}
    @atomic busy::Int
    @atomic seq::Int             # sequence lock for the descriptor: odd = being written
    gen::Int                     # last generation, only touched by the dispatcher
    # region descriptor
    @atomic fptr::Ptr{Cvoid}
    @atomic env::Any
    @atomic nchunks::Int
    @atomic skip::Int
    @atomic scope::Any
    @atomic dispatcher::Any
    @atomic waiting::Int
    @atomic nerrs::Int
    # the dispatcher's RNG state at the start of the region
    @atomic rng0::UInt64
    @atomic rng1::UInt64
    @atomic rng2::UInt64
    @atomic rng3::UInt64
    @atomic rng4::UInt64
    @atomic world::UInt
end

# The code of a worker task; also used to recognize a worker (`current_task().code isa ...`)
struct _PFWorkerCode <: Function
    pool::ParkedPool
    k::Int
end
(w::_PFWorkerCode)() = _pf_worker_loop(w.pool, w.k)

const _PF_POOL = Ref{Union{Nothing, ParkedPool}}(nothing)
const _PF_POOL_LOCK = Threads.SpinLock()

@inline _pf_wref(pool::ParkedPool, k::Int, f::Int) =
    Core.memoryrefnew(Core.memoryrefnew(pool.wslots), (k - 1) * _PF_LINE + f, false)
@inline _pf_cref(pool::ParkedPool, c::Int, f::Int) =
    Core.memoryrefnew(Core.memoryrefnew(pool.cslots), (c - 1) * _PF_LINE + f, false)
@inline _pf_load(r, order::Symbol) = Core.memoryrefget(r, order, false)
@inline _pf_store!(r, x::Int, order::Symbol) = (Core.memoryrefset!(r, x, order, false); nothing)
@inline _pf_cas!(r, old::Int, new::Int) =
    Core.memoryrefreplace!(r, old, new, :sequentially_consistent, :monotonic, false).success

@inline _pf_pause() = ccall(:jl_cpu_pause, Cvoid, ())
@inline function _pf_sched_yield()
    @static if Sys.iswindows()
        ccall(:SwitchToThread, stdcall, Cint, ())
    else
        ccall(:sched_yield, Cint, ())
    end
    nothing
end

# Is there another task this thread could run instead of spinning?
@inline function _pf_other_work()
    W = Base.workqueue_for(threadid())
    return !isempty(W) || !Base.Partr.multiq_check_empty()
end

# The RNG fork of `jl_rng_split` (src/task.c) on 5-word states, so that chunk `c` gets the
# same RNG state as the `c`-th task created by `@threads`.
@inline function _pf_rng_mix(c::UInt64, x::UInt64, a::UInt64, m::UInt64)
    w = x ⊻ a
    c += w * (2c + 1)
    c ⊻= c >> ((c >> 59) + 5)
    c *= m
    return c ⊻ (c >> 43)
end
@inline function _pf_rng_split(ct::Task)
    x = ct.rngState4
    x4 = x * 0xd1342543de82ef95 + 1
    ct.rngState4 = x4
    return (_pf_rng_mix(ct.rngState0, x, 0x214c146c88e47cb7, 0xaef17502108ef2d9),
            _pf_rng_mix(ct.rngState1, x, 0xa66d8cc21285aafa, 0xf34026eeb86766af),
            _pf_rng_mix(ct.rngState2, x, 0x68c7ef2d7b1a54d4, 0x38fd70ad58dd9fbb),
            _pf_rng_mix(ct.rngState3, x, 0xb053a7d7aa238c61, 0x6677f9b93ab0c04d),
            x4)
end
# The RNG state `jl_rng_split` gives to the `c`-th child of a task with RNG state `s`.
# Splitting only advances the LCG in word 5 of the parent, so the `c`-th child can be
# computed directly by whoever runs chunk `c`, off the dispatcher's critical path.
@inline function _pf_rng_child(s::NTuple{5, UInt64}, c::Int)
    x = s[5]
    for _ in 2:c
        x = x * 0xd1342543de82ef95 + 1
    end
    return (_pf_rng_mix(s[1], x, 0x214c146c88e47cb7, 0xaef17502108ef2d9),
            _pf_rng_mix(s[2], x, 0xa66d8cc21285aafa, 0xf34026eeb86766af),
            _pf_rng_mix(s[3], x, 0x68c7ef2d7b1a54d4, 0x38fd70ad58dd9fbb),
            _pf_rng_mix(s[4], x, 0xb053a7d7aa238c61, 0x6677f9b93ab0c04d),
            x * 0xd1342543de82ef95 + 1)
end
# Advance the RNG of `ct` as `n` splits do
@inline function _pf_rng_advance!(ct::Task, n::Int)
    x = ct.rngState4
    for _ in 1:n
        x = x * 0xd1342543de82ef95 + 1
    end
    ct.rngState4 = x
    nothing
end

@inline function _pf_rng_get(ct::Task)
    return (ct.rngState0, ct.rngState1, ct.rngState2, ct.rngState3, ct.rngState4)
end
@inline function _pf_rng_set!(ct::Task, s::NTuple{5, UInt64})
    ct.rngState0, ct.rngState1, ct.rngState2, ct.rngState3, ct.rngState4 = s
    nothing
end

function _pf_create_pool()
    n = threadpoolsize(:default)
    tid_offset = threadpoolsize(:interactive)
    maxchunks = _PF_MAX_CHUNKS_PER_THREAD * n
    spin_ns = UInt64(100_000)
    s = get(ENV, "JULIA_PARALLEL_SPIN_NS", nothing)
    if s !== nothing
        v = tryparse(Int, s)
        v !== nothing && v >= 0 && (spin_ns = UInt64(v))
    end
    wslots = AtomicMemory{Int}(undef, n * _PF_LINE)
    cslots = AtomicMemory{Int}(undef, maxchunks * _PF_LINE)
    for i in eachindex(wslots)
        Core.memoryrefset!(Core.memoryrefnew(Core.memoryrefnew(wslots), i, false), 0, :monotonic, false)
    end
    for i in eachindex(cslots)
        Core.memoryrefset!(Core.memoryrefnew(Core.memoryrefnew(cslots), i, false), 0, :monotonic, false)
    end
    pool = ParkedPool(n, tid_offset, maxchunks, spin_ns, Vector{Task}(undef, n), wslots, cslots,
                      Any[nothing for _ in 1:maxchunks],
                      0, 0, 0, C_NULL, nothing, 0, 0, nothing, nothing, 0, 0,
                      UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt(0))
    # creating the worker tasks must not change the RNG state of the caller
    ct = current_task()
    rng = _pf_rng_get(ct)
    for k in 1:n
        t = Task(_PFWorkerCode(pool, k))
        t.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid_offset + k - 1)
        pool.workers[k] = t
    end
    _pf_rng_set!(ct, rng)
    for t in pool.workers
        schedule(t)
    end
    return pool
end

@noinline function _pf_get_pool()
    pool = _PF_POOL[]
    pool === nothing || return pool
    lock(_PF_POOL_LOCK)
    try
        pool = _PF_POOL[]
        if pool === nothing
            pool = _pf_create_pool()
            _PF_POOL[] = pool
        end
    finally
        unlock(_PF_POOL_LOCK)
    end
    return pool
end

# Block the current task until somebody `schedule`s it, never throwing in between (an
# exception thrown into a parked task would otherwise leave a pending `schedule` that
# could later wake it from an unrelated `wait`).
function _pf_park()
    local err = nothing
    while true
        try
            wait()
            break
        catch e
            # e.g. an InterruptException; deliver it once we have been woken properly
            err = e
        end
    end
    return err
end

function _pf_worker_loop(pool::ParkedPool, k::Int)
    work = _pf_wref(pool, k, _PF_W_WORK)
    parked = _pf_wref(pool, k, _PF_W_PARKED)
    seen = 0
    spin_ns = pool.spin_ns
    while true
        g = _pf_load(work, :acquire)
        if g != seen
            seen = g
            _pf_worker_participate(pool, g, k)
            continue
        end
        # Spin for new work, then park.
        t0 = time_ns()
        tyield = t0
        i = 0
        while true
            _pf_load(work, :acquire) != seen && break
            _pf_pause()
            GC.safepoint()
            i += 1
            if i & 63 == 0
                now = time_ns()
                if now - t0 >= spin_ns || _pf_other_work()
                    _pf_store!(parked, 1, :sequentially_consistent)
                    if _pf_load(work, :sequentially_consistent) != seen &&
                       _pf_cas!(parked, 1, 0)
                        break  # work arrived before anyone saw us parked
                    end
                    _pf_park()  # woken by a dispatcher (which reset `parked`)
                    break
                elseif now - tyield >= 5_000
                    # give the CPU to other OS threads now and then (oversubscription)
                    _pf_sched_yield()
                    tyield = now
                end
            end
        end
    end
end

# The function pointer the workers call: `env` is a `RefValue{F}` holding the chunk function
function _pf_cfun(env::Base.RefValue{F}, c::Int) where {F}
    env[](c)
    return nothing
end

struct _PFDirect{F}
    f::F
end
@inline (r::_PFDirect)(c::Int) = r.f(c)
# Run `ex` in the dynamic scope `scope` (as `ScopedValues.@enter_scope`)
macro _pf_enter_scope(scope, ex)
    Expr(:tryfinally, esc(ex), nothing, esc(scope))
end

# A worker runs a chunk through the function pointer, in the dispatcher's dynamic scope and
# world age. A `@cfunction` runs in the latest world, so it is only used when that is the
# dispatcher's world (the common case); otherwise the chunk is run with `invoke_in_world`.
struct _PFIndirect
    fptr::Ptr{Cvoid}
    env::Any
    scope::Any
    world::UInt
end
@inline function (r::_PFIndirect)(c::Int)
    @_pf_enter_scope r.scope begin
        if r.world == Base.get_world_counter()
            ccall(r.fptr, Cvoid, (Any, Int), r.env, c)
        else
            Base.invoke_in_world(r.world, _pf_cfun, r.env, c)
        end
    end
end

# Run chunk `c` on the current task with a fresh task-local storage and the chunk's RNG
# state; capture exceptions. (Do not pass an argument of type `Any` here: together with
# the type parameter `R` that makes the call dynamic.)
@noinline function _pf_run_chunk(pool::ParkedPool, run::R, c::Int, ct::Task,
                                 rng::NTuple{5, UInt64}) where {R}
    old_storage = ct.storage
    old_rng = _pf_rng_get(ct)
    ct.storage = nothing
    _pf_rng_set!(ct, _pf_rng_child(rng, c))
    try
        run(c)
    catch e
        @inbounds pool.errs[c] = CapturedException(e, catch_backtrace())
        @atomic :monotonic pool.nerrs += 1
    finally
        ct.storage = old_storage
        _pf_rng_set!(ct, old_rng)
    end
    nothing
end

# Try to claim chunk `c` of region `g`; run it if claimed
@inline function _pf_try_chunk(pool::ParkedPool, run::R, g::Int, c::Int, ct::Task,
                               rng::NTuple{5, UInt64}) where {R}
    claim = _pf_cref(pool, c, _PF_C_CLAIM)
    x = _pf_load(claim, :monotonic)
    if x < g && _pf_cas!(claim, x, g)
        _pf_run_chunk(pool, run, c, ct, rng)
        _pf_store!(_pf_cref(pool, c, _PF_C_FIN), g, :sequentially_consistent)
        if (@atomic pool.waiting) == 1 && (@atomicreplace pool.waiting 1 => 0).success
            schedule((@atomic :acquire pool.dispatcher)::Task)
        end
    end
    nothing
end

# Claim and run chunk `home` of region `g`, then the unclaimed chunks among `steal:n`.
# The dispatcher steals all chunks (so that a late or busy worker never delays a region by
# more than one chunk); the workers only take the chunks beyond the first `nworkers`,
# which have no home participant (scanning the chunks of the other workers would move
# their cache lines around for nothing in the common case).
@inline function _pf_participate(pool::ParkedPool, run::R, g::Int, n::Int, home::Int,
                                 steal::Int, rng::NTuple{5, UInt64}) where {R}
    ct = current_task()
    _pf_try_chunk(pool, run, g, home, ct, rng)
    for c in steal:n
        c == home && continue
        _pf_try_chunk(pool, run, g, c, ct, rng)
    end
    nothing
end

function _pf_worker_participate(pool::ParkedPool, g::Int, k::Int)
    # Read a consistent descriptor for region `g` (sequence lock), or give up if the
    # region is already over.
    (@atomic :acquire pool.seq) == 2g || return
    fptr = @atomic :monotonic pool.fptr
    env = @atomic :monotonic pool.env
    n = @atomic :monotonic pool.nchunks
    skip = @atomic :monotonic pool.skip
    scope = @atomic :monotonic pool.scope
    world = @atomic :monotonic pool.world
    rng = (@atomic(:monotonic, pool.rng0), @atomic(:monotonic, pool.rng1),
           @atomic(:monotonic, pool.rng2), @atomic(:monotonic, pool.rng3),
           @atomic(:monotonic, pool.rng4))
    (@atomic :acquire pool.seq) == 2g || return
    home = k < skip ? k + 1 : k
    home > n && return
    _pf_participate(pool, _PFIndirect(fptr, env, scope, world), g, n, home, pool.nworkers + 1, rng)
    nothing
end

"""
    Threads.parked_run(fun, nchunks::Integer)

Call `fun(c)` for `c in 1:nchunks` in parallel on the parked worker pool and return
`nothing` after all calls finished. Internal building block of [`parallel_for`](@ref) and
`@threads :parked`.
"""
function parked_run(fun::F, nchunks::Integer) where {F}
    n = Int(nchunks)
    n <= 0 && return nothing
    if n == 1 || threadpoolsize(:default) == 1 || ccall(:jl_generating_output, Cint, ()) != 0
        _pf_serial(fun, n)
        return nothing
    end
    pool = _pf_get_pool()
    ct = current_task()
    if (isdefined(ct, :code) && ct.code isa _PFWorkerCode) || ct === (@atomic :monotonic pool.dispatcher)
        # nested parallel loop: run it serially on this task
        _pf_serial(fun, n)
        return nothing
    end
    if n > pool.maxchunks || !(@atomicreplace pool.busy 0 => 1).success
        # the pool is busy with a region of another task
        _pf_fallback(fun, n)
        return nothing
    end
    try
        _pf_region(pool, fun, n, ct)
    finally
        @atomic :release pool.busy = 0
    end
    return nothing
end

# Run the chunks one after the other on the current task. As in a parallel region, each
# chunk gets an empty task-local storage and a forked RNG state, an exception in a chunk
# does not stop the other chunks, and the exceptions are thrown as a `CompositeException`
# at the end.
@noinline function _pf_serial(fun::F, n::Int) where {F}
    ct = current_task()
    errs = nothing
    for c in 1:n
        rng = _pf_rng_split(ct)  # same as _pf_rng_child(state at start, c)
        old_storage = ct.storage
        old_rng = _pf_rng_get(ct)
        ct.storage = nothing
        _pf_rng_set!(ct, rng)
        try
            fun(c)
        catch e
            errs === nothing && (errs = Any[])
            push!(errs, CapturedException(e, catch_backtrace()))
        finally
            ct.storage = old_storage
            _pf_rng_set!(ct, old_rng)
        end
    end
    errs === nothing || throw(CompositeException(errs))
    nothing
end

@noinline function _pf_fallback(fun::F, n::Int) where {F}
    m = threadpoolsize()
    # `threading_run` calls `f(i)` for `i in 1:threadpoolsize()`
    threading_run(function (i)
                      for c in i:m:n
                          fun(c)
                      end
                  end, false)
end

function _pf_region(pool::ParkedPool, fun::F, n::Int, ct::Task) where {F}
    g = pool.gen + 1
    pool.gen = g
    env = Base.RefValue{F}(fun)
    fptr = @cfunction(_pf_cfun, Cvoid, (Ref{Base.RefValue{F}}, Int))
    # caller's position in the default pool: its worker is skipped and it runs chunk 1
    p = threadid() - pool.tid_offset
    skip = 1 <= p <= pool.nworkers ? p : pool.nworkers
    scope = Core.current_scope()
    # chunk `c` gets the RNG state of the `c`-th child task, as with `@threads`
    rng = _pf_rng_get(ct)
    # publish the descriptor
    @atomic :release pool.seq = 2g - 1
    @atomic :monotonic pool.fptr = fptr
    @atomic :monotonic pool.env = env
    @atomic :monotonic pool.nchunks = n
    @atomic :monotonic pool.skip = skip
    @atomic :monotonic pool.scope = scope
    @atomic :monotonic pool.dispatcher = ct
    @atomic :monotonic pool.nerrs = 0
    @atomic :monotonic pool.rng0 = rng[1]
    @atomic :monotonic pool.rng1 = rng[2]
    @atomic :monotonic pool.rng2 = rng[3]
    @atomic :monotonic pool.rng3 = rng[4]
    @atomic :monotonic pool.rng4 = rng[5]
    @atomic :monotonic pool.world = Base.tls_world_age()
    @atomic :release pool.seq = 2g
    # hand out the work
    for k in 1:pool.nworkers
        k == skip && continue
        home = k < skip ? k + 1 : k
        home > n && continue
        _pf_store!(_pf_wref(pool, k, _PF_W_WORK), g, :release)
    end
    # One fence for all workers (instead of a seq-cst store per worker) orders the stores of
    # the work slots before the loads of the parked flags (pairs with the worker's store of
    # `parked` and load of `work`).
    Core.Intrinsics.atomic_fence(:sequentially_consistent, :system)
    for k in 1:pool.nworkers
        k == skip && continue
        home = k < skip ? k + 1 : k
        home > n && continue
        parked = _pf_wref(pool, k, _PF_W_PARKED)
        if _pf_load(parked, :monotonic) == 1 && _pf_cas!(parked, 1, 0)
            schedule(@inbounds pool.workers[k])
        end
    end
    # run chunk 1, then steal
    _pf_rng_advance!(ct, n)
    _pf_participate(pool, _PFDirect{F}(fun), g, n, 1, 2, rng)
    # wait for the chunks run by other participants
    _pf_wait(pool, g, n)
    @atomic :monotonic pool.env = nothing
    @atomic :monotonic pool.scope = nothing
    @atomic :monotonic pool.dispatcher = nothing
    if (@atomic :acquire pool.nerrs) != 0
        errs = Any[]
        for c in 1:n
            e = @inbounds pool.errs[c]
            if e !== nothing
                push!(errs, e)
                @inbounds pool.errs[c] = nothing
            end
        end
        throw(CompositeException(errs))
    end
    nothing
end

function _pf_wait(pool::ParkedPool, g::Int, n::Int)
    spin_ns = pool.spin_ns
    t0 = UInt64(0)
    i = 0
    err = nothing
    for c in 1:n
        fin = _pf_cref(pool, c, _PF_C_FIN)
        while _pf_load(fin, :acquire) != g
            _pf_pause()
            GC.safepoint()
            i += 1
            if i & 63 == 0
                now = time_ns()
                t0 == 0 && (t0 = now)
                if now - t0 >= spin_ns || _pf_other_work()
                    @atomic pool.waiting = 1
                    if _pf_load(fin, :sequentially_consistent) == g &&
                       (@atomicreplace pool.waiting 1 => 0).success
                        continue
                    end
                    e = _pf_park()
                    e === nothing || (err = e)
                    t0 = time_ns()
                end
            end
        end
    end
    err === nothing || throw(err)
    nothing
end

# Contiguous chunk `c` of `nchunks` of the index range `1:len`
@inline function _pf_chunk(len::Int, c::Int, nchunks::Int)
    q, r = divrem(len, nchunks)
    lo = (c - 1) * q + min(c - 1, r) + 1
    hi = lo + q - 1 + (c <= r)
    return lo:hi
end

struct _PFForeach{F, I}
    f::F
    itr::I
    len::Int
    nchunks::Int
end
@inline function (x::_PFForeach)(c::Int)
    itr = x.itr
    first = firstindex(itr) - 1
    for i in _pf_chunk(x.len, c, x.nchunks)
        x.f(@inbounds itr[first + i])
    end
    nothing
end

"""
    Threads.parallel_for(f, itr; nchunks = Threads.threadpoolsize())

Call `f(x)` for every element `x` of the indexable collection `itr` (such as a range or a
vector) in parallel, and return `nothing` once all calls have finished.

The elements are split into `nchunks` contiguous chunks, which are run by the calling
task and by a pool of persistent worker tasks, one per thread of the `:default` thread pool.
Idle workers spin for a short time (100 µs, `JULIA_PARALLEL_SPIN_NS`) before parking, so
that a sequence of parallel loops does not pay for creating and scheduling tasks. The per
loop overhead is a fraction of a microsecond and the call does not allocate.

Semantics, compared to `Threads.@threads :dynamic` with the same number of chunks:

- the calling task runs (at least) one chunk itself, also when it is in the
  `:interactive` thread pool; the other chunks run on the workers of the `:default` pool;
  chunks are assigned dynamically, so do not rely on `threadid()` for a given chunk;
- each chunk runs within the dynamic scope (`ScopedValue`s) of the caller, with an empty
  `task_local_storage()` and with the task-local random number generator forked as for the
  tasks of `@threads`, but `current_task()` is the worker task (or the caller);
- exceptions are collected and thrown as a `CompositeException` of `CapturedException`s
  after all chunks finished;
- a `parallel_for` or `@threads :parked` loop called from inside the body runs serially;
  one called concurrently from another task falls back to `@threads :dynamic`;
- the body may block, yield or do I/O; a blocked chunk only blocks its own worker.

!!! compat "Julia 1.14"
    This function requires at least Julia 1.14.
"""
function parallel_for(f::F, itr; nchunks::Integer = threadpoolsize()) where {F}
    len = length(itr)
    len == 0 && return nothing
    m = min(Int(nchunks), len)
    m < 1 && throw(ArgumentError("nchunks must be positive"))
    parked_run(_PFForeach(f, itr, len, m), m)
    return nothing
end
