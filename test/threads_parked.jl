# This file is a part of Julia. License is MIT: https://julialang.org/license

# Tests of `Threads.parallel_for` and `@threads :parked` (parallel loops on parked worker
# tasks). Run by threads.jl with several thread configurations.

using Test
using Base.Threads
using Random

const NT = Threads.threadpoolsize()

# Before anything else: the first parallel loop starts the worker pool, which must not
# change the random numbers of the caller.
@testset "parallel_for RNG of the first loop" begin
    r1 = zeros(100); r2 = zeros(100)
    Random.seed!(99)
    Threads.@threads :parked for i in 1:100
        r1[i] = rand()
    end
    a1 = rand()
    Random.seed!(99)
    Threads.@threads :dynamic for i in 1:100
        r2[i] = rand()
    end
    a2 = rand()
    @test r1 == r2
    @test a1 == a2
end

@testset "parallel_for basics (threads = $(NT),$(Threads.threadpoolsize(:interactive)))" begin
    for n in (0, 1, 2, 3, NT - 1, NT, NT + 1, 64, 1000, 10007)
        n < 0 && continue
        x = zeros(Int, n)
        Threads.parallel_for(i -> (x[i] += i), 1:n)
        @test x == 1:n
        # through the macro
        y = zeros(Int, n)
        Threads.@threads :parked for i in 1:n
            y[i] += i
        end
        @test y == 1:n
        # elements of a vector, and an offset range
        v = collect(1:n) .* 2
        s = Threads.Atomic{Int}(0)
        Threads.parallel_for(x -> Threads.atomic_add!(s, x), v)
        @test s[] == sum(v)
        s[] = 0
        Threads.parallel_for(i -> Threads.atomic_add!(s, i), 5:(4 + n))
        @test s[] == sum(5:(4 + n))
    end
    # nchunks
    for nchunks in (1, 2, NT, 2NT, 8NT, 8NT + 1, 100NT)
        x = zeros(Int, 1000)
        Threads.parallel_for(i -> (x[i] += 1), 1:1000; nchunks)
        @test all(==(1), x)
    end
    @test_throws ArgumentError Threads.parallel_for(identity, 1:10; nchunks = 0)
    # comprehensions
    @test (Threads.@threads :parked [i^2 for i in 1:100]) == [i^2 for i in 1:100]
    @test (Threads.@threads :parked [i for i in 1:100 if isodd(i)]) == collect(1:2:100)
    @test (Threads.@threads :parked Float64[i for i in 1:10]) == collect(1.0:10.0)
    # many loops back to back
    x = zeros(Int, 64)
    for _ in 1:10_000
        Threads.parallel_for(i -> (@inbounds x[i] += 1), eachindex(x))
    end
    @test all(==(10_000), x)
    # all chunks run, on more than one thread if there are several
    tids = zeros(Int, 1000)
    Threads.parallel_for(i -> (tids[i] = Threads.threadid(); Libc.systemsleep(0.0001)), 1:1000)
    @test all(>(0), tids)
    if NT > 1
        @test length(unique(tids)) > 1
    end
end

@testset "parallel_for allocations" begin
    x = zeros(64)
    f(x) = Threads.parallel_for(i -> (@inbounds x[i] += 1.0), eachindex(x))
    f(x); f(x)
    # one `RefValue` holding the loop function (the count is process-wide, so take the
    # minimum of a few tries against noise from other tasks or a busy machine)
    @test minimum(_ -> (@allocations f(x)), 1:20) <= 1
end

@testset "parallel_for exceptions" begin
    for n in (1, NT, 64)
        ex = try
            Threads.parallel_for(i -> i == n && error("boom $i"), 1:n)
            nothing
        catch e
            e
        end
        # also a single chunk run on the calling task throws a CompositeException
        @test ex isa CompositeException
        @test length(ex.exceptions) == 1
        @test ex.exceptions[1] isa CapturedException
        @test ex.exceptions[1].ex == ErrorException("boom $n")
    end
    # exceptions in several chunks, the other chunks still run to completion
    # (one element per chunk; an exception ends its chunk, as for `@threads`)
    m = min(64, 8NT)
    x = zeros(Int, m)
    ex = try
        Threads.parallel_for(1:m; nchunks = m) do i
            x[i] = 1
            iseven(i) && throw(ArgumentError("bad $i"))
        end
        nothing
    catch e
        e
    end
    @test all(==(1), x)
    @test ex isa CompositeException
    @test length(ex.exceptions) == m ÷ 2
    @test all(e -> e.ex isa ArgumentError, ex.exceptions)
    # the same for a nested (serial) loop
    ex = try
        Threads.parallel_for(1:2; nchunks = 2) do _
            Threads.parallel_for(i -> iseven(i) && error("inner"), 1:4; nchunks = 4)
        end
        nothing
    catch e
        e
    end
    @test ex isa CompositeException
    @test length(ex.exceptions) == 2
    @test all(e -> e.ex isa CompositeException && length(e.ex.exceptions) == 2, ex.exceptions)
    # the pool keeps working after an exception
    y = zeros(Int, 100)
    Threads.parallel_for(i -> (y[i] = i), 1:100)
    @test y == 1:100
end

@testset "parallel_for nesting and concurrency" begin
    # nested loops run serially on the outer chunk's task
    n = 16
    A = zeros(Int, n, n)
    Threads.parallel_for(1:n) do i
        t = current_task()
        Threads.parallel_for(1:n) do j
            @test current_task() === t
            A[i, j] += 1
        end
        Threads.@threads :parked for j in 1:n
            A[i, j] += 1
        end
    end
    @test all(==(2), A)
    # @threads and @spawn inside the body
    B = zeros(Int, n, n)
    Threads.parallel_for(1:n) do i
        Threads.@threads for j in 1:n
            B[i, j] += 1
        end
        fetch(Threads.@spawn (B[i, 1] += 1))
    end
    @test sum(B) == n * n + n
    # concurrent calls from several tasks: one uses the pool, the others fall back
    m = 8
    C = zeros(Int, 1000, m)
    @sync for k in 1:m
        Threads.@spawn for _ in 1:20
            Threads.parallel_for(i -> (C[i, k] += 1), 1:1000)
        end
    end
    @test all(==(20), C)
end

@testset "parallel_for body blocks, yields, does IO" begin
    n = 4 * max(NT, 2)
    # yield and sleep
    x = zeros(Int, n)
    Threads.parallel_for(1:n) do i
        yield()
        sleep(0.001)
        x[i] = i
    end
    @test x == 1:n
    # wait on a task spawned inside the body
    y = zeros(Int, n)
    Threads.parallel_for(1:n) do i
        y[i] = fetch(Threads.@spawn i * 2)
    end
    @test y == 2 .* (1:n)
    # @async task sticky to the worker's thread
    z = zeros(Int, n)
    Threads.parallel_for(1:n) do i
        z[i] = fetch(@async i + 1)
    end
    @test z == (1:n) .+ 1
    # chunks that communicate through a channel (needs all chunks to run concurrently,
    # or the producer to finish first; with stealing, a chunk that is not started runs on
    # whichever participant is free)
    if NT >= 2
        ch = Channel{Int}(Inf)
        Threads.parallel_for(1:2; nchunks = 2) do i
            if i == 1
                foreach(k -> put!(ch, k), 1:10)
                close(ch)
            else
                @test sum(ch) == 55
            end
        end
    end
    # IO: write to a pipe and a file
    mktemp() do path, io
        lk = ReentrantLock()
        Threads.parallel_for(1:n) do i
            @lock lk println(io, i)
        end
        close(io)
        @test sort(parse.(Int, readlines(path))) == 1:n
    end
    # a timer / event that is notified by another task
    ev = Base.Event()
    t = Threads.@spawn (sleep(0.01); notify(ev))
    Threads.parallel_for(i -> wait(ev), 1:n)
    wait(t)
    @test ev.set
end

@testset "parallel_for GC in the body" begin
    n = 2 * max(NT, 2)
    refs = Vector{Any}(undef, n)
    Threads.parallel_for(1:n) do i
        refs[i] = [i]
        GC.gc(i == 1)
        refs[i] = (refs[i], Ref(i))
    end
    @test all(i -> refs[i][1] == [i] && refs[i][2][] == i, 1:n)
    # allocation-heavy body to trigger GC from several threads
    s = Threads.Atomic{Int}(0)
    Threads.parallel_for(1:10_000) do i
        v = [i, i]
        Threads.atomic_add!(s, sum(v) ÷ 2)
    end
    @test s[] == sum(1:10_000)
end

const PF_SV = Base.ScopedValues.ScopedValue(0)

@testset "parallel_for task semantics" begin
    # dynamic scope of the caller
    vals = zeros(Int, 64)
    Base.ScopedValues.with(PF_SV => 42) do
        Threads.parallel_for(i -> (vals[i] = PF_SV[]), 1:64)
    end
    @test all(==(42), vals)
    Threads.parallel_for(i -> (vals[i] = PF_SV[]), 1:64)
    @test all(==(0), vals)
    # empty task-local storage in every chunk, caller's storage restored
    # (one element per chunk)
    task_local_storage(:pf_key, :outer)
    m = min(64, 8NT)
    seen = Vector{Any}(undef, m)
    Threads.parallel_for(1:m; nchunks = m) do i
        seen[i] = get(task_local_storage(), :pf_key, nothing)
        task_local_storage(:pf_key, i)
    end
    @test all(isnothing, seen)
    @test task_local_storage(:pf_key) === :outer
    Threads.parallel_for(i -> (seen[i] = get(task_local_storage(), :pf_key, nothing)), 1:m; nchunks = m)
    @test all(isnothing, seen)
    # random numbers: reproducible and the same as with `@threads :dynamic`
    r1 = zeros(1000); r2 = zeros(1000); r3 = zeros(1000)
    Random.seed!(1234)
    Threads.@threads :parked for i in 1:1000
        r1[i] = rand()
    end
    a1 = rand()
    Random.seed!(1234)
    Threads.@threads :parked for i in 1:1000
        r2[i] = rand()
    end
    a2 = rand()
    Random.seed!(1234)
    Threads.@threads :dynamic for i in 1:1000
        r3[i] = rand()
    end
    a3 = rand()
    @test r1 == r2 == r3
    @test a1 == a2 == a3
    # chunks of a pool region run on the pool's threads or the caller's
    tps = Vector{Symbol}(undef, 64)
    Threads.parallel_for(i -> (tps[i] = Threads.threadpool()), 1:64)
    @test all(tp -> tp === :default || tp === Threads.threadpool(), tps)
    # called from a thread of the default pool
    z = fetch(Threads.@spawn :default begin
        w = zeros(Int, 100)
        Threads.parallel_for(i -> (w[i] = i), 1:100)
        w
    end)
    @test z == 1:100
    # called from the interactive pool, if there is one
    if Threads.threadpoolsize(:interactive) > 0
        z = fetch(Threads.@spawn :interactive begin
            w = zeros(Int, 100)
            Threads.parallel_for(i -> (w[i] = i), 1:100)
            w
        end)
        @test z == 1:100
    end
end

@testset "parallel_for after idle (workers parked)" begin
    x = zeros(Int, 64)
    for _ in 1:3
        sleep(0.01)  # longer than the spin time: all workers park
        Threads.parallel_for(i -> (x[i] += 1), 1:64)
    end
    @test all(==(3), x)
end

pf_world_g() = 1
function pf_world_h()
    r = zeros(Int, 64)
    Threads.parallel_for(i -> (r[i] = pf_world_g()), 1:64; nchunks = min(64, 8NT))
    return r
end

@testset "parallel_for world age" begin
    @test all(==(1), pf_world_h())
    w = Base.get_world_counter()
    @eval pf_world_g() = 2
    # all chunks run in the caller's world, as the tasks of `@threads` do
    @test all(==(1), Base.invoke_in_world(w, pf_world_h))
    @test all(==(2), pf_world_h())
end
