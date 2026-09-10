# This file is a part of Julia. License is MIT: https://julialang.org/license

# Stress test for the stackmap GC-root mode (--experimental --gc-roots=stackmap
# or --gc-roots=both). Run in a fresh process by test/gc.jl with those flags;
# every scenario keeps boxed values live across allocations, safepoint polls,
# gc-safe foreign calls, callbacks, task switches and non-local exits, and
# forces collections in between. Missing roots show up as corrupted values or
# crashes.

using Test

mutable struct Node
    val::Int
    next::Union{Node,Nothing}
end

@noinline function chain_sum(n::Node)
    s = 0
    x = n
    while x !== nothing
        s += x.val
        x = x.next
    end
    return s
end

# --- deep recursion with several live roots per frame ----------------------------------
@noinline function deep(n, acc)
    a = Node(n, acc)
    b = [n, n + 1]
    if n == 0
        GC.gc(true)
        return chain_sum(a) + length(b)
    end
    r = deep(n - 1, a) # a and b are live across the call
    GC.safepoint()
    return r + a.val + b[1]
end

@testset "deep recursion" begin
    N = 1500
    expected = sum(0:N) + 2 + sum(2 * (1:N))
    @test deep(N, nothing) == expected
end

# --- many live roots in a single frame (spill slots) -----------------------------------
@noinline function many_roots(k)
    a = Node(1, nothing); b = Node(2, a); c = Node(3, b); d = Node(4, c)
    e = Node(5, d); f = Node(6, e); g = Node(7, f); h = Node(8, g)
    i = Node(9, h); j = Node(10, i); l = Node(11, j); m = Node(12, l)
    GC.gc(false)
    GC.gc(true)
    s = chain_sum(m) + k
    GC.gc(false)
    return s + a.val + b.val + c.val + d.val + e.val + f.val + g.val + h.val + i.val + j.val + l.val + m.val
end

@testset "many roots" begin
    @test many_roots(0) == 78 + 78
end

# --- gc-safe foreign call while another thread collects --------------------------------
@noinline function gcsafe_sleep(v)
    x = Node(v, nothing) # live across the gc-safe call
    y = [v]
    @ccall gc_safe=true usleep(200::Cuint)::Cint
    return x.val + y[1]
end

@testset "gc-safe ccall" begin
    stop = Threads.Atomic{Bool}(false)
    collector = Threads.@spawn begin
        while !stop[]
            GC.gc(false)
            yield()
        end
    end
    s = 0
    for i in 1:200
        s += gcsafe_sleep(i)
    end
    stop[] = true
    wait(collector)
    @test s == 2 * sum(1:200)
end

# --- callbacks from foreign code (frames without frame pointers) -----------------------
function qsort_cmp(pa::Ptr{Int}, pb::Ptr{Int})
    tmp = Node(unsafe_load(pa), nothing) # allocate in the callback
    if tmp.val % 7 == 0
        GC.gc(false)
    end
    a = tmp.val
    b = unsafe_load(pb)
    return Cint(a < b ? -1 : a > b ? 1 : 0)
end

@noinline function sort_via_qsort(v::Vector{Int})
    keep = Node(length(v), nothing) # live across the foreign call
    cb = @cfunction(qsort_cmp, Cint, (Ptr{Int}, Ptr{Int}))
    GC.@preserve v begin
        @ccall qsort(pointer(v)::Ptr{Int}, length(v)::Csize_t, sizeof(Int)::Csize_t, cb::Ptr{Cvoid})::Cvoid
    end
    return keep.val
end

@testset "cfunction callbacks" begin
    v = collect(reverse(1:500))
    @test sort_via_qsort(v) == 500
    @test issorted(v)
end

# --- tasks yielding with live roots ------------------------------------------------------
@noinline function task_body(i)
    a = Node(i, nothing)
    b = Node(i + 1, a)
    yield()
    GC.gc(false)
    c = Node(i + 2, b)
    yield()
    return chain_sum(c)
end

@testset "tasks" begin
    tasks = [Threads.@spawn task_body(i) for i in 1:64]
    GC.gc(true)
    @test sum(fetch.(tasks)) == sum(3i + 3 for i in 1:64)
end

# --- non-local exits with roots live in the handler ----------------------------------------
@noinline function thrower(x)
    Node(x, nothing).val == -1 && return 0
    throw(ErrorException("boom $x"))
end

@noinline function catcher(n)
    keep = Node(n, nothing) # live across the setjmp/longjmp
    r = 0
    try
        thrower(n)
    catch e
        GC.gc(true)
        r = keep.val + length(e.msg)
    end
    return r
end

@testset "try/catch" begin
    @test catcher(5) == 5 + length("boom 5")
end

# --- stack overflow recovery then collection --------------------------------------------
@noinline function recurse_forever(n)
    a = Node(n, nothing)
    return recurse_forever(n + 1) + a.val
end

@testset "stack overflow" begin
    keep = Node(42, nothing)
    caught = try
        recurse_forever(1)
        false
    catch e
        e isa StackOverflowError
    end
    GC.gc(true)
    @test caught
    @test keep.val == 42
end

# --- interpreter frames mixed in ---------------------------------------------------------
@testset "interpreter" begin
    keep = Node(7, nothing)
    r = Base.invokelatest(() -> begin
        x = Node(1, keep)
        GC.gc(true)
        chain_sum(x)
    end)
    @test r == 8
end

# --- allocation heavy loop with collections ----------------------------------------------
@noinline function alloc_loop(n)
    head = nothing
    for i in 1:n
        head = Node(i, head)
        if i % 5000 == 0
            GC.gc(false)
        end
    end
    return chain_sum(head)
end

@testset "allocation loop" begin
    @test alloc_loop(100_000) == sum(1:100_000)
end
