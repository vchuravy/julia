# Exercises JIT code unloading. Run with JULIA_JIT_UNLOAD=1.
using Test

function unload_stats()
    s = zeros(Csize_t, 12)
    ccall(:jl_jit_unload_stats, Cvoid, (Ptr{Csize_t}, Csize_t), s, length(s))
    return (; enabled=Bool(s[1]), created=Int(s[2]), live=Int(s[3]), freed=Int(s[4]),
            bytes_freed=Int(s[5]), pending=Int(s[6]), cis_died=Int(s[7]),
            kept_stack=Int(s[8]), kept_dep=Int(s[9]), codegcs=Int(s[10]),
            reused=Int(s[11]), avail=Int(s[12]))
end

min_live_world() = ccall(:jl_min_live_world, Csize_t, ())

# Evict CodeInstances that are not valid in `world` or later, then collect.
function collect_code(world = Base.get_world_counter())
    evicted = ccall(:jl_evict_stale_code, Csize_t, (Csize_t,), world)
    GC.gc(true); GC.gc(true)
    released = ccall(:jl_jit_release_dead_code, Csize_t, ())
    return (; evicted=Int(evicted), released=Int(released))
end

@show unload_stats()
@test unload_stats().enabled
@show Int(Base.get_world_counter()), Int(min_live_world())

println("== 1. toplevel thunks")
s0 = unload_stats(); b0 = Base.jit_total_bytes()
for i in 1:200
    # a loop forces the thunk to be compiled rather than interpreted
    @eval begin
        local acc = 0
        for j in 1:$i
            acc += j
        end
        @assert acc == $(i * (i + 1) ÷ 2)
    end
end
r = collect_code()
s1 = unload_stats(); b1 = Base.jit_total_bytes()
@show r
@show s1.created - s0.created, s1.freed - s0.freed, b0, b1
@test s1.freed - s0.freed >= 150

println("== 2. redefined methods")
s0 = unload_stats(); b0 = Base.jit_total_bytes()
for i in 1:200
    @eval f2(x) = x + $i
    @eval g2(x) = f2(x) * 2
    @assert Base.invokelatest(g2, 1) == (1 + i) * 2
end
b_before = Base.jit_total_bytes()
r = collect_code()
s1 = unload_stats(); b1 = Base.jit_total_bytes()
@show r
@show s1.created - s0.created, s1.freed - s0.freed, b0, b_before, b1
@test r.evicted >= 190
@test s1.freed - s0.freed >= 190
@test b1 < b_before
@test g2(1) == 402
@test f2(1) == 201

println("== 3. memory is reused, and new code at old addresses works")
s0 = unload_stats()
for i in 1:100
    @eval h3(x) = x * $i + 1
    @assert Base.invokelatest(h3, 2) == 2i + 1
end
s1 = unload_stats()
@show s1.reused - s0.reused, s1.avail
@test s1.reused > s0.reused
# backtraces through code at a reused address
@eval thrower3(x) = x > 0 ? error("boom $x") : x
bt = try
    thrower3(1)
catch
    stacktrace(catch_backtrace())
end
@test any(fr -> fr.func === :thrower3, bt)

println("== 4. code that is still running is not freed")
@eval slow4(c1, c2) = (put!(c1, :started); x = take!(c2); x + 4000)
@eval outer4(c1, c2) = slow4(c1, c2) + 1
c1 = Channel{Any}(1); c2 = Channel{Any}(1)
t = Threads.@spawn outer4(c1, c2)
@test take!(c1) === :started
# invalidate both functions while the task is suspended inside them
@eval slow4(c1, c2) = :new
@eval outer4(c1, c2) = :new
s0 = unload_stats()
# evict everything that is not valid in the latest world, even though the task
# is still running in an older one
r = collect_code(Base.get_world_counter())
s1 = unload_stats()
@show r
@show s1.kept_stack - s0.kept_stack, s1.kept_dep - s0.kept_dep
@test s1.kept_stack > s0.kept_stack
for _ in 1:3
    collect_code(Base.get_world_counter())
end
put!(c2, 42)
@test fetch(t) == 4043
@test outer4(c1, c2) === :new
t = nothing
r = collect_code(Base.get_world_counter())
@show r
@show unload_stats()

println("== 5. eviction of code that is still needed just recompiles")
@eval k5(x) = x + 5
@eval caller5(x) = k5(x) + 1
w_old = Base.get_world_counter()
@test caller5(1) == 7
@eval k5(x) = x + 50
@test caller5(1) == 52
collect_code(Base.get_world_counter())
# the old world's code is gone, but the old world still works
@test Base.invoke_in_world(w_old, caller5, 1) == 7
@test caller5(1) == 52

println("== 7. code that is never unloaded keeps what it calls")
@eval @noinline callee7(x) = (Base.donotdelete(x); x + 7)
oc7 = Base.Experimental.@opaque (x::Int) -> callee7(x) + 1
@test oc7(1) == 9
@eval viacf7(x::Int) = callee7(x) + 2
cf7 = @cfunction(viacf7, Int, (Int,))
@test ccall(cf7, Int, (Int,), 1) == 10
@eval @noinline callee7(x) = (Base.donotdelete(x); x + 700)
for _ in 1:3
    collect_code()
end
# refill the freed memory with other code
for i in 1:50
    @eval filler7(x) = x - $i
    @assert Base.invokelatest(filler7, 0) == -i
end
@test oc7(1) == 9                          # runs in the world it was created in
@test ccall(cf7, Int, (Int,), 1) == 703    # runs in the latest world

println("== 8. backtraces of tasks suspended in unloadable code")
@eval inner8(c) = (take!(c); error("from inner8"))
@eval outer8(c) = inner8(c)
c8 = Channel{Any}(1)
t8 = Threads.@spawn outer8(c8)
while !istaskstarted(t8) || t8.state != :runnable || isready(c8); yield(); end
sleep(0.1)
@eval inner8(c) = :gone
@eval outer8(c) = :gone
for _ in 1:3
    collect_code()
end
put!(c8, 1)
err8 = try fetch(t8) catch e; e end
@test err8 isa TaskFailedException
bt8 = stacktrace(t8.backtrace)
@show bt8[1:min(end, 4)]
@test any(fr -> fr.func === :inner8, bt8)
@test any(fr -> fr.func === :outer8, bt8)
t8 = err8 = bt8 = nothing

println("== 6. churn with several threads")
function churn(n)
    for i in 1:n
        m = Symbol(:churn_, Threads.threadid(), :_, i % 7)
        @eval $m(x) = x + $i
        @assert Base.invokelatest(Base.invokelatest(getglobal, @__MODULE__, m), 1) == 1 + i
    end
end
s0 = unload_stats(); b0 = Base.jit_total_bytes()
for round in 1:5
    ts = [Threads.@spawn churn(100) for _ in 1:Threads.nthreads()]
    foreach(wait, ts)
    global ts = nothing
    @show collect_code()
end
s1 = unload_stats(); b1 = Base.jit_total_bytes()
@show s1.created - s0.created, s1.freed - s0.freed, b0, b1
@show unload_stats()
let n = Ref{UInt64}(), b = Ref{UInt64}(), t = Ref{UInt64}()
    ccall(:jl_gc_code_scan_stats, Cvoid, (Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}), n, b, t)
    println("stack scans: ", Int(n[]), ", ", round(b[] / 2^20, digits=1), " MB scanned, ", round(t[] / 1e6, digits=2), " ms total")
end
println("done")
