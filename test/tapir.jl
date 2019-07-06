using Libdl
dlopen("libcilkrts", Libdl.RTLD_GLOBAL)

nworkers() = ccall(:__cilkrts_get_nworkers, Cint, ())

"""
Call `end_cilk()` before you set the number of workers
"""
set_nworkers(N) = ccall(:__cilkrts_set_param, Cint, (Cstring, Cstring), "nworkers", string(N)) == 0

end_cilk() = ccall(:__cilkrts_end_cilk, Cvoid, ())
init_cilk() = ccall(:__cilkrts_init, Cvoid, ())

macro syncregion()
    Expr(:syncregion)
end

macro spawn(token, expr)
    Expr(:spawn, esc(token), esc(expr))
end

macro sync_end(token)
    Expr(:sync, esc(token))
end

macro loopinfo(args...)
    Expr(:loopinfo, args...)
end

const tokenname = gensym(:token)
macro sync(block)
    var = esc(tokenname)
    quote
        let $var = @syncregion()
            $(esc(block))
            @sync_end($var)
        end
    end
end

macro spawn(expr)
    var = esc(tokenname)
    quote
        @spawn $var $(esc(expr))
    end
end

macro par(expr)
    @assert expr.head === :for
    token = gensym(:token)
    body = expr.args[2]
    lhs = expr.args[1].args[1]
    range = expr.args[1].args[2]
    quote
        let $token = @syncregion()
            for $(esc(lhs)) = $(esc(range))
                @spawn $token $(esc(body))
                $(Expr(:loopinfo, (Symbol("tapir.loop.spawn.strategy"), 1)))
            end
            @sync_end $token
        end
    end
end

function f()
    let token = @syncregion()
        @spawn token begin
            1 + 1
        end
        @sync_end token
    end
end

function taskloop(N)
    let token = @syncregion()
        for i in 1:N
            @spawn token begin
                1 + 1
            end
        end
        @sync_end token
    end
end

function taskloop2(N)
    @sync for i in 1:N
        @spawn begin
            1 + 1
        end
    end
end

function taskloop3(N)
    @par for i in 1:N
        1+1
    end
end

function vecadd(out, A, B)
    @assert length(out) == length(A) == length(B)
    @inbounds begin
        @par for i in 1:length(out)
            out[i] = A[i] + B[i]
        end
    end
    return out
end

function fib(N)
    if N <= 1
        return N
    end
    token = @syncregion()
    x1 = Ref{Int64}()
    @spawn token begin
        x1[]  = fib(N-1)
    end
    x2 = fib(N-2)
    @sync_end token
    return x1[] + x2
end

###
# Interesting corner cases and broken IR
###

##
# Parallel regions with errors are tricky
# #1  detach within %sr, #2, #3
# #2  ...
#     unreachable()
#     reattach within %sr, #3
# #3  sync within %sr
#
# Normally a unreachable get's turned into a ReturnNode(),
# but that breaks the CFG. So we need to detect that we are
# in a parallel region.
#
# Question:
#   - Can we elimante a parallel region that throws?
#     Probably if the sync is dead as well. We could always
#     use the serial projection and serially execute the region.

function vecadd_err(out, A, B)
    @assert length(out) == length(A) == length(B)
    @inbounds begin
        @par for i in 1:length(out)
            out[i] = A[i] + B[i]
            error()
        end
    end
    return out
end

# This function is broken due to the PhiNode
@noinline function fib2(N)
    if N <= 1
        return N
    end
    token = @syncregion()
    x1 = 0
    @spawn token begin
        x1  = fib2(N-1)
    end
    x2 = fib2(N-2)
    @sync_end token
    return x1 + x2
end

# Spindle examples
# Use this for spindle analysis
@noinline function fib3(n)
    rv = n
    x = Ref{Int}()
    token = @syncregion()
    if (n > 2)
        @spawn token begin
          x[] = fib3(n-1)
        end
        y = fib3(n-2)
        @sync_end token
        rv = y + x[]
    end
    return rv
end

function tapir_2D(C, A, B)
    outer = @syncregion()
    for i in 1:size(A, 1) 
        @spawn outer begin
            inner = @syncregion()
            for j in 1:size(B, 2)
                @spawn inner begin
                    for k in 1:size(B, 1)
                        C[i, j] += A[i, k] * B[k, j]
                    end
                end
                @sync_end outer
            end
        end
    end
    @sync_end outer
end
