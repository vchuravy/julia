# This file is a part of Julia. License is MIT: https://julialang.org/license

# Abstract types with fields: `abstract type A; x::Int; end` declares fields
# that every subtype inherits (before its own fields).

using Test, InteractiveUtils

module AbstractFieldsTest

using Test, InteractiveUtils

abstract type AF_P{T}
    x::T
    AF_P{T}(x) where {T} = (x = x,)
end

struct AF_Q{T} <: AF_P{Vector{T}}
    y::T
end

@testset "declaration and reflection" begin
    @test isabstracttype(AF_P) && !isstructtype(AF_P)
    @test fieldnames(AF_P) == (:x,)
    @test fieldnames(AF_P{Int}) == (:x,)
    @test fieldtypes(AF_P{Int}) == (Int,)
    @test fieldtype(AF_P{Int}, :x) === Int
    @test fieldtype(AF_P{Int}, 1) === Int
    @test hasfield(AF_P, :x) && !hasfield(AF_P, :y)
    @test fieldname(AF_P{Int}, 1) === :x
    @test fieldindex(AF_P{Int}, :x, false) == 1
    @test_throws ArgumentError fieldindex(AF_P{Int}, :x)
    @test_throws ArgumentError fieldcount(AF_P)
    @test_throws ArgumentError fieldcount(AF_P{Int})
    @test AF_P{Int}.name.n_inherited == 0
    @test AF_P.body.types == Core.svec(AF_P.body.parameters[1])

    # abstract types without fields still report no fields
    @test fieldnames(Real) == ()
    @test fieldtypes(Real) == ()
    @test !hasfield(Real, :x)
end

@testset "inheritance into a struct" begin
    @test fieldnames(AF_Q) == (:x, :y)
    @test fieldtypes(AF_Q{Int}) == (Vector{Int}, Int)
    @test AF_Q{Int}.name.n_inherited == 1
    @test fieldcount(AF_Q{Int}) == 2
    @test !Base.issingletontype(AF_Q{Int})
    q = AF_Q([1], 2)
    @test q isa AF_Q{Int}
    @test q.x == [1] && q.y == 2
    @test getfield(q, :x) == [1] && getfield(q, 1) == [1]
    @test AF_Q{Int}([1], 2).y == 2
    @test_throws MethodError AF_Q(1, 2)
    @test isconst(AF_Q{Int}, :x) && isconst(AF_Q{Int}, :y)
    @test endswith(sprint(dump, AF_Q{Int}), "AF_P{Vector{Int64}}\n  x::Vector{Int64}\n  y::Int64\n")
end

@testset "parent constructors" begin
    @test AF_P{Int}(3) === (x = 3,)
    @test AF_P{Float64}(3) === (x = 3.0,)
    @test_throws MethodError AF_P{Int}("a")
    @test_throws MethodError AF_P(3)
end

struct AF_R <: AF_P{Int}
    z::Int
    AF_R(x, z) = new(; AF_P{Int}(x)..., z)
    AF_R(z) = new(; z, x = 0)
    AF_R(z, ::Val{:extra}) = new(; z, x = 0, w = 1)
    AF_R(z, ::Val{:missing}) = new(; z)
    AF_R(x, z, ::Val{:positional}) = new(x, z)
    AF_R(x, z, ::Val{:splat}) = new((x, z)...)
    AF_R(x, z, ::Val{:many}) = new(x, z, 1)
    AF_R(x, z, ::Val{:mixed}) = new(x, z, 1.0)
end

@testset "keyword and positional new with inherited fields" begin
    r = AF_R(1, 2)
    @test r.x == 1 && r.z == 2
    @test r === AF_R(1, 2, Val(:positional)) === AF_R(1, 2, Val(:splat))
    @test AF_R(5) === AF_R(0, 5)
    @test_throws ArgumentError AF_R(1, Val(:extra))
    @test_throws ArgumentError AF_R(1, Val(:missing))
    # too many positional arguments fail on the `fieldtype` lookup of the extra one
    @test_throws BoundsError AF_R(1, 2, Val(:many))
    @test_throws BoundsError AF_R(1, 2, Val(:mixed))
    @test AF_R(1.0, 2) === AF_R(1, 2)     # conversion through fieldtype
    @test_throws InexactError AF_R(1.5, 2)
    # the keyword form folds to a plain allocation
    src = only(code_typed(AF_R, (Int, Int)))[1]
    @test !any(x -> Meta.isexpr(x, :splatnew), src.code)
    @test any(x -> Meta.isexpr(x, :new), src.code)
end

abstract type AF_Mut
    const k::Int
    @atomic a::Int
    w
end

mutable struct AF_M <: AF_Mut
    m::Int
end

@testset "const and atomic declared by an abstract type" begin
    @test isconst(AF_Mut, :k) && !isconst(AF_Mut, :w)
    @test Base.isfieldatomic(AF_Mut, :a) && !Base.isfieldatomic(AF_Mut, :k)
    @test fieldnames(AF_M) == (:k, :a, :w, :m)
    @test isconst(AF_M, :k) && !isconst(AF_M, :w) && !isconst(AF_M, :m)
    @test Base.isfieldatomic(AF_M, :a) && !Base.isfieldatomic(AF_M, :m)
    m = AF_M(1, 2, 3, 4)
    @test_throws ErrorException m.k = 9
    m.w = 9
    @test m.w == 9
    @test (@atomic m.a) == 2
    @atomic m.a = 7
    @test (@atomic m.a) == 7
    @test_throws ConcurrencyViolationError m.a = 1
    @test occursin("const k::Int", sprint(dump, AF_Mut))
    @test occursin("@atomic a::Int", sprint(dump, AF_Mut))
    # an immutable struct cannot inherit an atomic field
    @test_throws ErrorException @eval struct AF_Imm <: AF_Mut end
    # but a mutable one may leave its own trailing fields uninitialized
    @test_throws ErrorException @eval mutable struct AF_Bad <: AF_Mut
        m::Int
        AF_Bad() = new()
    end
    @eval mutable struct AF_Ok <: AF_Mut
        m
        AF_Ok(k, a, w) = new(k, a, w)
    end
    @test !isdefined(AF_Ok(1, 2, 3), :m)
    @test isdefined(AF_Ok(1, 2, 3), :w)
end

abstract type AF_Base
    b::Int
end
abstract type AF_Mid <: AF_Base
    m::String
end
struct AF_Leaf <: AF_Mid
    l::Float64
end
struct AF_Empty <: AF_Mid end
primitive type AF_Prim 8 end

@testset "chains of abstract types" begin
    @test fieldnames(AF_Mid) == (:b, :m)
    @test AF_Mid.name.n_inherited == 1
    @test fieldtypes(AF_Mid) == (Int, String)
    @test fieldnames(AF_Leaf) == (:b, :m, :l)
    @test fieldtypes(AF_Leaf) == (Int, String, Float64)
    @test AF_Leaf(1, "a", 2.0).m == "a"
    @test fieldnames(AF_Empty) == (:b, :m)
    @test !Base.issingletontype(AF_Empty)
    @test AF_Empty(1, "x").b == 1
    @test sizeof(AF_Empty) == 16
    @test_throws ErrorException @eval primitive type AF_Prim2 <: AF_Base 8 end
end

@testset "name clashes" begin
    @test_throws ErrorException @eval struct AF_Clash <: AF_Base
        b::Int
    end
    @test_throws ErrorException @eval abstract type AF_Clash2 <: AF_Mid
        m
    end
end

@testset "redefinition" begin
    @eval abstract type AF_Re; x::Int; end
    T1 = AF_Re
    @eval abstract type AF_Re; x::Int; end
    @test AF_Re === T1
    @eval abstract type AF_Re; x::Float64; end
    @test AF_Re !== T1
    @test fieldtypes(AF_Re) == (Float64,)
    @eval abstract type AF_Re; x::Float64; y; end
    @test fieldnames(AF_Re) == (:x, :y)
end

"""
    AF_Doc

An abstract type with documented fields.
"""
abstract type AF_Doc
    "the x"
    x::Int
    "the y"
    y
end

@testset "field docstrings" begin
    fdocs = Base.Docs.meta(@__MODULE__)[Base.Docs.Binding(@__MODULE__, :AF_Doc)].docs[Union{}].data[:fields]
    @test fdocs[:x] == "the x" && fdocs[:y] == "the y"
    @test fieldnames(AF_Doc) == (:x, :y)
end

getx(p) = p.x
getx_nothrow(p::AF_P{Int}) = getfield(p, :x)
setx!(m::AF_Mut, v) = (m.w = v; m)

@testset "inference" begin
    @test Base.infer_return_type(getx, (AF_P{Int},)) === Int
    @test Base.infer_return_type(getx, (AF_P{Float64},)) === Float64
    @test Base.infer_return_type(getx, (AF_P,)) !== Union{}
    @test Base.infer_return_type(getx, (AF_Mid,)) === Any    # `x` is not declared by AF_Mid
    @test Base.infer_return_type(p -> p.m, (AF_Mid,)) === String
    @test Base.infer_return_type(p -> p.b, (AF_Mid,)) === Int
    @test Core.Compiler.is_nothrow(Base.infer_effects(getx_nothrow, (AF_P{Int},)))
    @test Base.infer_return_type(p -> isdefined(p, :x), (AF_P{Int},)) === Bool
    @test only(Base.return_types(p -> isdefined(p, :x), (AF_P{Int},))) === Bool
    @test Base.infer_return_type(p -> fieldtype(typeof(p), :x), (AF_P{Int},)) <: Type{Int}
    @test Base.infer_return_type(setx!, (AF_M, Int)) === AF_M
    @test Base.infer_return_type(() -> fieldtype(AF_P{Int}, :x), ()) <: Type{Int}
    # a declared `const` field cannot be assigned
    @test Base.infer_return_type(m -> (m.k = 1; m), (AF_Mut,)) === Union{}
    # generated code accesses declared fields by name on the runtime type
    llvm = sprint(code_llvm, getx, (AF_P{Int},))
    @test occursin("jl_field_index", llvm)
    @test !occursin("jl_f_getfield", llvm)
    @test getx(AF_R(3, 4)) == 3
end

end # module
