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

@testset "name clashes and refinement" begin
    # a redeclaration must be a subtype of the declared type
    @test_throws ErrorException @eval struct AF_Clash <: AF_Base
        b::String
    end
    @test_throws ErrorException @eval abstract type AF_Clash2 <: AF_Mid
        m::Int
    end
    # a redeclaration cannot add attributes
    @test_throws ErrorException @eval mutable struct AF_Clash3 <: AF_Base
        const b::Int
    end
    # covariant refinement: the field keeps its inherited slot and narrows
    @eval abstract type AF_Num; v::Number; tag::Symbol; end
    @eval struct AF_NumInt <: AF_Num; v::Int; end
    @test fieldnames(AF_NumInt) == (:v, :tag)
    @test fieldtypes(AF_NumInt) == (Int, Symbol)
    @test AF_NumInt.name.n_inherited == 2
    @test AF_NumInt(1, :a).v === 1
    @test_throws InexactError AF_NumInt(1.5, :a)
    @test Base.infer_return_type(x -> x.v, (AF_Num,)) === Number
    @test Base.infer_return_type(x -> x.v, (AF_NumInt,)) === Int
    @eval abstract type AF_NumReal <: AF_Num; v::Real; end   # an abstract type may refine too
    @test fieldtypes(AF_NumReal) == (Real, Symbol)
    @eval struct AF_NumF <: AF_NumReal; v::Float64; end
    @test fieldtypes(AF_NumF) == (Float64, Symbol)
    @test_throws ErrorException @eval struct AF_NumBad <: AF_NumReal; v::Complex{Int}; end
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
    # while the fields of AF_P{Int} are a stable prefix of every subtype's
    # fields, generated code reads them at a fixed offset; otherwise by name
    llvm = sprint(code_llvm, getx, (AF_P{Int},))
    @test !occursin("jl_field_index", llvm)
    @test !occursin("jl_f_getfield", llvm)
    @test getx(AF_R(3, 4)) == 3
end

stable_prefix(T) = ccall(:jl_typename_stable_field_prefix, Cint, (Any,), Base.unwrap_unionall(T).name) != 0

abstract type AF_S1; s1::Int; end
abstract type AF_S2; s2::Int; end
struct AF_S1Only <: AF_S1; extra::Int; end
gets1(x::AF_S1) = x.s1

@testset "stable field prefix" begin
    @test stable_prefix(AF_P) && stable_prefix(AF_Base) && stable_prefix(AF_Mid) && stable_prefix(AF_S1)
    @test !stable_prefix(Real)  # no fields
    @test gets1(AF_S1Only(1, 2)) == 1
    @test !occursin("jl_field_index", sprint(code_llvm, gets1, (AF_S1,)))
    # the inferred code records its dependency on the prefix
    ci = Base.code_typed(gets1, (AF_S1,))[1]
    mi = Base.method_instance(gets1, (AF_S1Only,))
    # a subtype whose fields start with another ancestor's fields breaks AF_S1's prefix
    @eval struct AF_S21 <: (AF_S2, AF_S1); c::Int; end
    @test fieldnames(AF_S21) == (:s1, :s2, :c)   # reversed linearization: AF_S1's field first
    @test stable_prefix(AF_S1) && !stable_prefix(AF_S2)
    @test gets1(AF_S21(1, 2, 3)) == 1
    @eval struct AF_S12 <: (AF_S1, AF_S2); c::Int; end
    @test fieldnames(AF_S12) == (:s2, :s1, :c)
    @test !stable_prefix(AF_S1)
    @test gets1(AF_S12(1, 2, 3)) == 2 && gets1(AF_S1Only(1, 2)) == 1 && gets1(AF_S21(1, 2, 3)) == 1
    @test occursin("jl_field_index", sprint(code_llvm, gets1, (AF_S1,)))
    # a single-parent subtype never breaks a prefix
    @eval abstract type AF_S3; s3::Int; end
    @eval struct AF_S3a <: AF_S3; a::Int; end
    @eval abstract type AF_S3b <: AF_S3; b::Int; end
    @eval struct AF_S3c <: AF_S3b; c::Int; end
    @test stable_prefix(AF_S3) && stable_prefix(AF_S3b)
    # refining a field to a different storage breaks the prefix of the ancestors holding it
    @eval abstract type AF_S4; v::Number; w::Int; end
    @eval abstract type AF_S4a <: AF_S4; a::Int; end
    @test stable_prefix(AF_S4) && stable_prefix(AF_S4a)
    @eval struct AF_S4b <: AF_S4a; v::Int; end
    @test !stable_prefix(AF_S4) && !stable_prefix(AF_S4a)
    @eval abstract type AF_S5; v::Number; end
    @eval struct AF_S5b <: AF_S5; v::Real; end   # both stored as pointers: the prefix holds
    @test stable_prefix(AF_S5)
end

mutable abstract type AF_MA
    count::Int
    const id::Int
end
mutable struct AF_MA1 <: AF_MA
    extra::Int
end
abstract type AF_MA2 <: AF_MA end
mutable struct AF_MA3 <: AF_MA2 end
bump!(m::AF_MA) = (m.count += 1; m.count)

@testset "mutable abstract type" begin
    @test ismutabletype(AF_MA) && ismutabletype(AF_MA2) && ismutabletype(AF_MA1) && ismutabletype(AF_MA3)
    @test isabstracttype(AF_MA) && !isstructtype(AF_MA)
    @test_throws ErrorException @eval struct AF_MABad <: AF_MA end
    @test_throws ErrorException @eval struct AF_MABad2 <: AF_MA2 end
    @test_throws ErrorException @eval primitive type AF_MABad3 <: AF_MA 8 end
    m = AF_MA1(0, 7, 1)
    @test bump!(m) == 1 && bump!(m) == 2 && m.count == 2
    @test bump!(AF_MA3(5, 1)) == 6
    @test_throws ErrorException m.id = 2
    @test Core.Compiler.is_nothrow(Base.infer_effects(bump!, (AF_MA,)))
    @test Base.infer_return_type(bump!, (AF_MA,)) === Int
    # a stable prefix gives a direct store too
    @test !occursin("jl_f_setfield", sprint(code_llvm, bump!, (AF_MA,)))
    @test occursin("mutable abstract type", sprint(dump, AF_MA))
    @test Base.remove_linenums!(Meta.parse("mutable abstract type X; a::Int; end")) ==
        Expr(:abstract, true, :X, Expr(:block, Expr(:(::), :a, :Int)))
    @test Meta.parse("mutable abstract type X end") == Expr(:abstract, true, :X, Expr(:block))
    @test string(Base.remove_linenums!(Meta.parse("mutable abstract type X; a::Int; end"))) == "mutable abstract type X\n    a::Int\nend"
end

end # module
