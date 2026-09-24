# This file is a part of Julia. License is MIT: https://julialang.org/license

# Multiple supertypes: `struct C <: (A, B) end` (Dylan-style, C3 linearized)

using Test, InteractiveUtils

module MultiSupersTest

using Test, InteractiveUtils

# the Dylan grid example (Barrett et al. 1996)
abstract type Grid end
abstract type HGrid <: Grid end
abstract type VGrid <: Grid end
abstract type HVGrid <: (HGrid, VGrid) end
abstract type VHGrid <: (VGrid, HGrid) end
struct HVCell <: HVGrid end
struct VHCell <: VHGrid end
mutable struct MutHV <: (HGrid, VGrid)
    x::Int
end
primitive type PrimHV <: (HGrid, VGrid) 8 end

@testset "declaration and reflection" begin
    @test supertype(HVGrid) === HGrid
    @test supertype(VHGrid) === VGrid
    @test Base.direct_supertypes(HVGrid) === (HGrid, VGrid)
    @test Base.direct_supertypes(VHGrid) === (VGrid, HGrid)
    @test Base.direct_supertypes(HGrid) === (Grid,)
    @test Base.direct_supertypes(Int) === (Signed,)
    @test HVGrid.supers === Core.svec(HGrid, VGrid)
    @test Base.ancestors(HVCell) === (HVGrid, HGrid, VGrid, Grid, Any)
    @test Base.ancestors(VHCell) === (VHGrid, VGrid, HGrid, Grid, Any)
    @test Base.ancestors(Int) === (Signed, Integer, Real, Number, Any)
    @test supertypes(HVCell) === (HVCell, HVGrid, HGrid, VGrid, Grid, Any)
    @test supertypes(Int) === (Int, Signed, Integer, Real, Number, Any)
    @test HVCell.name.linearization == Core.svec(HVCell.name, HVGrid.name, HGrid.name, VGrid.name, Grid.name, Any.name)
    @test !isdefined(Int.name, :linearization)
    @test HVGrid.name.parents === Core.svec(HGrid, VGrid)
    @test !isdefined(HGrid.name, :parents)
    @test HVCell.name.flags & 0x08 == 0x08 && HGrid.name.flags & 0x08 == 0
    @test HGrid.name.may_join == 0x01 && VGrid.name.may_join == 0x01 && Grid.name.may_join == 0x01
    @test Int.name.may_join == 0x00
    @test HVCell in subtypes(HVGrid) && HVGrid in subtypes(HGrid) && HVGrid in subtypes(VGrid)
    unqualify(str) = replace(str, r"[\w.]*MultiSupersTest\." => "")
    @test occursin("<: (HGrid, VGrid)", unqualify(sprint(dump, HVGrid)))
    @test occursin("HVCell <: HVGrid <: HGrid <: VGrid <: Grid <: Any", unqualify(sprint(Base.show_supertypes, HVCell)))
    @test fieldnames(MutHV) == (:x,)
    @test PrimHV <: HGrid && PrimHV <: VGrid
end

@testset "subtyping" begin
    @test HVCell <: HGrid && HVCell <: VGrid && HVCell <: Grid
    @test HVCell() isa VGrid && VHCell() isa HGrid
    @test HVGrid <: HGrid && HVGrid <: VGrid
    @test !(HGrid <: VGrid) && !(VGrid <: HGrid)
    @test !(HVGrid <: VHGrid) && !(VHGrid <: HVGrid)
    @test Union{HGrid,VGrid} >: HVCell
    @test Type{HVCell} <: Type{<:VGrid}
    @test !(Vector{HVCell} <: Vector{VGrid})
    @test Vector{HVCell} <: Vector{<:VGrid}
    @test Tuple{HVCell,Int} <: Tuple{VGrid,Integer}
    @test !(Base.morespecific(Tuple{HGrid}, Tuple{VGrid}))
    @test !(Base.morespecific(Tuple{VGrid}, Tuple{HGrid}))
    @test Base.morespecific(Tuple{HVGrid}, Tuple{VGrid}) && Base.morespecific(Tuple{HVGrid}, Tuple{HGrid})
    @test typejoin(HVCell, VHCell) === Any   # two minimal common ancestors (HGrid, VGrid)
    @test typejoin(HVCell, Grid) === Grid
    @test typejoin(HVGrid, HGrid) === HGrid
    @test typejoin(HVGrid, VGrid) === VGrid
    @test typejoin(HVCell, Int) === Any
end

abstract type Alone1 end
abstract type Alone2 end

@testset "intersection through joins" begin
    @test typeintersect(HGrid, VGrid) == Union{HVGrid, VHGrid, MutHV, PrimHV}
    @test typeintersect(VGrid, HGrid) == Union{HVGrid, VHGrid, MutHV, PrimHV}
    @test typeintersect(HGrid, Grid) === HGrid
    @test typeintersect(HVGrid, VHGrid) === Union{}
    @test typeintersect(HGrid, Alone1) === Union{}
    @test typeintersect(Tuple{HGrid,Int}, Tuple{VGrid,Integer}) == Tuple{Union{HVGrid,VHGrid,MutHV,PrimHV},Int}
    @test Base.hasintersect(HGrid, VGrid)
    @test !Base.hasintersect(Alone1, Alone2)
    # no join exists yet between these two
    @test typeintersect(Alone1, Alone2) === Union{}
    @eval struct Join12 <: (Alone1, Alone2) end
    @test typeintersect(Alone1, Alone2) === Join12
    @test typeintersect(Alone2, Alone1) === Join12
    @eval struct Join12b <: (Alone2, Alone1) end
    @test typeintersect(Alone1, Alone2) == Union{Join12, Join12b}
    # a subtype of an existing join is not a maximal join
    @eval abstract type AJoin <: (Alone1, Alone2) end
    @eval struct SubJoin <: AJoin end
    @test typeintersect(Alone1, Alone2) == Union{Join12, Join12b, AJoin}
end

abstract type PA{T} end
abstract type PB{T} end
struct PC{T} <: (PA{T}, PB{Vector{T}}) end
abstract type PQ{T} <: PA{T} end
abstract type PS{T} <: PA{T} end
struct PGood{T} <: (PQ{T}, PS{T}) end

@testset "parametric supertypes" begin
    @test PC{Int} <: PA{Int} && PC{Int} <: PB{Vector{Int}}
    @test !(PC{Int} <: PB{Int})
    @test Base.direct_supertypes(PC{Int}) === (PA{Int}, PB{Vector{Int}})
    @test Base.direct_supertypes(PC) == (PA{T} where T, PB{Vector{T}} where T)
    @test Base.ancestors(PC{Int}) === (PA{Int}, PB{Vector{Int}}, Any)
    @test typeintersect(PA{Int}, PB{Vector{Int}}) === PC{Int}
    @test typeintersect(PA{Int}, PB{Int}) === Union{}
    @test typeintersect(PA, PB) == PC
    @test typeintersect(PA{Int}, PB) === PC{Int}
    @test Base.ancestors(PGood{Int}) === (PQ{Int}, PS{Int}, PA{Int}, Any)
    @test typejoin(PGood{Int}, PQ{Int}) === PQ{Int}
    @test typejoin(PGood{Int}, PGood{Float64}) === PGood
    # the same ancestor must not be reached with different parameters
    @test_throws ErrorException @eval struct PBad{T} <: (PQ{T}, PS{Int}) end
    @test_throws ErrorException @eval struct PBad2 <: (PQ{Int}, PS{Float64}) end
    # types that are their own parameters
    @eval struct PSelf{T} <: (PQ{PSelf{Tuple{T}}}, PS{PSelf{Tuple{T}}}) end
    @test supertype(PSelf{Int}) === PQ{PSelf{Tuple{Int}}}
    @test Base.direct_supertypes(PSelf{Int}) === (PQ{PSelf{Tuple{Int}}}, PS{PSelf{Tuple{Int}}})
end

@testset "definition errors" begin
    @test_throws ErrorException @eval struct Bad1 <: (Int, HGrid) end
    @test_throws ErrorException @eval struct Bad2 <: (HGrid, HGrid) end
    @test_throws ErrorException @eval struct Bad3 <: (HGrid, Union{HGrid,VGrid}) end
    @test_throws ErrorException @eval struct Bad4 <: (HGrid, Tuple) end
    @test_throws ErrorException @eval struct Bad5 <: (HGrid, Type) end
    @test_throws Exception @eval struct Bad6 <: () end
    @test_throws ErrorException @eval abstract type Bad7 <: (HGrid, Bad7) end
    # inconsistent precedence graph (the "confused grid")
    @test_throws ErrorException @eval abstract type Confused <: (HVGrid, VHGrid) end
    @test_throws ErrorException @eval struct ConfusedS <: (HVGrid, VHGrid) end
end

# the pedalo example (Ducournau et al. via Barrett et al.): monotonicity
abstract type Boat end
abstract type DayBoat <: Boat end
abstract type WheelBoat <: Boat end
abstract type EngineLess <: DayBoat end
abstract type SmallMultihull <: DayBoat end
abstract type PedalWheelBoat <: (EngineLess, WheelBoat) end
abstract type SmallCatamaran <: SmallMultihull end
struct Pedalo <: (PedalWheelBoat, SmallCatamaran) end

@testset "C3 linearization" begin
    @test Base.ancestors(Pedalo) === (PedalWheelBoat, EngineLess, SmallCatamaran, SmallMultihull, DayBoat, WheelBoat, Boat, Any)
    # monotonicity: each ancestor's linearization is a subsequence
    lin = (Pedalo, Base.ancestors(Pedalo)...)
    for a in Base.ancestors(Pedalo)
        a === Any && continue
        sub = (a, Base.ancestors(a)...)
        idx = [findfirst(x -> x === s, lin) for s in sub]
        @test issorted(idx)
    end
    @test Base.ancestors(PedalWheelBoat) === (EngineLess, DayBoat, WheelBoat, Boat, Any)
end

abstract type FieldA
    a::Int
end
abstract type FieldB
    b::String
end
abstract type FieldRoot
    r::Int
end
abstract type FieldL <: FieldRoot
    l::Int
end
abstract type FieldR <: FieldRoot
    rr::Int
end
struct Diamond <: (FieldL, FieldR)
    d::Int
end
struct FieldAB <: (FieldA, FieldB)
    c::Float64
    FieldAB(a, b, c) = new(; FieldA(a)..., FieldB(b)..., c)
end
FieldA(a) = (a = a,)
FieldB(b) = (b = b,)

@testset "fields inherited through several parents" begin
    @test fieldnames(FieldAB) == (:b, :a, :c)   # reversed linearization: FieldB's first
    @test fieldtypes(FieldAB) == (String, Int, Float64)
    x = FieldAB(1, "two", 3.0)
    @test x.a == 1 && x.b == "two" && x.c == 3.0
    @test fieldnames(Diamond) == (:r, :rr, :l, :d)   # the shared root's field appears once
    @test Diamond(1, 2, 3, 4).r == 1
    @test Base.infer_return_type(x -> x.a, (FieldA,)) === Int
    @test Base.infer_return_type(x -> x.r, (FieldL,)) === Int
    @eval abstract type FieldA2; a::Float64; end
    @test_throws ErrorException @eval struct Clash2 <: (FieldA, FieldA2) end
end

fdispatch(::HGrid) = :h
fdispatch(::VGrid) = :v
gdispatch(::Grid) = :grid
gdispatch(::HGrid) = :h

@testset "method lookup through every parent" begin
    @test gdispatch(VHCell()) === :h     # via the secondary parent
    @test gdispatch(HVCell()) === :h
    # both methods apply; the linearization of the argument type orders them
    @test length(Base.methods_including_ambiguous(fdispatch, (HVCell,))) == 1
    @test length(Base.methods_including_ambiguous(fdispatch, (HVGrid,))) == 1
    @test length(Base.methods_including_ambiguous(fdispatch, (Grid,))) == 2
    @test length(methods(fdispatch, (HVCell,))) == 1
    @test hasmethod(fdispatch, (VHCell,))
    @test fdispatch(HVCell()) === :h && fdispatch(VHCell()) === :v
    @test fdispatch(MutHV(1)) === :h && fdispatch(reinterpret(PrimHV, 0x1)) === :h
    @test which(fdispatch, (HVCell,)).sig == Tuple{typeof(fdispatch), HGrid}
    @test which(fdispatch, (VHCell,)).sig == Tuple{typeof(fdispatch), VGrid}
    @test !Base.isambiguous(methods(fdispatch)...)
    @test Base.return_types(fdispatch, (HVCell,)) == [Symbol]
    @test Base.return_types(fdispatch, (HVGrid,)) == [Symbol]
end

# a join does not change dispatch for the concrete types that exist already
struct HOnlyCell <: HGrid end
@testset "existing concrete dispatch survives a join" begin
    @test fdispatch(HOnlyCell()) === :h
    mi = Base.method_instance(fdispatch, (HOnlyCell,))
    ci = mi.cache
    @test ci.max_world == typemax(UInt)
    @eval struct LateJoin <: (HGrid, VGrid) end
    @test fdispatch(HOnlyCell()) === :h
    @test Base.method_instance(fdispatch, (HOnlyCell,)) === mi
    @test mi.cache === ci && ci.max_world == typemax(UInt)
    @test fdispatch(LateJoin()) === :h
end

@testset "redefinition" begin
    @eval struct Redef <: (HGrid, VGrid); x::Int; end
    T1 = Redef
    @eval struct Redef <: (HGrid, VGrid); x::Int; end
    @test Redef === T1
    @eval struct Redef <: (VGrid, HGrid); x::Int; end
    @test Redef !== T1
    @test Base.direct_supertypes(Redef) === (VGrid, HGrid)
end

end # module
