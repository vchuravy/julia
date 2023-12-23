# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    @newinterp NewInterpreter

Defines new `NewInterpreter <: AbstractInterpreter` whose cache is separated
from the native code cache, satisfying the minimum interface requirements.
"""
macro newinterp(InterpName)
    InterpCompilerName = esc(Symbol(string(InterpName, "Compiler")))
    InterpName = esc(InterpName)
    C = Core
    CC = Core.Compiler
    quote
        struct $InterpCompilerName <: $CC.AbstractCompiler end
        $CC.abstract_interpreter(compiler::$InterpCompilerName, world::UInt) =
            $InterpName(;world, compiler)
        struct $InterpName <: $CC.AbstractInterpreter
            meta # additional information
            world::UInt
            inf_params::$CC.InferenceParams
            opt_params::$CC.OptimizationParams
            inf_cache::Vector{$CC.InferenceResult}
            code_cache::$CC.InternalCodeCache
            compiler::$InterpCompilerName
            function $InterpName(meta = nothing;
                                 world::UInt = Base.get_world_counter(),
                                 compiler::$InterpCompilerName = $InterpCompilerName(),
                                 inf_params::$CC.InferenceParams = $CC.InferenceParams(),
                                 opt_params::$CC.OptimizationParams = $CC.OptimizationParams(),
                                 inf_cache::Vector{$CC.InferenceResult} = $CC.InferenceResult[],
                                 code_cache::$CC.InternalCodeCache = $CC.InternalCodeCache(compiler))
                return new(meta, world, inf_params, opt_params, inf_cache, code_cache, compiler)
            end
        end
        $CC.InferenceParams(interp::$InterpName) = interp.inf_params
        $CC.OptimizationParams(interp::$InterpName) = interp.opt_params
        $CC.get_world_counter(interp::$InterpName) = interp.world
        $CC.get_inference_cache(interp::$InterpName) = interp.inf_cache
        $CC.code_cache(interp::$InterpName) = $CC.WorldView(interp.code_cache, $CC.WorldRange(interp.world))
        $CC.cache_owner(interp::$InterpName) = interp.compiler
    end
end
