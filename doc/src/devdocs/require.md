# Base.require

`Base.require` is responsible for loading modules and invalidating the precompilation cache.

### Module loading callbacks

It is possible to listen to the modules loaded by `Base.require`, by registering a callback.

```julia
loaded_packages = Channel{Symbol}()
callback = (mod::Symbol) -> put!(loaded_packages, mod)
push!(Base.package_callbacks, callback)
```

This functionality is considered experimental and subject to change.

