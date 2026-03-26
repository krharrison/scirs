"""
    JLLCompat

JLL (Julia Library Link) compatibility layer for SciRS2.

This module provides the library path resolution logic that supports both:
  1. Prebuilt JLL artifact binaries (installed via `Pkg.add` + `Pkg.build`).
  2. Locally compiled shared libraries (built from source with Cargo).
  3. Legacy path: `deps/libscirs2_core.<ext>` (for backward compatibility with
     the pre-JLL build system used up to SciRS2 0.3.x).

The resolved path is exposed as `JLLCompat.lib_path()` and used by the rest of
the SciRS2 module to form `ccall` library references.

# Resolution priority

1. `deps/deps.jl` (written by `deps/build.jl` at `Pkg.build` time).
   Contains the exact path discovered / compiled during the build step.
2. Environment variable `SCIRS2_JULIA_LIB` — lets advanced users point to a
   custom build without modifying the package.
3. JLL artifact via `Pkg.Artifacts` (requires populated Artifacts.toml).
4. Legacy `deps/libscirs2_core.<ext>` path (backward compat, 0.3.x builds).
5. System library search path (last resort, relies on LD_LIBRARY_PATH etc.).
"""
module JLLCompat

using Libdl

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

# Cached library path.  Resolved once during __init__.
const _LIB_PATH = Ref{String}("")

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

"""Return the platform-appropriate shared library file name for `libname`."""
function _lib_filename(libname::AbstractString)::String
    if Sys.isapple()
        return "$(libname).dylib"
    elseif Sys.iswindows()
        return "$(libname).dll"
    else
        return "$(libname).so"
    end
end

# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------

"""
Attempt to load the path from `deps/deps.jl` (written by `Pkg.build`).
Returns the library path string, or `nothing` on failure.
"""
function _try_deps_jl()::Union{String, Nothing}
    deps_file = joinpath(@__DIR__, "..", "deps", "deps.jl")
    if !isfile(deps_file)
        return nothing
    end
    # Evaluate in an isolated module to avoid polluting the JLLCompat namespace.
    m = Module(:_DepsLoader)
    try
        Base.include(m, deps_file)
        if isdefined(m, :_SCIRS2_JULIA_LIB_PATH)
            p = string(getfield(m, :_SCIRS2_JULIA_LIB_PATH))
            return isfile(p) ? p : nothing
        end
    catch err
        @debug "JLLCompat: deps.jl evaluation failed: $err"
    end
    return nothing
end

"""
Attempt to use the `SCIRS2_JULIA_LIB` environment variable.
Returns the library path string, or `nothing` if the variable is unset.
"""
function _try_env_var()::Union{String, Nothing}
    p = get(ENV, "SCIRS2_JULIA_LIB", "")
    if !isempty(p) && isfile(p)
        return p
    end
    return nothing
end

"""
Attempt to resolve the library via the JLL artifact declared in Artifacts.toml.
Returns the library path string, or `nothing` if unavailable.
"""
function _try_artifact()::Union{String, Nothing}
    try
        # Pkg.Artifacts is a stdlib available since Julia 1.6.
        import_expr = :(using Pkg.Artifacts: artifact_meta, artifact_path)
        eval(import_expr)
        artifacts_file = joinpath(@__DIR__, "..", "Artifacts.toml")
        if !isfile(artifacts_file)
            return nothing
        end
        meta = artifact_meta("SciRS2", artifacts_file)
        if isnothing(meta) || get(meta, "git-tree-sha1", "") == ""
            return nothing
        end
        artifact_dir = artifact_path(Base.SHA1(meta["git-tree-sha1"]))
        jll_name = _lib_filename("libscirs2_julia")
        p = joinpath(artifact_dir, jll_name)
        return isfile(p) ? p : nothing
    catch err
        @debug "JLLCompat: Artifact resolution failed: $err"
        return nothing
    end
end

"""
Try the legacy `deps/libscirs2_core.<ext>` path used by SciRS2 <= 0.3.x.
Returns the library path string, or `nothing` if not found.
"""
function _try_legacy_core()::Union{String, Nothing}
    deps_dir = joinpath(@__DIR__, "..", "deps")
    for name in ("libscirs2_core", "libscirs2_julia")
        p = joinpath(deps_dir, _lib_filename(name))
        if isfile(p)
            return p
        end
    end
    return nothing
end

"""
Try `Libdl.find_library` to search the system library path.
Returns the library path string (as reported by dlopen), or `nothing`.
"""
function _try_system_path()::Union{String, Nothing}
    for name in ("scirs2_julia", "scirs2_core")
        p = Libdl.find_library(name)
        if !isempty(p)
            return p
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    lib_path() -> String

Return the resolved path to the SciRS2 Julia shared library.

Raises an error if the library has not been found.  Call `Pkg.build("SciRS2")`
to trigger the build process if the library is missing.
"""
function lib_path()::String
    p = _LIB_PATH[]
    if isempty(p)
        error(
            "SciRS2: shared library not found.\n" *
            "Run `Pkg.build(\"SciRS2\")` to build or download the library.\n" *
            "Alternatively, set the SCIRS2_JULIA_LIB environment variable to the\n" *
            "full path of libscirs2_julia.dylib / .so / .dll."
        )
    end
    return p
end

"""
    resolve!() -> String

Resolve the library path using the priority chain and cache the result.
Called once during module `__init__`.  Returns the resolved path.
"""
function resolve!()::String
    strategies = [
        ("deps.jl",      _try_deps_jl),
        ("env var",      _try_env_var),
        ("JLL artifact", _try_artifact),
        ("legacy core",  _try_legacy_core),
        ("system path",  _try_system_path),
    ]

    for (label, fn) in strategies
        result = fn()
        if !isnothing(result)
            _LIB_PATH[] = result
            @debug "JLLCompat: Resolved library via '$label': $result"
            return result
        end
    end

    # All strategies failed — emit a helpful warning but do not error yet.
    # The error is deferred to `lib_path()` so that pure-Julia functionality
    # in PureAlgorithms.jl still works even without the native library.
    @warn(
        "SciRS2: Native library not found. Pure-Julia algorithms are still available.\n" *
        "To enable native acceleration, run: Pkg.build(\"SciRS2\")\n" *
        "Or set environment variable SCIRS2_JULIA_LIB to the library path."
    )
    return ""
end

"""
    abi_version() -> Union{UInt32, Nothing}

Query the ABI version from the loaded library, or return `nothing` if the
library is unavailable or the symbol is not present.
"""
function abi_version()::Union{UInt32, Nothing}
    p = _LIB_PATH[]
    if isempty(p)
        return nothing
    end
    try
        handle = Libdl.dlopen(p; throw_error=false)
        if handle === nothing
            return nothing
        end
        sym = Libdl.dlsym(handle, :scirs2_julia_abi_version; throw_error=false)
        result = isnothing(sym) ? nothing : ccall(sym, Cuint, ())
        Libdl.dlclose(handle)
        return result
    catch
        return nothing
    end
end

"""
    capability_flags() -> UInt32

Return the capability bitmask from the loaded library.

Bit layout:
  - bit 0 (0x01): linalg
  - bit 1 (0x02): stats
  - bit 2 (0x04): fft
  - bit 3 (0x08): optimize

Returns 0 if the library is unavailable.
"""
function capability_flags()::UInt32
    p = _LIB_PATH[]
    if isempty(p)
        return UInt32(0)
    end
    try
        handle = Libdl.dlopen(p; throw_error=false)
        if handle === nothing
            return UInt32(0)
        end
        sym = Libdl.dlsym(handle, :scirs2_julia_capabilities; throw_error=false)
        result = isnothing(sym) ? UInt32(0) : ccall(sym, Cuint, ())
        Libdl.dlclose(handle)
        return result
    catch
        return UInt32(0)
    end
end

"""
    has_linalg()   -> Bool
    has_stats()    -> Bool
    has_fft()      -> Bool
    has_optimize() -> Bool

Predicate helpers for capability testing.
"""
has_linalg()   = (capability_flags() & 0x01) != 0
has_stats()    = (capability_flags() & 0x02) != 0
has_fft()      = (capability_flags() & 0x04) != 0
has_optimize() = (capability_flags() & 0x08) != 0

# ---------------------------------------------------------------------------
# Module __init__
# ---------------------------------------------------------------------------

function __init__()
    resolve!()
end

end  # module JLLCompat
