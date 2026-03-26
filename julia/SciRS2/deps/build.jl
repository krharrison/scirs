"""
SciRS2 dependency build script — JLL-aware artifact resolver.

Execution order:
  1. Try to resolve a prebuilt JLL artifact from Artifacts.toml.
  2. If no artifact is available (git-tree-sha1 is empty or download fails),
     attempt a source build using Cargo.
  3. Write the resolved library path to `deps/deps.jl` so that SciRS2.jl can
     load it at runtime without repeating the resolution logic.

This file is executed automatically by `Pkg.build("SciRS2")`.
"""

using Pkg, Pkg.Artifacts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

const DEPS_DIR    = @__DIR__
const PKG_ROOT    = joinpath(DEPS_DIR, "..")
const WORKSPACE   = joinpath(PKG_ROOT, "..", "..")   # scirs/ workspace root
const DEPS_FILE   = joinpath(DEPS_DIR, "deps.jl")

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

"""Return the expected shared library file name for the current platform."""
function _libname()::String
    if Sys.isapple()
        return "libscirs2_julia.dylib"
    elseif Sys.iswindows()
        return "scirs2_julia.dll"
    else
        return "libscirs2_julia.so"
    end
end

"""Return the canonical platform triple used in Artifacts.toml keys."""
function _platform_key()::String
    arch = string(Sys.ARCH)
    if Sys.isapple()
        return "$(arch)-apple-darwin"
    elseif Sys.iswindows()
        return "$(arch)-w64-mingw32"
    else
        return "$(arch)-linux-gnu"
    end
end

# ---------------------------------------------------------------------------
# Step 1: JLL artifact resolution
# ---------------------------------------------------------------------------

"""
Try to obtain `libscirs2_julia` from the prebuilt artifact declared in
Artifacts.toml.

Returns the directory containing the shared library, or `nothing` if the
artifact is not (yet) available (e.g. git-tree-sha1 is empty).
"""
function _try_artifact()::Union{String, Nothing}
    artifacts_file = joinpath(PKG_ROOT, "Artifacts.toml")
    if !isfile(artifacts_file)
        @info "SciRS2 build: Artifacts.toml not found, skipping JLL artifact lookup."
        return nothing
    end

    meta = artifact_meta("SciRS2", artifacts_file)
    if isnothing(meta) || get(meta, "git-tree-sha1", "") == ""
        @info "SciRS2 build: Artifact metadata not yet populated (release pending)."
        return nothing
    end

    try
        artifact_dir = artifact_path(Base.SHA1(meta["git-tree-sha1"]))
        lib_path = joinpath(artifact_dir, _libname())
        if isfile(lib_path)
            @info "SciRS2 build: Found prebuilt JLL artifact at $lib_path"
            return artifact_dir
        end
        @info "SciRS2 build: Artifact directory exists but library not found: $lib_path"
    catch err
        @warn "SciRS2 build: Artifact resolution failed: $err"
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Step 2: Source build via Cargo
# ---------------------------------------------------------------------------

"""
Build `libscirs2_julia` from source using Cargo.

Requires the Rust toolchain to be present on PATH.  Returns the directory
containing the compiled library, or throws an error on failure.
"""
function _source_build(; release::Bool=true)::String
    cargo_exe = Sys.which("cargo")
    if isnothing(cargo_exe)
        error(
            "SciRS2 build: 'cargo' not found on PATH.\n" *
            "Install the Rust toolchain from https://rustup.rs/ and re-run Pkg.build(\"SciRS2\").\n" *
            "Alternatively, install a prebuilt SciRS2_jll package once it is available."
        )
    end

    # Workspace root must contain Cargo.toml
    cargo_toml = joinpath(WORKSPACE, "Cargo.toml")
    if !isfile(cargo_toml)
        # Try one level up in case we're not in the expected directory tree
        alt_workspace = joinpath(PKG_ROOT, "..", "..")
        if isfile(joinpath(alt_workspace, "Cargo.toml"))
            cargo_toml = joinpath(alt_workspace, "Cargo.toml")
        else
            error("SciRS2 build: Could not locate Cargo.toml. Expected at: $cargo_toml")
        end
    end

    workspace_dir = dirname(cargo_toml)
    @info "SciRS2 build: Starting Cargo source build in $workspace_dir"

    profile_flag = release ? "--release" : ""
    cmd = `$cargo_exe build --package scirs2-julia --features full $profile_flag`

    @info "SciRS2 build: Running $cmd"
    success_flag = (run(setenv(cmd, dir=workspace_dir); wait=true)).exitcode == 0
    if !success_flag
        error(
            "SciRS2 build: Cargo build failed.\n" *
            "See the output above for details.  Common causes:\n" *
            "  - Missing Rust nightly features (try: rustup update)\n" *
            "  - Missing system libraries (oxiblas requires no system BLAS — Pure Rust)\n" *
            "Run manually: cd $workspace_dir && cargo build -p scirs2-julia --features full"
        )
    end

    # Locate the compiled artifact
    target_subdir = release ? "release" : "debug"
    target_dir = joinpath(workspace_dir, "target", target_subdir)
    lib_path = joinpath(target_dir, _libname())

    if !isfile(lib_path)
        # Fallback: scan target dir for any matching library
        candidates = filter(
            f -> startswith(basename(f), "libscirs2_julia") || startswith(basename(f), "scirs2_julia"),
            readdir(target_dir, join=true),
        )
        if isempty(candidates)
            error("SciRS2 build: Library not found after successful build. Expected: $lib_path")
        end
        lib_path = first(candidates)
    end

    # Copy to deps/ for stable path
    dest = joinpath(DEPS_DIR, _libname())
    cp(lib_path, dest; force=true)
    @info "SciRS2 build: Library installed to $dest"
    return DEPS_DIR
end

# ---------------------------------------------------------------------------
# Step 3: Write deps.jl
# ---------------------------------------------------------------------------

"""
Write `deps/deps.jl` with the resolved library directory and path constants.
This file is `include()`d by `SciRS2.jl` at module load time.
"""
function _write_deps(lib_dir::String)
    lib_path = joinpath(lib_dir, _libname())
    # Escape backslashes on Windows
    lib_path_escaped = replace(lib_path, '\\' => "\\\\")
    lib_dir_escaped  = replace(lib_dir,  '\\' => "\\\\")

    open(DEPS_FILE, "w") do io
        println(io, """
# deps.jl — Auto-generated by deps/build.jl. Do not edit manually.
# Re-run Pkg.build("SciRS2") to regenerate.

const _SCIRS2_JULIA_LIB_DIR  = "$(lib_dir_escaped)"
const _SCIRS2_JULIA_LIB_PATH = "$(lib_path_escaped)"
""")
    end
    @info "SciRS2 build: Wrote $DEPS_FILE"
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

function _build()
    @info "SciRS2 build: Starting dependency resolution (JLL-aware)..."
    @info "SciRS2 build: Platform triple = $(_platform_key())"
    @info "SciRS2 build: Expected library name = $(_libname())"

    # Priority 1: prebuilt JLL artifact
    lib_dir = _try_artifact()

    # Priority 2: local source build
    if isnothing(lib_dir)
        @info "SciRS2 build: No prebuilt artifact found. Attempting source build..."
        lib_dir = _source_build(; release=true)
    end

    # Verify the resolved library actually loads
    lib_path = joinpath(lib_dir, _libname())
    if !isfile(lib_path)
        error("SciRS2 build: Resolution succeeded but library file not found: $lib_path")
    end

    # Quick sanity check: try dlopen
    try
        import Libdl
        handle = Libdl.dlopen(lib_path)
        ver_sym = Libdl.dlsym(handle, :scirs2_julia_version; throw_error=false)
        if !isnothing(ver_sym)
            ver_ptr = ccall(ver_sym, Ptr{UInt8}, ())
            ver_str = unsafe_string(ver_ptr)
            @info "SciRS2 build: Library loaded successfully. Version = $ver_str"
        end
        abi_sym = Libdl.dlsym(handle, :scirs2_julia_abi_version; throw_error=false)
        if !isnothing(abi_sym)
            abi_ver = ccall(abi_sym, Cuint, ())
            @info "SciRS2 build: ABI version = $abi_ver"
            if abi_ver < 1
                @warn "SciRS2 build: ABI version $abi_ver is older than expected (>= 1)."
            end
        end
        Libdl.dlclose(handle)
    catch err
        @warn "SciRS2 build: Library sanity check failed (non-fatal): $err"
        @warn "SciRS2 build: The library may still work at runtime."
    end

    _write_deps(lib_dir)
    @info "SciRS2 build: Dependency build completed successfully."
    @info "SciRS2 build: Library path: $lib_path"
end

_build()
