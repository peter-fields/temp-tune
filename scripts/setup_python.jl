# scripts/setup_python.jl
#
# =====================================================================
#  IMPORTANT:
#  This script MUST be run from the ROOT of the temp-tune repository:
#
#      cd path/to/temp-tune
#      julia --project=. scripts/setup_python.jl
#
#  Do NOT run this from another directory.
#  Do NOT run this with a different --project.
#
#  Purpose:
#   - set up project-local Conda at ./ .conda
#   - set up project-local Matplotlib config at ./ .mplconfig
#   - rebuild PyCall against that local Conda python
#   - install matplotlib into that same Conda environment
#
#  This is a one-time setup step for this repository.
# =====================================================================

import Pkg

# Defensive: ensure we are actually in the temp-tune repo root
expected_files = ["Project.toml", "src", "scripts"]
missing = filter(f -> !ispath(joinpath(pwd(), f)), expected_files)

if !isempty(missing)
    error("""
    This script must be run from the ROOT of the temp-tune repository.

    Expected to find: $(join(expected_files, ", "))
    Missing:           $(join(missing, ", "))

    Current directory:
        $(pwd())

    Correct usage:
        cd path/to/temp-tune
        julia --project=. scripts/setup_python.jl
    """)
end

# Make sure DrWatson is available (should already be in Project.toml)
try
    using DrWatson
catch
    Pkg.add("DrWatson")
    using DrWatson
end

proj = projectdir()
conda_home = joinpath(proj, ".conda")
mplconfig  = joinpath(proj, ".mplconfig")

# If PyCall is already loaded, rebuilding will not take effect
if "PyCall" in keys(Base.loaded_modules)
    error("""
    PyCall is already loaded in this Julia session.

    Please:
      1. Exit Julia
      2. cd into the temp-tune repo root
      3. Re-run: julia --project=. scripts/setup_python.jl
    """)
end

# Wire environment for project-local Conda + Matplotlib config
ENV["CONDA_JL_HOME"] = conda_home
ENV["MPLCONFIGDIR"]  = mplconfig
ENV["MPLBACKEND"]    = get(ENV, "MPLBACKEND", "Agg")

mkpath(conda_home)
mkpath(mplconfig)

println("== temp-tune Python/Matplotlib setup ==")
println("Project dir:   ", proj)
println("CONDA_JL_HOME: ", conda_home)
println("MPLCONFIGDIR:  ", mplconfig)
println("MPLBACKEND:    ", ENV["MPLBACKEND"])
println()

# Ensure required packages exist (should already be in Project.toml, but safe)
Pkg.instantiate()

# Force PyCall to use Conda.jl's python under CONDA_JL_HOME
ENV["PYTHON"] = ""

println("Building PyCall against project-local Conda python...")
Pkg.build("PyCall")

# Install matplotlib into that same Conda env
using Conda

println("Installing matplotlib into project-local Conda env...")
# If needed for compatibility, you can pin:
# Conda.add("matplotlib=3.7.*")
Conda.add("matplotlib")

println()
println("Setup complete.")
println("Now restart Julia and use:")
println("  include(joinpath(projectdir(), \"src\", \"matplotlib_helpers.jl\"))")
println("  setup_notebook_for_paper(); using_Py(true)")
