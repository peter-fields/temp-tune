# temp-tune

This code base uses the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to maintain a fully reproducible scientific project named

> **temp-tune**

The project also relies on **PyCall + PyPlot + a project-local Conda environment**
for all Python/Matplotlib functionality.

---

## Reproducing this project locally

To (locally) reproduce this project, do the following.

### 0. Obtain the code

Clone or download this repository. Note that large raw datasets are typically not
included in the git history and may need to be downloaded independently.

---

### 1. Open a Julia console and do:

```julia
using Pkg
Pkg.add("DrWatson")          # install globally, for using `quickactivate`
Pkg.activate("path/to/temp-tune")
Pkg.instantiate()

cd("path/to/temp-tune")
include(scriptsdir("setup_python.jl"))
```

This will:

- activate the `temp-tune` project
- instantiate all Julia dependencies
- set up the project-local Conda environment at `./.conda`
- set up the project-local Matplotlib config at `./.mplconfig`
- rebuild `PyCall` against the project-local Conda Python
- install `matplotlib` into that same Conda environment

**Important:**

- `path/to/temp-tune` must be the root of this repository.
- This setup step is **required once per clone**.
- After this completes, **restart Julia** before doing any analysis.

---

## Using this project

You may notice that most scripts start with:

```julia
using DrWatson
@quickactivate "temp-tune"
```

which auto-activates the project and enables DrWatson’s local path handling.

---

## Using Python / Matplotlib in this project

After the one-time setup is complete, initialize Python/Matplotlib in any script
or notebook with:

```julia
using DrWatson
@quickactivate "temp-tune"
include(srcdir("matplotlib_helpers.jl"))

setup_notebook_for_paper()
using_Py(true)
```

This ensures that:

- PyCall is bound to the project-local Conda environment
- Matplotlib uses the project-local `.mplconfig`
- all plotting settings are consistent and reproducible

A project local install matplotlib will be used via the Julia Conda package. The directory .conda/pkgs is a cache and may be safely deleted at any time to reclaim disk space.

---

## Notes on reproducibility

- The use of a **project-local Conda environment** is intentional and required for
  reproducibility.
- Do **not** use your system Python, a global Conda install, or another project’s
  Conda environment with this project.
- All Python dependencies must be installed into `temp-tune/.conda`.
- If you encounter Python/Matplotlib issues, re-running

  ```julia
  include(scriptsdir("setup_python.jl"))
  ```

  in a **fresh Julia session** usually resolves them.

---

