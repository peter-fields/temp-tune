# temp-tune

This repository contains the code and materials for the paper “Understanding temperature tuning in energy-based models” by Peter William Fields, Vudtiwat Ngampruetikorn, David J. Schwab, and Stephanie E. Palmer. arXiv ID: 2512.09152, ( https://www.arxiv.org/pdf/2512.09152 )

---

This code base uses the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to maintain a fully reproducible scientific project named

> **temp-tune**

The project also relies on **PyCall + PyPlot + a project-local Conda environment**
for all Python/Matplotlib functionality.

---

## Reproducing this project locally

The Jupyter notebooks use IJulia. This package is not in the project environment, so ensure that IJulia is installed in your global Julia environment if it is not already (e.g. via Pkg.add("IJulia")) before running the notebooks or activating the project. 

To (locally) reproduce this project, do the following.

#### 0. Obtain the code

Clone or download this repository.

---

#### 1. Open a Julia console and do:

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
- The directory .conda/pkgs is a cache and may be safely deleted at any time to reclaim disk space.

---

## Data availability

From the root of your cloned `temp-tune` repository, run:

```bash
# create data directory
mkdir -p data
cd data

# install git-lfs if not already installed
git lfs install

# clone the dataset directly into the expected folder
git clone https://huggingface.co/datasets/peter-fields/temp-tune-data ising_sweeps
```
This ensures the following directory structure required:
```
temp-tune/
└── data/
    └── ising_sweeps/
        (data files here)
```

---

## Minimum working examples

Can be found for both nearest-neighbor Ising distribution and toy model experiments at:
```
temp-tune/notebooks/min_working_examples.ipynb
```

---

## Using this project

Start each script/notebook with:

```julia
using DrWatson
@quickactivate "temp-tune"
```

to activate the project and ensure local path handling is correct. 

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

A project local install matplotlib will be used via the Julia Conda package. 

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

