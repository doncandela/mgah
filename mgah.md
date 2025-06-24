# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   6/24/25

- [Introduction](#intro)  
  
  - [What is this?](#what-is)
  - [Parallel computing in Python](#parcomp-python)
  - [Hardware used](#hardware)
    - [PCs](#pcs)
    - [GPUs](#gpus)
    - [Unity HPC cluster](#unity-intro)
  - [Pip, Conda, and APT](#pip-conda-apt)
  - [Conda environments and test code used in this document](#envs-testcode)
  - [Installing a local package on a PC](#local-package)
  - [Parallel execution on multiple cores](#multiple-cores)

- [Part 1: MPI, GPU, and Apptainer on a Linux PC](#on-linux-pc)
  
  - [MPI on a Linux PC](#mpi-pc)
    - [Why do it](#why-mpi-pc)    
    - [Installing OpenMPI and `mpi4py` on a Linux PC](#install-openmpi-pc)
    - [Simple MPI test programs: `mpi_hw.py`  and `osu_bw.py`](#mpi-testprogs)
    - [More elaborate MPI programs using the `dem21` package](#mpi-dem21)
    - [Hyperthreading and NumPy multithreading with MPI](#multithread-mpi)
  - [Using an NVIDIA GPU on a Linux PC](#gpu-pc)
    - [Why do it](#why-gpu-pc)
    - [Non-NVIDIA GPUs](#non-nvidia)
    - [Installing NVIDIA drivers](#nvidia-drivers)
    - [Installing CUDA-aware Python packages: PyTorch, CuPy...](#pytorch-cupy)
    - [A few of NVDIA's many GPUS, with test results](#gpu-list)
  - [Using Apptainer on a Linux PC](#apptainer-pc)
    - [Why do it](#why-apptainer-pc)
    - [Apptainer history](#apptainer-history)
    - [Installing Apptainer](#install-apptainer)
    - [Testing the install: An OS-only container](#os-only-container)
    - [A container including chosen Python packages](#packages-container)
    - [A container with a local Python package installed](#local-package-container)
    - [A container that can use MPI](#mpi-container)
    - [A container to run the more elaborate MPI package `dem21`](#dem21-container)
    - [A container that can use a GPU](#gpu-container)

- [Part 2: Moving code to a Slurm HPC cluster](#move-to-hpc)
  
  - [Why do it](#why-hpc)
  - [Unity cluster at UMass, Amherst](#unity-cluster)
    - [History](#unity-history)
    - [Using interactive apps: Jupyter, MATLAB...](#unity-interactive)
    - [Logging in (terminal mode)](#unity-login)
    - [Storage](#unity-storage)
    - [Transferring files to/from Unity](#unity-file-transfer)
    - [Slurm on Unity](#unity-slurm)
    - [Running jobs interactively: `salloc` or `unity-compute`](#run-interactive)
    - [Using `.bashrc` and `.bash_aliases`](#rc-files)
    - [Using modules and Conda](#unity-modules-conda)
    - [Running batch jobs: `sbatch`](#run-batch)
    - [Why won't my jobs run](#why-wont-run)
  - [Using MPI on Unity (without Apptainer)](#unity-mpi)
    - [Ways of running Python MPI programs on Unity](#ways-mpi-unity)
    - [Use `sbatch` to run a simple MPI job](#sbatch-mpi)
    - [Enabling NumPy multithreading in MPI batch jobs](#sbatch-multithread)
    - [Use `sbatch` to run `boxpct.py + dem21` with MPI](#sbatch-dem21)
  - [Using a GPU on Unity (without Apptainer)](#unity-gpu)
    - [Using PyTorch with a GPU on Unity](#pytorch-unity)
    - [A Conda environment with CuPy](#conda-gpu-unity)
    - [Run `gputest.py` on Unity interactively](#gputest-interactive)
    - [A batch job using a GPU](#gpu-sbatch)
    - [Picking a GPU on Unity](#pick-gpu)
  - [Using Apptainer on Unity](#unity-apptainer)
    - [Getting container images on the cluster](#images-to-unity)
    - [Running a container interactively or in batch job](#unity-run-container)
    - [Running containers that use MPI](#unity-mpi-container)
    - [Running a container the uses a GPU](#unity-gpu-container)
    - [Other ways of getting/running Apptainer containers](#other-apptainer)

- [Random notes on parallel computing in Python](#random-notes)  
  
  - [Wall time and CPU time](#wall-cpu-time)
  - [Strong and weak scaling](#strong-weak-scaling)
  - [Estimating MPI communication overhead](#estimate-mpi-overhead)

- [Summary and TODOs](#summary-todos)

## Introduction <a id="intro"></a>

### What is this?<a id="what-is"></a>

This the cheat sheet I that accumulated as I learned to combine several tools for **parallel computing in Python** on various **Linux** computer systems:

- [**MPI**](https://en.wikipedia.org/wiki/Message_Passing_Interface) allows multiple instances of Python to operate in parallel and communicate with each other, in the cores of a single computer or a cluster of connected computers. Code written to parallelize using MPI can utilize all the cores of a desktop computer and also scale to a larger number of cores in an HPC computer cluster.

- A [**GPU**](https://en.wikipedia.org/wiki/Graphics_processing_unit) installed in a single computer can carry out highly parallel computations, so it offers an alternative to "MPI on a cluster of computers" for parallelizing code - but the degree of parallel operation is limited by the model of GPU that is available (unless multiple GPUs and/or GPUs on multiple MPI-connected computers are used, things not discussed in this document).

- [**Apptainer**](https://apptainer.org/) (formerly called **Singularity**) is a **container** system that allows user code and most of its dependencies (OS version, packages like NumPy) to be packaged together into a single large "image" file, which should then be usable  without modification or detailed environment configuration on many different computer systems from a Linux PC to a large cluster.

- High-performance Computing ([**HPC**](https://en.wikipedia.org/wiki/High-performance_computing)) typically refers to using a large cluster of connected computers assembled and maintained by Universities and other organizations for the use of their communities.  This document only discusses an HPC cluster running Linux and managed by  [**Slurm**](https://slurm.schedmd.com/overview.html) scheduling software, with  the the [**UMass Unity cluster**](https://unity.rc.umass.edu/index.php) as the specific HPC system used here.

- A few references are made in this document to [GitHub](https://github.com/) as a place from which files can be downloaded.  Many resources are available on using Git/GitHub, including [this cheat sheet](https://github.com/doncandela/gs-git) that I put together.

Why Python?  Why Linux? Because those are what I use, and this is my cheat sheet.  So this document is geared towards this work flow:

- Write some Python code and get it working on a Linux PC.

- (If desired) to get some parallel speedup either:
  
  - add MPI code and get that working on the multiple cores of the PC, or
  
  - start using a GPU-aware package like CuPy or PyTorch and get that working using the PC's GPU.

- (If desired) to move the code to an HPC cluster like Unity:
  
  - Optionally use Apptainer to containerize the code - this document shows how to do this if the code uses MPI, a GPU, or neither.
  
  - Copy the (already working) code, containerized or not, to the HPC cluster and run it there.

Although there may be some information useful for the following topics, this document **does not cover:**

- Other than brief mentions, the use of OpenMP (a multithreading package not to be confused with OpenMPI) and/or the Python Mutiproccessing package for parallelization on the cores of a single computer. However, multithreading by NumPy (which may indirectly use OpenMP) is discussed.

- Operating systems other than Linux (Windows, macOS...).

- Computer languages other than Python such as C++.

- GPUs other than NVIDIA, except some brief mentions of AMD ROCm and Mac MPS support in the section [Non-NVIDIA GPUs](#non-nvidia) below.

- Direct, low-level programming of GPUs in CUDA-C++  (as opposed to the use of GPU-aware Python packages like CuPy and PyTorch, which are discussed).

- "Higher level" (than MPI) packages for using computer clusters such as Spark, Dask, Charm4Py/Charm++...).

- Cloud computing (Amazon Web Services, Microsoft Azure...). 

- The Docker container system, other than as a source for building Apptainer containers.

- The Kubernetes scheduling/management software typically used rather than Slurm in commercial settings, particularly with Docker.

**This document is quite long** because it shows explicit examples of commands and output for many different situations -- every time I figured out how to do something, I pasted an example here -- so it is not particularly readable.

### Parallel computing in Python<a id="parcomp-python"></a>

Python is a semi-interpreted language (compiled to a byte code, like Java) and so is much more slowly executed than a fully compiled language like C++, unless an add-on like [Numba](http://numba.pydata.org/) or [Cython](https://cython.org/) is used (neither of these is discussed further in this document, although they may certainly be useful).

Therefore good performance on large tasks is often achieved by using **packages** (typically written by others in a compiled language like C++) like [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [CuPy](https://cupy.dev/), and [PyTorch](https://pytorch.org/), to carry out the time-consuming **inner loops** of algorithms. The same is true of other high-level languages like [MATLAB](https://www.mathworks.com/products/matlab.html) and [Mathematica](https://www.mathematica.org/).   While some think Python is inherently slower than C++, if the time limiting factor is, for example, a large linear algebra operation then in either language it will likely be carried out by the same highly-optimized [BLAS](https://www.netlib.org/blas/) function on a CPU (via NumPy, for Python), or the corresponding [cuBLAS](https://developer.nvidia.com/cublas) function on a GPU (via CuPy).

There are however some murky intermediate situations. For example [NumPy advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html) allows many complicated operations (e.g. operations on elements meeting complicated conditions) on arrays to be carried out much faster than if they were coded directly in Python -- but maybe slower than would be possible in C++.

Be that as it may, the premise of this document is **speeding up (Python + packages) code** by using one or the other of the following strategies (or potentially both together, although that is not discussed in detail):

(a)  **by running many copies of the same (Python + packages) code at the same time, using MPI** -- on the multiple cores of one or more CPUs, or

(b) **by using GPU-aware packages** -- a GPU is a highly-parallel computational device which however does not directly run Python code (or C++ code, for that matter, although a specialized hybrid language called [CUDA C++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) is often used to program GPUs).

For case (a) the individual, simultaneously-executing copies of a Python program can each take advantage of packages like NumPy and SciPy.  (However, depending on how things are set up, when used with MPI packages like NumPy may be prevented from using multithreading on multiple cores as is common without MPI.)  For this approach, you need to figure out how to split your problem into many pieces that can profitably run in parallel, how the pieces will be set up, controlled, and communicate with each other, etc.

 Conversely for case (b) a  **GPU-aware Python package** like CuPy, PyTorch, or PyCUDA can be installed. The first two of these completely take care of parallelization in a manner transparent to the Python programmer, who however must keep track of which objects are on the GPU and which are on the CPU - a relatively simple thing.

It is important to distinguish between **multithreading** and **multiprocessing**:

- A **process** is an independently-running program with its own memory space and other resources. Each process can run an independent Python program. Each core of a CPU can run multiple processes, but only one at a time (i.e. serially) - running multiple processes in parallel requires multiple cores.

- A **thread** is part of a process, that can sometimes use multiple cores to run in parallel with other threads in the same process. For example BLAS which is called by NumPy to do linear algebra can use **multithreading** to run faster if multiple cores are available to the process. 

- Although a Python program can call packages like NumPy/BLAS that are sped up by doing multithreading on multiple cores, only one Python interpreter at a time can run in a process (for now - there is a [proposal](https://peps.python.org/pep-0703/) to relax this). Thus to carry out parallel *Python* operations **multiprocessing** is required. This can take several different forms:
  
  - Python has a standard package [**`multiprocessing`**](https://docs.python.org/3/library/multiprocessing.html) that can run parallel processes on the different cores of a single CPU (maybe on all the cores in the typically two CPUs in an HPC node? I’m a bit unclear on this).
  
  - The C++ package [**OpenMP**](https://www.openmp.org/) (Python bindings [**PyOMP**](https://github.com/Python-for-HPC/PyOMP)) can also run parallel processes on the different cores single CPU (or single node = typically two CPUs?).
  
  - [**MPI**](https://en.wikipedia.org/wiki/Message_Passing_Interface) can run parallel processes on the different cores of a single CPU and **also on multiple nodes connected by a network**. Implementations of MPI go under names like [**OpenMPI**](https://www.open-mpi.org/) (not to be confused with the non-MPI single-node multiprocessing package OpenMP) and [**MPICH**](https://www.mpich.org/). A Python interface to the installed version of MPI is provided by [**MPI for Python (`mpi4py`)**](https://mpi4py.readthedocs.io/en/stable/). 
    
    - Rather than directly use MPI, various higher-level applications like [**Spark**](https://spark.apache.org/), [**Dask**](https://www.dask.org/), or [**Charm4py**](https://charm4py.readthedocs.io/en/latest/) (perhaps no longer supported) can be used to coordinate parallel operations between cores and nodes. These applications can use MPI, and do not require MPI coding by the user. 
  
  - Parallel Python processes that do not need to communicate at all can be run on separate nodes of an HPC cluster as separately launched jobs, without using any of the things mentioned above.

- I haven’t used either of the Python packages `multiproccessing`, PyOMP but I believe the `multiprocessing` package has more high-level language support for Python. I believe this type of single-CPU, multi-core multiprocessing should be possible in an Apptainer container as described in this document, but I haven’t tried this.

Finally, some Python jargon: A text file with extension `.py` containing Python language statements is sometimes called

- a **program**, considering it to express an algorithm like a C++ program, or
- a **script**, considering it to be a set of high-level directives, or
- a **module**, considering it to be code that could be imported into another `.py` file.
  In this document these three terms are used interchangeably with apologies to those who make distinctions between them.

### Hardware used<a id="hardware"></a>

#### PCs<a id="pcs"></a>

The commands and test programs shown in this document were tested on one or more of these PCs (only two of these have GPUs):

- **candela-20:** Assembled from parts 9/19, 3.6 GHz AMD Ryzen 5 3600 6-core CPU, 32 GB RAM, GPU similar to NVIDIA GeForce GTX 1050 Ti, running Ubuntu 24.04.
- **candela-21:** Assembled from parts 11/22, 3.4 GHz AMD Ryzen 9 5950X 16-core CPU, 64 GB RAM, GPU similar to NVIDIA GeForce GTX 1060 6 GB VRAM, running Ubuntu 24.04.
- **hoffice:** Lenovo A10 510-23 purch 11/17, 2.4 GHz Intel Core i5 4-core CPU, 8 GB RAM, no NVIDIA-type GPU, running Ubuntu 24.04.

#### GPUs<a id="gpus"></a>

A more detailed version of this table is in the section [A few of NVDIA's many GPUS, with test results](#gpu-list) below. The GPUs in candela-20 and candela-21 were actually EVGA models equivalent to the NVIDIA models listed here.  Limited information on other GPUs available on the Unity cluster is in the section [Picking a GPU on Unity](#pick-gpu) below.

| NVIDIA Model: | GeForce GTX 1050 Ti | GeForce GTX 1660 | Tesla T4                     | Tesla V100 DGXS    | Tesla A100 SMX4    | Hopper H100 SXM5 |
| ------------- | ------------------- | ---------------- | ---------------------------- | ------------------ | ------------------ | ---------------- |
| Where:        | candela-20 GPU      | candela-21 GPU   | Best free GPU on Colab 10/23 | Available on Unity | Available on Unity | Exists on Unity  |
| Price:        | $250 9/19           | $297 11/22       | $2,000 10/23                 | $1,500 10/23       | $15,000 10/23      | $30,000 10/23    |

#### Unity HPC cluster<a id="unity-intro"></a>

The HPC commands shown in this document were tested on the [Unity cluster](https://unity.rc.umass.edu/index.php) at UMass, Amherst. Unity runs the [Slurm](https://slurm.schedmd.com/overview.html) job scheduling system and as of 1/25 had:

- About 350 general-access nodes plus another 350 “preempt” nodes belonging to groups but available for general access when not otherwise being used (also additional nodes never available for general access).
- About 20,000 total cores in the general-access and preempt nodes, with individual nodes mostly having two CPUs and between 24 and 192 cores.
- GPUs on about half of the nodes, with individual nodes having between 2 and 8 GPUs giving about 1000 total GPUs.  The GPU models (thus capabilities) varied widely, with the newest/best GPUs on the preempt modes as might be expected.

Detailed information on using Unity is in the section [Unity cluster at UMass, Amherst](#unity-cluster) below.

### Pip, Conda, and APT<a id="pip-conda-apt"></a>

- [**Pip**](https://pypi.org/project/pip/) (package installer for Python) installs Python packages by default from [**PyPI**](https://pypi.org/), the open-source Python Package Index.
  
  - Some Python packages installed by pip require libraries written in other languages; pip can download the source code for these libraries and compile it. Alternatively pip can download a **wheel** (replaces an earlier thing called an **egg**) that has the needed libraries pre-compiled. Pip will prefer a wheel to uncompiled libraries if available, and will cache wheels for later use. A few pip commands:
    
    ```
    $ pip install package            # install a package
    $ pip install package=2.3.4      # install a specific version
    $ pip list                       # list all installed packages
    $ pip freeze                     # list all installed packages with more info
    $ pip show package               # show info on package: where it is,what depends on it
    $ pip uninstall package          # uninstall package
    ```
  
  - Pip can also be used to install a package located on the local computer, which will allow the package to be imported from any directory on that computer. The local package will be a directory (repo in the example below) that contains the files `pyproject.toml` and `setup.py` and one or more subdirectories containing the Python modules that make up the package. Instructions for setting up a package so it can be installed with pip are [here](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder) (see the the answer “Solution without sys.path hacks”); also more up-to-date instructions including `pyproject.toml` are in Appendix B of this [Git/GitHub](https://github.com/doncandela/gs-git.git) tutorial. Once a package has been set up in this way, it can be installed as follows:
    
    ```
    $ cd ~/../repo               # switch to package directory
    $ pip install -e .           # install from current directory, -e means package
                                 # can be edited without re-installing
    ```
  
  - If a **virtual environment** has been created and activated by **venv** or by **Conda**, pip will install Python packages “in that environment”, i.e. they will only be visible and importable when that environment is activated.
  
  - You often see `python -m pip install package` used rather than the simpler commands listed above. I haven’t found this necessary when using pip inside a virtual environment; a discussion is [here](https://snarky.ca/why-you-should-use-python-m-pip/).
  
  - You will often see `pip3` rather than `pip`. If you are in an environment with Python 3 installed (vs. the earlier Python 2) they should point to the same command -- you can check this by comparing the output from `pip --version` with that from `pip3 --version`.

- [**Conda**](https://anaconda.org/anaconda/conda) combines and extends the package-management functions of **pip** and the environment-management functions of **venv**. Conda can install packages and libraries for any language, not just Python, which means Conda can install Python itself.
  
  - **Anaconda** and **Miniconda** are **distributions** of packages (many, and not so many respectively).   
  
  - It seems preferable to use Conda to install packages when they are available as Conda packages, but many packages (and more recent versions of packages) are not available as Conda packages and can only be installed using pip.
    
    - Frequently more up-to-date Conda packages are available from [**conda-forge**](https://conda-forge.org/) than from the default Conda channel -- see example below on how to use.
    - Here is an [article on using Conda and pip together](https://www.anaconda.com/blog/using-pip-in-a-conda-environment); it says **pip should be used *after* Conda**. In other words (I believe) creating a Conda environment automatically creates a venv environment of the same name, and activating the Conda environment switches to that venv.   
    - When using pip and Conda together, **the Conda environment should be created including Python** as in all the examples in this document.  Then, if pip is used in this environment it will install things in this environment.  If the Conda environment does not include Python, pip will try to modify the global environment which is usually not what is wanted (and is not allowed on an HPC cluster like Unity).
  
  - Some Conda commands:
    
    ```
    $ conda update conda                  # good to do this before other conda commands
    $ conda search openmpi                # search for openmpi packages in default channels
    $ conda search -c conda-forge openmpi # search in channel conda-forge
    $ conda create -n p39 python=3.9.12   # create environment p39 with specific Python version
    $ conda activate p39                  # activate environment p39, will change prompt
    (p39)$ conda install matplotlib numba numpy pyyaml scipy  # install packages to current
                                                              # environment p39
    (p39)$ conda install spyder=5.2.2 # install Spyder IDE to this evironment
    (p39)$ conda install -c conda-forge quaternion            # install a package from the
                                                              # Conda-Forge repository
    $ conda env list                     # list all defined environments
    $ conda create -n enew --clone eold  # create environment enew by cloning existing eold
    $ conda env remove -n p39            # get rid of environment p39
    $ conda rename -n eold enew          # rename environment eold to enew (actually clones
                                         # eold to enew then removes eold so not fast)
    ```
  
  - Sometimes `conda create` or `conda install` will fail with the message `Solving environment: failed`.  Tips to avoid or fix this situation:
    
    - Include all of the packages needed in the initial `conda create` command, rather than adding them later with `conda install` (or at least all of the packages that seem to be interacting).
    - Let Conda choose the version numbers rather than specifying them.
    - Get packages from `conda-forge` as shown above.
    - If you do specify a Python version, often an **earlier Python version** will be compatible with the available Conda versions of the other packages you need.
    - An IDE like Spyder has many complex dependencies. But when used only to edit files (as opposed to running them) Spyder can be run from the base or no environment, so there is no need to install it in your environments.
  
  - **Running a Jupyter Notebook in a Conda environment.**
    
    - On a **Linux PC**, it seems to work fine to activate the Conda environment, then issue the `jupyter notebook` command.
    - On an **HPC cluster like Unity** Jupyter Notebooks are can be run in a browser-based application like Unity OnDemand.  In this case `ipykernel` is used to make the environment available when a JN is launched, as detailed [here](#unity-interactive).

- [**APT**](https://documentation.ubuntu.com/server/how-to/software/package-management/index.html) (Advanced Packaging Tool) is used to **install software packages system-wide** (for all users, in all environments) and **must be run with root (sudo) privilege** except  when used to find out about installed packages.  The **`apt`** command is more modern than and basically replaces the earlier **`apt-get`** although you will see references to both, often mixed together.  Some common APT commands:
  
  ```
  $ sudo apt update             # update package index, run this before running other apt cmds
  $ sudu apt install <package>  # install a package
  $ sudo apt upgrade            # upgrade all installed packages
  $ sudo apt remove <package>   # remove a previously installed package
  $ apt list --installed        # list all installed packages
  $ apt list <package> --installed  # list installed packages with this name
  ```

### Conda environments and test code used in this document<a id="envs-testcode"></a>

- **Conda environments and Apptainer**. Often, and for all the examples in this document, there is no need to create a Conda environment when an Apptainer container is being used - the container serves as an environment.  The exception is for a set of containers running MPI, as in this case it is necessary to have an MPI system running outside the containers.

- The following Conda environments are created and used on a PC, when Apptainer is not being used (except for **`ompi`** which is used when an Apptainer container is run). 
  
  - **`p39`** (defined just above) has Python 3.9, NumPy, SciPy, etc but does not have OpenMPI, PyTorch, or CuPy.
  - **`dfs`** (defined in [Installing a local package](#local-package)) environment for trying out  the local package `dcfuncs`.
  - **`m4p`** (defined in [MPI on a Linux PC](#mpi-pc)) includes OpenMPI 5 and `mpi4py`, so MPI can be used by a Python program.
  - **`dem21`** (also defined in [MPI on a Linux PC](#mpi-pc)) is like `m4p` but additionally includes the locally-installed package `dem21` and additional packages that `dem21` imports.
  - **`pyt`** (defined in [Installing CUDA-aware Python packages...](#pytorch-cupy)) adds PyTorch, which can be run with or without a GPU.
  - **`gpu`** (also defined in [Installing CUDA-aware Python packages...](#pytorch-cupy)) adds CuPy, which has NumPy/SciPy-like functions that run on a GPU.
  - **`ompi`** (defined in [A container that can use MPI](#mpi-container)) includes only OpenMPI and Python as an example of a minimal environment for running a container that uses MPI.

- The following Conda environments are created and used on the Unity HPC cluster, when Apptainer is not being used (except for **`ompi`** which can be used when an Apptainer container is run).  They generally do the same things as the corresponding environments defined for PCs listed just above.
  
  - **`npsp-jn`** (defined in [Using interactive apps: Jupyter, MATLAB... ](#unity-interactive)) has NumPy, SciPy, and ipykernel so the environment can be used in a Jupyter Notebook on Unity.
  - **`npsp`** (defined in [Using modules and Conda](#unity-modules-conda)) has NumPy, SciPy, and Matplotlib, but not CuPy.
  - **`dfs`** (also defined in [Using modules and Conda](#unity-modules-conda)) has NumPy and the local package `dcfuncs` installed.
  - **`m4p`** (defined in [Using MPI on Unity (without Apptainer)](#unity-mpi)) includes OpenMPI 5.0.3, and `mpi4py`, so MPI can be used.  We also define **`m4pe`** which is like `m4p` except that it links to an external OpenMPI package which requires loading an OpenMPI module.
  - **`dem21`** (also defined in [Using MPI on Unity (without Apptainer)](#unity-mpi)) is like `m4p` but additionally includes the locally-installed package `dem21` and additional packages that `dem21` imports.  We also define **`dem21e`** which (like `m4pe`) links to an external OpenMPI package.
  - **`pyt`** (defined in [Using PyTorch on Unity](#pytorch-unity) adds PyTorch, which can be run with or without a GPU.  It also includes **`ipykernel`** so PyTorch can be run interactively in an JN using Unity OnDemand.
  - **`gpu`** (defined in [Using a GPU in Unity (without Apptainer)](#unity-gpu)) includes CuPy, which has NumPy/SciPy-like functions that run on a GPU.
  - **`ompi`** (defined in  [Running containers that use MPI](#unity-mpi-container)) includes only OpenMPI and Python as an example of a minimal environment for running a container that uses MPI on Unity, when an OpenMPI module is not loaded.

- The following test code is used:
  
  - **`mpi_hw.py`** is an MPI "Hello world" program that verifies that a functional MPI system is installed that can run multiple copies of a Python program in parallel.
  - **`osu_bw.py`** measures the communication speed between two MPI ranks.
  - **`count.py`** times how fast a Python program can count.
  - **`count_mpi.py`** times the counting speeds of multiple Python processes running simultaneously using MPI.
  - **`threadcount.py`** uses timing to estimate the number of threads in use while NumPy is multiplying matrices.
  - **`threadcount_mpi.py`** estimates the number of threads in use in each rank of an MPI run.
  - **`gputest.py`** makes dense and sparse matrices of various sizes and floating-point types, and times operations using these matrices on the CPU and (if available) the GPU. If run in an environment without CuPy like **`p39`**, only CPU tests will be run. But if run in **`gpu`** and a GPU can be initialized, will also run GPU tests.
  - **`np-version.py`** is a very short program that imports NumPy and prints out its version.
  - **`dcfuncs`** is small package of utility functions, used in this document as an example of a Python package [installed locally](#local-package).  It is available from the public GitHub repo [doncandela/dcfuncs](https://github.com/doncandela/dcfuncs), which also includes the test programs **`test_util.py`**, etc, mentioned in this document.
  - **`dem21`** is a complex package for doing DEM simulations of granular media using MPI parallelism. It is stored  in the currently private GitHub repo [doncandela/dem21](https://github.com/doncandela/dem21).  It is used in this document as a test and example of how a large, complex MPI code can be run.  The `dem21` repo includes the test program `boxpct.py` mentioned in this document. Also mentioned is a much more complex granular-memory simulation program called `mx2.py`.  While these codes are not available publicly, the examples here may be generally useful to show how an MPI program using many parallel ranks can be run on a PC or an HPC cluster, in both cases either non-containerized or containerized using Apptainer. The following shell scripts are used to run `mx2.py` on a PC, which requires various additional files not detailed in this document:
    - **`mx2.sh`** runs `mx2.py` without Apptainer.
    - **`mx2-app.sh`** runs `mx2.py` using the Apptainer container built by **`dem21.def`**.

- The following Apptainer definition files are used. They are all discussed in [Using Apptainer on a Linux PC](#apptainer-pc) below.  They have been all been used to build container images (`.sif` files) on PCs, which can then be run successfully both on the PCs and on Unity.
  
  - **`os-only.def`** makes a container that contains only the **Ubuntu OS**.
  
  - **`pack.def`** makes a container that contains Linux, Conda, and the **Miniconda** package distribution, and installs a few selected packages in the container.
  
  - **`dfs.def`** makes a container with the local package **`dcfuncs`** installed in it.
  
  - **`m4p.def`** makes a container with **OpenMPI** and **MPI for Python** installed in it, so it can be used to run MPI programs.
  
  - **`dem21.def`** makes an MPI-enabled container like the one made by `m4p.def`, but it also has the more elaborate MPI-using package `dem21` installed in the container.
  
  - **`gpu.def`** makes a container that imports **CuPy** so it can be used to run Python programs that use CuPy to run a GPU.

- The following sbatch scripts are defined for use with Slurm on the Unity cluster:
  
  - **`simple.sh`** (defined in [Example of a simple batch job](#simple-batch)) runs job that uses none of MPI, a GPU, or Apptainer.  **`simple2.sh`** has an additional command to print the sbatch script into the output file.
  - **`osu_bw.sh`** (defined in [Using MPI on Unity (without Apptainer)](#unity-mpi)) runs the MPI messaging-bandwidth test program `osu_bw.py`.
  - **`threadcount_mpi.sh`** and **`threadcount_mpi2.sh`** (both also defined in [Using MPI on Unity (without Apptainer)](#unity-mpi)) run `threadcount_mpi.py` to demonstrate the use NumPy multithreading along with MPI parallelism.
  - **`boxpct_mpi.sh`** (also defined in [Using MPI on Unity (without Apptainer)](#unity-mpi)) runs `boxpct.py` (which uses the `dem21` package) in MPI-parallel mode.
  - **`gputest.sh`** (defined in [Using a GPU on Unity (without Apptainer)](#unity-gpu)) runs `gputest.py` which uses a GPU.
  - **`simple-app.sh`** (defined in [Running a container interactively or in batch job](#unity-run-container)) uses an Apptainer container to run `gputest.py` without a GPU.
  - **`osubw-app.sh`** (defined in [Running containers that use MPI](#unity-mpi-container)) uses an Apptainer container to run the MPI messaging-bandwidth test program `osu_bw.py`.
  - **`boxpct-app.sh`** (also defined in [Running containers that use MPI](#unity-mpi-container)) uses an Apptainer container to run the test program for the `dem21` package `boxpct.py` in MPI-parallel ranks.
  - The following sbatch scripts run the granular-memory simulation program `mx2.py`, which requires various additional files not detailed in this document:
    - **`mx2-unity.sh`** runs `mx2.py` on Unity without Apptainer.
    - **`mx2-unity-app.sh`** runs `mx2.py` on Unity using the Apptainer container built by **`dem21.def`**.
  - **`gputest-app.sh`** (defined in [Running a container the uses a GPU](#unity-gpu-container)) uses an Apptainer container to run `gputest.py`  which uses a GPU.

### Installing a local package on a PC<a id="local-package"></a>

Sometimes it is convenient to write or otherwise come by a **package of Python modules** (containing class and function definitions), copy the package somewhere on the computer being used, and then make it possible to import the package from any directory on the same computer -- this is a **local package**, as opposed to a package downloaded from a repository of published packages like Anaconda or PyPi.  A way to structure such a local package is outlined in Appendix B of the cheat sheet  [Getting started with Git and GitHub](https://github.com/doncandela/gs-git).

In other sections of this document it is shown how a local package like this can be [installed on an HPC cluster](#local-package-unity) like Unity (in user space), and how it can be [installed in an Apptainer container](#local-package-container) which can then be used on a PC or on an HPC cluster.  As a starting point this section shows how a local package can be installed on a Linux PC, not using Apptainer.

- The package used for these examples is **`dcfuncs`**, a small set of utility functions that can be downloaded from <https://github.com/doncandela/dcfuncs> -- hit `Download Zip` under the `<> Code` tab.  Alternatively, the package can be cloned into the current directory on the local PC by doing
  
  ```
  $ git clone https://github.com/doncandela/dcfuncs.git
  ```
  
  This repository has the following structure:
  
  ```
  dcfuncs/
     src/
        dcfuncs/            # installable package
           util.py          # utility functions: error exit, profiling, etc.
           configs.py       # reading yaml configuration files
        test/               # code to test if package is installed and usable
           test-util.py
           test-configs.py
           test-util.ipynb
     pyproject.toml
     setup.py
  ```
  
  (Read the comments in `util.py` and `configs.py` to find out what the functions do -- not relevant for present purposes.)

- Make a Conda environment `dfs` in which to install and test the `dcfuncs` package:
  
  ```
  $ conda update conda
  $ conda create -n dfs python=3
  $ conda activate dfs
  (dfs)..$ conda install numpy          # needed by dcfuncs
  ```

- Download this package and go to the subdirectory **`test`**. Before the package is installed, running any of the `test-...` programs will give a `ModuleNotFound` error:
  
  ```
  (dfs)..test$ python test-util.py
  Traceback (most recent call last):
    File "test-util.py", line 7, in <module>
    import dcfuncs.util as dutil
  ModuleNotFoundError: No module named 'dcfuncs'
  ```

- Go to the top directory in the repository and use this `pip` command to install the package (the optional `-e` makes the package editable without reinstalling, while the `.` means install from the current directory).
  
  ```
  (dfs)...dcfuncs$ pip install -e .
  (dfs)...dcfuncs$ pip list               # this will show dcfuncs installed
  ```

- Now the test programs run without error:
  
  ```
  (dfs)..test$ python test-util.py
  This is: dutil.py 8/19/24 D.C.
  Using: util.py 8/18/24 D.C.
  
  Testing zz:
  - WARNING from util.py test code - This is just a warning.
  
  Testing stoi:
  stoi results = 93801881091158, 6318, 249613385242335
         ...
  ```

### Parallel execution on multiple cores<a id="multiple-cores"></a>

Modern CPU chips have multiple **cores** (as of 1/25 CPUs in consumer PCs have 4-10 cores while CPUs for HPC have up to 128 cores), with each core able to execute an independent **thread** within a process, or alternatively a full **process** which can execute Python code.  Some fine points about **counting the number of cores**:

- In an HPC cluster like Unity, a **node** is typically a board with two **sockets** each holding a CPU chip with shared memory between the CPUs, often along with one or more GPUs  -- so the cores per node is typically twice the cores per CPU chip. 
- Typical CPU chips can use ["hyperthreading" or "hardware multithreading"](https://en.wikipedia.org/wiki/Hyper-threading) to make the number of **virtual cores** available to software to be twice the number of **physical cores**, giving a modest increase (less than two) in parallel throughput.  AMD calls this technology "simultaneous multithreading" while Intel calls it "hyperthreading" -- for simplicity only the latter term is used here.
- In some SLURM and MPI settings, "cpu" refers to a **core** not a CPU chip. For example the `-c`,`-cpus-per-task` option to the SLURM `sbatch` command  and the `--cpus-per-proc` option to the OpenMPI `mpirun` command both set the the number of cores (not CPU chips) allocated to a process

It might be thought that a PC using *n* cores should get *n* times as much computation done per second, provided the code can be efficiently split up to have *n* different things going on at the same time.  However, there are (at least) three factors that often make the gain in computation speed from using multiple cores less than the number of cores, even with efficiently parallel code: (a) the **clock speeds** of the cores may depend on how many are in use, due to automatic **thermal management** by the CPU chip, (b) the cores may need to **contend for memory access**, and (c) the available cores can be used for **multithreading by NumPy and similar packages**, or for **MPI multiprocessing**, or for **both of these things simultaneously**.

- **Clock speeds and thermal management.**  To  show this effect we run the the program **`count_mpi.py`** on the **16-core PC candela-21** [mentioned above](#pcs). The The AMD Ryzen 9 5950X CPU chip in this PC has a **base clock speed of 3.4 GHz** and a **boost clock speed up to 4.9 GHz**.  `count_mpi.py` uses MPI to run a simple program that times how long it takes to count up to a specified number (1,000,000,000 in the examples shown here) on a chosen number of cores.  First we run on one core (the environment `m4p` is defined in [MPI on a Linux PC](#mpi-pc) below):
  
  ```
  (m4p)$ mpirun -n 1 -use-hwthread-cpus python count_mpi.py 1000000000
  This is rank 0 of 1 on candela-21 running Open MPI v5.0.3
  (rank 0) Counting up to 1,000,000,000...
  (rank 0)...done, took 1.815e+01s, 5.510e+07counts/s
                        ...
  ```
  
  If we check the clock speeds in another terminal while `count_mpi.py` is running, we see that a single core is running at 4.7 GHz, near the maximum boost clock speed:
  
  ```
  $ cat /proc/cpuinfo | grep 'cpu MHz'
  cpu MHz        : 4657.204
  cpu MHz        : 2200.000
  cpu MHz        : 2200.000
  cpu MHz        : 2200.000
  cpu MHz        : 2200.000
  cpu MHz        : 2200.000
           ...
  ```
  
    Next `count_mpi.py`  is run on all 32 virtual cores (`-use-hwthread-cpus` is used to turn on hyperthreading):
  
  ```
  (m4p)$ mpirun -n 32 -use-hwthread-cpus python count_mpi.py 1000000000
  This is rank 5 of 32 on candela-21 running Open MPI v5.0.3
  This is rank 4 of 32 on candela-21 running Open MPI v5.0.3
  (rank 4) Counting up to 1,000,000,000...
  This is rank 16 of 32 on candela-21 running Open MPI v5.0.3
                         ...
  (rank 1)...done, took 3.957e+01s, 2.527e+07counts/s
  (rank 14)...done, took 4.001e+01s, 2.499e+07counts/s
  (rank 16)...done, took 4.002e+01s, 2.499e+07counts/s
  ```
  
  In this case all cores are running at 3.75 GHz, showing that the CPU will not use the maximum boost clock when all cores are in use:
  
  ```
  $ cat /proc/cpuinfo | grep 'cpu MHz'
  cpu MHz        : 3750.022
  cpu MHz        : 3750.079
  cpu MHz        : 3750.026
  cpu MHz        : 3750.091
  cpu MHz        : 3750.062
  cpu MHz        : 3750.102
  cpu MHz        : 3750.084
             ...
  ```
  
  As 16 physical cores are being used, one could expect the total count rate for all processes to be about 16*(3.75GHz/4.66GHz) = 12.8 times more than the single-core count rate.  In fact it is 32*2.5e7/s which is 14.6 times more than the single-core rate, suggesting a modest advantage from using hyperthreading.  So for this simple program which requires no memory access, the advantage of using all cores along with hyperthreading (14.6) was nearly equal to the number of cores (16) despite the reduced clock rate due to using all cores.
  
  It was also found for this particular combination of software and hardware:
  
  - Running on 16 cores with hyperthreading disabled (i.e. not using `use-hwthread-cpus`) gave a somewhat smaller throughput advantage (11.9) over using one core.  So in this case hyperthreading gives a 23% speadup.
  - Runinng on 16 cores with hyperthreading worked poorly (throughput advantage 7.6). From examining the clock speeds it seems that the 16 tasks were not using all 16 physical cores in this case.  In other words some physical cores ran 2 tasks in their two virtual cores, while other physical cores ran none.

- **Memory contention.**  The cores in a CPU chip must ultimately read inputs and save outputs of computations in RAM external to the chip, and the bandwidth for moving data between RAM and the CPU cores can be the time-limiting factor for code, rather than the processing speed of the cores.  CPUs have a [hierarchy of caches](https://en.wikipedia.org/wiki/Cache_hierarchy) to help mitigate this bottleneck, but it seems that memory access can still dominate over core processing speed for operations like sparse-matrix multiplication (see [CPU and GPU test results](#gpu-list) below). Here is an interesting [article](https://siboehm.com/articles/22/Fast-MMM-on-CPU) showing how the memory-intensive operation of dense-matrix multiplication is optimized to largely eliminate the memory bottleneck.

- **NumPy multithreading.**
  
  - The (non-MPI) Python script **`threadcount.py`** uses (process time)/(wall time) to estimate the number of cores in use when NumPy makes and then multiplies matrices filled with random numbers. On the 16-core PC [candela-21](#pcs) (the environment `npsp` is defined in [Using modules and Conda](#unity-modules-conda) below):
    
    ```
    $ conda activate npsp
    (npsp)$ python threadcount.py
    Making 10,000 x 10,000 random matrices...
    ...took 1.815e+00s, average threads = 1.000
    Multiplying matrices 3 times...
    ...took 5.426e+00s per trial, average threads = 15.959
    ```
    
    It can be seen that NumPy (specifically using `rng.normal` with `rng` a random number generator returned by `numpy.random.default_rng`) uses only one core to make a random-filled matrix, but (specifically using `numpy.matmul`) uses all 16 cores to multiply the matrices. By default the multithreaded NumPy functions (or rather the underlying linear algebra packages) are **greedy**, using all available cores in this the example above.
    
    According to the NumPy docs, it may be possible to control the the number of threads used by the linear algebra packages called by NumPy by setting the environment variable **`OMP_NUM_THREADS`**.  This worked on candela-21:
    
    ```
    (npsp)$ export OMP_NUM_THREADS=4
    (npsp)$ python threadcount.py
    Making 10,000 x 10,000 random matrices...
    ...took 1.809e+00s, average threads = 1.000
    Multiplying matrices 3 times...
    ...took 8.235e+00s per trial, average threads = 3.998
    
    (npsp)$ export OMP_NUM_THREADS=1
    (npsp)$ python threadcount.py
    Making 10,000 x 10,000 random matrices...
    ...took 1.806e+00s, average threads = 1.000
    Multiplying matrices 3 times...
    ...took 3.112e+01s per trial, average threads = 1.000
    ```
    
    For this particular example, compared to using a single core, `np.matmul` was 3.8 times faster using 4 cores and 5.7 times faster using 16 cores -- so `np.matmul` made excellent use of 4 cores but had only modest gains beyond that point.
  
  - In the non-MPI experiments on the [PCs used for this document](#pcs) Numpy never used hyperthreading -- the number of threads reported by `threadcount.py` and the number of cores in use shown by the Ubuntu System Monitor never exceeded the number of physical cores, even though the System Monitor showed twice this number of CPU's.  It may be possible to control this behavior using additonal **`OMP_...`** environment variables, as discussed [here](https://theartofhpc.com/pcse/omp-affinity.html); this was not tried.
  
  - **Tradeoff with MPI.** If MPI is used, and the total number of cores available is limited, there may be a trade off between giving each MPI rank more cores (so NumPy can multithread) and running a program in more MPI ranks.  This is discussed further in [Hyperthreading and NumPy multithreading with MPI](#multithread-mpi) below.

## Part 1: MPI, GPU, and Apptainer on a Linux PC<a id="on-linux-pc"></a>

### MPI on a Linux PC<a id="mpi-pc"></a>

#### Why do it<a id="why-mpi-pc"></a>

Using MPI, multiple copies of a Python program can run in parallel on the cores of a PC.  This can also be accomplished with the [Python `multiprocessing` package](https://docs.python.org/3/library/multiprocessing.html), which I haven't tried.

What MPI can do (and `multiprocessing` cannot do) is increase the parallelism to copies of Python running on **multiple computers connected by a network** - i.e. multiple nodes of an HPC cluster. Therefore a possible reason for developing MPI-parallel code on a PC is to enable eventual expansion to a higher degree of parallelism on an HPC cluster.  Note, however, that parallelism across all the cores of any single node of an HPC cluster could be accomplished without MPI by using the `multprocessing` package -- Unity nodes currently have up to 128 cores.

The most popular open-source MPI packages seem to be [**OpenMPI**](https://www.open-mpi.org/) and [**MPICH**](https://www.mpich.org/). Also, there are some other versions of MPI that are derived from MPICH: [**MVAPICH2**](https://mvapich.cse.ohio-state.edu/), [**Intel MPI**](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)... For brevity **only OpenMPI is discussed in this document**.

#### **Installing OpenMPI and MPI for Python (`mpi4py`) on a Linux PC**<a id="install-openmpi-pc"></a>

To run MPI-parallel programs written in Python, it is necessary to have available (a) a working MPI installation such as [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/) and (b) a Python package such as [MPI for Python (`mpi4py`)](https://mpi4py.readthedocs.io/en/stable/) to provide Python versions of the MPI functions. This section describes how to install OpenMPI and `mpi4py` on a PC, when an Apptainer container is not being used. Using OpenMPI + `mpi4py` in other situations is described in separate sections below: [with Apptainer on a PC](#mpi-container), [without Apptainer on the Unity HPC cluster](#unity-mpi), and [with Apptainer on Unity](#unity-mpi-container).

The docs for OpenMPI and `mpi4py` describe various ways of obtaining and installing these things, including building from source, but for the purpose of running an MPI-parallel Python program on a Linux PC the easiest thing is to use Conda to install both OpenMPI and `mpi4py` in an environment, here called **`m4p`**.  We also include some other packages likely  to be needed by programs running in this environment -- here we choose NumPy, SciPy, and Matplotlib. The version of OpenMPI installed can be found using `mpirun --version`, while much more detailed info is returned by `ompi_info`:

```
$ conda create -n m4p -c conda-forge openmpi=5 mpi4py python=3.12
$ conda activate m4p
(m4p)$ conda install -c conda-forge numpy scipy matplotlib
(m4p)$ mpirun --version
mpirun (Open MPI) 5.0.7
(m4p)$ ompi_info | head
                 Package: Open MPI conda@f424a898794e Distribution    
                Open MPI: 5.0.7                          
  Open MPI repo revision: v5.0.7                 
   Open MPI release date: Feb 14, 2025                 
                 MPI API: 3.1.0                   
            Ident string: 5.0.7                
                  Prefix: /home/dc/anaconda3/envs/m4p    
 Configured architecture: x86_64-conda-linux-gnu       
           Configured by: conda                    
           Configured on: Mon Feb 17 07:57:35 UTC 2025
```

#### Simple MPI test programs: `mpi_hw.py`  and `osu_bw.py` <a id="mpi-testprogs"></a>

- MPI works by **loading and starting $n$ copies of the same code**, each in its own process running on a separate core (or cores if, for example [NumPy multithreading is enabled](multithread-mpi))  Each process has a unique **rank**  in the range $0\dots n-1$, and a process can find its rank via an MPI function call -- and then use the rank to figure out what it should do (this is up to the programmer!).  To simultaneously run four copies (ranks) of  the Python program `myprog.py` we do
  
  ```
  $ mpirun -n 4 python myprog.py
  ```

- The "Hello world" of MPI programs, **`mpi_hw.py`** simply prints a message including the rank and other information.  Running `mpi_hw.py` in four ranks gives us four such messages, in an indeterminate order:**
  
  ```
  (m4p)..$ cd python-scripts; ls    # cd to directory containing test programs
  mpi_hw.py  osu_bw.py ...
  (m4p)..python-scripts$ mpirun -n 4 python mpi_hw.py
  Hello world from rank 0 of 4 on candela-21 running Open MPI v5.0.7
  Hello world from rank 3 of 4 on candela-21 running Open MPI v5.0.7
  Hello world from rank 1 of 4 on candela-21 running Open MPI v5.0.7
  Hello world from rank 2 of 4 on candela-21 running Open MPI v5.0.7
  ```
  
  Some fine points about the number of ranks:
  
  - Omitting `-n 4` above will set the number of ranks equal to the  number of available cores, while setting the number of ranks to more than the available cores will cause `mpirun` to fail.
  
  - Including the option  `--use-hwthread-cpus`  on `mpirun` will use **hyperthreading**, if available for the CPU used, to double the effective number of cores and thus double the allowed number of ranks.

- In `mpi_hw.py` the code in each rank runs independently, with no communication between ranks.  The main purpose of MPI is to carry out inter-rank communication, to enable non-trivial parallel algorithms.  **`osu_bw.py`** tests the **speed of communication between two ranks**, for messages of various sizes.  Here this test program was run on the PC candela-21.  As the two ranks are on cores on the same CPU chip (on a PC like this -- not necessarily true on an HPC cluster), the communication speed should be quite fast:
  
  ```
  (m4p)..python-scripts$  mpirun -n 2 python osu_bw.py
  2
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
           1                3.69
           2                7.34
           4               14.63
           8               29.26
          16               57.95
          32              115.64
          64              209.67
         128              379.26
         256              707.62
         512            1,523.25
       1,024            2,900.98
       2,048            5,324.45
       4,096            2,881.93
       8,192            4,948.91
      16,384            7,633.05
      32,768           10,637.36
      65,536           13,375.59
     131,072           14,828.03
     262,144           15,466.52
     524,288           16,143.36
   1,048,576           16,809.99
   2,097,152           17,052.31
   4,194,304           17,227.80
   8,388,608           16,646.96
  16,777,216           11,017.93
  ```
  
  It can be seen that the maximum inter-rank communication speed was about 17 GB/s on this PC ([candela-21](#pcs)).

#### More elaborate MPI programs using the `dem21` package<a id="mpi-dem21"></a>

Here we use the discrete-element-method (DEM) simulation package **`dem21`**  (not currently publicly available) as an example of a much more elaborate MPI program.  It is assumed that OpenMPI has been installed on the PC as [described above](#install-openmpi-pc).

- **Environment for running `dem21`.** With access the `dem21` repo is cloned from GitHub to a directory `foo/dem21`.  Then a suitable environment **`dem21`** is created similar to  `m4p` defined above but including the additional packages needed according to the instructions in the documentation `dem21.pdf` (in the repo).  Finally, the `dem21` package is installed in this environment (it was helpful to set the Python version to 3.11 and to break up the Conda install commands as shown here, otherwise Conda got stuck trying to solve the environment):
  
  ```
  (base)..foo$ git clone git@github.com:doncandela/dem21.git
  
  (base)..foo$ conda create -n dem21 -c conda-forge openmpi=5 mpi4py python=3.11
  (base)..foo$ conda activate dem21
  (dem21)..foo$ conda install -c conda-forge numpy scipy matplotlib dill numba pyaml
  (dem21)..foo$ conda install -c conda-forge quaternion
  (dem21)..foo$ cd dem21
  (dem21)..foo/dem21$ pip install -e .
  ```

- **Quick test program `boxpct.py`.**  Now it is possible to run the test program `boxpct.py` (included in the repo) in MPI-parallel mode:
  
  ```
  (dem21)..foo/dem21$ cd tests/box
  (dem21)..foo/dem21/tests/box$ export pproc=mpi # this tells boxpct.py to use MPI
  (dem21)..foo/dem21/tests/box$ mpirun -n 4 python boxpct.py
  - Started MPI on master + 3 worker ranks.
  THIS IS: boxpct.py 12/3/22 D.C., using dem21 version: v1.2 2/11/25
  Parallel processing: MPI, GHOST_ARRAY=True
  - Read 1 config(s) from /home/dc/Documents/RES/COMPUTERS/foo/dem21/tests/box/box.yaml
  
  SIM 1/1:
  Using inelastic 'silicone' grainlets with en=0.7 and R=0.500mm
  343 'sphere' grains in (7.66)x(7.66)x(7.66)mm box (phig=0.4), vrms=10.0m/s
  No gravity, 'hertz' normal force law, Coulomb w. Hookean spring friction with GG mu=0.1, GW mu=0
  -     Writing grain ICs x,v to /tmp/tmpkduz52n8/temp.grains
  - READYING SIM with 343 grains and 6 walls
                         ...
  ```

- **More resource-intensive runs with `mx2.py`.**<a id="mx2py"></a> Finally we run a much bigger, longer-running DEM simulation which will be repeated below using Apptainer and on Unity (and both), to see if there is any performance impact from containerizing the code, and to investigate the speed-ups that can be obtained from the larger core counts available on Unity.
  
  - The tested code is a simulation of a "granular memory" experiment on a dense pack of 10,240 tetrahedral grains (each composed of four spherical grainlets) with 450,000 time steps.
  
  - The simulation is carried out by the program `mx2.py`, continuing a sample-preparation simulation (not shown here) carried out by `ms.py` -- these programs call the `dem21` package which is run in run in MPI-parallel mode here.
  
  - `mx2.py` needs the input-signals package `msigs` so we install it in the `dem21` environment:
    
    ```
    ..$ conda activate dem21
    (dem21)..$ cd GMEM/msigs; ls   # cd to where msigs package is kept
    'msigs README'   pyproject.toml   setup.py   src   test
    (dem21)..GMEM/msigs$ pip install -e .
    ```
  
  - A rather complicated directory structure not detailed here is used to organize the simulations so they can generate the granular samples and then carry out memory simulations on these samples.  Each memory simulation is carried out in a subdirectory called **`cc-expts..`** by a shell script (or on Unity, an sbatch script) called **`cc-expts/mx2...sh`** , which runs the program  **`mx2.py`** located in a higher directory.
    
    Each of the various memory simulations described in this document (on a PC or on Unity, containerized or not, using different numbers of cores and/or hyperthreading) was uses a different  **`mx2..sh`** script. For this simple non-Apptainer, PC case the script is called **`mx2.sh`**:
    
    ```
    #!/bin/bash
    # cc-expts/mx2.sh 4/6/25 D.C.
    # Shell script to run granular-memory simulation program mx2.py non-containerized
    # on a PC, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
    #
    # Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
    # Reads default config file mx2.yaml in grandparent directory modified by
    # mx2mod.yaml in current directory.
    #
    # To run on N cores 1st activate environment 'dem21' then do
    #
    # ./mx2.sh N
    #
    export pproc=mpi
    mpirun -n $1 python ../../mx2.py mx2mod |& tee output
    # Alt version allows hyperthreading:
    #mpirun -n $1 python --use-hwthreads-cpus ../../mx2.py mx2mod |& tee output
    ```
    
    The simulation is run in the Conda environment **`dem21`**, in the directory `cc-expts` containing `mx2.sh` and other needed files not explained here:
    
    ```
    (dem21)..$ cd ..cc-expts; ls
    bw6-sigs.yaml  bw6.svg  mx2mod.yaml  mx2.sh signals.sh
    (dem21)...cc-expts$ ./mx2.sh 15 # run mx2.py in 15 MPI ranks
    - Started MPI on master + 14 worker ranks.
    This is: mx2.py 7/29/24 D.C.
    Using dem21 version: v1.2 2/11/25
    Imput signals made by: memsigs.py 8/23/24 D.C.
                    ...
    ```
  
  - With the chosen parameters (set by `.yaml` config files read by `ms.py` and `mx2.py`) the simulation spatial domain was divided into 216 "boxes",  each of which does independent work during each simulation step.  Thus up to 216 operations could be carried out in parallel, if sufficient MPI ranks were allocated.
  
  - The boxes are distributed by the `dem21` package as evenly as possible across the available MPI ranks: rank 0 is used by the overall control program, and each remaining rank holds a "crate" which in turn holds zero or more boxes. Computation by the boxes in each crate is sequential for each time step, thus one might expect the overall execution time to be roughly proportional to the number of boxes per crate.  For the maximum possible parallelism (one box per crate) the number of MPI ranks must be at least one greater than the number of boxes, i.e. at least 217 in the present case.
  
  - When run in 15 MPI ranks on the  16-core PC [candela-21](#pcs) with [hyperthreading disabled](#multithread-mpi) as it is by default:
    
    - There were 13 crates with 16 boxes and 1 crate with 8 boxes, requiring 16 serial box calculations at each time step.
    
    - The 450,000 step simulation required 15,937 s = 4.427 hr, or 3.46 microsec/step-grain.
    
    - Running the simulation required about 2.1 GB of memory beyond what the PC was using before the simulation was run.
  
  - The simulation was also run in 30 MPI ranks on the same PC with [hyperthreading enabled](#multithread-mpi) by supplying  `--use-hwthread-cpus` to the `mpirun` command in `mx2.sh`.
    
    - Now there were 27 crates with 8 boxes and 2 crates with 0 boxes, requiring 8 serial box calculations at each time step, i.e. half as many as without hyperthreading.
    
    - Now the simulation required 16,567 s = 4.602 hr, or 3.60 microsec/step-grain.  So despite the greater degree of parallelism with hyperthreading, in this case there was no overall advantage in in fact the program ran slightly slower.  It seems that whatever speed advantage was provided by hyperthreading ([expected to be of order 25%](#multiple-cores)) was negated by the the increased MPI communication required or other unknown factors.
  
  - **An even bigger simulation.**<a id="even-bigger-sim"></a> To provide more things to compare with Unity the same code was used to run an even bigger sim,  with ten times as many grains (100,450) but otherwise identical, on `candela-21` using 16 cores with hyperthreading disabled.  Now there were 1,728 boxes giving up to 116 boxes per crate, so other factors being equal one would expect the sim to take (116/16) = 7.3 times as long to run.
    
    - This simulation required 165,900 s = 46.1 hr to run (3.69 microsec/step-grain) -- 10.4 times longer than the smaller sim. This 
    
    - Only 5.5 GB of memory was required, showing that this is a very low-memory application.

#### Hyperthreading and NumPy multithreading with MPI<a id="multithread-mpi"></a>

- As discussed in [Parallel execution on multiple cores](#multiple-cores) above, "hyperthreading" or "simultaneous multithreading" is a feature of many CPU chips which makes each physical core act like two virtual cores.
  - It seems that hyperthreading is often turned off (or disabled?) in HPC clusters, for example some information on this for the Unity cluster is [here](https://docs.unity.rc.umass.edu/documentation/get-started/hpc-theory/threads-cores-processes-sockets/) and [here](https://docs.unity.rc.umass.edu/news/2023/06/june-5-maintenance-concluded/).
  - In trials running OpenMPI on Linux PCs, hyperthreading was turned off by default but is could be turned on by supplying the option **`--use-hwthread-cpus`** to **`mpirun`**.  I was unable to achieve significant performance improvements with hyperthreading (see previous section for an example), so it is not discussed further here.
- As also discussed in [Parallel execution on multiple cores](#multiple-cores), some NumPy functions (like matrix multiplication) can take advantage of multithreading on multiple cores to speed up.  I believe this is typcially via the use of OpenMP by the underlying BLAS functions employed by NumPy.
  - For non-MPI programs [it was found](#multiple-cores) that some NumPy functions will greedily use all available cores unless the environment variable `OMP_NUM_THREADS` is used to reduce the cores used.
  - For MPI programs on PCs, I found that the number of cores used by NumPy is controlled by the `mpirun` option `-cpus-per-proc` in concert with `OMP_NUM_THREADS`.  However, it seemed difficult to get much advantage in this way so detailed trials are not shown here.
  - When neither `OMP_NUM_THREADS` nor `-cpus-per-proc` is used, it seems  that NumPy functions called in a Python MPI program will be limited to a single core.
  - The section [Enabling NumPy multithreading in MPI batch jobs](#sbatch-multithread) below shows how NumPy multithreading can be controlled in and MPI program running on the Unity HPC cluster.

### Using an NVIDIA GPU on a Linux PC<a id="gpu-pc"></a>

#### Why do it<a id="why-gpu-pc"></a>

A relatively inexpensive GPU can offer significant speedups. For example in test results [shown below](#gpu-list) on the candela-21 PC assembled in 2022 the \$300 GPU was about four times faster than the \$550 16-core CPU chip for operations on large dense and sparse matrices.

If the code might eventually be transferred to an HPC cluster, the more capable GPUs on the HPC nodes should offer even greater speed-ups than this.

#### Non-NVIDIA GPUs<a id="non-nvidia"></a>

Both PyTorch and CuPy can use AMD GPUs, which use [ROCm](https://rocm.docs.amd.com/en/latest/what-is-rocm.html) drivers (vs CUDA drivers for NVIDIA GPUs).  It seems PyTorch can also use an MPS GPU on a Mac. I don't have an AMD GPU  or a Mac and haven't tried these things.

#### Installing NVIDIA drivers<a id="nvidia-drivers"></a>

These steps are only need once on a given PC, unless updating to newer versions.

- Find out if (and which) NVIDIA GPU hardware is installed:
  
  ```
  $ sudo lshw -C display
  ```

- By default Ubuntu will use the open-source X driver for NVIDIA hardware, but this seems not to be appropriate when using the GPU for computation, and sometimes even just for graphics (e.g. Mayavi did not function correctly on candela-21 with the X driver). Thus a **proprietary NVIDIA driver** should be installed.

- It’s possible to [download and install drivers from NVIDIA](https://www.nvidia.com/download/index.aspx) but this will require stopping the X driver, etc (didn’t try).

- Instead it’s easier to install from Linux channels following e.g. [instructions here](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/). This can be done in Terminal, but easiest to use the Ubuntu graphical **Software and Updates** app: The **Additional Drivers** tab should show the GPU and available drivers (from NVIDIA as well as X) – pick most recent (proprietary, tested) NVIDIA driver, Apply Changes, then when completed reboot system.

- Now you can use the text-based and/or graphical apps supplied by NVIDIA to check that the driver is functioning and get info on the GPU:
  
  ```
  $ nvidia-smi         # text-based
  $ nvidia-settings    # graphical
  ```
  
  Note the **CUDA Version** from nvida-smi (was 12.2 on candela-20 PC 1/25)

#### Installing  CUDA-aware Python packages: PyTorch, CuPy...<a id="pytorch-cupy"></a>

- To use the GPU in Python programs, a Python package must be installed that knows how to access and use the GPU.  Three such packages are discussed briefly here:
  
  - **PyCUDA**, for running CUDA-C++ code from a Python program.
  - **PyTorch**, a popular machine-learning platform that **can be used with or without a GPU**.
  - **CuPy**, which provides GPU-accelerated replacements for many NumPy and SciPy functions.

- [**PyCUDA**](https://pypi.org/project/pycuda/) enables a Python program to directly operate an NVIDIA GPU in the [**CUDA-C++**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) language.  CUDA-C++ code is passed to PyCUDA functions as multiline strings, for example:
  
  ```
  mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
  ```
  
  To used PyCUDA you will need to learn about the NVIDIA GPU architecture with its kernels, threads, streams, etc.
  I have not used PyCUDA and it is not discussed further in this document.

- [**PyTorch**](https://pytorch.org/) is a machine-learning platform that can be used with or without a GPU. 
  
  - **Installing PyTorch on a Linux PC.**  First we make a Conda environment `pyt`  in which to run PyTorch,  with other packages that will be used -- here we have chosen NumPy, SciPy, Matplotlib, and Jupyter Notebook but I think none of these are required:
    
    ```
    (base)..$ conda update conda 
    (base)..$ conda create -n pyt python=3                   # 1/25 installed python 3.13.1
    (base)..$ conda activate pyt
    (pyt)..$ conda install numpy scipy matplotlib jupyter    # this downgraded python to 3.12.8
    (pyt)..$ jupyter notebook                                # check that JN works
    ```
    
    Next install PyTorch by running the appropriate command from the [PyTorch Getting Started page](https://pytorch.org/get-started/locally/). The installation command depends on which version of CUDA is installed, if any -- since CUDA 12.2 was installed I selected the nearest version no later than 12.2 which was 12.1:
    
    ```
    (pyt-gmem)..$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    
    As mentioned above in remarks on [Conda](#pip-conda-apt), on a Linux PC a Conda environment can be used with Jupyter Notebook simply by doing `conda activate ...` to activate the environment then `jupyter notebook` to start a JN. To check that PyTorch is usable, run this code in a JN which creates a small tensor filled with random numbers (can also run this code in Python started at the terminal):
    
    ```
    import torch
    x = torch.rand(5, 3)
    print(x)
    
    tensor([[0.3366, 0.6316, 0.7751],
            [0.6390, 0.9224, 0.9920],
            [0.3413, 0.4789, 0.9909],
            [0.3987, 0.7204, 0.1965],
            [0.7794, 0.5492, 0.6764]])
    ```
  
  - **Using the GPU in PyTorch.** The fundamental objects in PyTorch are **tensors**, which are NumPy arrays with additional features: the ability to store gradients, participate in backpropagation, etc.  A tensor can exist on either the CPU or the GPU, and you cannot do operations between tensors in two different places (e.g. multiply a tensor on the CPU by a tensor on the GPU).  PyTorch makes it easy to move tensors to the GPU if it exists, otherwise leave them on the CPU as this example (run in a JN) shows.
    
    First we set a string `DEVICE` to be `'cuda'` if an NVIDIA GPU is available, otherwise `'cpu`'
    
    ```
    import torch
    DEVICE = ('cuda' if torch.cuda.is_available()              # Nvidia GPU with CUDA
              else 'mps' if torch.backends.mps.is_available()  # MAC GPU with MPS
              else 'cpu')                                      # no GPU
    print(f'Availble device: {DEVICE}')
    
    Availble device: cuda
    ```
    
    Next we make two $3\times3$ tensors on the CPU, and get versions of them that are moved to the available device (CPU or GPU):
    
    ```
    x = torch.rand(3,3)
    y = torch.rand(3,3)
    xdvc = x.to(DEVICE)
    ydvc = y.to(DEVICE)
    x.device,xdvc.device
    
    (device(type='cpu'), device(type='cuda', index=0))
    ```
    
    We can multiply the two tensors moved to the available device.  **This code works whether or not a GPU is available.**
    
    ```
    torch.mm(xdvc,ydvc)
    
    tensor([[0.7678, 0.7135, 0.3886],
            [0.9471, 0.4305, 0.3549],
            [0.5254, 0.4289, 0.2599]], device='cuda:0')
    ```
    
    If we try to multiply a tensor on the CPU by a tensor on the GPU, we get an error:
    
    ```
    torch.mm(x,ydvc)
    
    ---------------------------------------------------------------------------
    RuntimeError                              Traceback (most recent call last)
    Cell In[10], line 1
    ----> 1 torch.mm(x,ydvc)
    
    RuntimeError: Expected all tensors to be on the same device...
    ```
    
    Going the other way, we can use `xdvc.cpu().numpy()` to get the NumPy array part of a tensor or `xdvc[1,1].item()` to get the float value of a tensor element -- the results will be on the CPU, whether or not `xdvc` is on the GPU.
    
    This is as far as we will go in explaining PyTorch in this document, apart from some installation/useage instructions:
    
    - [Using PyTorch on Unity](#pytorch-unity) Shows how to create an environment that can run PyTorch on Unity.

- [**CuPy**](https://cupy.dev/) provides "drop-in" substitutes for many NumPy and SciPy array functions (many with the same function names and call signatures) which however run on an NVIDIA GPU and use the highly-optimized [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) libraries.  Thus CuPy provides a fairly painless way to achieve GPU acceleration for NumPy/SciPy code, although one must keep track of where arrays are (GPU vs CPU) and transfer them as necessary.  CuPy also provides some "low level" GPU capabilities which have no NumPy/SciPy analogs such as creating CUDA **events**, writing CUDA **kernels**, etc.
  
  - **Installing CuPy on a Linux PC.** First we make a Conda environment `gpu`  in which to run Python programs using CuPy,  with other packages that will be used -- here we have chosen Numpy and SciPy so they can be compared with CuPy, and Jupyter Notebook to use for the examples below.  We install CuPy from [conda-forge](https://conda-forge.org/) because (as of 1/25) only a very outdated version of CuPy was available from the default Conda channel.
    
    ```
    (base)..$ conda update conda 
    (base)..$ conda create -n gpu python=3
    (base)..$ conda activate gpu
    (gpu)..$ conda install numpy scipy jupyter
    (gpu)..$ conda install -c conda-forge cupy
    (gpu)..$ conda list              # see versions of installed packaged
    ```
    
    In 1/25 after these commands the installed versions were: Python 3.11.11, NumPy 2.2.1, SciPy 1.15.1, Jupyter 1.1.0, CuPy 13.3.0.
  
  - **Simple examples of CuPy.**  This code run in a Jupyter Notebook imports `cupy` as well as `cupyx` (needed if SciPy-equivalent functions will be used) and prints out the [CUDA compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) of the installed NVIDIA GPU -- this was the small GPU on the candela-20 PC with a compute capability of 6.1.  This code will fail if there is no GPU, or CUDA is not installed.
    
    ```
    import numpy as np
    import scipy
    import cupy as cp         # numpy-equiv and CUDA GPU funcs
    import cupyx              # scipy-equive GPU funcs
    
    # Optional code to get some info on the GPU.
    gpudevice = cp.cuda.Device()
    print(f'GPU has compute capability {gpudevice.compute_capability}')
    
    GPU has compute capability 61
    ```
    
    Next (after doing the imports as above), we make two $3\times3$ arrays of random numbers -- one using `np.random.rand` which gives a regular NumPy array (type `numpy.ndarray`) and the other using `cp.random.rand` which gives a CuPy array on the GPU (type `cp.ndarray`).  Notes: (a) the syntax is identical for the NumPy and CuPy functions, and (b) CuPy arrays always live on the GPU (unlike PyTorch tensors, which can be either on the CPU or on the GPU).
    
    ```
    x = np.random.rand(3,3)
    ygpu = cp.random.rand(3,3)
    print('x=\n',x,type(x))
    print('\nygpu=\n',ygpu,type(ygpu))
    
    x=
      [[0.69443384 0.79801011 0.30639659]
      [0.73035583 0.42256363 0.8435655 ]
      [0.05977241 0.69179361 0.03713523]] <class 'numpy.ndarray'>
    
    ygpu=
      [[0.17523124 0.84222265 0.18135991]
      [0.9385187  0.03738684 0.48680856]
      [0.96396622 0.4795829  0.97488979]] <class 'cupy.ndarray'>
    ```
    
    We cannot multiply `x` by `ygpu` without first moving one or the other of them so they are on the same device (CPU or GPU).  First we use the `get` method of a CuPy array, which returns a copy of `ygpu`  as a NumPy array (on the CPU) so we can multiply the matrices using `np.matmul`:
    
    ```
    z = np.matmul(x,ygpu.get())
    print('z=\n',z,type(z))
    
    z=
      [[1.16598987 0.76164555 0.81312352]
      [1.33773368 1.03548014 1.16054826]
      [0.69553234 0.09401508 0.38381413]] <class 'numpy.ndarray'>
    ```
    
    Next we use the function `cp.asarray` to copy the NumPy array `x` to a CuPy array on the GPU, so we can multiply the matrices using `cp.matmul`:
    
    ```
    zgpu = cp.matmul(cp.asarray(x),ygpu)
    print('zgpu=\n',zgpu,type(zgpu))
    
    zgpu=
      [[1.16598987 0.76164555 0.81312352]
      [1.33773368 1.03548014 1.16054826]
      [0.69553234 0.09401508 0.38381413]] <class 'cupy.ndarray'>
    ```

- **A more elaborate CuPy program: `gputest.py`.**<a id="gputest-py"></a>  This program tests the speed of the CPU and (if available) the GPU by timing the muliplication of matrices -- both dense and sparse -- of various sizes and data types.  See the comments in `gputest.py` for details:
  
  ```
  (gpu) $ python gputest.py
  Running: gputest.py 11/22/23 D.C.
  Local time: Mon Jan  6 14:24:54 2025
  GPU 0 has compute capacity 6.1, 6 SMs, 4.23 GB RAM, guess model = GeForce GTX 1050
  CPU timings use last 10 of 11 trials
  GPU timings use last 25 of 28 trials
  
  ***************** Doing test dense_mult ******************
  Multiply M*M=N element dense matrices
  *********************************************************
  
  ************ Using float64 **************
           N     flop make mats  CPU test *CPU op/s*  GPU test *GPU op/s*  GPU xfer xfer rate
      99,856 6.30e+07 3.39e-03s 5.05e-04s 1.25e+11/s 1.23e-03s 5.12e+10/s 3.27e-02s  0.10GB/s
   1,000,000 2.00e+09 2.92e-02s 8.69e-03s 2.30e+11/s 2.85e-02s 7.02e+10/s 1.01e-02s  3.17GB/s
                                          ...
  ```

#### A few of NVDIA's many GPUS, with test results<a id="gpu-list"></a>

NVIDA has made many different GPUs. This table shows includes the relatively small GPUs in my PCs, a somewhat bigger GPU available for free on Google Colab, and a few more powerful GPUs available on the Unity HPC cluster. Limited information on other GPUs available on the Unity cluster is in the section [Picking a GPU on Unity](#pick-gpu) below.

| NVIDIA Model:                      | GeForce GTX 1050 Ti    | GeForce GTX 1660        | Tesla T4                 | Tesla V100 DGXS    | Tesla A100 SMX4    | Hopper H100 SXM5 |
| ---------------------------------- | ---------------------- | ----------------------- | ------------------------ | ------------------ | ------------------ | ---------------- |
| Where:                             | candela-20 GPU         | candela-21 GPU          | Best free on Colab 10/23 | Available on Unity | Available on Unity | Exists on Unity  |
| Price:                             | $250 9/19              | $297 11/22              | $2,000 10/23             | $1,500 10/23       | $15,000 10/23      | $30,000 10/23    |
| Release date:                      | 10/16                  | 3/19                    | 9/18                     | 3/18               | 5/20               | 3/22             |
| GPU chip, arch:                    | Pascal                 | TU116, Turing           | TU104, Turing            | GV100, Volta       | GA 100, Ampere     | GH100, Hopper    |
| Fab procces:                       | 14 nm                  | 12 nm                   | 12 nm                    | 12 nm              | 7 nm               | 4 nm             |
| Transistors:                       | 3.3e9                  | 6.6e9                   | 1.4e10                   | 2.1e10             | 5.4e10             | 8e10             |
| CUDA compute:                      | 6.1                    | 7.5                     | 7.5                      | 7.0                | 8.0                | 9.0              |
| Streaming Multiprocs:              | 6                      | 22                      | 40                       | 80                 | 108                | 144              |
| CUDA cores:                        | 768 (128/SM)           | 1,408 (64/SM)           | 2,560 (64/SM)            | 5,120 (64/SM)      | 6,912 (64/SM)      | 16,896           |
| Tensor cores:                      | 0                      | 0                       | 320                      |                    |                    |                  |
| Ray tracing cores:                 | 0                      | 0                       | 40                       |                    |                    |                  |
| Memory:                            | 4 GB                   | 6 GB GDDR5              | 15 GB GDDR6              | 16 GB HBM2         | 40 GB HBM2e        | 96 GB HBM3       |
| Memory width:                      | 128-bit                | 192-bit                 | 256-bit                  | 4,096-bit          | 5,120-bit          | 5,120-bit        |
| Memory BW:                         | 112 GB/s               | 192 GB/s                | 320 GB/s                 | 897 GB/s           | 1,555 GB/s         | 1,681 GB/s       |
| Max float64 FLOPS:                 | 0.067 TF               | 0.157 TF                | ?                        | 6.6 TF             | 9.7 TF             | 33.5 TF          |
| **Max float32 FLOPS:**             | **2.14 TF**            | **5.03 TF**             | **8.1 TF**               | **13.2 TF**        | **19.5 TF**        | **66.9 TF**      |
| Max tf32 FLOPS:                    |                        |                         | 64.8 TF                  | 105.7 TF           | 312 TF             | 989 TF           |
| Max int32 OPS:                     |                        |                         |                          | 17.7 TO            | 19.7 TO            |                  |
| **Test results from `gputest.py`** |                        |                         |                          |                    |                    |                  |
| **System:**                        | **candela-20**         | **candela-21**          | **Google Colab**         | **Unity HPC**      | **Unity HPC**      |                  |
| **CPU or GPU:**                    | **6-core 3.8 GHz CPU** | **16-core 3.4 GHz CPU** | **Tesla T4 GPU**         | **Tesla V100 GPU** | **Tesla A100 GPU** |                  |
| Dense matrix multiplication        |                        |                         |                          |                    |                    |                  |
| float64 CPU / GPU:                 | 0.22 / 0.07 TF         | **0.34 /0.14 TF**       | 0.06 /0.128 TF           | 0.64 /4.8 TF       | **0.41 /11.6 TF**  |                  |
| float32 CPU /GPU:                  | 0.55 /1.86 TF          | **0.78 /3.00 TF**       | 0.06 / 5.6 TF            | 1.26 / 9.2 TF      | **1.28 / 15.2 TF** |                  |
| Bond vectors using Numpy Indexing  |                        |                         |                          |                    |                    |                  |
| float64 CPU / GPU:                 | 0.12 / 0.40 GF         | **0.12 /0.66 GF**       | 0.06 / 1.17 GF           | 0.07 /6.2 GF       | **0.115 /15.6 GF** |                  |
| float32 CPU /GPU:                  | 0.19 /0.52 GF          | **0.22 /0.97 GF**       | 0.05 / 3.3 GF            | 0.11 / 3.5 GF      | **0.21 / 24 GF**   |                  |
| Bond vectors using CSR matrix      |                        |                         |                          |                    |                    |                  |
| float64 CPU / GPU:                 |                        | **0.24 /0.67 GF**       | 0.07 / 1.12 GF           | 0.07 /5.6 GF       | **0.22 /13.7 GF**  |                  |
| float32 CPU /GPU:                  |                        | **0.26 /0.94 GF**       | 0.09 / 3.0 GF            | 0.10 / 11.9 GF     | **0.26 / 21 GF**   |                  |
| Node forces using CSR matrix       |                        |                         |                          |                    |                    |                  |
| float64 CPU / GPU:                 |                        | **0.55 /1.28 GF**       | 0.14 / 1.96 GF           | 0.13 /7.9 GF       | **0.36 /31 GF**    |                  |
| float32 CPU /GPU:                  |                        | **0.66 /1.41 GF**       | 0.22 / 3.82 GF           | 0.23 / 13.8 GF     | **0.43 / 55 GF**   |                  |

These test results used arrays with $10^6$ elements.  Here TF = TFLOPS = $10^{12}$ FP ops/s, GF = GFLOPS = $10^9$ FP ops/s. 

- Conclusions from this testing:
  
  - **float32 vs float64:** This only makes a big difference when doing dense matrix calculations, which operate at close to theoretical maximum FLOPS for each type of GPU.  Sparse operations (bond vectors and of node forces) have 300-1,000 times less FLOPS, suggesting they are dominated by memory access rather than FLOP capability, and the speed doesn’t depend much on float32/float64. For **dense matrix operations**:
    
    - Using **float64**, candela-21 GPU is terrible and shouldn’t be used. A100 GPU is 34 times faster than candela-21 CPU.
    - Using **float32** (already 2.3 times faster than float64 on candela-21 CPU), candela-21 GPU is 3.8 times faster and A100 GPU is 19 times faster than candela-21 CPU  (so using GPUs A100 is only 5 times faster than candela-21).
  
  - **Speedup of float32 over float64** on various systems:
    
    - **candela-21** float32 speedup is **8.8 for dense operations** (float32 on GPU vs float64 on CPU), only **1.1-1.5 for sparse operations** (all on GPU).
    - **Unity using A100 GPU** float32 speedup is **1.3 for dense operations, 1.5-1.8 for sparse operations**.
  
  - **CSR matrix vs numpy indexing:** When calculating bond vectors from node positions, no advantage to using CSR matrix over numpy indexing.
  
  - **Sparse operations.**  These are float32 speedups with numpy indexing for bond vectors, float64 and or CSR for bond vectors only modestly different. CPU-GPU transfer times are ignored here:
    
    - **Calculating bond vectors from differences of node positions.**  candela-21 GPU is 3.6 times faster and A100 GPU is 81 times faster than candela-21 CPU (so using GPUs A100 is 22 times faster than candela-21).
    - **Calculating node forces by summing bond forces.**  candela-21 GPU is 2.1 times faster and A100 GPU is 83 times faster than candela-21 CPU (so using GPUs A100 is 39 times faster than candela-21).
    - **Operations on larger sparse matrices.** For operations on sparse matrices with $10^7$ elements (ten times bigger than in the table above), the advantage of using a GPU over a CPU were somewhat bigger than with $10^6$ elements.
  
  - **Comparison of available GPUs** using **float32** and **candela-21 GPU** as reference:
    
    - **candela-20** is **1.5 times slower for dense and sparse operations**.
    - **T4** (best GPU available free on Colab as of 11/23) is **1.9 times faster for dense operations, 1.3-3.2 times faster for sparse operations**. But **CPU operations were much slower on Colab+T4 than on candela-21.**
    - **V100** (good GPU sometimes available on Unity as of 11/23) is **3 times faster for dense operations, 4-10 times faster for sparse operations**. But again **CPU operations were slower than on candela-21.**
    - **A100** (very good GPU sometimes available on Unity as of 11/23) is **5 times faster for dense operations, 20-40 times faster for sparse operations. CPU operations were similar** to candela-21.
    - The section [Picking a GPU on Unity](#pick-gpu) below has some summary specs for the various GPUs available on Unity, as requesting the better ones listed in the table above can result in long queue times for Unity jobs.

- **Running CuPy in a Google Colab notebook.** This was quite simple, as it seems compatible Numpy, CUDA and CuPy are installed by default:
  
  - Paste Python code that imports CuPy into a code cell (for example the GPU test program `testgpu.py` can simply be pasted into a cell).
  - Select a GPU runtime.  As of 10/23, the only GPU type available for free on Colab was the Tesla T4. Tesla V100 and Tesla A100 GPUs were available but only on a paid tier.
  - Hit run.

### Using Apptainer on a Linux PC<a id="apptainer-pc"></a>

#### Why do it<a id="why-apptainer-pc"></a>

Code that is containerized using Apptainer should be usable on various PCs without setting up environments with the correct packages (with compatible versions) installed.  However, this does require Apptainer itself to be installed on the PCs, which is not necessarily trivial or commonly done.

Probably the best reason for containerizing code is to make it easy to run the code on an HPC cluster, which is likely to have Apptainer pre-installed and ready to use (as Unity does).  In the examples below, **containers developed and usable on a PC could also be used without modification on Unity**.

#### Apptainer history<a id="apptainer-history"></a>

- **Singularity** (the original name) developed in 2015 at LBL by Gregory Kurtzer et al. and became popular on HPC clusters 2016-2018.
- 2018 Kurtzer founded [Sylabs](https://sylabs.io/) and released Singularity 3.0 rewritten in Go.
- 5/21 Sylabs forked Singularity 3.8.? to **SingularityCE** “community edition”, they also have a pay-for version called SingularityPRO:
  - First version SingularityCE 3.8.0 5/26/21
  - Current version SingularityCE 4.2 as of 1/25
- 11/21 Singularity renamed [**Apptainer**](https://apptainer.org/).
  - Last Singularity version 3.8.7 3/17/22
  - First Apptainer version 1.0.1 3/2/22
  - Current Apptainer version 1.3.6 12/24
- Apptainer versions on various systems, as of 1/25:
  - My Linux PCs at work and home, installed as below: Apptainer 1.3.4
  - Unity: Apptainer 1.3.4

#### Installing Apptainer<a id="install-apptainer"></a>

1/25 Installed Apptainer 1.3.4 on [my three PCs  running Ubuntu 24.04](#pcs):

- Followed the instructions in the Install Ubuntu Packages section in the Installing Apptainer section of the Apptainer Admin Guide in the [documentation](https://apptainer.org/documentation/) at the Apptainer site.  These were the steps for a **non-setuid** installation, which generally seems to be recommended:
  
  ```
  $ sudo add-apt-repository -y ppa:apptainer/ppa
  $ sudo apt update
  $ sudo apt install -y apptainer
  ```

- This should eventually be unnecessary, but as of 1/25 trying to run Apptainer freshly installed as above as a regular (non-sudo) user under Ubuntu 24.04 fails with the following messages:
  
  ```
  ERROR  : Could not write info to setgroups: Permission denied
  ERROR  : Error while waiting event for user namespace mappings: no event received
  ```
  
  - The source of this problem - a security upgrade when Ubuntu went to 24.04 that requires an apparmor profile for Apptainer which was omitted from the Apptainer distribution for Ubuntu - is discussed in these links:
    
    - https://github.com/apptainer/apptainer/issues/2608
    - https://github.com/apptainer/apptainer/blob/main/INSTALL.md#apparmor-profile-ubuntu-2404
    - https://github.com/apptainer/apptainer/issues/2360
  
  - From these links I found the problem could be fixed by adding a file called `apptainer` to the directory `/etc/apparmor.d` with these contents…
    
    ```
    # Permit unprivileged user namespace creation for apptainer starter
    abi <abi/4.0>,
    include <tunables/global>
    profile apptainer /usr/libexec/apptainer/bin/starter{,-suid} 
        flags=(unconfined) {
      userns,
      # Site-specific additions and overrides. See local/README for details.
      include if exists <local/apptainer> 
    }
    ```
    
    ...then re-booting or running this command:
    
    ```
    $ sudo systemctl reload apparmor
    ```
  
  - On **Unity** this problem does not exist or was fixed by administrators -- non-superusers are allowed to run Apptainer containers.

- Check the Apptainer version.  I also checked the Singularity version, which on my PCs was still the pre-fork version installed in 2022 (not used in the rest of this document):
  
  ```
  $ apptainer --version
  apptainer version 1.3.4
  $ singularity --version
  singularity version 3.7.4
  ```

- On **Unity** `singularity` is aliased to `apptainer`, meaning Singularity commands actually run Apptainer.  Logged into Unity:
  
  ```
  $ apptainer --version
  apptainer version 1.3.4
  $ singularity --version
  apptainer version 1.3.4
  ```
  
  For this document I decided not to do this, but rather to use `apptainer` commands directly.

#### Testing the install: An OS-only container<a id="os-only-container"></a>

- Here is a [short tutorial](https://medium.com/@dcat52/making-your-first-container-81b832d82a6f).

- The specifications for building a container are in a small text file (a `.def` file), while the container itself is a large Singularity image file (a `.sif` file). Typically containers are “bootstrapped” from a **Docker** container found on [Docker Hub](https://hub.docker.com/) -- the `apptainer build` command will convert the Docker container to an Apptainer container, then add more things to it as specified by the `.def` file. For a minimal definition file that bootstraps the official container for Ubuntu 24.04 and doesn’t add anything further to it, make a file **`os-only.def`** with these contents:
  
  ```
  Bootstrap: docker
  From: ubuntu:24.04
  ```
  
  These statements tell Apptainer to use the container `ubuntu:24.04` from Docker Hub, where you can find many other starting containers to use for your builds (for example, containers  with other flavors or versions of Linux, with Conda and the Miniconda distribution, with other applications...).

- Because container images (`.sif` files) can be fairly large (up to several GB), it is often helpful to keep them in a separately backed up directory. To avoid repetitive typing set a shell variable `SIFS` to point to this directory:
  
  ```
  $ export SIFS="/home/..."        # directory where .sif files will be put
  ```
  
  If this path includes any spaces it must be quoted as shown here and below, otherwise the quotes can
  be omitted.  Since this `export` command must be done every time a new terminal is opened, it is handy to create an alias `sifs`:
  
  ```
  # Add this to ~/.bash_aliases:
  alias sifs="export SIFS="/home/..."
  ```

- Build the container image: In the directory that contains the definition file `os-only.def` do the following command. **Root privilege is necessary to build a container image, but not to use it.** (According to the Apptainer docs it should be possible to build a container without root privilege by using the `--fakeroot` option, but I haven’t tried this.)
  
  ```
  $ sudo apptainer build "$SIFS"/os-only.sif os-only.def
  ```

- This created a 30 MB image file **`os-only.sif`** in the directory `$SIFS`. We can open a shell running inside the container using the `apptainer shell` command.  This gives a prompt `Apptainer>` from which we can issue shell commands that will run inside the container:
  
  ```
  $ apptainer shell "$SIFS"/os-only.sif
  Apptainer> pwd
  /home/...
  ```
  
  The current working directory inside the container is the same as is was before the container was started (more on this below), making it easy for commands run in the container to work with files outside the container.

- The following example shows
  
  - Python is not installed inside this bare-bones container.
  
  - The OS inside the container is Ubuntu 24.04 as set in the definition file `os-only.def`, independent of what the OS is outside the container. Either of the outside and inside OS versions can be newer (within broad limits set by kernel versions) or they can be the same.
  
  - A command (here `touch`) run inside the container accesses the directory outside the container from which the container was started.  
  
  - To exit the container shell do `ctrl-D`. Outside the container we see that Python is installed (unlike inside), and the file created by the `touch` command now exists.
    
    ```
    Apptainer> python --version
    bash: python: command not found
    Apptainer> cat /etc/os-release
    PRETTY_NAME="Ubuntu 24.04.1 LTS"
           ...
    Apptainer> pwd
    /home/..          (will be directory in which “apptainer shell” was run)
    Apptainer> touch foo
    Apptainer>        (hit ctrl-D to exit the container)
    exit
    $ python --version
    Python 3.11.11
    $ ls
    ...  foo  ...
    ```
    
    Possible point of confusion: The container has its own file system with its own root, independent of the file system outside the container.  Thus `/etc/os-release` in the `cat` command above prints a file that exists inside the container with OS information.  Once a container is built, **the file system inside the container is read-only**. However, for convenience, **certain directories inside the container are automatically bound to directories outside the container with the same names** when the container is run. Typically these will include **the user's current working and home directories** -- making it possible to access and write to the same files and directories from inside the container as outside.
    
    I believe that these two things: (a) the ability to run a container as a non-superuser, and (b) the ability to access files outside the container are among the primary differences between Apptainer and Docker containers, making Apptainer more suitable for use on a shared HPC system.  I think Docker containers are most frequently run in cloud-based virtual machines for which the user will have superuser access.
    
    **Note on the initial working directory in an Apptainer container.**  In 2023 when Unity was newer, the current working directory (CWD) inside an Apptainer container was not automatically bound to the CWD from which the container was run, if it was run from a subdirectory of `/work` (the normal place for job I/O) - I think because `/work` is not under `/home` unlike the work directories for some other HPC clusters like the former USMC.  But as of 1/25 this seems to have been fixed -- even when run from a directory under `/work`, the CWD is bound as usual.

#### A container including chosen Python packages<a id="packages-container"></a>

- It is possible to install Pip, Python, etc. into a bare-bones Ubuntu container as created above, but it is easier to start with a container that has Python, pip, Conda, and the Miniconda package distribution pre-installed. Docker images of such containers are available from the Anaconda folks. Make a file **`pack.def`** with these contents:
  
  ```
  Bootstrap: docker
  From: continuumio/miniconda3
  
  %post
    conda install numpy matplotlib scipy
  ```
  
  This has a new **`%post`** section which lists Linux commands that will be **run during the container-build process**, establishing an "environment" with the needed packages inside the container which will be available wherever the container is run. There is no need to make a Conda environment -- the container itself provides the environment.

- Build the container as above:
  
  ```
  $ sudo apptainer build "$SIFS"/pack.sif pack.def
  ```
  
  The resulting container **`pack.sif`**  is now considerably in larger – without the `conda install` command the container size is 235 MB, and with the full definition shown above it is 1.4 GB.

- Shelling into the container we can find the versions of things and verify that we have a working Python installation:
  
  ```
  $ apptainer shell "$SIFS"/pack.sif
  
  Apptainer> cat /etc/os-release
  PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
         ...
  Apptainer> python --version
  Python 3.12.8
  Apptainer> pip list
  Package                  Version
  ------------------------ -----------
  matplotlib               3.9.2
  numpy                    2.1.3
  scipy                    1.14.1
           ...
  Apptainer> python
           ...
  >>> print('Hello from this container')
  Hello from this container
  ```
  
   From this we see that in the container we are running Debian 12 Linux (not Ubuntu as above), Python 3.12.8, etc. If an different version of Python is needed for compatibility, there are more specific versions of continuumio/miniconda3 that can be bootstrapped from Docker Hub.

- **Using the container to run Python scripts outside the container.**
  
  - Here is simple program **`np-version.py`** that imports NumPy and prints out its version:
    
    ```
    import numpy as np
    print(f'numpy version = {np.__version__}')
    ```
    
    Running it in a terminal showed that NumPy 1.24.3 was installed on my PC:
    
    ```
    $ python np-version.py
    numpy version = 1.24.3
    ```
  
  - Rather than shelling into the container with **`apptainer shell`**, we we can use **`apptainer exec`** (invoked outside the container) to run `np-version.py` inside the container -- even though the file `np-version.py` is located outside the container. This reports the version of NumPy inside the container:
    
    ```
    $ apptainer exec "$SIFS"/pack.sif python np-version.py
    numpy version = 2.1.3
    ```
    
    This works because the actual program being run is `python`, which is installed inside the container, and `np-version.py` is just an input file to `python`.  Thus our program will be run with the Python version and package environment that exists inside the container.

#### A container with a local Python package installed<a id="local-package-container"></a>

Next we make a container with the local package **`dcfuncs`** installed inside it, so this package can be used by Python code outside the container. See [Installing a local package](#local-package) above, which shows how **`dcfuncs`** is installed and used without a container.

- Make a definition file **`dfs.def`** with the following contents:  
  
  ```
  Bootstrap: docker
  From: continuumio/miniconda3
  
  %files
      dcfuncs /dcfuncs
  
  %post
      conda install numpy matplotlib scipy
      cd /dcfuncs
      pip install .
  
  %runscript
      echo foo!
  ```
  
  In addition to the %post section with Linux commands introduced above, this `.def` file has two new sections:
  
  - The **`%files`**  section copies files into the container before the` %post` commands are run -- each line gives a source directory or file outside the container and a copy location inside the container. 
    
    - To **build** (as opposed to run) the container, the `dcfuncs` repository (directory `dcfuncs` with its files and subdirectories) must be available on the PC.  For this example copy the repository under the directory containing the definition file:
      
      ```
      dfs.def
      dcfuncs/
         src/
          ...
      ```
    
    - As written in `dfs.def` above the files from `dcfuncs`  will be written into a directory `/dcfuncs` in the container.  When the container is built, files can be written anywhere in the container file system as specified by the `%files` section of the `.def` file, including in the container's root directory `/`
  
  - The **`%post`** section runs pip to install the `dcfuncs` package inside the container.
  
  - An Apptainer container is an executable file, and the **`%runscript`** section gives the commands that will be executed if it is run.

- Build the container, resulting in the 1.4 GB image file **`dfs.sif`**:
  
  ```
  $ sudo apptainer build "$SIFS"/dfs.sif dfs.def
  ```
  
  You will see a warning about running `pip` as root -- not a problem here as we are in the confines of the container (I believe -- although you can find an extensive discussion of this point online).

- The simplest thing we can do with this new container is to run it
  
  ```
  $ "$SIFS"/dfs.sif
  foo!
  ```

- Shelling into the container we can use `pip list` to verify that the dcfuncs package is installed:
  
  ```
  $ apptainer shell "$SIFS"/dfs.sif
  Apptainer> pip list
  Package                  Version
  ------------------------ -----------
  dcfuncs                  1.0
               ...
  ```

- If we exit the container with `ctrl-D` and try to run the test code for `dcfuncs` with Python outside the container, it will fail with `module not found` (assuming we have not also installed `dcfuncs` outside the container):
  
  ```
  $ python dcfuncs/test/test-util.py
  Traceback (most recent call last):
            ...
  ModuleNotFoundError: No module named 'dcfuncs'
  ```
  
  But now we can use `python` in the container to run the same test code (which lies outside the container) without error:
  
  ```
  $ apptainer exec "$SIFS"/dfs.sif python dcfuncs/test/test-util.py
  This is: dutil.py 8/19/24 D.C.
  Using: util.py 8/18/24 D.C.
  
  Testing zz:
  - WARNING from util.py test code - This is just a warning.
                   ...
  ```
  
    The upshot is that **the** **`dcfuncs` package is available to `python` running in the container, whether or not `dcfuncs` is installed outside the container** – and `python` running in the container can run Python code (`.py` files) outside the container.

- As [shown earlier in this document](#gputest-py) if we run `gputest.py` in the Conda environment `gpu` then the GPU will be found and used:
  
  ```
  ..$ cd ...                 # change to directory where gputest.py is located
  ..$ conda activate gpu
  (gpu)..$ python gputest.py 
  Running: gputest.py 11/22/23 D.C.
  Local time: Tue Jan 7 16:05:53 2025
  GPU 0 has compute capacity 6.1, 6 SMs, 4.23 GB RAM, guess model = GeForce GTX 1050
  CPU timings use last 10 of 11 trials
  GPU timings use last 25 of 28 trials
        ...
  ```
  
  Conversely, since we have not installed CuPy inside this container, running the same program with `python` inside the container will not find CuPy and the GPU, even  if the container is run in the  `gpu` environment:
  
  ```
  (gpu)..$ apptainer exec "$SIFS"/dfs.sif python gputest.py
  Running: gputest.py 11/22/23 D.C.
  Local time: Tue Jan 7 16:09:24 2025
  Import cupy failed, using CPU only
  CPU timings use last 10 of 11 trials
        ...
  ```
  
  The upshot is that **the environment and packages installed outside the container are not available to `python` running in the container.** While this may seem like a disadvantage, it fits in with the main objective of containerization -- providing a consistent environment for running code independent of what is installed outside the container.

#### A container that can use MPI<a id="mpi-container"></a>

- [Apptainer and MPI applications](https://apptainer.org/docs/user/latest/mpi.html) in the Apptainer docs explains in some detail how MPI works with Apptainer. Here we give some rather simpler examples showing how Python programs using **`mpi4py`** for MPI parallelism can be run with Apptainer.  I have found that if Conda is used to install OpenMPI and MPI for Python (`mpi4py`) in the container, then it is possible for Python run in multiple containers started by `mpirun` (from OpenMPI installed outside the containers) to use MPI via `mpi4py` calls.  This did not require setting any environment variables, unlike the examples in the Apptainer docs referenced above.

- **Memory usage.** In the examples below the command
  
  ```
  $ mpirun -n <n> apptainer exec <container>.sif python <pyscricpt>.py...
  ```
  
  is used to run `<n>` copies of the container image `<container>.sif`, each of which runs `python <pyscript>.py`.  As `.sif` files are one or more GB in size, it might be thought this would require a lot of memory when the number of MPI ranks`<n>` is large but this is found not be true, see [below](#dem21-container).  I believe this is because the Apptainer container images including their internal file systems are read-only, which allows multiple containers on the same PC to share one copy of the container image.

- Make a definition file **`m4p.def`** with the following contents. The `%post` commands in this file are similar to those shown above to [install OpenMPI on a PC](#install-openmpi-pc):
  
  ```
  Bootstrap: docker
  From: continuumio/miniconda3
  
  %post
      conda install -c conda-forge openmpi=5 mpi4py python=3.12
      conda install -c conda-forge numpy scipy matplotlib
  ```

- Build the container, resulting in the 1.2 GB image file **`m4p.sif`**:
  
  ```
  $ sudo apptainer build "$SIFS"/m4p.sif m4p.def
  ```

- While the container `m4p.sif` can be built as above without a Conda environment in which MPI is installed, to run the container **MPI must be installed outside the container**.  This is because, as shown in examples below, MPI outside the container is used to run multiple copies of the container on separate cores and to handle communication between these copies. It will work to activate the environment **`m4p`** with OpenMPI and other packages [described above]() before running the container, but this environment contains things that are not needed outside the container (`mpi4py`, `numpy`...).  Here we make a simpler environment **`ompi`** that includes only OpenMPI and Python (so pip could be used to install additional things in this environment):
  
  ```
  $ conda deactivate
  $ conda create -n ompi -c conda-forge openmpi=5 python=3.12
  $ conda activate ompi
  (ompi)$ mpirun --version
  mpirun (Open MPI) 5.0.7
  ```

- With `ompi` activated we can use the the container to run `mpi_hw.py` and `osu_bw.py`, after switching to a directory that contains these programs:
  
  ```
  (ompi)..$ cd python-scripts; ls
  mpi_hw.py  osu_by.py ...
  (ompi)..python-scripts$ mpirun -n 6 apptainer exec "$SIFS"/m4p.sif python mpi_hw.py
  Hello world from rank 0 of 6 on candela-21 running Open MPI v5.0.7
  Hello world from rank 3 of 6 on candela-21 running Open MPI v5.0.7
  Hello world from rank 5 of 6 on candela-21 running Open MPI v5.0.7
  Hello world from rank 4 of 6 on candela-21 running Open MPI v5.0.7
  Hello world from rank 2 of 6 on candela-21 running Open MPI v5.0.7
  Hello world from rank 1 of 6 on candela-21 running Open MPI v5.0.7
  (ompi)..python-scripts$ mpirun -n 2 apptainer exec "$SIFS"/m4p.sif python osu_bw.py
  2
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
           1                3.64
           2                7.27
           4               14.58
           8               29.10
          16               56.49
          32              115.74
          64              204.76
         128              372.67
         256              698.36
         512            1,491.48
       1,024            2,902.60
       2,048            5,360.16
       4,096            7,361.31
       8,192           13,089.36
      16,384           21,916.65
      32,768           29,655.28
      65,536           36,365.06
     131,072           40,048.58
     262,144           42,727.54
     524,288           44,050.55
   1,048,576           44,995.58
   2,097,152           45,577.86
   4,194,304           45,258.97
   8,388,608           45,191.11
  16,777,216           44,483.20
  ```
  
  Things to note in this example:

- The command `mpirun -n 6 ...` is running six separate copies of the container on six cores of the PC.  This `mpirun` command is running outside the container. This is an example of the "Hybrid model" for running MPI described in the [Apptainer docs](https://apptainer.org/docs/user/latest/mpi.html).

- For reasons I don't understand, `osu_bw.py` reports inter-rank communication speeds about three times faster when run using Apptainer (up to 45 GB/s), than when run [directly by MPI without Apptainer](#mpi-testprogs). 

- On the [hoffice PC](#pcs), a simple all-in-one PC (but not on the [candela-21 PC](#pcs) , assembled from parts) `mpirun` gives warning messages about the absence of TCP - these can be suppressed by supplying `--mca btl ^tcp` to `mpirun`. I think TCP should be irrelevant when running MPI on a single PC, as it is concerned with communication between nodes.

#### A container to run the more elaborate MPI package `dem21`<a id="dem21-container"></a>

Here we use containerized code to duplicate the non-containerized tests shown in [More elaborate MPI programs using the `dem21` package](#mpi-dem21) above. Make a definition file **`dem21.def`** with the following contents. This is like `m4p.def` in the previous section, but with the following additions to install the `dem21`  and `msigs` packages in the container (see [A container with a local Python package installed](#local-package-container) above):

- There is a `%files` section that will copy the `dem21` and `msigs` packages, assumed to be in the directory from which the `apptainer build` will be run, into the container. 

- The `%post` section includes commands that install additional remote packages needed by `dem21` and install `dem21`  and `msigs` as a local packages, as is done without a container in [More elaborate MPI programs using the `dem21` package](#mpi-dem21) above.
  
  ```
  # dem21.def 4/1/25 D.C.
  # Apptainer .def file for running dem21 DEM simulation package, also
  # includes msigs package to creat input signals for granular-memory sims.
  Bootstrap: docker
  From: continuumio/miniconda3
  
  # Must build this container from a directory containing both dem21
  # and msigs repos.
  %files
      dem21 /dem21
      msigs /msigs
  
  %post
      conda install -c conda-forge openmpi=5.0.3 mpi4py
      conda install -c conda-forge dill matplotlib numba numpy pyaml scipy
      conda install -c conda-forge quaternion
      cd /dem21
      pip install .
      cd /msigs
      pip install .
  ```

- A directory `buid-dem21` is created and `dem21.def` and the `msigs`  repo (not publicly available) are copied into it. Then the `dem21` package (not publicly available) is cloned into this directory and the container is built. This made the 1.3 GB image file **`dem21.sif`**:
  
  ```
  ...build-dem21$ git clone git@github.com:doncandela/dem21.git
  ...build-dem$ ls
  dem21  dem21.def msigs
  ...build-dem$ sudo apptainer build "$SIFS"/dem21.sif dem21.def
  ```

- Next we test  that the `dem21` package can run in the container (see [More elaborate MPI programs using the `dem21` package](#mpi-dem21) above for the corresponding steps without a container, and the previous section [A container that can use MPI](#mpi-container) for how MPI is run with a container). We start by activating **`ompi`** so OpenMPI will be available outside the containers, and cd'ing to the directory containing the test program `boxpct.py` and its config file `box.yaml`:
  
  ```
  ...buid-dem$ conda activate ompi
  (ompi)...build-dem$ cd dem21/tests/box; ls
  boxpct.py box.yaml ...
  ```
  
  Then we set `pproc=mpi` so `boxpct.py` will run in MPI-parallel mode, and use `mpirun` to run 16 copies of `apptainer exec python` with `boxpct.py` as the argument (as usual, `SIFS` has been set to the directory containing the container image `dem21.sif`): 
  
  ```
  (ompi)...tests/box$ export pproc=mpi   # this tells boxpct.py to use MPI
  (ompi)...tests/box$ mpirun -n 16 apptainer exec "$SIFS"/dem21.sif python boxpct.py
  - Started MPI on master + 15 worker ranks.
  THIS IS: boxpct.py 12/3/22 D.C., using dem21 version: v1.2 2/11/25
  Parallel processing: MPI, GHOST_ARRAY=True
  - Read 1 config(s) from /home/dc/.../tests/box/box.yaml
  
  SIM 1/1:
  Using inelastic 'silicone' grainlets with en=0.7 and R=0.500mm
                                  ....
  ```

- Finally, as in the section [More resource-intensive runs...](#mx2py) above, we use the container image `dem21.sif` with `mx2.py` to run a larger simulation.   The only change required in the shell file `mx2.sh` in that section that runs the simulation is in the last line, which now uses `mpirun` to run multiple copies of `apptainer exec dem21.sif python` rather than multiple copies of `python`.  This modified shell file is called **`mx2-app.sh`**:
  
  ```
  #!/bin/bash
  # cc-expts-app/mx2-app.sh 4/6/25 D.C.
  # Shell script to run granular-memory simulation program mx2.py containerized
  # on a PC, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
  #
  # Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
  # Reads default config file mx2.yaml in grandparent directory modified by
  # mx2mod.yaml in current directory.
  #
  # To run on N cores 1st activate environment 'ompi' then do
  #
  # ./mx2-app.sh N
  #
  export pproc=mpi
  mpirun -n $1 apptainer exec "$SIFS"/dem21.sif python ../../mx2.py mx2mod |& tee output
  ```

- The simulation is run in precisely the same way as when not containerized, except that it can be run in the bare-bones OpenMPI environment `ompi` rather than the environment `dem21` which had `dem21`, `msigs`, and other packages installed:
  
  ```
  ...$ conda activate ompi
  (ompi)...$ cd cc-expts-app; ls
  mx2mod.yaml  mx2-app.sh ...
  (ompi)...cd-expts-app$ sifs               # alias sets SIFS to directory with dem21.sif
  (ompi)...cd-expts-app$ ./mx2-app.sh 15    # run containerized code in 15 MPI ranks
  - Started MPI on master + 14 worker ranks.
  This is: mx2.py 7/29/24 D.C.
  Using dem21 version: v1.2 2/11/25
  Imput signals made by: memsigs.py 8/23/24 D.C.
  Parallel processing mode: MPI, GHOST_ARRAY=True
                 ...
  ```
  
  This simulation required 17,054s (3.701e-6s/step-grain), which was 7% slower than the [identical simulation done without Apptainer](#mx2py).  Internal timing reported by `mx2.py` suggested that only 0.9% additional time was used for inter-process communication when Apptainer was used, so it is not clear if the 7% slowdown is actually due to containerization.
  
  Running the simulation in this containerized mode required 2.2 GB of memory, beyond that used before the simulation was started.  This was only slightly more than the memory required to run the simulation without Apptainer, 2.1 GB. This demonstrates clearly that running 16 copies of the 1.3 GB container image `dem21.sif` does not require 16 times as much memory, due I think to the read-only property of container images.

#### A container that can use a GPU<a id="gpu-container"></a>

- Make a definition file **`gpu.def`** with the following contents. As in the previous section we start with a Docker image that includes Python, Conda, and the Miniconda distribution and we install NumPy, Matplotlib, and SciPy. Now we also install CuPy, getting it from conda-forge to be sure it is up to date:
  
  ```
  Bootstrap: docker
  From: continuumio/miniconda3
  
  %post
      conda install numpy matplotlib scipy
      conda install -c conda-forge cupy
  ```

- Use `gpu.def` to build the container **`gpu.sif`**:
  
  ```
  $ sudo apptainer build "$SIFS"/gpu.sif gpu.def
  ```
  
  Due to the inclusion of CuPy, this container is considerably larger (3.5 GB vs 1.4 GB without CuPy).  Interestingly, I was able to build this container on  a PC that does not have a GPU (hoffice). Nevertheless when `gpu.sif` was copied to other computers that did have GPUs (my other PCs and Unity), it was able to use the GPU .

- A GPU, the NVIDIA drivers, and CUDA must all be installed **outside** the container on the computer on which the container is to be run -- but it is not necessary for Python packages that will be using the GPU, in this case CuPy, to be installed outside the container.
  
  - The section [Installing NVIDIA drivers](#nvidia-drivers) above shows how to install these things on a PC.
  - When successfully installed it will be possible to run the shell command **`nvidia-smi`** outside the container, which will print information on the installed GPU and CUDA version.

- As documented in [this Apptainer page](https://apptainer.org/docs/user/latest/gpu.html), `apptainer` commands that run the container (`shell`, `exec`..) must include the **`--nv`** option to make CUDA installed outside the container available inside.  Here we are in a directory containing `gputest.py`, which will successfully import CuPy and use the GPU when run by `python` in the container `gpu.sif`:
  
  ```
  (base)$ apptainer exec --nv "$SIFS"/gpu.sif python gputest.py
  Running: gputest.py 11/22/23 D.C.
  Local time: Wed Jan 15 17:54:50 2025
  GPU 0 has compute capacity 6.1, 6 SMs, 4.23 GB RAM, guess model = GeForce GTX 1050
  CPU timings use last 10 of 11 trials
  GPU timings use last 25 of 28 trials
  
  ***************** Doing test dense_mult ******************
  Multiply M*M=N element dense matrices
  *********************************************************
  
  ************ Using float64 **************
           N     flop make mats  CPU test *CPU op/s*  GPU test *GPU op/s*  GPU xfer xfer rate
      99,856 6.30e+07 3.31e-03s 4.72e-04s 1.34e+11/s 1.25e-03s 5.04e+10/s 4.00e-03s  0.80GB/s
                                      ...
  ```
  
  Using the container, we can run the program `gputest.py` from the `base` environment, even though `base` does not have the packages `numpy`, `scipy`, `cupy` that `gputest.py` imports installed.

## Part 2: Moving code to a Slurm HPC cluster<a id="move-to-hpc"></a>

### Why do it<a id="why-hpc"></a>

In general one uses an HPC cluster to get more computational power (e.g. if doing simulations carry out bigger and/or more simulations) than might be otherwise possible, due to one or more of the following factors:

- For code can only run on one computer but can use multiple cores, an HPC cluster typically will have computers with high core count (up to 128 cores per node on Unity).
- For code that can use MPI to run on more than one networked computer, an HPC cluster can offer even larger core counts since a cluster consists of many computers networked together.
- For code that can use a GPU to speed up computations, an HPC cluster may include higher-performance models of GPU than otherwise available.
- For code that requires a lot of memory, HPC clusters are typically configured with considerable memory.
- Even if none of the situations above is true, multiple jobs can be run simultaneously on an HPC cluster.

However, the individual CPUs in an HPC cluster are typically no faster than those in a desktop PC (sometimes slower, as HPC computers are optimized for reliability) -- so there may be little advantage to running code that is not parallelized with either MPI or utilization of a GPU on an HPC cluster, apart from the higher core count of individual computers and the possibility of running multiple jobs simultaneously.

Finally, the computational resources of an HPC cluster are only useful if available within reasonably short times waiting in queue. Jobs requiring many computers or the fastest GPUs may sit a long time before starting.

### Unity cluster at UMass, Amherst<a id="unity-cluster"></a>

#### History<a id="unity-history"></a>

- Before Unity was created the HPC cluster available to UMass Amherst researchers was the **UMass Shared Cluster (UMSC)**.  UMSC was administered by UMass Medical School, and ran Redhat Linux and the IBM LSF scheduling system.  UMSC was shut down in 3/23.
- As of 1/25 the HPC cluster for general UMass use is [**Unity**](https://unity.rc.umass.edu/) (started in early 2022) located (as UMSC was) at the [**MGHPCC**](https://www.mghpcc.org/) in Holyoke. Unity runs [**Ubuntu**](https://ubuntu.com/) (24.04 as of 1/25) and the [**Slurm**](https://slurm.schedmd.com/documentation.html) scheduling system. 

#### Using interactive apps: Jupyter, MATLAB...<a id="unity-interactive"></a>

- **Unity OnDemand.** For the most part, this document describes how to connect to and use the Unity cluster in **terminal mode**, by connecting from a terminal program on a remote PC.  But it is also possible to run interactive (non-terminal) applications like **JupyterLab**, **MATLAB**, **Mathematica**... as well as a GUI (desktop environment).  These things are all accessed via the **Interactive Apps** menu of the web application [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard).
  
  - Unlike a terminal mode login, these interactive applications on Unity can be accessed via a **web browser** on any PC or laptop, without setting up SSH keys.
  - Using Unity OnDemand requires an institutional login (for UMass, netid and password). It seems that this will then automatically direct any file activity (saved `.ipynb` files, for example) to the user's Unity home directory `/home/<netid>_umass_edu`. 
  - Interactive sessions using Unity OnDemand are limited to eight hours and require continuous internet connection.  If your Jupyter Notebook job might fail due to these limitations, it can be run non-interactively from a terminal-mode login using `nbconvert` or `papermill` as described [here](https://docs.unity.rc.umass.edu/documentation/software/ondemand/jupyterlab-ondemand/).

- **Jupyter confusion.** So far as I understand:
  
  - **Juptyer Notebook** is an application you run on a PC which creates and uses `.ipynb` (Ipython Notebook) files with a notebook interface.
  - **JuptyerLab** is a newer application than JN for using `.ipynb` files with more features (I haven't used much).  Like JN, it is an application you install locally.
  - **Google Colab** is a browser-based application (no local installation needed) with similar functions as JN - create and run `.ipynb` files - but with a rather different interface (more graphical, less keyboard-shortcut oriented).
  - **JuptyerHub** is an application run by servers (like the Unity folks) to provide a browser-based interface for creating and running `.ipynb` files.  JupyerHub is like Google Colab in being browers-based but (a) the interface is more like JuptyerLab and (b) the accessed `.ipynb` files are in Unity filespace, not on Google Drive.  I think JuptyerLab/MATLAB tab on the Interactive Apps page of Unity OnDemand (or maybe all the tabs) are running JupyterHub.
  - There are ways of accessing these things through **VSCode** but I haven't done that yet.

- **Using a Conda environment with Jupyter on Unity.**  As rather cryptically explained in [the Unity docs for Conda](https://docs.unity.rc.umass.edu/documentation/software/conda/), the preferred way to do this is to install and use **`ipykernel`** in the environment to create a **kernel specification** for the environment which will then be available for you to select when you start a Jupyter Notebook.
  
  - The Conda environment is created on Unity in terminal mode, either logged in with SSH as described below or by using the Shell tab of ion [Unity OnDemand](https://ood.unity.rc.umass.edu/pun/sys/dashboard).  Here we make a simple environment **npsp-jn** with NumPy, SciPy and ipykernel, then we use the `ipykernel install` command to make a kernel spec for the environment (note the `ipykernel install` command seems optional, see end of this section):
    
    ```
    $ unity-compute                     # get shell on a compute node
    (wait for the compute-node shell to come up)
    $ module load conda/latest          # needed to run Conda commands
    $ conda create -n npsp-jn python numpy scipy ipykernel
    $ conda activate npsp-jn
    (npsp-jn)$ python -m ipykernel install --user --name npsp-jn --display-name="NumPy-Scipy"
    ```
  
  - Then, when the  Interactive Apps tab of Unity OnDemand is used to start a JupyterLab/MATLAB session the **Launcher** will offer a button to start a `Numpy-Scipy` notebook (as it was called in the `ipykernel install` command above) and in the notebook it will be possible to import and use SciPy functions:
    
    ```
    [1]: from scipy.special import jv
    [2]: jv(3,1)     # Bessel function J_1(3)
          np.float64(0.019563353982668414)
    ```
  
  - It seems that a created  Conda environment is available as a kernel launch option for JupyterLab sessions in Unity OnDemand provided `ipykernel` is  installed in the environment, with a default display name like `Python [conda env:conda-<env name>]`, even if an `ipykernel install` command is not done.
  
  - It also has sometimes seemed that a kernel launch option created using `ipykernel install` remains available in JupyterLab OnDemand sessions even after the corresponding Conda environment is removed -- not sure how that is working.

#### Logging in (terminal mode)<a id="unity-login"></a>

- **Logging with SSH.** To login to Unity from a terminal program on a remote PC, **SSH keys must be set up**.  Here are the [instructions in the Unity docs](https://docs.unity.rc.umass.edu/documentation/connecting/ssh/). While a bit of a pain to set up, SSH is convenient to use and is necessary to enable usage of the `scp` and `rsync` file transfer commands described below.
  
  - **Setting up keys to allow login to unity from PC.** Here **`<user>`** is a user name on a Linux PC, while **`<userc>`** is a user name on Unity (assigned by Unity admins, typically of form `netid_umass_edu`):
    
    - In a browser go to https://docs.unity.rc.umass.edu/ and login with netid/pw.
    
    - In the  **Account Setting** page click **`[   +   ]`**. 
    
    - In the **Add New Key** popup selected **Generate key**, **OpenSSH**.
    
    - **Save the key on your PC in `~/.ssh`** with desired name, e.g. `~/.ssh/unity.key`.
    
    - Set the permissions of the key file so only the owner can read/write, otherwise ssh will refuse to use it:
      
      ```
      user:~/.ssh$ chmod 600 unity.key
      ```
    
    - Add following lines to `~/.ssh/config`, creating this file if necessary :
      
      ```
      Host unity
          HostName unity.rc.umass.edu
          User <userc>
          IdentityFile ~/.ssh/unity.key
      ```
    
    - SSH will set up and maintain the file `~/.ssh/known_hosts`
    
    - There is a way (not shown here) to store the private key in encrypted form, such that a password will be required to use it.
  
  - Now we can **login to Unity the PC** (will not request a password, unless the private key is encrypted):
    
    ```
    <user>:~$ ssh unity      # we start in a bash shell on the PC
    <userc>@login2:~$        # now we are in bash shell on Unity, ctrl-d to logout
    ```
  
  - Login to Unity from one of my PCs was set up as above and previously working -- but when I tried to login 12/24 I got warning and refusal to login:
    
    ```
    (base) dc:~$ ssh unity
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                         ...
    ```
    
    Later (2/25), from another PC, I got a different error message that sounded like there was a temporary problem that someone else would fix, but that was not the case:
    
    ```
    (base) dc:~$ ssh unity
    ssh: Could not resolve hostname unity: Temporary failure in name resolution
    ```
    
    In both cases this was fixed this by following the instructions above to generate a new key (including the `chmod 600 ... command`).

- **Logging in with Unity OnDemand.**  The **Shell, >_Unity Shell Access** menu item of [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard) opens a login shell (in a browser window, not a terminal) without using SSH (Unity OnDemand is accessed with `netid` and `pw`). This is reasonably convenient for all platforms and also allows logging into Unity from Windows without setting up ssh keys and without software beyond a browser.  But there seem to be some limitations on what can be done from this browser-window shell.

#### Storage<a id="unity-storage"></a>

- Here is the [Unity doc page](https://docs.unity.rc.umass.edu/documentation/cluster_specs/storage/) on storage.
- As of 1/25, the “Base quotas” (free initial tier) given to users are:
  - 50 GB of HDD in **`/home/<userc>`** with `<userc>` the Unity username, intended for user init files (like scripts for sbatch, I guess), not large-scale job I/O.
  - 1 TB of SDD in **`/work/pi_<userc>`** intended for job I/O, shared between users in a PI’s group.
  - Upon request, 5 TB HDD in **`/project/pi_<userc>`** intended to store e.g. job output from /work but not for direct job I/O.
  - Without special permission users can allocate up to 50 TB of high-performance (job I/O) scratch space lasting up to 30 days in **`/scratch`**.
  - Previously (7/22) there were **`/work/<userc>`** directories  rather than **`work/pi_<userc>`** for job I/O, but these have been discontinued.

#### Transferring files to/from Unity <a id="unity-file-transfer"></a>

- **Graphical:** Files can be transferred using the graphical [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard).  There may be some limitations on file size - using drag-and-drop I successfully transferred a 1.4 GB file from home (slower internet than work) but attempts to transfer a 3.5 GB file repeatedly failed.

- **CLI:** The following commands are part of OpenSSH and **require ssh keys to be set up** as shown above. Then the **`rsync`** and **`scp`** commands shown here can be run on the PC. Some info on using these commands with Unity is on [this page](https://docs.unity.rc.umass.edu/documentation/managing-files/cli/).
  
  - Use **`scp`** to copy an individual file `localfile` on the PC to a the cluster:
    
    ```
    $ scp <localfile> unity:~/<subdirec>                  # copy under directory /home/<userc>/
    $ scp <localfile> unity:/work/<direc>/<subdirec>      # copy under group working directory
    ```
    
    Unlike Unity OnDemand drag-and-drop, scp successfully copied a 3.5 GB file to Unity from my home PC with its slow internet  uplink (took 23 min).
  
  - Similarly use **`scp`** to copy an individual file `remotefile` on the cluster to the current directory on the PC (note the period indicating the current directory):
    
    ```
    $ scp unity:/work/<direc>/remotefile .
    ```
  
  - To **copy a directory and all its contents**, use the **`-r`** flag on scp
    
    ```
    $ scp -r <source directory> <target directory>
    ```
  
  - Be careful when using **`scp`**, especially with the **`-r`** flag, as **it will overwrite existing files on the destination computer** -- it seems not to have a flag to inhibit this.
  
  - You can also use **`rysnc`** to transfer a whole directory including subdirectories between a Linux PC and Unity.  Unlike `scp`, with appropriate flags `rsync` can be run repeatedly and will only update files that have been changed.  I used `rsync` on the old UMSC HPC cluster, but I haven't tried it yet  on Unity.

- **Cloning repos from GitHub (GH).**  As of 2/25 I don't see information about this in the [Unity docs](https://docs.unity.rc.umass.edu/documentation/), but this was figured out by consulting with someone at Unity help and looking at at [this webpage](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding) from GitHub. As in other places in this document `<user>..$` denotes input to a shell on the PC, while `<userc>..$` denotes input to a shell on Unity -- I have only tried these commands on a Unity login node, but they might also work on a compute node.
  
  - **Cloning from public GH repo.**  This can be done with no special setup by using the `git clone` command  on Unity with the  HTTPS address of the repo, available in the **`<> Code`** tab of the GH page for the repo.  For example, to clone the public repo `doncandela/dcfuncs` into a directory `foo` under `/work/pi_<userc>` do
    
    ```
    <userc>..foo$ git clone https://github.com/doncandela/dcfuncs.git
    ```
    
    This command resulted in the creation on Unity of the subdirectory `foo/dcfuncs`, containing this GH repo including its `.git` file (which is managed by Git and holds the repo history). 
  
  - **Cloning from private GH repo.** It's assumed here this a private repo to which you have SS access -- for example a repo that belongs to you.  It seemed that the easiest way was to use **SSH agent forwarding**, which allows Unity (while you are logged in) to use the SSH keys from your PC to authenticate to GH.  First check that you have SSH access to GH from your PC:
    
    ```
    <user>..$ ssh -T git@github.com
    Hi doncandela! You've successfully authenticated, but GitHub does not provide shell access.
    ```
    
    Still on the PC, edit (or create) the file  `~/.ssh/config` to enable SSH agent forwarding. From previously setting up SSH access to GH I found that this file was present on my PCs with a block for Unity,  to which I added the last line shown here:
    
    ```
    Host unity
         HostName unity.rc.umass.edu
         User candela_umass_edu                  # will be your Unity username <userc>
         IdentityFile ~/.ssh/2025-01-unity.key   # will be your SSH key file for Unity
         ForwardAgent yes                        # this is the new line added
    ```
    
    It will be necessary to do this on **every PC** that you will use to SSH into Unity and then tell Unity to clone from GH, and you will need to disconnect form Unity and SSH into it again for this to take effect.
    
    After you do this you will be able do the same check on Unity as was done on the PC that you have SSH access to GH:
    
    ```
    <userc>..$ ssh -T git@github.com
    Hi doncandela! You've successfully authenticated, but GitHub does not provide shell access.
    ```
    
    However, the *first* time you try to access GH from your Unity account you will need to answer `yes` to a multiline message like this:
    
    ```
    <userc>..$ ssh -T git@github.com
    The authenticity of host 'github.com (140.82.114.3)' can't be established.
    ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
    This key is not known by any other names.
    Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
    ```
    
    Now, on Unity you can clone a private GH repo you have access to like `doncandela/dem21` by using the repo's SSH address (also available the **`<> Code`** tab of the GH page for the repo):
    
    ```
    <userc>..foo$ git clone git@github.com:doncandela/dem21.git
    ```
    
    This creates the subdirectory `foo/dem21` on Unity with a clone of the repo.

- **Seeing how big the transferred files are.** To see the size on Unity storage of a whole set of directories and subdirectories with their contents (for example as transferred using `scp -r ...` of by cloning a GitHub repo) use the `du` command:
  
  ```
  <userc>...foo$ du -h
  0    ./dem21/.git/branches
       ...
  6.5K    ./dem21/tests/mpi
  473K    ./dem21/tests
  10M    ./dem21
  10M    .
  ```
  
  Variants: `du -h` as shown above will show the sizes of directories only, while `du -ah` will also show the sizes of all files in the directories.

#### Slurm on Unity<a id="unity-slurm"></a>

- Some Slurm resources:
  
  - [Quick Start User Guide](https://slurm.schedmd.com/quickstart.html) in the [Slurm docs](https://slurm.schedmd.com/documentation.html).
  - [Overview of threads, cores, and sockets in Slurm](https://docs.unity.rc.umass.edu/documentation/get-started/hpc-theory/threads-cores-processes-sockets/) in the [Unity docs](https://docs.unity.rc.umass.edu/documentation/).
  - Stanford tutorial [SLURM Basics](https://stanford-rc.github.io/docs-earth/docs/slurm-basics).
  - A list of [Convenient Slurm Commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/)  with detailed instructions from Harvard.
  - [Research Computing User's Guide](http://acadix.biz/RCUG/HTML/index.html) (esp Ch. 11 "Job Scheduling with SLURM").
  - A few more advanced resources are linked in [Running batch jobs](#run-batch) below.

- The nodes in a Slurm cluster are assigned to **partitions**, and one or more partitions are specified when a job is submitted.  Slurm allows a node to be assigned to multiple partitions, but I don’t think this is done much on Unity except that some `gpu` nodes are also in `cpu`, etc.  Here are the **x86_64 general-access and preempt [partitions on Unity](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/)** (numbers and best GPUs as of 1/25):
  
  | Partition     | Nodes | Total cores | Cores/node | Mem/node     | GPUs/node | Best GPUs         |
  | ------------- | ----- | ----------- | ---------- | ------------ | --------- | ----------------- |
  | `cpu`         | 181   | 10,240      | 24-64      | 180-1,000 GB |           |                   |
  | `cpu-preempt` | 146   | 9,424       | 24-192     | 30-1,510 GB  |           |                   |
  | `gpu`         | 156   |             | 12-128     | 180-370 GB   | 2-4       | NVIDIA V100       |
  | `gpu-preempt` | 198   |             | 64-128     | 500-2010 GB  | 6-8       | NVIDIA A100, H100 |
  
  - Job [wall-time](#wall-cpu-time) limits:
    
    - All partitions have a **default time limit of one hour**, in effect when `#SBATCH -t=..` is not used.
    
    - The nodes in **preempt partitions** belong to specific groups and jobs submitted outside the owning groups can be killed after two hours.  So, unless checkpointing is used to enable a job to pick up where it left off, general users should typically **submit jobs with time limits less than 2 hours to the preempt partitions**.
    
    - The **maximum time limit is 2 days** for all of these general-access partitions unless **`-q=long`** is set in the sbatch script which case the maximum time limit is **14 days (336 hours)**.
    
    - From the Unity docs I think the following are true but I’m not 100% sure:
      
      - If a job does start on a preempt partition, it will not be killed before 2 hours of run time.
      
      - To get more chance of scheduling sbatch script can list more than one partition, e.g
        
        ```
        #SBATCH -p=cpu,cpu-preempt
        ```
  
  - Jobs submitted to the `gpu` or `gpu-preempt` partitions will be rejected if they do not request GPUs using e.g. `#SBATCH -G=..`.
  
  - In my recent (4/25) experience, jobs submitted to the preempt partitions typically are scheduled more quickly than for the non-preempt partitions -- so it is worth doing this if the time limit can be kept less than two hours.  Also, it is possible that the Unity limits for total cores and total GPUs in use are not enforced for jobs on the preempt partitions.

- Some useful Slurm commands:
  
  ```
  $ squeue --me                    # show info on my jobs
  $ sacct -j <jobid>               # show more detailed info on a specific job
  $ scancel <jobid>                # kill one of my jobs.
  $ sacct -b                       # show brief info on recent jobs
  $ sacct -S 0601                  # list my jobs started June 1 or later
  $ seff <jobid>                   # show utilization efficiency of a completed job
  $ sinfo -p cpu -r -l             # show status of nodes in partition cpu
  $ scontrol show partition cpu    # detailed info on partition cpu
  $ scontrol show node cpu029      # detailed info on node cpu029
  $ scontrol show config           # show slurm configuration including default values
  ```

#### Running jobs interactively: `salloc` or `unity-compute`<a id="run-interactive"></a>

- For this section **interactive** means running a program in the shell (terminal), directly seeing any output, and waiting for it to complete before the shell prompt returns -- just as one might run a program in a terminal window on a PC, and different from running a program in the background using `sbatch` as [described in detail below](#run-batch). There is a different way of running programs "interactively" in Unity, using non-terminal apps like Jupyter Notebook as [described above](#unity-interactive).

- You are not supposed to run interactive jobs on a login node.  Instead, from the login-node shell, issue a command like
  
  ```
  $ salloc -c 6 -p cpu
  ```
  
  to allocate 6 cores on a node in the `cpu` partition and start an interactive shell on that node -- when the compute node is allocated (which seems to take at least a few seconds, sometimes longer) you will see the node name in the prompt. Similarly to allocate 6 cores and one GPU on a node in the `gpu` partition do
  
  ```
  $ salloc -c 6 -G 1 -p gpu
  ```
  
  In either case use `ctrl-d` to exit back to the login-node shell.

- On Unity I think `salloc` jobs have a default time limit of one hour, and default memory allocation of 1 GB per core.  These can be increased as follows, using the [same options as for `sbatch`](#run-batch):
  
  ```
  $ salloc -c 6 -p cpu -t 3:00:00          # set time limit to 3 hours
  $ salloc -c 6 -p cpu --mem-per-cpu=20G   # allocate 20 GB/core = 120 GB total
  ```

- There is another way to do this using `srun -pty bash`, but the way shown above with `salloc` seems to be recommended.  However Unity does provide a command `unity-compute` that gets a shell on a compute node in this way:
  
  ```
  $ unity-compute         # get a compute-node shell with default 2 cores
  $ unity-compute 6       # get a compute-node shell with 6 cores
  ```

#### Using `.bashrc` and `.bash_aliases`<a id="rc-files"></a>

- The file **`~/.bashrc`** is executed whenever an interactive shell is started (when logging in, when using `salloc` to open an interactive shell on a compute node...).

- If it exists **`~/.bash_aliases`** will be run by `~./bash_rc`, so this file is a good place to add alias commands without messing with the more complex `.bashrc`.
  
  - This shows the contents of a typical .bash_aliases file:
    
    ```
    $ cat .bash_aliases
    # ~/.bash_aliases for unity 1/11/25 D.C.
    
    # go to work directory
    alias wcd='cd /work/pi_candela_umass_edu'
    
    # get 6 cores on a non-gpu compute node and start an interactive shell
    alias ish='salloc -c 6 -p cpu'
    
    # get 6 cores and a GPU on a gpu compute node and start an interactive shell
    alias ishg='salloc -c 6 -G 1 -p gpu'
    ```
  
  - To see a list of defined aliases: `$ alias`

#### Using modules and Conda<a id="unity-modules-conda"></a>

- **Modules on an HPC system.**
  
  - HPC systems like Unity have a [**`modules`**](https://docs.unity.rc.umass.edu/documentation/software/modules/) system installed  that sets the environment so desired versions of software are available.  Loading a module is much like activating a Conda environment: It typically sets `PATH` and other environment modules and can also take other actions.   
    
    - The main difference is that modules are created by the HPC system admins to point to software packages installed by them in system space, and to run those packages correctly on the HPC system.
    - Conversely Conda environments are created by the user, install packages in user space, and must be configured by the user to work correctly.
    - It's not unusual to load some modules to make things like MPI, CUDA, and Conda itself available, and then to activate a Conda environment with specific code and packages needed.
  
  - A few useful module commands (to be entered in a shell before running code; also `module purge` and `module load` commands are typically included in [sbatch scripts](#run-batch):
    
    ```
    $ module av python               # show all python modules available
    $ module spider python           # another way to show available python modules
    $ module spider python/3.11.7    # shows which other modules must be loaded before this one (in this case none)
    $ module load python/3.11.7      # load one of the python modules
    $ module show python/3.11.7      # show what is in a particular module
    $ module list                    # list which modules are loaded
    $ module purge                   # unload all modules
    ```

- **Using Conda on an HPC system.**<a id="conda-hpc"></a>
  
  - You must start by loading the **Conda module**.  Then `conda` commands can be used to create and activate Conda environments. Both `conda install` and `pip install` can be used in a Conda environment to install packages in that environment (see [Pip, Conda, and APT](#pip-conda-apt) above for some typical commands). 
  
  - In this example an environment **`npsp`** is created with NumPy, SciPy, Matplotlib, and a version of Python earlier than the one installed outside of environments. These commands take some time and probably should be **[run on a compute node](#run-interactive)**, not a login node. 
    
    ```
    $ python --version
    Python 3.12.3
    $ module load conda/latest
    $ conda create -n npsp python=3.11
    $ conda activate npsp
    (npsp)$ conda install numpy scipy matplotlib
    (npsp)$ python --version
    Python 3.11.11
    (npsp)$ pip list
    Package         Version
    --------------- --------y
    matplotlib      3.10.0
    numpy           2.2.1
    scipy           1.15.1
    ```
  
  - A created environment like `npsp` will persist across logins to Unity, but the `module load conda/latest`  command must be executed in every new shell before a `conda` command such as activating a previously created environment can be given.
  
  - It seems that on Unity Conda environments are always stored in `/work/<userc>/.conda` no matter which directory they were created from, and (conveniently) they are usable from both `/home` and `/work` directories.
  
  - Unity has commands available to create several different **preset Conda environments** as shown [here](https://docs.unity.rc.umass.edu/documentation/software/conda/).  This is discussed more in the section below [Using Pytorch on Unity](#pytorch-unity).

- **Installing a local package on Unity.**
  
  - This is done much the same way as installing a local package on a PC, as [shown above](#local-package).
  
  - Following the same example as in that section, the repository for the **`dcfuncs`** package is cloned to a directory `work/pi_<userc>...clones/dcfuncs` on Unity. Then a Conda environment **`dfs`** is created and NumPy and  `dcfuncs` are installed in that environment.
    
    ```
    $ unity-compute                          # get shell on a compute node
    (wait for the compute-node shell to come up)
    $ module load conda/latest
    $ conda create -n dfs python=3.12
    $ conda activate dfs
    (dfs)..$ conda install numpy              # needed by dcfuncs
    (dfs)..$ cd ..clones                      # go to directory where will put clone
    (dfs)..clones$ git clone https://github.com/doncandela/dcfuncs.git
    (dfs)..clones$ cd dcfuncs; ls             # go into the cloned repo
    LICENSE  README.md  pyproject.toml  setup.py  src  test
    (dfs)...clones/dcfuncs$ pip install -e .  # install dcfuncs to current environment
    (dfs)...clones/dcfuncs$ pip list
    Package    Version Editable project location
    ---------- ------- ------_---------------------
    dcfuncs    1.0     /work/pi_<userc>/.../dcfuncs
    numpy      2.2.2
    pip        24.3.1
    setuptools 75.8.0
    wheel      0.45.1
    ```
  
  - Now that the environment `dfs` has been created, we can log completely out of Unity.  The environment it will persist and with it activated `dcfuncs` is available for importing in any directory. In a new login:
    
    ```
    $ unity-compute                       # get shell on a compute node
    (wait for the compute-node shell to come up)
    $ module load conda/latest
    $ conda activate dfs
    (dfs)...$ cd ...clones/dcfuncs/test   # go to test-code directory in cloned repo
    (dfs)...test$ ls
    test-configs.py  test-util.ipynb  test-util.py  test0.yaml  test1.yaml
    (dfs)...test$ python test-util.py     # we can run this program that imports dcfuncs
    This is: dutil.py 8/19/24 D.C.
    Using: util.py 8/18/24 D.C.
    
    Testing zz:
    - WARNING from util.py test code - This is just a warning.
    
    Testing stoi:
    stoi results = 93801881091158, 6318, 249613385242335
                           ...
    ```

#### Running batch jobs: `sbatch`<a id="run-batch"></a>

- To run a background job on Unity an **sbatch script** e.g `myjob.sh` is created then the job is submitted to Slurm using
  
  ```
  $ sbatch myjob.sh
  ```

- If desired the `sbatch` command can be run on a login node as it simply submits the job for scheduling -- the job itself is run on compute nodes chosen based on the parameters in the sbatch script.

- If the script results in job output being written to the CWD then it will typically be run in a subdirectory of `/work/pi_<userc>`, which has room for large job output.

- Here are the contents of a simple sbatch script:
  
  ```
  #!/bin/bash
  # Example sbatch script - can put comments like this on any line.
  #SBATCH -c 4                  # use 4 cores per task
  #SBATCH -p cpu                # submit job to partition cpu
  # #SBATCH -p cpu-preempt        # submit job to partition cpu-preempt (this is commented out)
  module purge                  # unload all modules
  module load python/3.12       # load version of Python needed
  python myscript.py > output   # run myscript.py sending its output to a file
  ```
  
    Notes on this script:
  
  - The first line `#!/bin/bash` indicates Bash should be used to interpret the file (you could use a different shell).
  
  - The sbatch script is a regular shell file except that lines that **start with `#SBATCH`** (exactly like this, in all caps) are interpreted specially by the `sbatch` command.  This means `#SBATCH` lines can be commented out by doubling the initial `#` (with or without a space as shown above).
  
  - The `#SBATCH` lines, which give information to Slurm on how to schedule the job, must come before any regular shell commands.  Any `#SBATCH` lines after other shell commands other than comments are ignored.
  
  - The remainder of the sbatch script is ordinary shell commands.  Typically these could load modules as needed, then run the desired program.
  
  - The Conda module must be loaded before activating a Conda environment, so the sbatch script might have lines like:
    
    ```
    module load conda/latest
    conda activate myenv
    ```
  
  - Programs can be run directly in the script as shown above, or as arguments to the Slurm command `srun` on a line in the script like this:
    
    ```
    srun python myscript.py > output
    ```
    
    Using `srun` establishes a **job step** and can also launch **multiple copies of the program** as separate tasks if `#SBATCH -n ..` was used to specify more than one task – see [this page](https://groups.oist.jp/scs/advanced-slurm) for more info.
  
  - MPI programs can be run using `srun` or `mpirun`.  Note `srun` is a Slurm command that can start multiple copies of any program, whether or not MPI is available.  Conversely `mpirun` is only available in an environment in which MPI is available, see [Using MPI on Unity](#unity-mpi) below.
    
    This starts 10 copies of `myscript.py` as separate tasks:
    
    ```
    mpirun -n 10 python myscript.py > output
    ```
    
    while this sets the number of copies (MPI tasks) according to the `#SBATCH -n=...` setting:
    
    ```
    mpirun python myscript.py > output
    ```
    
    The second form is probably preferable unless there is some reason to have `mpirun` to create less tasks than were allocated by `#SBATCH -n ..`.

- Some resources for writing sbatch scripts, especially on choosing and setting #SBATCH parameters:
  
  - In Unity docs: [Introduction to batch jobs](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/),  [Overview of threads, cores and sockets](https://docs.unity.rc.umass.edu/documentation/get-started/hpc-theory/threads-cores-processes-sockets/), and [Slurm cheat sheet](https://docs.unity.rc.umass.edu/documentation/jobs/slurm/).
  - In official Slurm docs: [sbatch options](https://slurm.schedmd.com/sbatch.html) (there are many).
  - A [quick-start guide](https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-quick-start-guide) from Lawrence Livermore National Lab. Warning: some things here are particular to LLNL, won’t apply to Unity.
  - [Examples of sbatch scripts for different kinds](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/) of jobs from Berkeley (same warning).
  - Some [more advanced Slurm topics](https://groups.oist.jp/scs/advanced-slurm) (array jobs, using srun in sbatch scripts to run multiple copies of one or more programs…) from Okinawa Institute of Science and Technology.

- In Slurm, a **task** can use one or more **cores** (which are called cpu’s in #SBATCH settings), but a task always lives on a single **node**, which typically means a single "computer" -- one or two CPU chips on a board with shared RAM and other resources like GPUs.  On the other hand, a single node can run multiple tasks if it has enough cores. 
  
  - For a **non-MPI job** there might be a **single task** (possibly multi-core), which therefore uses part or all of a **single node**.  These match the Slurm default `#SBATCH n=1` which will automatically live on one node.
  - For an **MPI job** (for which Slurm was originally conceived) there will ordinarily be **more than one task** and **each task corresponds to an MPI rank, running an independent copy of the code.**  I believe these tasks might run on one or more nodes unless `#SBATCH -N=...` is used to fix the number of nodes.
  - [This page](https://groups.oist.jp/scs/advanced-slurm) shows how **`srun`** can be used to run multiple copies of a program that don't communicate using MPI -- thus **more than one task** (I haven't tried this).

- Here are some #SBATCH settings useful for both single-task (non-MPI) and multi-task (MPI) jobs.  Note many command have two equivalent forms: a **single-dash+single-letter form that does not need an equals sign**, and a **double-dash+multi-letter form that takes an equal sign** before any supplied value.
  
  ```
  #SBATCH -J <name>           # set a name for job, otherwise will be script filename
  #SBATCH --job-name=<name>   # “ “
  
  #SBATCH -o <ofname>         # set filename for output, otherwise will be slurm-<jobid>.out
  #SBATCH --output=<ofname>   # “ “
  #SBATCH -e <efname>         # set filename for error output, otherwise will go to output file
  #SBATCH --error=<name>      # “ “
  
  #SBATCH --mail-type=END     # send email to submitting user when job ends
  #SBATCH --mail-type=ALL     # send email when job starts, ends, or fails
  
  #SBATCH -p <pname>          # run the job on the nodes in partition <pname> 
  #SBATCH --partition=<pname> # “ “
  #SBATCH -p <pname1>,<pname2>  # use nodes in either of two partitions
  
  #SBATCH -t 10               # set wall-time limit for job to complete to 10 minutes
  #SBATCH --time=10           # “ “
  #SBATCH -t 3:30             # set wall-time limit to 3.5 minutes
  #SBATCH -t 3:30:00          # set wall-time limit to 3.5 hours (note needs seconds)
  #SBATCH -t 0-3:30           # " "
  #SBATCH -t 2-3              # set wall-time limit to 2 days + 3 hours
  
  #SBATCH -c 6                # allocate 6 cores (not cpus!) per task
  #SBATCH --cpus-per-task=6   # “ “
  
  #SBATCH -G 1                # allocate one GPU for the whole job
  #SBATCH --gpus=1            # “ “
  
  #SBATCH --mem-per-cpu=4G   # allocate 4 GB of memory per core (not cpu!)
  
  #SBATCH -q <qos>            # request quality of service <qos>
  #SBATCH --qos=<qos>         # “ “
  #SBATCH -q long             # on unity allow time limit up to 14 days
  #SBATCH -q short            # on unity get higher priority for a single job per user
                              # on <=2 nodes with time limit <=4 hours
  
  #SBATCH -C ”<cons>”         # only use nodes that have constraint <cons>, quotes may be unnec.
  #SBATCH --constraint=”<cons”> # “ “
  #SBATCH -C ”<cons1>&<cons2>”  # only use nodes that match both constraints
  #SBATCH -C ”v100|a100”      # on unity only use nodes that have V100 or A100 GPUs
  ```
  
  Notes on these: 
  
  - For all of the Unity general-access partitions: The **default time limit is one hour**, when `-t ..` is not used. **Jobs submitted to preempt partitions can be killed after two hours**.
  - The **maximum time limit** that can be set using `-t ..` is **two days** unless `-q long` is used in which case it is **14 days (336 hours)**.
  - By running `scontrol config` it is seen that on Unity the **default memory per core is 1024 MB**.
  - [This page](https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/) shows the **constraints** available on Unity.
  - To set multiple constraints I think a **single `#SBATCH C ..` line** using the `&` or `|` operators must be used as shown above.

- Here are some #SBATCH settings more useful for **multi-task (e.g. MPI) jobs**:
  
  ```
  #SBATCH -n 100            # allocate resources for 100 tasks (100 MPI ranks)
  #SBATCH --ntasks=100      # “ “
  #SBATCH -C ib             # only use nodes with InfiniBand networking...
  #SBATCH -C "ib&mpi"       # .. also ensure consistent CPU type across nodes
  #SBATCH --gpus-per-task=1 # allocate one GPU per task
  #SBATCH --nodelist=cpu[049-068] # run on specific nodes
  #SBATCH -N 10             # run the job on 10 nodes
  #SBATCH --nodes=10        # “ “
  #SBATCH --exclusive       # use entire nodes (don’t share nodes with other jobs)
  # When using the following probably should be setting the number of nodes with -N
  #SBATCH --mem=5G          # allocate 5 GB of memory per node
  #SBATCH --mem=0           # allocate all available memory on nodes used
  ```
  
  The nodes on Unity are very heterogeneous, with between 12 and 192 cores per node, and there are various ways jobs can be constrained to run on nodes well-matched to the job:
  
  - The combination of `-n` (cores, typically) and `-N` (nodes) settings can be used to limit the job to high core-count nodes.  For example `-n 64` along with `-N 1`, or `-n 128` along with `-N 2`, will limit the job to nodes with at least 64 cores per node.
  
  - Setting `--exclusive` and `-n` without setting `-N` will also most likely result in nodes with certain minimum core counts.  For example `-n 128` with `--exclusive` will most likely run on two 64-core nodes, or one 128-core node.  As shown in examples below, this seems like a useful way to run jobs on even multiples of 64 cores.
  
  - `--nodelist` can limit the job to a specific set of nodes.
  
  - `-C` can limit the job to nodes having specific features. 

- **Example of a  simple batch job**<a id="simple-batch"></a> not using MPI, or a GPU, or Apptainer.
  
  - As a container is not being used, a Conda environment must be set up on Unity with the needed packages. This example uses the environment **`npsp`** [set up above](#conda-hpc) with Python, Numpy, and Scipy, but not CuPy.
  
  - The program `gputest.py` described above (which won't try to use a GPU if CuPy cannot be imported) was put in a directory `/work/.../try-gputest` along with an sbatch script **`simple.sh`** with these contents:
    
    ```
    #!/bin/bash
    # simple.sh 2/27/25 D.C.
    # One-task sbatch script runs gputest.py using none of MPI, a GPU, or Apptainer.
    #SBATCH -c 6                         # use 6 CPU cores
    #SBATCH -p cpu                       # submit to partition cpu
    echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
    module purge                         # unload all modules
    module load conda/latest             # need this to use conda commands
    conda activate npsp                  # environment with NumPy and SciPy but not CuPy
    python gputest.py                    # run gputest.py, output will be in slurm_<jobid>.out  
    ```
    
    As written above both the output of the commands like `echo nodelist...` and `module load...` will go to a file `slurm-<jobid>.out` in the directory from which `sbatch` is run, along with any output to sdout or stderr by `gputest.py`.  If desired the stdout+stderr output of `gputest.py` can be directed to a different file by replacing the last line above with `python gputest.py &> <filename>`. The `echo nodelist...`  command in this script is not necessary but they will put useful information in the output file -- in this case the specific nodes used. Here is a [list of Slurm environment variables](https://hpcc.umd.edu/hpcc/help/slurmenv.html) that can be used in this way.
  
  - To submit the job we do
    
    ```
    try-gputest$ sbatch simple.sh
    Submitted batch job 29258794
    ```
    
      Notes:
    
    - With this way of running, `sbatch` is run from the directory `try-gputest` and the output will go there as well -- this is why we are using a directory under `/work`.
    
    - The `sbatch` command returns immediately with the jobid, no matter how long the actual job takes.
    
    - `sbatch` can be run from a login node, since the job runs on nodes allocated according to the `#SBATCH` lines in `simple.sh`, not on the node where `sbatch` was run.
    
    - Since `simple.sh` sets the Conda environment, there is no need to set it before running `sbatch`.
  
  - To see our pending and running jobs do
    
    ```
    try-gputest$ sbatch squeue --me
                JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              29258794       cpu simple.s candela_  R       0:10      1 cpu001
    ```
    
    Jobs that haven't started running yet will have `PD` in the `ST` column.  Here `R` means the job is running (and has used 10 seconds so far).  To cancel the job (for example, if it runs much longer than expected) do
    
    ```
    try-gputest$ scancel 29258794
    ```
    
    Unlike an interactive job, **a job submitted using `sbatch` will not be cancelled when you log out** -- it will continue running until your code ends, or the job time limit (**default one hour**) is exceeded, or you cancel it with `scancel`.
  
  - Once the job has completed, it will no longer be shown by `squeue`, but we can get info on the job's efficiency by doing
    
    ```
    try-gputest$ seff 29258794
    Job ID: 29258794
    Cluster: unity
    User/Group: candela_umass_edu/candela_umass_edu
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 00:02:36
    CPU Efficiency: 18.57% of 00:14:00 core-walltime
    Job Wall-clock time: 00:02:20
    Memory Utilized: 1.08 GB
    Memory Efficiency: 18.02% of 6.00 GB
    ```
    
      We can get more info on completed jobs by using `sacct` as detailed in [this page](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/) from Harvard.
  
  - We can look at the output file when the job is completed (or while the job is still running, to see the output so far):
    
    ```
    nodelist=cpu001
    Loading conda
    Running: gputest.py 11/22/23 D.C.
    Local time: Thu Feb 27 23:57:35 2025
    Import cupy failed, using CPU only
    CPU timings use last 10 of 11 trials
    
    ***************** Doing test dense_mult ******************
    Multiply M*M=N element dense matrices
                                ...
    ```

- **Including the sbatch script in the `slurm-<jobid>.out` file**<a id="keep-sbatch"></a>.
  
  - Sometimes it is useful to keep a record of the sbatch script in the output file (called `slurm-<jobid>.out` by default). For example, if several jobs are run by making small changes to the sbatch script, your may want to keep a record in each output file of the precise sbatch script used to produce that output. This can be done by including the command `scontrol write batch_script $SLURM_JOB_ID -` in the sbatch file (note the `-` at the end of this command).  This sbatch script **`simple2.sh`** includes this command, followed by `echo` to make a blank line:
    
    ```
    #!/bin/bash
    # # simple2.sh 4/9/25 D.C.
    # One-task sbatch script using none of MPI, a GPU, or Apptainer.
    # This version writes the batch script to the ouput file.
    #SBATCH -c 6                         # use 6 CPU cores
    #SBATCH -p cpu                       # submit to partition cpu
    scontrol write batch_script $SLURM_JOB_ID -;echo # print this script to output
    echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
    module purge                         # unload all modules
    module load conda/latest             # need this to use conda commands
    conda activate npsp                  # environment with NumPy and SciPy but not CuPy
    python gputest.py                    # run gputest.py, output will be in slurm-<jobid>.out
    ```
  
  - Here is the start of the `slurm-<jobid>.out` file produced by running `sbatch simple2.sh`:
    
    ```
    #!/bin/bash
    # simple2.sh 4/8/25 D.C.
    # One-task sbatch script using none of MPI, a GPU, or Apptainer.
    # This version writes the batch script to the ouput file.
    #SBATCH -c 6                         # use 6 CPU cores
    #SBATCH -p cpu                       # submit to partition cpu
    scontrol write batch_script $SLURM_JOB_ID -;echo # print this script to output
    echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
    module purge                         # unload all modules
    module load conda/latest             # need this to use conda commands
    conda activate npsp                  # environment with NumPy and SciPy but not CuPy
    python gputest.py                    # run gputest.py, output will be in slurm-<jobid>.out
    
    nodelist=cpu023
    Loading conda
    Running: gputest.py 11/22/23 D.C.
    Local time: Tue Apr  8 22:10:44 2025
    Import cupy failed, using CPU only
    CPU timings use last 10 of 11 trials
                        . . . 
    ```

#### Why won't my jobs run<a id="why-wont-run"></a>

- Here are some useful scripts to see current usage/availablilty of resources on Unity, [more are here](https://docs.unity.rc.umass.edu/documentation/jobs/helper_scripts/).
  
  ```
  $ unity-slurm-partition-usage  # show how many idle cores and gpus in each partition
  $ unity-slurm-node-usage       # for each node show idle cores and gpus, partition is in
  $ unity-slurm-account-usage    # show cores and gpus my group is currently using
  $ unity-slurm-find-nodes a100  # show nodes with specified constraint (here A100 GPUs)
  $ unity-slurm-find-nodes a100 | unity-slurm-node-usage    # show idleness of nodes with
                                                            # specified constraint
  ```

- The Slurm **`sprio`** command returns information about the priorities of all pending jobs, and the factors that enter into those priorities.  One of those factors is the current  [fairshare score](https://docs.crc.ku.edu/how-to/fairshare-priority/) for the PI group, and the Slurm **`sshare`** shows how that is computed.  Here is an example that I encountered when trying to run some jobs in 4/25:
  
  - Running **`squeue --me`** shows that I have five jobs stuck pending:
    
    ```
    $ squeue --me
                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              32556933       cpu mx2-unit candela_ PD       0:00      4 (Resources)
              32556435       cpu mx2-unit candela_ PD       0:00      2 (Priority)
              32555853       cpu mx2-unit candela_ PD       0:00      1 (Priority)
              32223401       gpu gputest. candela_ PD       0:00      1 (Priority)
              32586974       gpu gputest. candela_ PD       0:00      1 (Priority)
    ```
  
  - Looking at the output of **`sprio`** for one of these jobs (last line here) we see that it's priority (10284) is low, due in part I guess to the low fairshare number (213):
    
    ```
    # Run sprio twice, one to show the head of the table and a second time to pull out
    # the line for one of my stuck jobs 32556933:
    $ sprio|head;sprio|grep 32556933
              JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE  PARTITION        QOS
           28472863 building       11082          0       1000        344         31      10000       1000
           28958275 building        9233          0       1000        344         10      10000       1000
           28958276 building        9229          0       1000        344         10      10000       1000
           29898988 gpu-preem      11456          0        980        301        176      10000          0
           30151398 building       10733          0        832        344         31      10000       1000
           31002153 gpu            10870          0        507        301         62      10000          0
           31184153 gypsum-rt      10611          0        424        174         13      10000          0
           31259028 gpu            10799          0        405        301         94      10000          0
           31259100 gpu            10767          0        405        301         62      10000          0
           32556933 cpu            10284          0         13        213         59      10000          0
    ```
  
  - Looking at the output of **`sshare`** for my PI group we see a FairShare score of 0.425616, due perhaps to fairly heavy usage of the cluster recently.  According to [this site](https://docs.crc.ku.edu/how-to/fairshare-priority/) a FairShare score less than 0.5 indicates our group has over-utilized our share:
    
    ```
    $ sshare|head;sshare|grep candela
    Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare 
    -------------------- ---------- ---------- ----------- ----------- ------------- ---------- 
    root                                          0.000000  5509757909      1.000000            
     pi_aakhavanmasoumi+                     1    0.001330           0      0.000000            
     pi_aakulkarni_umas+                     1    0.001330         786      0.000000            
     pi_abangert_mtholy+                     1    0.001330       29573      0.000005            
     pi_abecker_uri_edu                      1    0.001330          46      0.000000            
     pi_abiti_adili_uml+                     1    0.001330           0      0.000000            
     pi_abuttenschoe_um+                     1    0.001330           0      0.000000            
     pi_acheung_umass_e+                     1    0.001330       91209      0.000017            
     pi_candela_umass_e+                     1    0.001330     4729271      0.000858            
      pi_candela_umass_+ candela_u+          1    0.500000     3037218      0.642217   0.425616 
    ```
    
    Presumably we will need to wait for our FairShare score to recover back to 0.5 or above before our jobs will run with much priority. (In this particular example the two `gpu` jobs were asking for one of the better GPUs, a V100, and the three `cpu` jobs were asking for a lot of cores (64, 128, and 256) -- these factors may also have been holding up the jobs and it's not clear how if at all they are visible in the `sprio` and `sshare` output.)
  
  - These jobs ran after sitting in queue for a couple of days (perhaps when there was a break in usage by higher-priority users), after which my usage was up and my FairShare score was even lower:
    
    ```
    $ sshare | head -n 2; sshare | grep candela
    Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare 
    -------------------- ---------- ---------- ----------- ----------- ------------- ---------- 
     pi_candela_umass_e+                     1    0.001328    12676309      0.002223            
      pi_candela_umass_+ candela_u+          1    0.500000    10984521      0.866539   0.269433 
    ```

### Using MPI on Unity (without Apptainer)<a id="unity-mpi"></a>

#### Ways of running Python MPI programs on Unity<a id="ways-mpi-unity"></a>

Using Python MPI program requires (a) a working MPI installation - as in [Part 1](#on-linux-pc) we only consider [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/) here, and (b) Python MPI functions - also as in Part 1 we only consider [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/) here.  Even with these restrictions there are various choices involving modules and Conda environments that seem to work on Unity (and others that don't).  In this section we use the basic [MPI test programs described above](#mpi-testprogs): **`mpi_hw.py`** to check that there is a functional MPI + `mpi4py` setup, and **`osu_bw.py`** to measure the speed of communication between MPI ranks.

- There are **two different ways of running MPI** on Unity, both of which seem to work.
  
  - OpenMPI can be **installed in the environment** used, in which case it is not necessary to load an OpenMPI module when the program is run.  This is the case with the Conda environments **`m4p`** and **`dem21`** for non-containerized jobs, and **`ompi`** for containerized jobs (all defined below).
  
  - Conversely, as explained on [this conda-forge page](https://conda-forge.org/docs/user/tipsandtricks/#using-external-message-passing-interface-mpi-libraries), a "dummy" version of OpenMPI can be installed in the Conda environment, which **links to the OpenMPI installed outside the environment**, which must be made available by loading an OpenMPI module.  This is the case with the Conda environments **`m4pe`** and **`dem21e`** for non-containerized jobs -- when running containers, no Conda environment is needed when an OpenMPI module is loaded.
  
  From tests done 4/25 it appears that both methods work equally well on Unity, with no appreciable speed difference.  These tests were done with a moderately large MPI job (about one hour of run time on 128 cores) and for both non-containerized and containerized jobs.  The only difference noticed  was when the second method was used (OpenMPI module loaded), `mpirun --display bindings...` was unable to report which cores tasks were bound to.

- First we see which OpenMPI modules are available on Unity (as of 3/25). In the examples shown below when an OpenMPI module is used the latest version without CUDA `openmpi/5.0.3` is selected.
  
  ```
  $ module av openmpi
  
  ---------------------- /modules/modulefiles/spack/latest/linux-ubuntu24.04-x86_64/Core ------------------
     openmpi/4.1.6-cuda12.6    openmpi/4.1.6    openmpi/5.0.3-cuda12.6    openmpi/5.0.3
  ```

- Next we create two Conda environments **`m4p`** and **`m4pe`** with OpenMPI, `mpi4py`, Python, and other packages likely to be needed by programs running in this environment (here we show `numpy`, `scipy`, and `matplotlib`).  Things to note:
  
  - It seems necessary to load a Unity OpenMPI module before creating these environments, as shown here, for everything shown later to work, whether or not the OpenMPI module will be loaded when the environments are used.
  
  - We use `conda-forge`, which seems to have relatively up-to-date packages, and for **`m4pe`** (but not **`m4p`**) we follow the [conda-forge prescription](https://conda-forge.org/docs/user/tipsandtricks/#using-external-message-passing-interface-mpi-libraries) to specify that an external MPI implementation will be used. 
  
  ```
  $ unity-compute
  (wait for the compute-node shell to come up)
  $ module load conda/latest
  $ module load openmpi/5.0.3
  
  # Create environment m4p and check that OpenMPI is available in it.
  $ conda create -n m4p -c conda-forge openmpi=5 mpi4py python=3.12
  $ conda activate m4p
  (m4p)$ conda install numpy scipy matplotlib
  (m4p)$ mpirun --version
  mpirun (Open MPI) 5.0.7
  (m4p)$ ompi_info | head
                   Package: Open MPI conda@f424a898794e Distribution
                  Open MPI: 5.0.7
    Open MPI repo revision: v5.0.7
     Open MPI release date: Feb 14, 2025
                   MPI API: 3.1.0
              Ident string: 5.0.7
                    Prefix: /work/candela_umass_edu/.conda/envs/m4p
   Configured architecture: x86_64-conda-linux-gnu
             Configured by: conda
             Configured on: Mon Feb 17 07:57:35 UTC 2025
  (m4p)$ conda deactivate
  
  # Create environment m4pe and check that OpenMPI is available in it.
  $ conda create -n m4pe -c conda-forge "openmpi=5.0.3=external_*" mpi4py python=3.12
  $ conda activate m4pe
  (m4pe)$ conda install numpy scipy matplotlib
  (m4pe)$ mpirun --version
  mpirun (Open MPI) 5.0.3
  (m4pe)$ ompi_info | head
                   Package: Open MPI package-maintainer@gpu005 Distribution
                  Open MPI: 5.0.3
    Open MPI repo revision: v5.0.3
     Open MPI release date: Apr 08, 2024
                   MPI API: 3.1.0
              Ident string: 5.0.3
                    Prefix: /modules/spack/packages/linux-ubuntu24.04-x86_64/gcc-13.2.0/openmpi-5.0.3-bj572zbkduba5ueea4uwnhhgbi422h55
   Configured architecture: x86_64-pc-linux-gnu
             Configured by: package-maintainer
             Configured on: Sat Aug 24 21:50:27 UTC 2024
  ```

- Next we run **`mpi_hw.py`** interactively to check that MPI is functional.  Here and below we supply **`--display bindings`** to `mpirun` which should show which core(s) on which node each MPI rank is bound to, before running the supplied code (`python` here).
  
  ```
  # Get an interactive compute node with four cores.
  $ salloc -n 4 
  (wait for the compute-node shell to come up)
  $ cd python-scripts; ls    # cd to directory with mpi_hw.py
  mpi_hw.py  ...
  python-scripts$ module load conda/latest
  python-scripts$ conda activate m4p
  (m4p)..python-scripts$ mpirun --display bindings python mpi_hw.py
  [gypsum-gpu001:4010527] Rank 0 bound to package[0][core:4]
  [gypsum-gpu001:4010527] Rank 1 bound to package[0][core:5]
  [gypsum-gpu001:4010527] Rank 2 bound to package[1][core:8]
  [gypsum-gpu019:3278150] Rank 3 bound to package[0][core:3]
  Hello world from rank 2 of 4 on gypsum-gpu001 running Open MPI v5.0.7
  Hello world from rank 1 of 4 on gypsum-gpu001 running Open MPI v5.0.7
  Hello world from rank 0 of 4 on gypsum-gpu001 running Open MPI v5.0.7
  Hello world from rank 3 of 4 on gypsum-gpu019 running Open MPI v5.0.7
  
  # To run in environmenet m4pe, need to load OpenMPI module.
  (m4p)..python-scripts$ conda deactivate
  ..python-scripts$ conda activate m4pe
  (m4pe)..python-scripts$ mpirun --display bindings python mpi_hw.py
  bash: mpirun: command not found
  (m4pe)..python-scripts$ module load openmpi/5.0.3
  (m4pe)..python-scripts$ mpirun --display bindings python mpi_hw.py
  [gypsum-gpu001:4014610] Rank 0 is not bound (or bound to all available processors)
  [gypsum-gpu001:4014610] Rank 1 is not bound (or bound to all available processors)
  [gypsum-gpu001:4014610] Rank 2 is not bound (or bound to all available processors)
  [gypsum-gpu019:3281256] Rank 3 is not bound (or bound to all available processors)
  Hello world from rank 1 of 4 on gypsum-gpu001 running Open MPI v5.0.3
  Hello world from rank 2 of 4 on gypsum-gpu001 running Open MPI v5.0.3
  Hello world from rank 0 of 4 on gypsum-gpu001 running Open MPI v5.0.3
  Hello world from rank 3 of 4 on gypsum-gpu019 running Open MPI v5.0.3
  ```
  
  We see:
  
  - We can run `mpirun` in environment `m4p` without first loading an OpenMPI module, but not in environment `m4pe` which requires the OpenMPI module to be loaded.
  
  - `--display bindings` shows both node and core for each rank when an OpenMPI module is *not* loaded (top example above), but only shows node and `Rank n is not bound...` when an OpenMPI module *is* loaded (bottom example above).  Although not shown above, this holds true even if environment `m4p` is used.  Even when `Rank n is not bound...` is displayed, the program still functions -- so this may be a problem of communicating information rather than an actual problem.

- Finally we run **`osu_bw.py`** to check inter-rank communication speed.
  
  - We get an interactive node with four cores (`-n 4`) on two nodes (`-N 2`) so intra- and inter-node communication could be compared, see below. For other purposes it may not be necessary to specify the number of nodes (`-N` can be omitted) or it may be wished to force all the ranks to be on one node (`-N 1`).  We also specify **`-C ib` to constrain the job to nodes with InfiniBand connectivity**; if this was not done the reported communication speeds were sometimes about 100 times slower.
    
    To start with we activate `m4p` and do not load the OpenMPI module, then use `mpirun -n 2...` to run `osu_bw.py` on two ranks -- it is necessary to supply `-n 2` to `mpirun` here as otherwise it would default to the four ranks allocated by the `salloc` command, and `osu_bw.py` insists on exactly two ranks.
    
    ```
    $ salloc -n 4 -N 2 -C ib
    (wait for the compute-node shell to come up)
    $ cd python-scripts; ls    # cd to directory with osu_bw.py
    osu_bw.py  ...
    python-scripts$ module load conda/latest
    python-scripts$ conda activate m4p
    (m4p)..python-scripts$ mpirun -n 2 --display bindings python osu_bw.py
    [cpu045:1260236] Rank 1 bound to package[1][core:47]
    [cpu045:1260236] Rank 0 bound to package[1][core:42]
    2
    2
    # MPI Bandwidth Test
    # Size [B]    Bandwidth [MB/s]
             1                1.95
             2                3.81
             4                7.51
             8               15.48
            16               31.47
            32               61.05
            64              123.70
           128              220.22
           256              439.24
           512              815.56
         1,024            1,498.98
         2,048            2,401.25
         4,096            3,096.10
         8,192            4,489.08
        16,384            6,877.37
        32,768            8,597.59
        65,536           14,706.91
       131,072           17,796.39
       262,144           12,374.38
       524,288            7,485.89
     1,048,576            8,596.79
     2,097,152            8,909.45
     4,194,304            9,040.91
     8,388,608            9,107.42
    16,777,216            6,882.39
    ```
    
    The maximum speed seen here, 18 GB/s, varied by about +/-2 GB/s on repeated runnings.
  
  - The example above used OpenMPI's [default binding strategy](https://docs.open-mpi.org/en/v5.0.x/launching-apps/scheduling.html) "by slot", which placed both MPI ranks on the same node. To measure the communication speed between nodes, we can supply `--map-by node` to `mpirun`:
    
    ```
    (m4p)..python-scripts$ mpirun -n 2 --map-by node --display bindings python osu_bw.py
    [cpu045:1260382] Rank 0 bound to package[1][core:42]
    [cpu047:2425563] Rank 1 bound to package[1][core:53]
    2
    # MPI Bandwidth Test
    # Size [B]    Bandwidth [MB/s]
    2
             1                1.15
             2                2.31
             4                4.58
             8                8.94
            16               18.27
            32               36.85
            64               73.54
           128              145.92
           256              280.03
           512              549.85
         1,024            1,058.59
         2,048            2,222.78
         4,096            4,392.00
         8,192            6,205.18
        16,384            9,066.75
        32,768           10,469.50
        65,536           11,322.56
       131,072           11,803.02
       262,144           12,076.67
       524,288           12,209.42
     1,048,576           12,275.90
     2,097,152           12,205.76
     4,194,304           12,288.37
     8,388,608           12,310.17
    16,777,216           12,282.66
    ```
    
    Now the maximum speed is 12 GB/s, i.e.  35% slower than the speed between ranks on the same node - but for some reason the largest message sizes are actually faster between nodes than between ranks on the same node.
    
    The OpenMPI version of `mpirun` has [many other options](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html).  Note that in other MPI packages such as MPICH  `mpirun` has different, incompatible options.
  
  - The speed tests were repeated using the environment **`m4pe`** (external MPI installation) and gave apparently identical speed results, both between ranks on the same node and on different nodes.  As noted above this required loading the OpenMPI module and then `--display bindings` no longer showed the core numbers used.

#### Using `sbatch` to run a simple MPI job<a id="sbatch-mpi"></a>

Make an sbatch script **`osu_bw.sh`** with the following contents, and put it in a directory `try_mpi` along with the test program `osu_bw.py`:

```
#!/bin/bash
# osu_bw.sh 4/3/25 D.C.
# sbatch script to run osu_bw.py, which times the speed of MPI messaging
# between two MPI ranks.
#SBATCH -n 2                         # run 2 MPI ranks
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate m4p                   # environment with OpenMPI, mpi4py, NumPy and SciPy
mpirun --display bindings python osu_bw.py
```

  We go to a directory `try-mpi` into which we have copied both `osu_bw.sh` and `osu_bw.py` and run the script (as usual when running sbatch scripts, this can be done from a login shell and there is no need to set a Conda environment as the script does this):

```
$ cd try-mpi; ls
osu_bw.sh  osu_bw.py  ...
try_mpi$ sbatch osu_bw.sh
Submitted batch job 31253216
(wait until 'squeue --me' shows that job has completed)
try-mpi$ cat slurm-31253216.out
nodelist=cpu046
Loading conda
[cpu046:2330110] Rank 0 bound to package[0][core:14]
[cpu046:2330110] Rank 1 bound to package[1][core:47]
2
2
# MPI Bandwidth Test
# Size [B]    Bandwidth [MB/s]
         1                0.93
         2                3.11
         4                6.24
         8               12.56
        16               25.16
        32               47.17
        64              101.24
       128              183.91
       256              365.55
       512              709.00
     1,024            1,232.34
     2,048            1,688.76
     4,096            2,238.17
     8,192            4,456.95
    16,384            6,828.24
    32,768            8,058.85
    65,536           14,153.93
   131,072           15,690.97
   262,144           11,528.39
   524,288            8,487.05
 1,048,576            8,582.14
 2,097,152            8,361.06
 4,194,304            8,990.32
 8,388,608            8,792.12
16,777,216            7,536.62
```

#### Enabling NumPy multithreading in MPI batch jobs<a id="sbatch-multithread"></a>

**Note:** As explained at the very end of this section, **this has not been fully worked out** and more work would be needed to get this working reliably.

For background see [Parallel execution on multiple cores](#multiple-cores) and [Hyperthreading and NumPy multithreading with MPI](#multithread-mpi) above.  Make an sbatch script **`threadcount_mpi.sh`**  with the following contents and put it in the directory `try-mpi` along with `threadcount_mpi.py`, which is an MPI-parallel version of `threadcount.py`  [discussed above](#multiple-cores). 

```
#!/bin/bash
# threadcount_mpi.sh 4/4/25 D.C.
# sbatch script to run threadcount_mpi.py, which uses MPI to time matrix
# multiplications in parallel in several MPI tasks.
#SBATCH -n 4                         # run 4 MPI ranks
#SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate m4p                   # environment with OpenMPI, mpi4py, NumPy and SciPy
mpirun --display bindings python threadcount_mpi.py
```

Run this sbatch script, and examine its output file and efficiency as reported by `seff`. With the `sbatch` defaults, NumPy only uses one thread (one core) in each MPI rank:

```
try-mpi$ sbatch threadcount_mpi.sh
Submitted batch job 29270594
(wait until 'squeue --me' shows that job has completed)
try-mpi$ cat slurm-29270594.out
nodelist=cpu[046-048],uri-cpu006
Loading conda
[cpu046:3233583] Rank 0 bound to package[1][core:44]
[cpu048:812940] Rank 2 bound to package[0][core:12]
[uri-cpu006:1412965] Rank 3 bound to package[0][core:5]
[cpu047:1048628] Rank 1 bound to package[0][core:31]
This is rank 0 of 4 on cpu046 running Open MPI v5.0.7
This is rank 3 of 4 on uri-cpu006 running Open MPI v5.0.7
This is rank 1 of 4 on cpu047 running Open MPI v5.0.7
This is rank 2 of 4 on cpu048 running Open MPI v5.0.7
(rank 0) Making 10,000 x 10,000 random matrices...
(rank 3) Making 10,000 x 10,000 random matrices...
(rank 1) Making 10,000 x 10,000 random matrices...
(rank 2) Making 10,000 x 10,000 random matrices...
(rank 0) ...took 3.043e+00s, average threads = 1.000
(rank 0) Multiplying matrices 3 times...
(rank 1) ...took 3.078e+00s, average threads = 1.000
(rank 1) Multiplying matrices 3 times...
(rank 3) ...took 3.284e+00s, average threads = 1.000
(rank 3) Multiplying matrices 3 times...
(rank 2) ...took 3.292e+00s, average threads = 1.000
(rank 2) Multiplying matrices 3 times...
(rank 3) ...took 2.574e+01s per trial, average threads = 1.000
(rank 2) ...took 4.091e+01s per trial, average threads = 1.000
(rank 0) ...took 4.332e+01s per trial, average threads = 1.000
(rank 1) ...took 4.397e+01s per trial, average threads = 1.000
try-mpi$ seff 29270594
Job ID: 29270594
Cluster: unity
User/Group: candela_umass_edu/candela_umass_edu
State: COMPLETED (exit code 0)
Nodes: 4
Cores per node: 1
CPU Utilized: 00:08:01
CPU Efficiency: 85.28% of 00:09:24 core-walltime
Job Wall-clock time: 00:02:21
Memory Utilized: 7.31 GB (estimated maximum)
Memory Efficiency: 22.84% of 32.00 GB (8.00 GB/core)
```

Here is a modified sbatch script **`threadcount_mpi2.sh`** that will enable NumPy to use two threads (two cores) in each MPI rank.  The modifications are (a) setting `#SBATCH -c 2` to allocate two cores per MPI rank, and (b) supplying the option `--cpus-per-rank 2` to `mpirun`.  Experimentally both of these modifications are needed to allow NumPy to multithread; conversely setting `OMP_NUM_THREADS` as  [discussed above](#multiple-cores) for a non-MPI setting seems to have no effect here.

```
#!/bin/bash
# threadcount_mpi2.sh 4/4/25 D.C.
# sbatch script to run threadcount_mpi.py, which uses MPI to time matrix
# multiplications in parallel in several MPI tasks.
# threadcount_mpi2.sh has been modified from threadcount_mpi.sh so NumPy can
# use 2 threads.
#SBATCH -n 4                         # run 4 MPI ranks
#SBATCH -c 2                         # give each rank two cores
#SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate m4p                   # environment with OpenMPI, mpi4py, NumPy and SciPy
mpirun --display bindings --cpus-per-rank 2 python threadcount_mpi.py
```

Run the modified sbatch script, and examine its output file and efficiency:

```
try-mpi$ sbatch threadcount_mpi2.sh
Submitted batch job 29273079
(wait until 'squeue --me' shows that job has completed)
/try-mpi$ cat slurm-29273079.out
nodelist=cpu[045-046],uri-cpu[009,049]
Loading conda
[cpu045:1329365] Rank 0 bound to package[0][core:15-16]
[cpu046:3239344] Rank 1 bound to package[1][core:40-41]
[uri-cpu049:3879069] Rank 3 bound to package[0][core:11-12]
[uri-cpu009:313846] Rank 2 bound to package[0][core:3,7]
This is rank 0 of 4 on cpu045 running Open MPI v5.0.7
This is rank 1 of 4 on cpu046 running Open MPI v5.0.7
This is rank 2 of 4 on uri-cpu009 running Open MPI v5.0.7
This is rank 3 of 4 on uri-cpu049 running Open MPI v5.0.7
(rank 1) Making 10,000 x 10,000 random matrices...
(rank 0) Making 10,000 x 10,000 random matrices...
(rank 2) Making 10,000 x 10,000 random matrices...
(rank 3) Making 10,000 x 10,000 random matrices...
(rank 0) ...took 2.847e+00s, average threads = 1.000
(rank 0) Multiplying matrices 3 times...
(rank 1) ...took 3.030e+00s, average threads = 1.000
(rank 1) Multiplying matrices 3 times...
(rank 3) ...took 3.227e+00s, average threads = 1.000
(rank 3) Multiplying matrices 3 times...
(rank 2) ...took 3.364e+00s, average threads = 1.000
(rank 2) Multiplying matrices 3 times...
(rank 3) ...took 1.293e+01s per trial, average threads = 1.999
(rank 2) ...took 1.298e+01s per trial, average threads = 1.999
(rank 0) ...took 2.095e+01s per trial, average threads = 1.999
(rank 1) ...took 2.166e+01s per trial, average threads = 2.000
try-mpi$ seff 29273079
Job ID: 29273079
Cluster: unity
User/Group: candela_umass_edu/candela_umass_edu
State: COMPLETED (exit code 0)
Nodes: 4
Cores per node: 2
CPU Utilized: 00:07:10
CPU Efficiency: 72.64% of 00:09:52 core-walltime
Job Wall-clock time: 00:01:14
Memory Utilized: 7.41 GB (estimated maximum)
Memory Efficiency: 11.57% of 64.00 GB (8.00 GB/core)
```

Notes:

- The modified job used twice as many cores (8 rather than 4), and it ran about twice as fast because the time-consuming part of `threadcount_mpi.py` (a call to `np.matmul` to do the matrix multiplications) was able to use two threads rather than one.

- Only some NumPy functions can use multithreading to run faster -- for example the creation of the matrices using `rng.normal` only used one thread even though two cores were available.

- The [OpenMPI docs for `mpirun`](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html) say `--cpus-per-rank` is deprecated in favor of something like `--map-by <obj>:PE=n` but I haven't tried to figure this out.

- If the settings for `#SBATCH -c ..` and `mpirun --cpus-per-rank ..` do not agree, strange things happen that I do not understand.

- **This job did not work reliably and probably requires additional `#SBATCH` settings.**  For example, sometimes one MPI rank took much longer to run, or used less threads than the other ranks.  It may be necessary to constrain the job to certain nodes, but this has not been tried.

#### Using `sbatch` to run `boxpct.py + dem21` with MPI<a id="sbatch-dem21"></a>

See [More elaborate MPI programs...](#mpi-dem21) above for the corresponding steps on a PC.

- As in that section, the **`dem21`** package (not public) is cloned into a directory `try-dem21` on Unity. Also the **`msigs`** package (also not public), which is used by `mx2.py` to carry out larger simulations, has been copied into the same directory:
  
  ```
  try-dem21$ git clone git@github.com:doncandela/dem21.git
  try-dem21$ ls
  dem  msigs ...
  ```

- As in [Ways of running Python MPI programs on Unity](#ways-mpi-unity) above a Conda environment **`dem21`** is defined on Unity, similar to the **`m4p`**  environment defined in that section but now including additional packages required by `dem21` and with the `dem21` and `msigs` packages installed.  Also an environment **`dem21e`** is created similar to **`mype`** which includes a dummy version of OpenMPI that links to the system OpenMPI:
  
  ```
  try-dem21$ unity-compute             # get an interactive shell on a compute node
  (wait for the compute-node shell to come up)
  try-dem21$ module load conda/latest
  try-dem21$ module load openmpi/5.0.3
  
  # Make environment dem21:
  try-dem21$ conda create -n dem21 -c conda-forge openmpi=5 mpi4py python=3.12
  try-dem21$ conda activate dem21 
  (dem21)..try-dem21$ conda install numpy scipy matplotlib dill numba pyaml
  (dem21)..try-dem21$ conda install -c conda-forge quaternion
  (dem21)..try-dem21$ cd dem21
  (dem21)..try-dem21/dem21$ pip install -e .
  (dem21)..try-dem21/dem21$ cd ../msigs
  (dem21)..try-dem21/msigs$ pip install -e .
  
  # Make environment dem21e:
  (dem21)..try-dem21/conda deactivate
  try-dem21$ conda create -n dem21e -c conda-forge "openmpi=5.0.3=external_*" mpi4py python=3.12
  try-dem21$ conda activate dem21e 
  (dem21e)..try-dem21$ conda install numpy scipy matplotlib dill numba pyaml
  (dem21e)..try-dem21$ conda install -c conda-forge quaternion
  (dem21e)..try-dem21$ cd dem21
  (dem21e)..try-dem21/dem21$ pip install -e .
  (dem21e)..try-dem21/dem21$ cd ../msigs
  (dem21e)..try-dem21/msigs$ pip install -e .
  ```

- At this point we can log out and back into Unity, and there will be no need to activate the environment `dem21` interactively as it will be activated by sbatch scripts as needed.

We go back to `try-dem21` and copy the test program `boxpct.py` and its configuration file `box.yaml` there from the `tests` subdirectory of the cloned repo:

```
try-dem21$ ls dem21/tests/box
box.yaml  boxmod.yaml  boxpct.py  boxpct.sh  heap3.yaml  output  plots
try-dem21$ cp dem21/tests/box/boxpct.py .
tri-dem21$ cp dem21/tests/box/box.yaml .
```

- An sbatch script **`boxpct.sh`** is created in the directory `try-dem21` with these contents:
  
  ```
  #!/bin/bash
  # boxpct.sh 4/4/25 D.C.
  # sbatch script to run boxpct.py, using the dem21 package in MPI-parallel mode.
  #SBATCH -n 4                         # run 4 MPI ranks
  #SBATCH -N 1                         # all ranks on one node
  #SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
  #SBATCH -p cpu                       # submit to partition cpu
  #SBATCH -C ib                        # require inifiniband connectivity
  echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
  module purge                         # unload all modules
  module load conda/latest             # need this to use conda commands
  conda activate dem21                 # environment with OpenMPI, dem21, and dependencies
  export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
  mpirun --display bindings python boxpct.py
  ```

- Now we can use this script to run `boxpct.py` in MPI-parallel mode:
  
  ```
  try-dem21$ sbatch boxpct.sh
  Submitted batch job 31294505
  (wait until 'squeue --me' shows that job has completed)
  try-dem21$ cat slurm-31294505.out
  nodelist=uri-cpu007
  Loading conda
  [uri-cpu007:472048] Rank 0 bound to package[0][core:0]
  [uri-cpu007:472048] Rank 1 bound to package[0][core:1]
  [uri-cpu007:472048] Rank 2 bound to package[1][core:32]
  [uri-cpu007:472048] Rank 3 bound to package[1][core:33]
  - Started MPI on master + 3 worker ranks.
  THIS IS: boxpct.py 12/3/22 D.C., using dem21 version: v1.2 2/11/25
  Parallel processing: MPI, GHOST_ARRAY=True
  - Read 1 config(s) from /work/pi_.../try-dem21/box.yaml
  
  SIM 1/1:
  Using inelastic 'silicone' grainlets with en=0.7 and R=0.500mm
                     ...
  ```
  
  As of 4/25, under some circumstances, each rank emitted an warning message like...
  
  ```
  Warning: program compiled against libxml 213 using older 209
  ```
  
  ...before running successfully.  This was seen when `-c conda-forge` was supplied to the `conda install numpy...` command used in creating the Conda environment `dem21`, unlike what is shown above. However, I don't know if this is really the reason for such warnings or if, for example, it depends on the library versions installed on the Unity node allocated to the job.

- **Larger `dem21` sims using `mx2.py`.**  To try out a more time-consuming simulation using an sbatch job on Unity, we follow the steps shown for a PC in [A more intensive run with `mx2.py`](#mx2py) above. 
  
  - In the directory `cc-expts-unity` containing the other needed files (not explained here) we make an sbatch script **`mx2-unity.sh`** as follows.  Since this file will be modified slightly to test various ways of running on Unity, this script [includes an `scontrol` command](#keep-sbatch) to print the sbatch script used in the `slurm-<jobid>.out` file: 
    
    ```
    #!/bin/bash
    # cc-expts-unity/mx2-unity.sh 4/12/25 D.C.
    # sbatch script to run granular-memory simulation program mx2.py non-containerized on
    # the Unity cluster, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
    # Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
    # Reads default config file mx2.yaml in grandparent directory modified by
    # mx2mod.yaml in current directory.
    #SBATCH -n 15                        # run on 128 cores
    #SBATCH -N 1                         # use 1 node
    # #SBATCH --exclusive                  # don't share nodes with other jobs
    #SBATCH --mem=100G                   # allocate 100GB memory per node
    # #SBATCH --mem=0                      # allocate all available memory on nodes used
    #SBATCH -t 10:00:00                  # time limit 10 hrs (default is 1 hr)
    #SBATCH -p cpu                       # submit to partition cpu 
    # #SBATCH -p cpu,cpu-preempt           # submit to partition cpu or cpu-preempt (<2 hrs)
    # #SBATCH --nodelist=cpu[022-029],cpu[049-068]  # restrict to 128-core nodes 
    #SBATCH -C ib                        # require inifiniband connectivity
    scontrol write batch_script $SLURM_JOB_ID -;echo # print this batch script to output
    echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used 
    module purge                         # unload all modules
    module load conda/latest             # need this to use conda commands
    conda activate dem21                 # environment with OpenMPI, then don't need OpenMPI module
    # module load openmpi/5.0.3            # load OpenMPI module, use with environment dem21e
    # conda activate dem21e                # environment with external OpenMPI, use with OpenMPI module
    export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
    mpirun --display bindings python ../../mx2.py mx2mod  # run displaying bindings
    # mpirun python ../../mx2.py mx2mod    # run without displaying bindings
    ```
  
  - All of the MPI trials show in this document used `#SBATCH -C ib` to request nodes with **InfiniBand networking**.  But for **multi-node MPI jobs** (typically jobs on more than 64 cores) probably should have used `#SBATCH -C "ib&mpi"` to additionally ensure **consistent CPU type across nodes**.  I haven't tried this yet.
  
  - We submit the job from this directory  (`mx2.py` has been written to be run from the directory where its output should go -- this may not be true for other programs):
    
    ```
    (start the job)
    $ cd ..cc-expts-unity; ls
    bw6-sigs.yaml  bw6.svg  mx2mod.yaml  mx2-unity.sh signals.sh
    ..cc-expts-unity$ sbatch mx2-unity.sh
    Submitted batch job 31446485
    (use 'squeue --me' to see when job starts, time running, if done)
    ..cc-expts-unity$ cat slurm-31446485.out  # can do while running to see output so far
    (for long jobs no need to stay logged in to Unity while they run)
    
    (see when job is done, check efficiency)
    $ sacct -b        # show brief summary of recent jobs
    $ seff 31446485   # check cores, memory, time used by completed job
    Job ID: 31446485
    Cluster: unity
    User/Group: candela_umass_edu/candela_umass_edu
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 15
    CPU Utilized: 3-10:37:41
    CPU Efficiency: 99.95% of 3-10:40:00 core-walltime
    Job Wall-clock time: 05:30:40
    Memory Utilized: 3.34 GB
    Memory Efficiency: 3.34% of 100.00 GB
    
    (look at job output)
    cd ..cc-expts-unity        # cd to where job was run from...
    ..cc-expts-unity$ cat slurm-31446485.out
    (batch script is printed here)
    nodelist=cpu054
    Loading conda
    [cpu054:261073] Rank 0 bound to package[0][core:0]
    [cpu054:261073] Rank 1 bound to package[0][core:1]
    [cpu054:261073] Rank 2 bound to package[0][core:6]
    [cpu054:261073] Rank 3 bound to package[0][core:7]
    [cpu054:261073] Rank 4 bound to package[0][core:8]
    [cpu054:261073] Rank 5 bound to package[0][core:9]
    [cpu054:261073] Rank 6 bound to package[0][core:10]
    [cpu054:261073] Rank 7 bound to package[0][core:11]
    [cpu054:261073] Rank 8 bound to package[1][core:73]
    [cpu054:261073] Rank 9 bound to package[1][core:75]
    [cpu054:261073] Rank 10 bound to package[1][core:76]
    [cpu054:261073] Rank 11 bound to package[1][core:77]
    [cpu054:261073] Rank 12 bound to package[1][core:78]
    [cpu054:261073] Rank 13 bound to package[1][core:79]
    [cpu054:261073] Rank 14 bound to package[1][core:80]
    - Started MPI on master + 14 worker ranks.
    
    This is: mx2.py 7/29/24 D.C.
    Using dem21 version: v1.2 2/11/25
    Imput signals made by: memsigs.py 8/23/24 D.C.
    Parallel processing mode: MPI, GHOST_ARRAY=True
    - Read 1 config(s) from /work/pi_../cc-expts-unity/../../mx2.yaml << mx2mod.yaml
    - Read 1 config(s) from bw6-sigs.yaml
    
    **** SIM 1/1   forces n='hertz' t='coulh' mu=1.0 signal binwords-43 (sig 0 in 'bw6')
    ```
  
  Some stats from running in various ways. Here external MPI = no means the environment **`dem21`** was activated and the module `openmpi/5.0.3` was not loaded. Conversely external MPI = yes means the environment **`dem21e`** was activated and the module `openmpi/5.0.3` was loaded:
  
  | system                 | candela-21        | Unity             | Unity             | Unity            | Unity                   | Unity                                     | Unity            | Unity     |
  | ---------------------- | ----------------- | ----------------- | ----------------- | ---------------- | ----------------------- | ----------------------------------------- | ---------------- | --------- |
  | external MPI?          | -                 | no                | no                | no               | no                      | no                                        | no               | yes       |
  | cores (`-n`)           | 15                | 15                | 64                | 128              | 128                     | 256                                       | 256              | 256       |
  | max boxes/crate        | 16                | 16                | 4                 | 2                | 2                       | 1                                         | 1                | 1         |
  | req. nodes (`-N`)      | -                 | 1                 | 1                 | 1                | no `-N`                 | no `-N`                                   | 2                | no `-N`   |
  | `--exclusive` ?        | -                 | no                | yes               | yes              | yes                     | yes                                       | yes              | yes       |
  | nodes used, cores/node |                   | 1                 | 1, 64             | 1, 128           | 2, 64                   | 4, 64                                     | 2, 128           | NOT TRIED |
  | which nodes used       | -                 | cpu 054           | umd-cscdr 045     | cpu 061          | umd-cscdr-cpu [045-046] | umd-cscdr-cpu [022-023, 025], uri-cpu 050 | cpu [061, 064]   |           |
  | inter-rank comm time   |                   | 3.3%              | 9.8%              | 18.8%            | 19.7%                   | 35.0%                                     | 35.1%            |           |
  | memory used            | 2.1 GB            | 3.3 GB            | 10.8 GB           | 26.5 GB          | 11.1 GB                 | 34.0 GB                                   | 28.0 GB          |           |
  | sim wall time          | 266 min = 4.43 hr | 331 min = 5.51 hr | 119 min = 1.99 hr | 64 min = 1.07 hr | 72 min = 1.19 hr        | 50 min = 0.83 hr                          | 47 min = 0.78 hr |           |
  | time/(step-grain)      | 3.46e-6 s         | 4.30e-6 s         | 1.56e-6 s         | 0.83e-6 s        | 0.93e-6 s               | 0.65e-6 s                                 | 0.61e-6          |           |
  | speed/candela-21       | 1.00              | 0.80              | 2.2               | 4.2              | 3.7                     | 5.3                                       | 5.7              |           |
  
  Notes:
  
  - The speed scales rather less than linearly with the number of cores (less than [strong scaling](#strong-weak-scaling)). It appears useful for this particular situation to use up to 128 cores (which however is only about 5 times faster than using 15 cores, rather than the strong-scaling expectation of 8 times faster) -- but going beyond this to 256 cores did not help much, and queue times were longer for 256 cores.  Running on the same bank of nodes `umd-cscdr-cpu`, using 256 cores was only  1.4 times faster than using 128 cores rather than the strong-scaling speedup of 2.0.
  
  - Allowing the 128-core sims to run on 2 64-code nodes was only about 15% slower than running on 1 128-core node, and again the queue time was shorter for 64-code nodes.
  
  - In summary the best way to run here was to used `--exclusive` and `--mem=0` to get full use of nodes with all of their memory, but to not specify `-N`  (which typically resulting in 64-core nodes, but occasionally 128-core nodes) or to explicitly specify `-N` that gives 64-core nodes as Unity currently (4/25) has a lot of 64-core nodes.
  
  - Here are some typical `#SBATCH` settings that someone in my group used successfully for a script that submitted many jobs similar to the job described above:
    
    ```
    #SBATCH -q long                     # required for jobs running more than 2 days
    #SBATCH -N 1                        # run on a single node
    #SBATCH --nodelistcp=cpu[049-068]     # limit to nodes with 128 cores
    #SBATCH -n 109                      # allocated for 109 MPI ranks
    #SBATCH --mem=15000                 # allocate 15 GB memory (per node, one node here)
    #SBATCH -t 300:00:00                # time limit 300 hrs (max allowed is 14 days = 336 hrs)
    #SBATCH --constraint=ib
    ```
    
    The [docs for `sbatch`](https://slurm.schedmd.com/sbatch.html) seem to imply that *all* of the nodes listed in `--nodelist=..` will be allocated to the job, but experimentally only the number of nodes specified by `-N...` will be allocated.  With the `#SBATCH` settings shown above all MPI ranks were on a single node, which proved to be more efficient than some other ways of running.
    <a id="unity-dem21-bigger"></a>
  
  - Finally some runs were done on a simulation with ten times as many grains, as described for a PC in [An even bigger simulation](#even-bigger-sim) above. 
    
    | system                 | candela-21          | Unity             | Unity             | Unity             | Unity                                        |
    | ---------------------- | ------------------- | ----------------- | ----------------- | ----------------- | -------------------------------------------- |
    | external MPI?          | -                   | no                | no                | no                | yes                                          |
    | cores (`-n`)           | 16                  | 64                | 128               | 256               | 256                                          |
    | max boxes / crate      | 116                 | 28                | 14                | 7                 | 7                                            |
    | req. nodes (`-N`)      | -                   | no `-N`           | no `-N`           | no `-N`           | no `N`                                       |
    | `--exclusive` ?        | -                   | yes               | yes               | yes               | yes                                          |
    | nodes used, cores/node | -                   | 1, 64             | 2, 64             | 2, 128            | 4, 64                                        |
    | which nodes used       | -                   | umd-cscdr-cpu 041 | uri-cpu [011-012] | cpu [056, 065]    | umd-cscdr-cpu [039, 046], uri-cpu [007, 012] |
    | inter-rank comm time   | 13.6%               | 13.7%             | 12.0%             | 20.3%             | 24.0%                                        |
    | memory used            | 5.5 GB              | 17.3 GB           | 16.6 GB           | 37.4 GB           | 16.0 GB                                      |
    | sim wall time          | 2,777 min = 46.1 hr | 958 min = 16.0 hr | 460 min = 7.66 hr | 274 min = 4.56 hr | 284 min = 4.74 hr                            |
    | time / (step-grain)    | 3.67e-6 s           | 1.27e-6 s         | 0.61e-6 s         | 0.36e-6 s         | 0.38e-6 s                                    |
    | speed / candela-21     | 1.00                | 2.9               | 6.0               | 10.1              | 9.9                                          |
    
    Notes:
    
    - It worked well use `--exclusive` but omit `-N` specifying the number of nodes to be used -- Slurm typically allocated 64-core nodes, but sometimes allocated 128-core nodes.
    
    - Comparing the last two columns, the run using external MPI was 2% slower which is not a significant difference especially as different nodes (and a different numbers of nodes) were used.
    
    - Compare with the results for the 10-times smaller simulation tabulated above, the 128-core run took 6.4 times longer while the 256-core run took 5.5 times longer. These scalings with simulation size were even better than the corresponding reductions in boxes/crates (increases in code parallelism) of 14/2=7.0 and 7/1=7.0, suggesting that [weak scaling](#strong-weak-scaling) applies here.

### Using a GPU on Unity (without Apptainer)<a id="unity-gpu"></a>

#### Using PyTorch with a GPU on Unity<a id="pytorch-unity"></a>

This section shows how to set up and test an environment to run PyTorch on Unity, using the simple [PyTorch test code for a PC](#pytorch-cupy) shown above.  It seems that the correct way to use PyTorch on Unity is to create a Conda environment, then use pip to install PyTorch into that environment.  As on a PC, PyTorch can be run on Unity without a GPU but that is not shown here.

[Here](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) is a Princeton page showing how to set up and use PyTorch on their HPC clusters.  The Princeton page has information on several additional topics not covered here: Running batch PyTorch jobs, using multiple GPUs for big ML jobs, etc.  Before I found the needed info in the Unity docs I simply copied the Princeton instructions for creating a PyTorch environment, and this did work on Unity (although the installed PyTorch required GPUs with higher compute capabilities than some GPUs on Unity).

More recently (5/25) the Unity help folks pointed me to [their page on Conda](https://docs.unity.rc.umass.edu/documentation/software/conda/)  which shows commands for creating pre-set Conda environments on Unity including some for PyTorch:

```
# Get list of available pre-sets
$ unity-conda-list
No argument provided. Listing all available presets:
python3-10
python3-11
python3-8
python3-9
pytorch-arm
pytorch-latest
pytorch-power9
pytorch
tensorflow

# See what's in the pre-set pytorch-latest
$ unity-conda-list pytorch-latest
Reporting packages for the following conda environment(s):
     pytorch-latest
Environment Name: pytorch-latest
Conda Channels:
     pytorch
Packages to be installed: 
     python=3.9.*
     pip
     numpy=1.21.5
     pandas=1.3.5
     {'pip': ['torch', 'torchvision', 'torchaudio']}
```

The command `unity-conda-create` can be used to create an environment `pytorch-latest` in which PyTorch is installed (it may be possible to give the created environment a different name by supplying a second argument to `unity-conda-create`, haven't tried this):

```
userc@login4:~$ salloc -c 6 -G 1 -p gpu
(wait for compute-node shell to come up)
userc@gypsum-gpu140:~$ unity-conda-create -n pytorch-latest
```

Rather than using `unity-conda-create`, it works to directly create the desired environment using Conda and pip.  While it worked to specify the specific version numbers reported by the `unity-conda-list pytorch-latest` command as shown above, this resulted in incompatibility problems when another module (`pickle` or `dill`) was installed into the environment.  The steps shown here created an environment **`pyt`** for running PyTorch on Unity that did not have this compatibility problem:

```
userc@login4:~$ salloc -c 6 -G 1 -p gpu      # get shell on compute node with a GPU
(wait for compute-node shell to come up)
userc@gypsum-gpu073:~$ module load conda/latest
userc@gypsum-gpu073:~$ conda create -n pyt python=3
userc@gypsum-gpu073:~$ conda activate pyt                      
(pyt) userc@gypsum-gpu073:~$ conda install numpy matplotlib                      
(pyt) userc@gypsum-gpu073:~$ pip install torch torchvision torchaudio
```

In this environment `pyt` it is possible to import PyTorch, and PyTorch can use the GPU:

```
(pyt) userc@gypsum-gpu073:~$ python
>>> import torch
>>> torch.cuda.is_available()           # check that GPU is available
True
>>> torch.cuda.get_device_capability()  # find CUDA compute capability of GPU (here 5.2)
(5, 2)
>>> x = torch.rand(5, 3)
>>> print(x)
tensor([[0.7946, 0.7594, 0.5193],
        [0.4011, 0.7268, 0.3369],
        [0.0739, 0.0982, 0.7924],
        [0.4226, 0.0937, 0.2713],
        [0.8465, 0.3324, 0.1237]])
>>> xgpu = x.to('cuda')
>>> xgpu
tensor([[0.3522, 0.7894, 0.9021],
        [0.3567, 0.0228, 0.9312],
        [0.9708, 0.9888, 0.7526],
        [0.8859, 0.3973, 0.9353],
        [0.8659, 0.6805, 0.9278]], device='cuda:0')
```

The version of PyTorch installed as above seems capable of working with the least capable GPUs on Unity ([compute capability 5.2](#pick-gpu)) as shown above. Earlier, when I installed PyTorch as shown on a Princeton page, that version of PyTorch required a compute capability of 7.5 or above.  Unity jobs can be restricted to nodes with GPUs meeting such a requirement by supplying the constraint `-C sm_75` to `salloc` or as an `#SBATCH` line.

If it is desired to use  **PyTorch on Unity in a Jupyter Notebook** launched via n [Unity OnDemand](https://ood.unity.rc.umass.edu/pun/sys/dashboard), then **ipykernel** should be installed in the Conda environment `pyt` and the `ipykernel install` command should be used to make the environment available in JNs as explained [here](#unity-interactive):

```
(pyt)$ conda install ipykernel
(pyt)$ python -m ipykernel install --user --name pyt --display-name="PyTorch (pyt)"
```

#### A Conda environment with CuPy<a id="conda-gpu-unity"></a>

This was called **`gpu`** was created on Unity as follows (as of 1/25 it seemed the current version of Python, 3.13, was incompatible with CuPy - hence the specification here python=3.12):

```
$ module load conda/latest
$ conda create -n gpu python=3.12
$ conda activate gpu
(gpu)$ python --version
Python 3.12.8
(gpu)$ conda install numpy scipy matplotlib cupy
(gpu)$ pip list
Package         Version
--------------- -----------
cupy            13.3.0
matplotlib      3.10.0
numpy           2.2.1
scipy           1.15.1
```

Unlike on my PCs, on Unity it was not necessary to explicitly specify `-c conda-forge` to get an up-to-date version of CuPy (see [Installing CUDA-aware Python packages](#pytorch-cupy) above).  This may be because [on Unity, Conda uses Minforge](https://docs.unity.rc.umass.edu/documentation/software/conda/) rather than Anaconda.

#### Run `gputest.py` on Unity interactively<a id="gputest-interactive"></a>

- Here we get an interactive shell with 6 cores and one GPU on a compute node in the `gpu` partition, and load a CUDA module (although CUDA typically seems to be loaded already on GPU nodes). Then we run `nvidia-smi` to check that the GPU and CUDA are available and get info on them (not sure why CUDA version reported by `nvidia-smi` doesn’t match module loaded):
  
  ```
  $ salloc -c 6 -G 1 -p gpu
  (wait for the compute-node shell to come up)
  $ module load cuda/12.6
  $ nvidia-smi
  Wed Jan 15 16:48:06 2025       
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
                                       ...
  ```

- Next we activate the environment `gpu` created as above.  Then the program [`gputest.py`](#gputest-py) can be run and it will use the GPU:
  
  ```
  $ module load conda/latest
  $ conda activate gpu
  (gpu)$ cd ..                # cd to directory containing gputest.py
  (gpu)$ python gputest.py
  Running: gputest.py 11/22/23 D.C.
  Local time: Sun Jan 12 23:25:36 2025
  GPU 0 has compute capacity 6.1, 28 SMs, 11.71 GB RAM, guess model = None
  CPU timings use last 10 of 11 trials
  GPU timings use last 25 of 28 trials
  
  ***************** Doing test dense_mult ******************
  Multiply M*M=N element dense matrices
  
  *********************************************************
  
  ************ Using float64 **************
       N     flop make mats  CPU test *CPU op/s*  GPU test *GPU op/s*  GPU xfer xfer rate
  99,856 6.30e+07 7.61e-03s 1.20e-02s 5.25e+09/s 2.79e-04s 2.26e+11/s 4.42e-03s  0.72GB/s
                                  ...
  ```

#### A batch job using a GPU<a id="gpu-sbatch"></a>

- As in the non-GPU background job example above, here we again run `gputest.py` in the directory `/work/...test_gpu` but now we activate the Conda environment `gpu` which does include CuPy, so `gputest.py` will try to use a GPU.  We will also need to ensure the CUDA module is loaded, request a GPU, and run the job in a GPU partition.  So we will use an sbatch script **`gputest.sh`** with these contents:
  
  ```
  #!/bin/bash
  # gputest.sh 4/12/25 D.C.
  # One-task sbatch script using a GPU but not Apptainer, runs gputest.py
  #SBATCH -c 6                  # allocate 6 CPU cores
  #SBATCH -G 1                  # use one GPU
  # #SBATCH -C v100               # insist on a V100 GPU
  # #SBATCH -C a100               # insist on an A100 GPU (not avail on partition gpu)
  #SBATCH -t 0:10:00            # time limit 10 min (default is 1 hr)
  #SBATCH -p gpu                # submit to partition gpu
  # #SBATCH -p gpu,gpu-preempt    # submit to partition gpu or gpu-preempt (<2 hrs)
  scontrol write batch_script $SLURM_JOB_ID -;echo # print this batch script to output
  echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
  module purge                  # unload all modules
  module load conda/latest
  module load cuda/12.6         # need CUDA to use a GPU
  conda activate gpu            # environment with NumPy, SciPy, and CuPy
  python gputest.py
  ```
  
  Here is an [article in the Unity docs](https://docs.unity.rc.umass.edu/documentation/tools/gpus/) with extensive information on the GPUs available on Unity and how to select them.  The examples of selecting particular GPUs shown (commented out) in the sbatch script are not very practical as V100 GPUs can be hard to get and A100 GPUs nearly impossible. Before submitting a job that requests specific GPU models, it may be worth checking on their current availability:
  
  ```
  # Show idleness of nodes that have V100 GPUs:
  $ unity-slurm-find-nodes v100 | unity-slurm-node-usage 
  ```

- Here we submit the job, then examine its output. Note this can be run in any (or no) Conda environment, and if desired on a login node, as it is the node allocated by `sbatch` that will be used to run the job:
  
  ```
  try-gputest$ sbatch gputest.sh
  Submitted batch job 29282756
  (wait until 'squeue --me' shows that job has completed)
  $ cat slurm-29282756.out
  nodelist=gypsum-gpu168
  Loading conda
  Loading cuda version 12.6
  Running: gputest.py 11/22/23 D.C.
  Local time: Fri Feb 28 22:30:40 2025
  GPU 0 has compute capacity 7.5, 68 SMs, 11.54 GB RAM, guess model = None
  CPU timings use last 10 of 11 trials
  GPU timings use last 25 of 28 trials
  
  ***************** Doing test dense_mult ******************
  Multiply M*M=N element dense matrices
  *********************************************************
                             ...
  ```
  
    Looking at the [Unity node list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/nodes/) we see that node `gypsum-gpu168` has NVIDIA RTX 2080ti GPUs, which according to the [Unity GPU info](https://docs.unity.rc.umass.edu/documentation/tools/gpus/) has compute capability 7.5 in agreement with the output of `gputest.py`.

#### Picking a GPU on Unity<a id="pick-gpu"></a>

As of 5/25 requesting (as a general-access user) a V100 GPU on Unity resulted in a several-day queue time, while a job requesting an A100 GPU seemed unwilling to run at all. Therefore this table was compiled from the Unity docs with some summary performance specs of the NVIDIA GPUs on Unity as of 5/25 (the first line shows the inexpensive GPU in the [candela-21 PC](#pcs), for comparison). This information is from the [GPUs on Unity](https://docs.unity.rc.umass.edu/documentation/tools/gpus/) page, the [Unity GPU Summary List](https://docs.unity.rc.umass.edu/documentation/cluster_specs/gpu_summary/), and the Wikipedia article [List of Nvidia graphics processing units ](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units):

| NVIDIA model, constraint           | Total GPUs, `gpu` | Total GPUs, `gpu-preempt` | GPUs per node | Compute capability, constraint | VRAM per GPU, constraint | Mem BW         | float64, float32 FLOPS |
| ---------------------------------- | ----------------- | ------------------------- | ------------- | ------------------------------ | ------------------------ | -------------- | ---------------------- |
| GeForce GTX 1660                   | (c-21)            | (c-21)                    | 1             | 7.5                            | 6 GB                     | **0.19 TB/s**  | 0.16, **5.0 TF**       |
| GeForce GTX TITAN X, `titanx`      | 188               | 188                       | 4             | 5.2, `sm_52`                   | 12 GB, `vram12`          | **0.48 TB/s**  | 0.35, **10.9 TF**      |
| Tesla M40 24GB, `m40`              | 76                | 76                        | 4             | 5.2m `sm_52`                   | 23 GB, `vram23`          | **0.29 TB/s**  | 0.17, **6.8 TF**       |
| GeForce GTX 1080 Ti, `1080ti`      | 312               | --                        | 8             | 6.1, `sm_61`                   | 11 GB, `vram11`          | **0.32 TB/s**  | 0.25, **8.0 TF**       |
| V100-PCIE-16GB, `v100`             | --                | --                        |               | 7.0, `sm_70`                   | 16 GB, `vram16`          | **0.90 TB/s**  | 7.0, **14.0 TF**       |
| V100-SXM2-16GB, `v100`             | 17                | --                        | 2-3           | 7.0, `sm_70`                   | 16 GB, `vram16`          | **0.90 TB/s**  | 7.5, **14.9 TF**       |
| V100-SXM2-32GB, `v100`             | 8                 | --                        | 4             | 7.0, `sm_70`                   | 32 GB, `vram32`          | **0.90 TB/s**  | 7.5, **14.9 TF**       |
| GeForce RTX 2080, `2080`           | --                | 368                       | 8             | 7.5, `sm_75`                   | 8 GB, `vram8`            | **0.45 TB/s**  | 0.32, **10.1 TF**      |
| GeForce RTX 2080 Ti, `2080ti`      | 144               | --                        | 8             | 7.5, `sm_75`                   | 11 GB, `vram11`          | **0.62 TB/s**  | 0.37, **11.8 TF**      |
| Quadro RTX 8000, `rtx8000`         | --                | 18                        | 6             | 7.5, `sm_75`                   | 48 GB, `vram48`          | **0.67 TB/s**  | 0.35, **11.1 TF**      |
| A100-PCIE-40GB, `a100`, `a100-40g` | 32 tot            | 8                         | 4-8           | 8.0, `sm_80`                   | 40 GB, `vram40`          | **1.56 TB/s**  | 9.7, **19.5 TF**       |
| A100-SXM4-80GB, `a100`, `a100-80g` | --                | 133                       |               | 8.0, `sm_80`                   | 80 GB, `vram80`          | **1.56 TB/s**  | 9.7, **19.5 TF**       |
| A16, `a16`                         | 4                 | 64                        | 4-8           | 8.6, `sm_86`                   | 16 GB, `vram16`          | **4x0.2 TB/s** | 1.09, **4x4.6 TF**     |
| A40, `a40`                         | 8                 | 4                         | 4             | 8.6, `sm_86`                   | 48 GB, `vram48`          | **0.70 TB/s**  | 1.1, **37 TF**         |
| L40S, `l40s`                       | 40                | 56                        | 4             | 8.9, `sm_89`                   | 48 GB, `vram48`          | **0.86 TB/s**  | 1.41, **91 TF**        |
| L4, `l4`                           | 8                 | --                        | 8             | 8.9, `sm_89`                   | 23 GB, `vram23`          | **0.30 TB/s**  | 0.49, **67 TF**        |
| H100 80GB HBM3                     | 12                | 4                         | 4             |                                |                          |                |                        |
| GH200, `gh200`                     | --                | --                        |               | 9.0, `sm_90`                   | 102 GB, `vram102`        | **3.35 TB/s**  | 33.5, **67 TF**        |

Notes:

- With `#SBATCH -C a40` the job will only run on a node with an A40 GPU.  To allow running on a node with either an A16 or and A40 GPU use `#SBATCH -C "a16|a40"`.
- With `#SBATCH -C sm_86` the job will run on a node with compute capability **8.6 or higher**. Except for the first line, the table above is ordered by compute capability.
- With `#SBATCH -C vram16` the job will run on a node with compute capability **16 GB or more VRAM**.
- I believe an sbatch script should include only one `#SBATCH -C ...` line; multiple constraints should be combined in a single `#SBATCH -C ...` command with and's or or's as [shown here](https://slurm.schedmd.com/sbatch.html).
- Some of the more recent NVIDIA GPUs are particularly optimized for lower-precision FP operations used for AI training, not shown above.

### Using Apptainer on Unity<a id="unity-apptainer"></a>

#### Getting container images on the cluster<a id="images-to-unity"></a>

To run on Unity, a suitable container image (`.sif` file) must be present in a Unity job I/O location under `/work/pi_<userc>`.  Note `.sif` files are typically one to several GB in size.

- An image can be built on a Linux PC as described in [Using Apptainer on a Linux PC](#apptainer-pc) above, then [tranferred to Unity](#unity-file-transfer) using `scp` (the graphical [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard) does not seem able to transfer files this big).  All of the containers used in this document were built on a Linux PC in this way.  It is remarkable that these PC-built containers all worked without modification when transferred to Unity.

- It may be possible to build an image directly on Unity using the **`--fakeroot`** option to `apptainer build`, I haven’t tried this.

- Due to their large sizes, I find it handy to put all my `.sif` files in one directory on Unity.  For interactive jobs it is useful to define an alias that sets the environment variable `SIFS` to the path to this directory:
  
  ```
  # Add this to ~/.bash_aliases:
  alias sifs='export SIFS=/work/pi_..../sifs'
  ```
  
  If the **path has any spaces** `"$SIFS"` must be used rather than `$SIFS`  but this is not shown below. Then when a new shell is obtained `SIFS` can easily be set (and if desired, the list of container images and their sizes seen): 
  
  ```
  $ sifs             # sets SIFS
  $ du -ah $SIFS
  1.3G    /work/pi_..../sifs/dfs.sif
  1.3G    /work/pi_..../sifs/pack.sif
  1.2G    /work/pi_..../sifs/m4p.sif
  1.3G    /work/pi_..../sifs/dem21.sif
  3.3G    /work/pi_..../sifs/gpu.sif
               ...
  ```
  
  It is generally  assumed below that **`SIFS` is set** and the **container images `dem21.sf`..`pack.sif` have been built** as detailed in [Using Apptainer on a Linux PC](#apptainer-pc)  and transferred to Unity, as shown just above.
  
  For big batch container jobs, it seems better to to set the location of  `.sif` files directly in the sbatch script -- otherwise if   it is accidentally forgotten to set the location before submitting a job with a long queue time the job will die when it finally runs:
  
  ```
  (typical lines in an sbatch script using Apptainer)
  SIFS='/work/pi_.../sifs'          # where <mycontainer>.sif is kept
  mpirun apptainer exec $SIFS/<mycontainer>.sif python <myprogram>.py
  ```

#### Running a container interactively or in a batch job<a id="unity-run-container"></a>

This section describes how to run a container that **does not use MPI or a GPU** -- the additional steps needed for those things are in separate sections below.

- **Environment for running Apptainer containers.**  Unless MPI or a GPU is used, for many purposes it should not be necessary to load modules other than the Apptainer module or to set a Conda environment:
  
  - Python and packages typically loaded with Conda like NumPy and SciPy should be pre-loaded in the container, all in the desired versions.  
  
  - User packages installed locally should also be pre-loaded in the container.
  
  - It should not be necessary to set a Conda environment before running the container, unless this is required for code running outside the container.
  
  - Both MPI and use of a GPU require non-OS code outside the container  (to manage communication between copies of the container, or to run the GPU) so in these cases a suitable Conda environment and/or module must be loaded, as shown in sections below. 

- **Running a container interactively.** Obtain a shell on a compute node, and in this shell load the Apptainer module.  Also set `SIFS` to point to directory where `.sif` (container image) files are kept.
  
  ```
  $ salloc -c 6 -p cpu    # Get 6 cores on a compute node in the cpu partition
  (wait for the compute-node shell to come up)
  $ module load apptainer/latest
  $ sifs
  ```
  
   Here we have made a directory `try-tprogs` on Unity and copied into it:
  
  - The short program `np-version.py` that imports Numpy and prints its version number.
  - The program `test-util.py` that imports the `dcfuncs` package and tests that it can be run.
  
  First we check the version of Python loaded on the Unity node we are using:
  
  ```
  $ cd try-tprogs; ls
  np-version.py  test-util.py
  test-tprogs$ python --version
  Python 3.12.3
  ```
  
  Next we run the container **`dfs.sif`** that was built in the section [A container with a local Python package installed](#local-package-container), which can run programs that import the **`dcfuncs`** package.  Running the container like this  executes the commands in the `%runscript` section of the container definition file:
  
  ```
  try-tprogs$ $SIFS/dfs.sif
  foo!
  ```
  
  Shelling into the container we see the version of Python installed inside when it was built:
  
  ```
  try-tprogs$ apptainer shell $SIFS/dfs.sif
  Apptainer> python --version
  Python 3.12.8
  Apptainer>             # hit ctrl-d to get out of container
  exit
  ```
  
  Next we use `python` inside the container to run the scripts `np-version.py` and `test-util.py` that are outside the container.  The first script `np-version.py` uses NumPy installed in the container when it was built, independent of what Numpy if any exists outside the container.  The second script `test-util.py` imports and uses the package `dcfuncs`, which was installed locally inside the container when it was built:
  
  ```
  try-tprogs$ apptainer exec $SIFS/dfs.sif python np-version.py
  numpy version = 2.2.1
  try-tprogs$ apptainer exec $SIFS/dfs.sif python test-util.py
  This is: dutil.py 8/19/24 D.C.
  Using: util.py 8/18/24 D.C.
    ...
  ```
  
  It is interesting to see what outside files can be accessed from inside a container  -- this depends on how the system admins have set things up.  Poking around a container on Unity after doing `apptainer shell..` from a directory under `/work` , it seemed that from inside the container I could access all files under `/work` , but under `/home` I could only see the files under my own subdirectory of `/home` (as of 3/25).

- **Running a  (non-MPI, non-GPU) container with a batch job.**<a id="app-sbatch"></a> For this purpose we have copied the python script `gputest.py` to  a Unity directory `/work/.../try-gputest` (this was also done in earlier sections showing how to run batch jobs without Apptainer). For now we will run `gputest.py` from a container that that does not contain CuPy, which will cause it not to use a GPU.  The container **`dfs.sif`** used just above would work, but here we use the even simpler container **`pack.sif`** defined in [A container including chosen Python packages](#packages-container) .  An sbatch script **`simple-app.sh`** with the following contents is put in the directory `try-gputest`:
  
  ```
  #!/bin/bash
  # simple-app.sh 4/6/25 D.C.
  # One-task sbatch script uses an Apptainer container pack.sif
  # that doesn't have CuPy or OpenMPI to run gputest.py (which detects
  # CuPy is not present and so doesn't use a GPU).
  # Must set SIFS to directory containing pack.sif before running this
  # script in a directory containing gputest.py.
  #SBATCH -c 6                       # use 6 CPU cores
  #SBATCH -p cpu                     # submit to partition cpu
  echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
  module purge                       # unload all modules
  module load apptainer/latest
  # Use python in pack.sif to run gputeset.py in CWD.
  apptainer exec $SIFS/pack.sif python gputest.py
  ```
  
  We can run the job from a login node, and there is no need to load any modules or set an environment as these are taken care of by the sbatch script above. Examining the output file,  it can be seen that `gputest.py` was unable to import `cupy` and so did not try to use a GPU:
  
  ```
  try-tprogs$ sifs          # sets SIFS to directory containing pack.sif 
  try-tprogs$ ls
  simple-app.sh  gputest.py
  try-tprogs$ sbatch simple-app.sh
  Submitted batch job 29310376
  (wait until 'squeue --me' shows that job has completed)
  try-gputest$ cat slurm-29310376.out
  nodelist=gypsum-gpu080
  Loading apptainer version latest
  Running: gputest.py 11/22/23 D.C.
  Local time: Sun Mar  2 02:41:10 2025
  Import cupy failed, using CPU only
  CPU timings use last 10 of 11 trials
  
  ***************** Doing test dense_mult ******************
  Multiply M*M=N element dense matrices
  *********************************************************
                   ...
  ```

#### Running containers that use MPI<a id="unity-mpi-container"></a>

Here we combine things from the sections above on [running a non-MPI container on Unity](#unity-run-container), [running a container with MPI on a PC](#mpi-container), and [running a non-containerized MPI job on Unity](#unity-mpi).

For the examples here it assumed that the needed image file (**`m4p.sif`** or **`dem21.sif`**) has been built on a PC as shown in the sections referenced above and transferred to a directory under `/work/pi...` on Unity -- also that **`sifs`** has been aliased to a command that sets the environment variable `SIFS` to point to this directory, as [discussed here](#images-to-unity).

- First we make a Conda environment **`ompi`** in which to run the container, with OpenMPI but not including `mpi4py`, `numpy`... as these things are installed in the container. We do include Python so pip could be used to install things to this environment.  It works equally well to use the environment **`m4py`** defined above which does include `mpi4py`, etc.
  
  ```
  $ unity-compute
  (wait for the compute-node shell to come up)
  $ module load conda/latest
  $ module load openmpi/5.0.3
  $ conda create -n ompi -c conda-forge openmpi=5 python=3.12
  $ conda activate ompi
  (ompi)$ mpirun --version
  mpirun (Open MPI) 5.0.7
  (m4p)$ ompi_info | head
                   Package: Open MPI conda@f424a898794e Distribution
                  Open MPI: 5.0.7
    Open MPI repo revision: v5.0.7
     Open MPI release date: Feb 14, 2025
                   MPI API: 3.1.0
              Ident string: 5.0.7
                    Prefix: /work/candela_umass_edu/.conda/envs/ompi
   Configured architecture: x86_64-conda-linux-gnu
             Configured by: conda
             Configured on: Mon Feb 17 07:57:35 UTC 2025
  ```

- Here we use the container **`m4p.sif`** to run **`mpi_hw.py`** in a manner similar to the [non-Apptainer section on MPI](#ways-mpi-unity) above.  The difference is that here we are using `mpirun` to start multiple copies of `apptainer exec`, each of which then runs `python mpi_hw.py`.
  
  ```
  $ salloc -n 4              # get interactive shell on compute node with 4 cores
  (wait for the compute-node shell to come up)
  $ module load conda/latest
  $ conda activate ompi
  (ompi)..$ sifs             # sets SIFS to directory with m4p.sif
  $ cd python-scripts; ls    # cd to directory with mpi_hw.py
  mpi_hw.py  ...
  (ompi)..python-scripts$ mpirun --display bindings apptainer exec $SIFS/m4p.sif python mpi_hw.py
  [cpu005:01143] Rank 0 bound to package[1][core:18]
  [cpu005:01143] Rank 1 bound to package[1][core:19]
  [cpu007:2945042] Rank 2 bound to package[1][core:21]
  [cpu008:4032683] Rank 3 bound to package[0][core:3]
  Hello world from rank 1 of 4 on cpu005 running Open MPI v5.0.7
  Hello world from rank 3 of 4 on cpu008 running Open MPI v5.0.7
  Hello world from rank 2 of 4 on cpu007 running Open MPI v5.0.7
  Hello world from rank 0 of 4 on cpu005 running Open MPI v5.0.7
  ```

- Here we use the container **`m4p.sif`** to run **`osu_bw.py`**  in two ranks on the same node, then on two ranks on different nodes.  Again we copy the [non-Apptainer section on MPI](#ways-mpi-unity) above, but again we are using `mpirun` to run multiple copies of `apptainer exec` rather than directly running `python`.
  
  ```
  $ salloc -n 4 -N 2 -C ib    # get interactive shell with 4 cores on 2 nodes, infiniband
  (wait for the compute-node shell to come up)
  $ module load conda/latest
  $ conda activate ompi
  (ompi)..$ sifs             # sets SIFS to directory with m4p.sif
  $ cd python-scripts; ls    # cd to directory with osu_bw.py
  osu_bw.py  ...
  
  # Run with default binding, puts both ranks on same node.
  (ompi)..python-scripts$ mpirun -n 2 --display bindings apptainer exec $SIFS/m4p.sif python osu_bw.py
  [cpu049:742033] Rank 0 bound to package[1][core:83]
  [cpu049:742033] Rank 1 bound to package[1][core:84]
  2
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
           1                2.59
           2                5.47
           4               11.13
           8               22.20
          16               44.19
          32               86.90
          64              174.54
         128              308.25
         256              603.50
         512            1,202.67
       1,024            2,355.65
       2,048            4,450.55
       4,096            7,887.71
       8,192            5,930.39
      16,384           13,396.33
      32,768            9,352.70
      65,536           13,328.49
     131,072           16,870.32
     262,144           18,602.76
     524,288           20,919.15
   1,048,576           21,668.60
   2,097,152           21,891.10
   4,194,304           22,074.93
   8,388,608           22,823.64
  16,777,216           21,265.93
  
  # Run with --map-by node, puts ranks on diffent nodes.
  (ompi)..python-scripts$ mpirun -n 2 --map-by node --display bindings apptainer exec $SIFS/m4p.sif python osu_bw.py
  [cpu049:742176] Rank 0 bound to package[1][core:83]
  [cpu050:4179815] Rank 1 bound to package[0][core:2]
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
  2
           1                1.76
           2                3.48
           4                7.06
           8               14.21
          16               28.43
          32               56.85
          64              114.17
         128              224.43
         256              387.01
         512              781.21
       1,024            1,478.60
       2,048            3,056.77
       4,096            4,353.85
       8,192            4,458.93
      16,384            5,331.77
      32,768            7,504.09
      65,536           11,265.97
     131,072           11,891.43
     262,144           12,117.05
     524,288           12,148.33
   1,048,576           12,211.63
   2,097,152           12,285.85
   4,194,304           12,315.35
   8,388,608           12,317.98
  16,777,216           12,310.61
  ```
  
  The peak speeds seem about the same as observed in the the [non-Apptainer tests](#ways-mpi-unity) above.

- **Running a containerized MPI batch job.** Here is an sbatch script **`osubw-app.sh`** for running `osu_bw.py`  with the container `myp.sif` in a batch job. The `#SBATCH` settings are the same as used for `salloc` when `osu_bw.py` was run interactively in the previous section:
  
  ```
  #!/bin/bash
  # osubw-app.sh 4/6/25 D.C.
  # Two-task sbatch script uses an Apptainer container m4p.sif that
  # has OpenMPI to run osu_bw.py, which measures the commnunication
  # speed between two MPI ranks. Activates the Conda environment ompi to
  # make OpenMPI available outside the container. Must set SIFS to directory
  # containing m4p.sif before running script in a directory containing osu_bw.py
  #SBATCH -N 2                       # allocate two nodes
  #SBATCH -n 4                       # allocate for up to 4 MPI ranks
  #SBATCH -p cpu                     # submit to partition cpu
  #SBATCH -C ib                      # require inifiniband connectivity
  echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
  module purge                       # unload all modules
  module load apptainer/latest
  module load conda/latest
  conda activate ompi
  mpirun -n 2 --display bindings apptainer exec $SIFS/m4p.sif python osu_bw.py
  
  # Alternative mpirun command has '--map-by node' to make run on two nodes.
  #mpirun -n 2 --display bindings --map-by node \
  #    apptainer exec $SIFS/ompi5.sif python osu_bw.py
  ```
  
  The job is submitted from a directory containing this sbatch script and `osu_bw.py`.  As usual it can be submitted from a login shell, without activating a Conda environment:
  
  ```
  $ sifs                # sets SIFS to directory containing m4p.sif 
  $ cd try-mpi; ls      # cd to directory where osubw-app.sh and osu_bw.py are located
  osubw-app.sh osu_bw.py...
  try-mpi$ sbatch osubw-app.sh
  Submitted batch job 31303947
  (wait until 'squeue --me' shows that job has completed)
  try-mpi$ cat slurm-31303947.out
  nodelist=cpu[046-047]
  Loading apptainer version latest
  Loading conda
  [cpu046:691666] Rank 0 bound to package[0][core:1]
  [cpu046:691666] Rank 1 bound to package[0][core:2]
  2
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
           1                2.43
           2                4.81
           4                9.80
           8               19.52
          16               39.17
          32               77.89
          64              155.72
         128              288.67
         256              578.99
         512            1,155.67
       1,024            2,242.96
       2,048            4,224.08
       4,096            7,576.89
       8,192            5,053.99
      16,384            7,661.24
      32,768            9,452.70
      65,536           16,074.42
     131,072           20,191.90
     262,144           14,337.65
     524,288            9,070.66
   1,048,576            8,696.12
   2,097,152            9,255.01
   4,194,304            9,152.04
   8,388,608            9,188.64
  16,777,216            8,734.19
  ```

- **Using a container to run the `dem21` test program `boxpct.py`.**  Here we follow the steps shown for a PC in [A container to run the more elaborate MPI package `dem21`](#dem21-container) above, but now running sbatch jobs on Unity.
  
  - As [described earlier](#images-to-unity) the container **`dem21.sif`** built on a PC as in [A container to run the more elaborate...](#dem21-container)  was copied to a Unity directory under `/work/pi..` and the alias `sifs` was set up to set the environment variable `SIFS` to point to this directory.
  
  - As was done earlier for a [non-containerized run on Unity](#sbatch-dem21),  `boxpct.py`  and its configuration file `box.yaml` are copied to a directory `try-dem21`. Now we also put in this directory an sbatch script **`boxpct-app.sh`** with these contents:
    
    ```
    #!/bin/bash
    # boxpct-app.sh 4/6/25 D.C.
    # n-task sbatch script uses an Apptainer container dem21.sif that
    # has OpenMPI and the dem21 package to run boxpct.py, which is a
    # test program for the dem21 simulation package.
    # Must set SIFS to directory containing dem21.sif before running this
    # script in a directory containing boxpct.py
    #SBATCH -n 4                       # allocate for n MPI ranks
    #SBATCH -p cpu                     # submit to partition cpu
    #SBATCH -C ib                      # require inifiniband connectivity
    echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
    module purge                       # unload all modules
    module load apptainer/latest
    module load conda/latest
    conda activate ompi
    export pproc=mpi                   # tells dem21 to run in MPI-parallel mode
    mpirun --display bindings apptainer exec $SIFS/dem21.sif python boxpct.py
    ```
  
  - The job is run in the same manner as `osu_bw.py` was run just above:
    
    ```
    $ sifs                # sets SIFS to directory containing m4p.sif 
    $ cd try-dem21; ls      # cd to directory where boxpct-app.sh, boxpct.py, and box. are located
    boxpct-app.sh boxpct.py  box.yaml  ...
    try-dem21$ sbatch boxpct-app.sh
    Submitted batch job 31304489
    (wait until 'squeue --me' shows that job has completed)
    try-mpi$ cat slurm-31304489.out
    nodelist=cpu046
    Loading apptainer version latest
    Loading conda
    [cpu046:717593] Rank 0 bound to package[0][core:1]
    [cpu046:717593] Rank 1 bound to package[0][core:2]
    [cpu046:717593] Rank 2 bound to package[1][core:32]
    [cpu046:717593] Rank 3 bound to package[1][core:33]
    - Started MPI on master + 3 worker ranks.
    THIS IS: boxpct.py 12/3/22 D.C., using dem21 version: v1.2 2/11/25
    Parallel processing: MPI, GHOST_ARRAY=True
    - Read 1 config(s) from /work/pi.../try-dem21/box.yaml
    
    SIM 1/1:
    Using inelastic 'silicone' grainlets with en=0.7 and R=0.500mm
                              ...
    ```

- **Using a container to run a larger DEM simulation with `dem21`.**   Here we continue to follow the corresponding steps shown for a PC in [A container to run the more elaborate MPI package `dem21`](#dem21-container) above.
  
  - Here is an appropriate sbatch script, called **`mx2-unity-app.sh`**.  As in the other examples above, this script does not load an OpenMPI module but does activate the environment `ompi` with OpenMPI installed (these choices are discussed in [Ways of running Python MPI programs on Unity](#ways-mpi-unity) above).  As noted in that section, this script probably should have use `#SBATCH -C "ib&mpi"` rather than `#SBATCH -C ib` as shown here to ensure **consistent CPU type** when more than one node is used (typically when more than 64 cores are requested), but I haven't tried that:
    
    ```
    #!/bin/bash
    # cc-expts-unity-app/mx2-unity-app.sh 4/12/25 D.C.
    # sbatch script to run granular-memory simulation program mx2.py containerized on the
    # Unity cluster, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
    # Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
    # Reads default config file mx2.yaml in grandparent directory modified by
    # mx2mod.yaml in current directory.
    #SBATCH -n 15                        # run on 15 cores
    #SBATCH -N 1                         # use 1 node
    # #SBATCH --exclusive                  # don't share nodes with other jobs
    #SBATCH --mem=100G                   # allocate 100GB memory per node
    # #SBATCH --mem=0                      # allocate all available memory on nodes used
    #SBATCH -t 10:00:00                  # time limit 10 hrs (default is 1 hr)
    #SBATCH -p cpu                       # submit to partition cpu
    # #SBATCH -p cpu,cpu-preempt           # submit to partition cpu or cpu-preempt (<2 hrs)
    #SBATCH -C ib                        # require inifiniband connectivity
    scontrol write batch_script $SLURM_JOB_ID -;echo # print this script to output
    echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
    module purge                         # unload all modules
    module load apptainer/latest
    module load conda/latest             # need this to use conda commands
    conda activate ompi                  # environment with OpenMPI, then don't need OpenMPI module
    # module load openmpi/5.0.3            # load OpenMPI module, then don't need ompi environment 
    export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
    SIFS='/work/pi_candela_umass_edu/dcstuff/sifs'   # where dem21.sif is kept
    mpirun --display bindings apptainer exec $SIFS/dem21.sif \
         python ../../mx2.py mx2mod       # run displaying bindings
    # mpirun apptainer exec $SIFS/dem21.sif \
    #    python ../../mx2.py mx2mod       # run without displaying bindings
    ```
  
  - As was done for a [non-containerized run on Unity](#sbatch-dem21) we can run this script from on a login node in the directory containing the script:
    
    ```
    $ cd ..cc-expts-unity; ls
    bw6-sigs.yaml  bw6.svg  mx2mod.yaml  mx2-unity-app.sh signals.sh
    ..cc-expts-unity$ sbatch mx2-unity-app.sh
    Submitted batch job 31844213
    (use 'squeue --me' to see when job starts, time running, if done)
    ..cc-expts-unity$ cat slurm-31844213.out  # can do while running to see output so far
    (for long jobs no need to stay logged in to Unity while they run)
    ```
    
    Some stats from running in various ways (all containerized).  Here external MPI = no means the environment **`ompi`** was activated and the module `openmpi/5.0.3` was not loaded.  Conversely external MPI = yes means no environment was activated and the module `openmpi/5.0.3` was loaded:
    
    | system                 | candela-21        | Unity        | Unity             | Unity                   | Unity                       | Unity                      |
    | ---------------------- | ----------------- | ------------ | ----------------- | ----------------------- | --------------------------- | -------------------------- |
    | external MPI?          | -                 | no           | no                | no                      | no                          | yes                        |
    | cores (`-n`)           | 15                | 15           | 64                | 128                     | 256                         | 256                        |
    | max boxes/crate        | 16                | 16           | 4                 | 2                       | 1                           | 1                          |
    | req. nodes (`-N`)      | -                 | 1            | 1                 | 2                       | no `-N`                     | no `-N`                    |
    | `--exclusive` ?        | -                 | no           | yes               | yes                     | yes                         | no                         |
    | nodes used, cores/node |                   | 1            | 1, 64             | 2, 64                   | 4, 64                       | 4, 64                      |
    | which nodes used       | -                 | TODO         | umd-cscdr-cpu 042 | umd-cscdr-cpu [033-034] | uri-cpu [017, 022, 024-025] | uri-cpu [019, 022,024-025] |
    | inter-rank comm time   |                   | X%           | 9.9%              | 26.4%                   | 48.8%                       | 47.2%                      |
    | memory used            | 2.2 GB            | TODO GB      | 16.9 GB           | 16.9 GB                 | 50.4 GB                     | 17.0 GB                    |
    | sim wall time          | 284 min = 4.74 hr | X min = X hr | 119 min = 1.99 hr | 80 min = 1.33 hr        | 67 min = 1.12 hr            | 66 min = 1.10 hr           |
    | time/(step-grain)      | 3.70e-6 s         | Xe-6 s       | 1.56e-6 s         | 1.04e-6 s               | 0.88e-6 s                   | 0.86e-6 s                  |
    | speed/candela-21       | 1.00              | 0.X          | 2.4               | 3.6                     | 4.2                         | 4.3                        |
    | speed / non-container  | 0.94              |              | 1.00              | 0.89                    | 0.74                        |                            |
    
    Notes:
    
    - The next-to-last line compares the speed on Unity to a [containerized run on the `candela-21` PC](#dem21-container).
    - The last line compares the speed to [non containerized runs](#sbatch-dem21) on the same system (PC or Unity) using the same number of cores and nodes.  It appears that the speed penalty for containerization is up to about 25%, but some of this penalty may be due to the particular nodes allocated.
    - The last two columns seem to show that the speed does not to depend on whether external OpenMPI (from a module load) or OpenMPI from a Conda environment was used.
  
  - Finally some containerized runs were done on a simulation with ten times as many grains, to compare with corresponding [non-containerized runs](#unity-dem21-bigger) above.
    
    | system                 | candela-21 | Unity             | Unity                    | Unity                                     | Unity                                             |
    | ---------------------- | ---------- | ----------------- | ------------------------ | ----------------------------------------- | ------------------------------------------------- |
    | external MPI?          | -          | no                | no                       | no                                        | yes                                               |
    | cores (`-n`)           | 16         | 64                | 128                      | 256                                       | 256                                               |
    | max boxes / crate      | 116        | 28                | 14                       | 7                                         | 7                                                 |
    | req. nodes (`-N`)      | -          | no `-N`           | no `-N`                  | no `-N`                                   | no `N`                                            |
    | `--exclusive` ?        | -          | yes               | yes                      | yes                                       | yes                                               |
    | nodes used, cores/node | -          | 1, 64             | 2, 64                    | 4, 64                                     | 4, 64                                             |
    | which nodes used       | -          | umd-cscdr-cpu 040 | umd-cscdr-cpu [040, 046] | umd-cscdr-cpu 039, uri-cpu [007, 011-012] | nodelist=umd-cscdr-cpu039, uri-cpu [007, 011-012] |
    | inter-rank comm time   |            | 14.8%             | 12.8%                    | 26.1%                                     | 25.3%                                             |
    | memory used            |            | 23.0 GB           | 21.9 GB                  | 21.1 GB                                   | 21.3 GB                                           |
    | sim wall time          |            | 944 min = 15.7 hr | 473 min = 7.88 hr        | 288 min = 4.90 hr                         | 289 min = 4.81 hr                                 |
    | time / (step-grain)    |            | 1.25e-6 s         | 0.63e-6 s                | 0.38e-6 s                                 | 0.38e-6 s                                         |
    | speed / candela-21     |            |                   |                          |                                           |                                                   |
    | speed / non-container  |            | 1.02              | 0.97                     | 0.95                                      |                                                   |
    
    Notes:
    
    - From this table we see that this large (5 hrs on 256 cores) sim, **the speed is nearly identical for non-containerized jobs, and when external MPI vs Conda-environment MPI was used**.
    - Therefore, the same maximum speedup is obtained as [tabulated above](#unity-dem21-bigger) for non-containerized runs, which is **a speedup of about 10 using 256 cores on unity compared with the 16-core PC candela-21** (i.e. 16 times as many cores).

#### Running a container the uses a GPU<a id="unity-gpu-container"></a>

- This is very similar to running a non-GPU container as described above, with these differences:
  
  - Obviously this must be done on a node with GPU(s), with a GPU allocated to the job by SLURM.
  - Both CUDA and Apptainer modules should be loaded (although both packages seem to be pre-loaded on GPU nodes).
  - The container (`.sif file`) must have been built with CUDA libraries. Installing CuPy in the container build definition seems to accomplish this.
  - Apptainer commands running the container must have the --nv flag to make the external CUDA libraries available.

- Here is an example of these things in action for **interactive use of a GPU with a container**:  
  
  - As [described earlier](#images-to-unity) the container **`gpu.sif`** (built on a PC as in [A container that can use a GPU](#gpu-container)) was copied to a Unity directory under `/work/pi..` and the alias `sifs` was set up to set the environment variable `SIFS` to point to this directory.
  
  - `gputest.py` was copied to the directory `try-gputest` (this was already done for other sections above).
  
  - An interactive shell is allocated on a compute node with 6 cores and one GPU, then CUDA and Apptainer modules are loaded (`nvidia-smi` checks that the GPU is available but is not necessary here):
    
    ```
    $ salloc -c 6 -G 1 -p gpu
    (wait for the compute-node shell to come up)
    $ module load apptainer/latest
    $ module load cuda/12.6
    $ nvidia-smi
    Tue Jan 14 16:57:25 2025
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4
                               ...
    ```
  
  - Now `apptainer exec` with the flag `--nv` is able to use the GPU.  Note that it is not necessary to set a Conda environment -- running in the container replaces this:
    
    ```
    $ cd try-gputest; ls
    gputest.py ...
    try-gputest$ sifs                  # set SIFS
    try-gputest$ apptainer exec --nv $SIFS/gpu.sif python gputest.py
    Running: gputest.py 11/22/23 D.C.
    Local time: Tue Jan 14 17:00:08 2025
    GPU 0 has compute capacity 6.1, 28 SMs, 11.71 GB RAM, guess model = None
    CPU timings use last 10 of 11 trials
    GPU timings use last 25 of 28 trials
    
    ***************** Doing test dense_mult ******************
    Multiply M*M=N element dense matrices
    
    *********************************************************
    
    ************ Using float64 **************
           N     flop make mats  CPU test *CPU op/s*  GPU test *GPU op/s*  GPU xfer xfer rate
      99,856 6.30e+07 5.57e-03s 1.04e-03s 6.04e+10/s 2.76e-04s 2.28e+11/s 9.48e-03s  0.34GB/s
                                        ...
    ```

- **A batch job using a container, with a GPU.**
  
  - This is similar to the [non-GPU batch container job shown earlier](#app-sbatch), with these differences: 
    
    - We request a GPU, and submit to a GPU partition.
    - In addition to Apptainer, we need to load the module for CUDA.
    - We use the container `gpu.sif` that was built including Cupy.
    - We need the `–-nv` flag on apptainer exec.
  
  - Here is an sbatch script **`gputest-app.sh`** that incorporates these changes, which we put in the same directory `try-gputest` as `gputest.py`:  TODO request specific GPU - A100?
    
    ```
    #!/bin/bash
    # gputest-app.sh 4/6/25 D.C.
    # One-task sbatch script uses an Apptainer container gpu.sif
    # that has CuPy to run gputest.py, which will use a GPU.
    # Must set SIFS to directory containing gpu.sif before running this
    # script in a directory containing gputest.py
    #SBATCH -c 6                       # use 6 CPU cores
    #SBATCH -G 1                       # use one GPU
    #SBATCH -p gpu                     # submit to partition gpu
    # #SBATCH -p gpu,gpu-preempt         # submit to partition gpu or gpu-preempt (<2 hrs)
    scontrol write batch_script $SLURM_JOB_ID -;echo # print this batch script to output
    echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
    module purge                       # unload all modules
    module load apptainer/latest
    module load cuda/12.6              # need CUDA to use a GPU
    # Use python in gpu.sif to run gputest.py in CWD; need --nv flag
    # on apptainer exec to use a GPU.
    apptainer exec --nv $SIFS/gpu.sif python gputest.py
    ```
    
    As usual for `sbatch` jobs, we can run the job from a login node if desired:  TODO print full results
    
    ```
    $ cd try-gputest; ls
    gputest-app.sh gputest.py ...
    try-gputest$ sifs                    # set SIFS
    try-gputest$ sbatch gputest-app.sh
    Submitted batch job 29323951
    (wait until 'squeue --me' shows that job has completed)
    try-gputest$ cat slurm-29323951.out
    (prints the sbatch file)
    nodelist=gypsum-gpu096
    Loading apptainer version latest
    Loading cuda version 12.6
    Running: gputest.py 11/22/23 D.C.
    Local time: Sun Mar  2 14:40:17 2025
    GPU 0 has compute capacity 5.2, 24 SMs, 12.79 GB RAM, guess model = None
    CPU timings use last 10 of 11 trials
    GPU timings use last 25 of 28 trials
    
    ***************** Doing test dense_mult ******************
    Multiply M*M=N element dense matrices
    *********************************************************
    
    ************ Using float64 **************
             N     flop make mats  CPU test *CPU op/s*  GPU test *GPU op/s*  GPU xfer xfer rate
        99,856 6.30e+07 5.75e-03s 4.16e-04s 1.52e+11/s 5.77e-04s 1.09e+11/s 4.67e-03s  0.68GB/s
                           ...
    try-gputest$ seff 29323951
    Job ID: 29323951
    Cluster: unity
    User/Group: candela_umass_edu/candela_umass_edu
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 00:03:15
    CPU Efficiency: 19.12% of 00:17:00 core-walltime
    Job Wall-clock time: 00:02:50
    Memory Utilized: 2.76 GB
    Memory Efficiency: 46.07% of 6.00 GB
    ```
    
    When most of the job time is due to computations on the GPU (vs computations on the CPU, or GPU-CPU data transfers), I see no reason why a containerized GPU job would run at a different speed than the corresponding non-containerized job -- but I haven't run tests for this.

#### Other ways of getting/running Apptainer containers<a id="other-apptainer"></a>

This document showed how to build an Apptainer container on a Linux PC (bootstrapping from Docker Hub), then transfer it to the Unity HPC cluster and run it there.  But there are other possibilities not detailed here:

- Apptainer containers can be built from an Apptainer `.def` file directly on Unity -- this might be useful if a suitable Linux PC were not available.

- Apptainer containers can be obtained from others.

- Apptainer can directly "run" Docker containers, which I believe consists in an automatic Apptainer build with cached output, so it can be done repeatedly without rebuilding.

## Random notes on parallel computing with Python<a id="random-notes"></a>

### Wall time and CPU time<a id="wall-cpu-time"></a>

- “Wall time” or “real time” refers to time that progresses at the same rate as physical time, independent of whether a computer program is running or is suspended, or how many cores the program is running on.  The Python function `time.perf_counter` returns wall time in seconds as a float with high resolution (order of ns) but with arbitrary origin, hence only differences between successive `time.perf_counter` calls are meaningful. Alternatively `time.perf_counter_ns` returns the wall time in nanoseconds as an integer, again with arbitrary origin.
- “CPU time” (e.g. from `time.process_time`) may exclude time when a process is suspended, and thus advance slower than wall time.  For parallel processing CPU time (e.g. `CPU Utilized` reported by Slurm `seff`) may be the sum of the time used by all processes, and thus advance faster than wall time.
- Elapsed wall time between program start and end might be the most appropriate measure of program execution time for discussing parallel speedup, as it gives the actual time the user must wait for the program to finish.  However it does not include the time a job waits in a cluster scheduling queue; when this is included the total queued+execution time may have a minimum for some number of cores, if (as is typical) the time waiting in queue increases with the number of cores requested.

### Strong and weak scaling<a id="strong-weak-scaling"></a>

- See this [helpful article](https://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling)

- Let $Q$ be the **size of the computational task**, defined so that the expected computation time is at least roughly proportional to $Q$. For a granular simulation $Q$ could be the number of grains (roughly scaling with computational time) times the number of time steps (precisely scaling with computational time). Also let $t_N(Q)$ be the **wall time to complete task $Q$ on $N$ cores**.

- **Parallel speedup** can be defined as $S(N,Q)= t_1(Q)/t_N(Q)$. A plot of $S(N,Q)$ vs $N$ is called a **strong scaling plot**, and shows how a given task $Q$ can be speeded up by using more cores $N$. Ideally $S(N,Q)$ would be proportional to $N$, but **Amdahl’s law** suggests $S(N,Q)$ should level off at finite value for large $N$ due to the fraction of the code that is not parallelized. Even if the non-parallel code is negligible, $S(N,Q$) might level off at large $N$ due to the costs of breaking the problem into more and more (smaller and smaller) parallel tasks. For example in various types of simulation, if space is broken into smaller and smaller domains there will be additional costs for communication between the domains.

- Often the number of cores $N$ is increased not to decrease the execution time, but rather to increase the size $Q$ of the task that can be carried out. If we take $Q = Q_1 N$ and $N$ is varied keeping the size per core $Q_1$ constant, we can define the **weak parallel speedup** $S_W(N,Q_1) = Nt_1(Q_1)/t_N(NQ_1)$. A plot of $S_W(N,Q_1)$ vs $N$ is called a **weak scaling plot**. The weak speedup $S_W$ measures how much problem can be done per unit time with $N$ cores compared with the same thing for one core, $S_W(N,Q_1)= [NQ_1/t_N(NQ_1)]/[Q_1/t_1(Q_1)]$. **Gustafson’s law** suggests that $S_W$ can sometimes increase indefinitely with $N$, unlike the strong speedup $S$.  This is trivially the case for "embarrassingly parallel" tasks that can simply be broken up into chunks that require little or no communication.

### Estimating MPI communication overhead<a id="estimate-mpi-overhead"></a>

Here is a simple model of a parallel computation that describes the (non-public) simulation package `dem21` used for testing in several sections above: [on a PC](#mpi-dem21), [on the Unity cluster](#sbatch-dem21), and also when containerized with Apptainer. Many parallel codes would be more complicated than this, but the idea used here might still apply:

- Timers (e.g. differences between `time.perf_counter` return values) are used to measure the wall time required for various sections of the code to complete, including the total execution time for the program $t_t$.

- *N* cores are used to carry out *N* processes (i.e. one process per core), of which one is a master or control process and $N−1$ are worker processes.

- During some periods totaling $t_{np}$ the single master process is computing and the $N-1$ worker processes are idle. This is the time used exclusively for **non-parallel** processing.

- During other periods the master process is idle, and during these periods some or all of the worker processes may be computing. Let the total time during which the $j^{th}$ worker process is computing be $t_j$. Let $t_p$ be the sum of the $N−1$ $t_j$’s. This is the total **parallel-processing time**, summed over the processes that can potentially operate in parallel (but may not, for example when one of these processes is waiting for another to reach some point, or when the master process is waiting for all of the worker processes to reach some point).

- Communication between the $N$ processes may be difficult to attribute to specific processes, particularly when one process is waiting on another. To handle this simply, all communications operations (e.g. all MPI function calls) are excluded from both $t_t$ and the $t_j$’s.

- If we imagine running the computation on a single core (hence no communication calls), the $N−1$ worker computations would need to be done sequentially so the total execution time should be $t_1 = t_{np} + t_p$. Thus we define the **actual, measured speedup** to be $S = t_t/( t_{np} + t_p)$.

- In an ideal situation there would be negligible time spent on communication (e.g. all MPI function calls would return instantaneously) and each of the $N−1$ worker processes would require precisely the same time, $t_j = t_p/(N-1)$ for every $j$. In this case the (ideal) total execution time should be $t_{t,I} = t_{np} + t_j = t_{np} + t_p/(N−1)$. Thus the **ideal speedup** would be $S_I = t_1/ t_{t,I} = (t_{np} + t_p)/(t_{np} + t_p/(N−1))$. This ideal speedup is still limited by Amdahl’s law: So long as the non-parallel portion of the execution time $t_{np}$ is negligible, $S_I \approx N−1$, the number of parallel processes. But as $N$ increases $S_I$ tends to a finite value due to $t_{np}$.

- Apart from non-negligible non-parallel processing time $t_{np}$, another reason the actual speedup $S$ is often less than the ideal speedup $S_I$ is that the $N−1$ worker computation times $t_j$ are not all equal. This is a problem of **overall load balancing** between the parallel processes, which must be addressed by writing an efficient parallel code (not discussed in this document). Let the maximum $t_j$ be $t_{j,\mathit{max}}$. If we can still ignore communication time we expect the total execution time to be $t_{t,NC} = t_{np} + t_{j,\mathit{max}}$ and the expected **speedup with no communication cost** would be $S_{NC} = t_1/t_{t,NC} = (t_{np} + \sum_j t_j)/(t_{np} + t_{j,\mathit{max}})$ which is less than $S_I$ unless all of the $t_j$’s are equal (perfect load balancing).

- Finally, we can estimate the total **time required for interprocess communication and synchronization** $t_C$ as the difference between the measured total execution time and the expected total time including load imbalance, $t_C = t_t − t_{t,NC}$. What is nice about this way of measuring communication time is that it sidesteps questions of accounting for and attributing the time required for various types of communication calls (blocking or non-blocking sends and receives, broadcasts and gathers/reductions, waits for completion, synchronizations…). And provided the inferred total communication time $t_C$ is for example, less than 20% of the measured total execution time $t_t$, we know that the speedup $S$ cannot be improved by more than 20% by reducing $t_C$. This can be significant as there may be thorny issues in optimizing $t_C$, as discussed elsewhere in this document (linking to the correct communication libraries for the hardware in use, for example) and below (synchronization inefficiencies).

- The ideal speedup $S_I$, the actual speedup $S$, and the fraction of execution time used by interproccess communication $t_C/t_t$ can all be measured by timings of the wall time required for execution of code blocks during single parallel run, thus sidestepping to some degree questions about variable CPU clock speed and variations of execution time with processor model.

- This measure of interproccess communication time does include (in addition to actual time required to carry out communication operations such as transmitting data from one node to another) waiting time due to inefficiencies other than overall load imbalance in the structure of the parallel code. In the absence of such inefficiencies, with instantaneous interproccess communications the parallel portion of the code would take no longer than the computation time for the slowest of the $N−1$ parallel processes, $t_{j,\mathit{max}}$.
  
  An example of maximally inefficient code is if each parallel process $j$ waits (e.g. via an MPI receive call) for process $j−1$ to complete before it starts – in this case the code will not run in parallel at all and most of the execution time will be counted as interproccess communication time contributing to $t_C$. We can characterize such inefficiencies as **synchronization inefficiencies**: Processes cannot run because they are waiting for other processes to complete tasks.
  
  Here is a more realistic example of synchronization inefficiencies: All $N−1$ parallel processes are required to periodically synchronize with each other, for example to carry out time steps of a simulation. Early in the simulation one set of processes runs more slowly, causing other processes to wait (perhaps due to an imbalance in the number of particles handled by each process). Later in the simulation a different set of processes runs more slowly. In this scenario, all processes might use the same total computation time $t_j$, but the time required for the code to run will be significantly larger than $t_{np} + t_{j,\mathit{max}}$, and the excess time will contribute to $t_C$.

- Clearly such synchronization inefficiencies do not dominate the overall computation time and actual speedup $S$ when $t_C$ is comparable to or smaller than $t_p$. But in the reverse case $t_C \gg t_p$ both inefficient communications operations and inefficient code structure must be investigated as possible causes.

## Summary and TODOs<a id="summary-todos"></a>

### A long cheat sheet

This "cheat sheet" on MPI, GPUs, Apptainer, and HPC (**MGAH**) has ended up longer than anticipated.  But finally this shows with examples how the elements of  MGAH can be used together in various useful combinations, including:

- Using a Slurm cluster, both [without (**H**)](#unity-cluster) and [with (**AH**)](#unity-apptainer) Apptainer.

- Running an MPI program on a PC, both [without (**M**)](#mpi-pc) and [with (**MA**)](#mpi-container) Apptainer.

- Running an MPI program on a Slurm cluster, both [without (**MH**)](#unity-mpi) and [with (**MAH**)](#unity-mpi-container) Apptainer.

- Running a GPU program on a PC, both [without (**G**)](#gpu-pc) and with [(**GA**) Apptainer](#gpu-container).

- Running a GPU program on a Slurm cluster, both [without (**GH**)](#unity-gpu) and [with (**GAH**)](#unity-gpu-container) Apptainer.

What haven't been shown explicitly here are examples of using MPI and GPUs together with or without Apptainer (**MGH**, **MGAH**)-- but I think the necessary ingredients to do this are in the sections above.

The main advantages that emerged for each of the elements of MGAH were:

- **MPI (M)** enables parallel computation that can use all cores of a PC, and then if needed scale to using all cores on multiple nodes in a cluster -- at the expense of figuring out in detail how to divide your code into multiple tasks that coordinate send messages to coordinate.
- A **GPU (G)** parallelizes code more easily than MPI, but only if the code can make use of existing GPU-aware packages like CuPy.  While the degree of parallelism is limited by the GPU available, in practice this seems no worse than what can be done using MPI (unless HPC resources beyond those discussed in this document were obtained).
- Containerization of code with **Apptainer (A)** does seem effective.  For example, all of the containers built and tested on PCs (using MPI, a GPU, or neither) worked without difficulty when simply copied to the Unity cluster.
- Moving code to an **HPC cluster (H)** like Unity has proved useful to me mainly when the *volume* of work (e.g. number of simulations) increased, to enable completion of a study.  Individual jobs could be made to complete about 10 times faster than on a PC (if many cores were used for MPI, or a high-quality GPU could be used) -- but the big advantage over the PC was the ability to queue up many such jobs all at once.

### TODOs

- **Backup/archival storage** of files from Unity.  It seems that the storage provided in `/home` and `/work` directories is not considered suitable for this.

- **Using multiple GPUs.**  Most GPU nodes on a cluster like Unity have multiple GPUs, which could be in principle be used together to do jobs that are too big or too slow when done on a single GPU.  How to do this?  Within CuPy, for example, it seems straightforward to put objects like arrays on specific GPUs, but how can multiple GPUs used like a single large GPU might be? CuPy does have a `distributed_array` type and can use "NVIDIA NCCL" for multi-GPU communication, but how are these things used, and how efficient are they for various types of computation?

- **Ways of submitting multiple `sbatch` jobs.** It might be useful to find out about **array jobs** and **checkpointing**.  Some information on these topics is [here](https://groups.oist.jp/scs/advanced-slurm) and [here](https://jhpce.jhu.edu/slurm/crafting-jobs/).
