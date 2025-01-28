# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   1/28/25

- [Introduction](#intro)  
  
  - [What is this?](#what-is)
  - [Parallel computing in Python](#parcomp-python)
  - [Hardware used](#hardware)
    - [PCs](#pcs)
    - [GPUs](#gpus)
    - [Unity HPC cluster](#unity-intro)
  - [Pip and Conda](#pip-conda)
  - [Conda environments and test code used in this document](#envs-testprogs)
  - [Installing a local package](#local-package)

- [Part 1: MPI, GPU, and Apptainer on a Linux PC](#on-linux-pc)
  
  - [MPI on a Linux PC](#mpi-pc)
    - [Why do it](#why-mpi-pc)    
    - [Installing OpenMPI](#install-openmpi)
    - [Simple MPI test programs: `mpi_hw.py` and `osu_bw.py`](#mpi-testprogs)
    - [A more elaborate MPI program: `boxpct.py` with the `dem21` package](#boxpct-dem21)
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
    - [A container that can use a GPU](#gpu-container)

- [Part 2: Moving code to a Slurm HPC cluster](#move-to-hpc)
  
  - [Why do it](#why-hpc)
  - [Unity cluster at UMass, Amherst](#unity-cluster)
    - [History](#unity-history)
    - [Logging in](#unity-login)
    - [Storage](#unity-storage)
    - [Transferring files to/from Unity](#unity-file-transfer)
    - [Slurm on Unity](#unity-slurm)
    - [Running jobs interactively: `salloc` or `unity-compute`](#run-interactive)
    - [Using `.bashrc` and `.bash_aliases`](#rc-files)
    - [Using modules and Conda](#unity-modules-conda)
    - [Running batch jobs: `sbatch`](#run-batch)
    - [Using MPI](#unity-mpi)
    - [Using a GPU](#unity-gpu)
  - [Using Apptainer on the Unity HPC cluster](#unity-apptainer)
    - [Getting container images on the cluster](#images-to-unity)
    - [Running a container interactively or in batch job](#unity-run-container)
    - [Running a container that uses MPI](#unity-mpi-container)
    - [Running a container the uses a GPU](#unity-gpu-container)

- [Random notes on parallel speedup](#speedup-notes)
  
  - [Wall time and CPU time](#wall-cpu-time)
  - [Factors other than parallelism affecting execution speed](#other-speed-factors)
  - [Strong and weak scaling](#strong-weak-scaling)
  - [Estimating MPI communication overhead](#estimate-mpi-overhead)

## Introduction <a id="intro"></a>

### What is this?<a id="what-is"></a>

This the cheat sheet I that accumulated as I learned to combine several tools for **parallel computing in Python** on various **Linux** computer systems:

- [**MPI**](https://en.wikipedia.org/wiki/Message_Passing_Interface) allows multiple instances of Python to operate in parallel and communicate with each other, in the cores of a single computer or a cluster of connected computers. Code written to parallelize using MPI can utilize all the cores of a desktop computer and also scale to a larger number of cores in an HPC computer cluster.

- A **GPU** installed in a single computer can carry out highly parallel computations, so it offers an alternative to "MPI on a cluster of computers" for parallelizing code - but the degree of parallel operation is limited by the model of GPU that is available (unless multiple GPUs and/or GPUs on multiple MPI-connected computers are used, things not discussed in this document).

- [**Apptainer**](https://apptainer.org/) is a **container** system that allows user code and most of its dependencies (OS version, packages like NumPy) to be packaged together into a single large "image" file, which should then be usable  without modification or detailed environment configuration on many different computer systems from a Linux PC to a large cluster.

- High-performance Computing (**HPC**) typically refers to using a large cluster of connected computers assembled and maintained by Universities and other organizations for the use of their communities.  This document only discusses an HPC cluster running Linux and managed by  [**Slurm**](https://slurm.schedmd.com/overview.html) scheduling software, with  the the [**UMass Unity cluster**](https://unity.rc.umass.edu/index.php) as the specific HPC system used here.

Why Python?  Why Linux? Because those are what I use, and this is my cheat sheet.  So this document is geared towards this work flow:

- Write some Python code and get it working on a Linux PC.

- (If desired) to get some parallel speedup either:
  
  - add MPI code and get that working on the multiple cores of the PC, or
  
  - start using a GPU-aware package like CuPy or PyTorch and get that working using the PC's GPU.

- (If desired) to move the code to an HPC cluster like Unity:
  
  - Optionally use Apptainer to containerize the code - this document shows how to do this if the code uses MPI, a GPU, or neither.
  
  - Copy the (already working) code, containerized or not, to the HPC cluster and run it there.

Although there may be some information useful for the following topics, this document **does not cover:**

- Other than brief mentions, the use of OpenMP (a multithreading package not to be confused with OpenMPI) and/or the Python Mutiproccessing package for parallelization on the cores of a single computer.

- Operating systems other than Linux (Windows, macOS...).

- Computer languages other than Python such as C++.

- GPUs other than NVIDIA, except some brief mentions of AMD ROCm and Mac MPS support in the section [Non-NVIDIA GPUs](#non-nvidia) below.

- Direct, low-level programming of GPUs in CUDA-C++  (as opposed to the use of GPU-aware Python packages like CuPy and PyTorch, which are briefly covered).

- "Higher level" (than MPI) packages for using computer clusters such as Spark, Dask, Charm4Py/Charm++...).

- Cloud computing (Amazon Web Services, Microsoft Azure...). 

- The Docker container system, other than as a source for building Apptainer containers.

- The Kubernetes scheduling/management software typically used rather than Slurm in commercial settings, particularly with Docker.

### Parallel computing in Python<a id="parcomp-python"></a>

Python is a semi-interpreted language (compiled to a byte code, like Java) and so is much more slowly executed than a fully compiled language like C++, unless an add-on like [Numba](http://numba.pydata.org/) or [Cython](https://cython.org/) is used (neither of these is discussed further in this document, although they may certainly be useful).

Therefore good performance on large tasks is often achieved by using **packages** (typically written by others in a compiled language like C++) like [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [CuPy](https://cupy.dev/), and [PyTorch](https://pytorch.org/), to carry out the time-consuming **inner loops** of algorithms. The same is true of other high-level languages like [MATLAB](https://www.mathworks.com/products/matlab.html) and [Mathematica](https://www.mathematica.org/).   While some think Python is inherently slower than C++, if the time limiting factor is, for example, a large linear algebra operation then in either language it will likely be carried out by the same highly-optimized [BLAS](https://www.netlib.org/blas/) function on a CPU (via NumPy, for Python), or the corresponding [cuBLAS](https://developer.nvidia.com/cublas) function on a GPU (via CuPy).

There are however some murky intermediate situations. For example [NumPy advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html) allows many complicated operations (e.g. operations on elements meeting complicated conditions) on arrays to be carried out much faster than if they were coded directly in Python -- but maybe slower than would be possible in C++.

Be that as it may, the premise of this document is **speeding up Python code** by using one or the other of the following strategies (or potentially both together, although that is not discussed in detail):

(a)  **by running many copies of the same Python code at the same time, using MPI** -- on the multiple cores of one or more CPUs, or

(b) **by using a GPU** which is a highly-parallel computational device which however does not directly run Python code (or C++ code, for that matter, although a specialized hybrid language called [CUDA C++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) is often used to program GPUs).

For case (a) the individual, simultaneously-executing copies of a Python program can each take advantage of packages like NumPy and SciPy, providing parallel speedup in addition to that provided by such packages.  **TODO only true if Numpy multithreading allowed in MPI, haven't seen how to do this on PCs yet and haven't tried on Unity yet** For this approach, you need to figure out how to split your problem into many pieces that can profitably run in parallel, how the pieces will be set up, controlled, and communicate with each other, etc.

 Conversely for case (b) a  **GPU-aware Python package** like CuPy, PyTorch, or PyCUDA can be installed. The first two of these completely take care of parallelization in a manner transparent to the Python programmer, who however must keep track of which objects are on the GPU and which are on the CPU - a relatively simple thing.

It is important to distinguish between **multithreading** and **multiprocessing**:

- A **process** is an independently-running program with its own memory space and other resources. Each process can run an independent Python program. Each core of a CPU can run multiple processes, but only one at a time (i.e. serially) - running multiple processes in parallel requires multiple cores.

- A **thread** is part of a process, that can sometimes use multiple cores to run in parallel with other threads in the same process. For example BLAS which is called by NumPy to do linear algebra can use **multithreading** to run faster if multiple cores are available to the process.  We can seen this in action by running **`threadcount.py`**, which estimates the number of threads in use when NumPy is used to multiply matrices.  Here it is run on the 6-core PC [candela-20](#pcs)...
  
  ...and here it is run on the 16-core PC [candela-21](#pcs):
  
  ```
  $ python threadcount.py
  Making two 3,000 x 3,000 random matrices...
  ...took 2.094e-01s, average threads = 1.000
  Multiplying matrices 3 times...
  ...took 2.144e-01s per trial, average threads = 5.968
  ```
  
  We see that `threadcount.py` accurately estimates the number of cores, and also...

- Although a Python program can call packages like NumPy/BLAS that are sped up by doing multithreading on multiple cores, only one Python interpreter at a time can run in a process (for now - there is a [proposal](https://peps.python.org/pep-0703/) to relax this). Thus to carry out parallel *Python* operations **multiprocessing** is required. This can take several different forms:
  
  - Python has a standard package [**`multiprocessing`**](https://docs.python.org/3/library/multiprocessing.html) that can run parallel processes on the different cores of a single CPU (maybe on all the cores in the typically two CPUs in an HPC node? I’m a bit unclear on this).
  
  - The C++ package [**OpenMP**](https://www.openmp.org/) (Python bindings [**PyOMP**](https://github.com/Python-for-HPC/PyOMP)) can also run parallel processes on the different cores single CPU (or single node = typically two CPUs?).
  
  - [**MPI**](https://en.wikipedia.org/wiki/Message_Passing_Interface) can run parallel processes on the different cores of a single CPU and **also on multiple nodes connected by a network**. Implementations of MPI go under names like [**OpenMPI**](https://www.open-mpi.org/) (not to be confused with the non-MPI single-node multiprocessing package OpenMP) and [**MPICH**](https://www.mpich.org/).
    
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

A more detailed version of this table is in the section [A few of NVDIA's many GPUS, with test results](#gpu-list) below. The GPUs in candela-20 and candela-21 were actually EVGA models equivalent to the NVIDIA models listed here.

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

### Pip and Conda<a id="pip-conda"></a>

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

- [**Conda**](https://anaconda.org/anaconda/conda) combines and extends the package-management functions of **pip** and the environment-management functions of **venv**. Conda can install packages and libraries for any language, not just Python, which means Conda can install Python itself.
  
  - **Anaconda** and **Miniconda** are **distributions** of packages (many, and not so many respectively).   
  
  - It seems preferable to use Conda to install packages when they are available as Conda packages, but many packages (and more recent versions of packages) are not available as Conda packages and can only be installed using pip.
    
    - Here is an [article on using Conda and pip together](https://www.anaconda.com/blog/using-pip-in-a-conda-environment); it says **pip should be used *after* Conda**.    
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
    $ conda env list                    # list all defined environments
    $ conda create -n enew –clone eold  # create environment enew by cloning existing eold
    $ conda env remove -n p39           # get rid of environment p39
    ```
  
  - Sometimes `conda create` or `conda install` will fail with the message `Solving environment: failed`.  Tips to avoid this situation:
    
    - Include all of the packages needed in the initial `conda create` command, rather than adding them later with `conda install` (or at least all of the packages that seem to be interacting).
    - Let Conda choose the version numbers rather than specifying them.
    - If you do specify a Python version, use an earlier version more likely to be compatible with the available Conda versions of the other packages you need.
    - An IDE like Spyder has many complex dependencies. But when used only to edit files (as opposed to running them) Spyder can be run from the base or no environment, so there is no need to install it in your environments.

### Conda environments and test code used in this document<a id="envs-testcode"></a>

- The following Conda environments are created and used on a PC in this document:
  - **`p39`** (defined just above) has Python 3.9, NumPy, SciPy, etc but does not have OpenMPI, PyTorch, or CuPy.
  - **`dfs`** (defined in [Installing a local package](#local-package) below) environment for trying out  the local package `dcfuncs`.
  - **`pyt`** (defined in [Installing CUDA-aware Python packages...](#pytorch-cupy) below) adds PyTorch.
  - **`gpu`** (also defined in [Installing CUDA-aware Python packages...](#pytorch-cupy) below) adds CuPy.
- The following Conda environments are created and used on the Unity HPC cluster:
  - **`npsp`** (defined in [Using modules and Conda](#unity-modules-conda)) has NumPy, SciPy, and Matplotlib, but not CuPy.
  - **`dfs`** (also defined in [Using modules and Conda](#unity-modules-conda)) has NumPy and the local package `dcfuncs` installed.
  - **`gpu`** (defined in [Running batch jobs: `sbatch`](#run-batch)) includes CuPy, so a GPU can be used.
- The following test code is used:
  - **`threadcount.py`** uses timing to estimate the number of threads in use while Numpy is multplying matrices.
  - **`threadcount_mpi.py`** esimates the numer of threads in use in each rank of an MPI run.
  - **`gputest.py`** makes dense and sparse matrices of various sizes and floating-point types, and times operations using these matrices on the CPU and (if available) the GPU. If run in an environment without CuPy like **`p39`**, only CPU tests will be run. But if run in **`gpu`** and a GPU can be initialized, will also run GPU tests.
  - **`np-version.py`** is a very short program that imports Numpy and prints out its version.
  - **`dcfuncs`** is small package of utility functions, used in this document as an example of a Python package [installed locally](#local-package). 
- The following Apptainer definition files are used. They are all discussed in [Using Apptainer on a Linux PC](#apptainer-pc) below:
  - **`os-only.def`** makes a container that contains only the **Ubuntu OS**.
  - **`pack.def`** makes a container that contains Linux, Conda, and the **Miniconda** package distribution, and installs a few selected packages in the container.
  - **`dfs.def`** makes a container with the local package **`dcfuncs`** installed in it
  - **`gpu.def`** makes a container that imports **CuPy** so it can use a GPU.
- The following sbatch scripts are defined for use with Slurm on the Unity cluster:
  - **`noapp-nogpu.sh`** (defined in [Running batch jobs: `sbatch`](#run-batch)) runs non-Apptainer job that doesn't use a GPU.
  - **`noapp-gpu.sh`** (defined in [Using a GPU](#unity-gpu)) runs non-Apptainer job that uses a GPU.
  - **`app-nogpu.sh`** (defined in [Running a container interactively or in batch job](#unity-run-container)) runs an Apptainer (containerized) job that doesn't use a GPU.
  - **`app-gpu.sh`** (defined in [Running a container the uses a GPU](#unity-gpu-container)) runs an Apptainer job that uses a GPU.

### Installing a local package<a id="local-package"></a>

Somtimes it is convenient to write or otherwise come by a **package of Python modules** (containing class and function definitions), copy the package somewhere on the computer being used, and then make it possible to import the package from any directory on the same computer -- this is a **local package**, as opposed to a package downloaded from a repository of published packages like Anaconda or PyPi.  A way to structure such a local package is outlined in Appendix B of the cheat sheet  [Getting started with Git and GitHub](https://github.com/doncandela/gs-git).

In other sections of this document it is shown how a local package like this can be [installed on an HPC cluster](#local-package-unity) like Unity (in user space), and how it can be [installed in an Apptainer container](#local-package-container) which can then be used on a PC or on an HPC cluster.  As a starting point this section shows how a local package can be installed on a Linux PC , not using Apptainer.

- The package used for these examples is **`dcfuncs`**, a small set of utility functions that can be downloaded from [GitHub - doncandela/dcfuncs](https://github.com/doncandela/dcfuncs) -- hit `Download Zip` under the `<> Code` tab (read the comments to find out what the functions do -- not relevant for present purposes). This repository has the following structure:
  
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

## Part 1: MPI, GPU, and Apptainer on a Linux PC<a id="on-linux-pc"></a>

### MPI on a Linux PC<a id="mpi-pc"></a>

#### Why do it<a id="why-mpi-pc"></a>

Using MPI, multiple copies of a Python program can run in parallel on the cores of a PC, but the same thing can be accomplished with the [Python `multiprocessing` package](https://docs.python.org/3/library/multiprocessing.html), probably more easily (I haven't tried `multiproccesing`).

What MPI can do (and `multiprocessing` cannot do) is increase the parallelism to copies of Python running on **multiple computers connected by a network** - i.e. multiple nodes of an HPC cluster. Therefore a possible reason for developing MPI-parallel code on a PC is to enable eventual expansion to a higher degree of parallelism on an HPC cluster.

Note, however, that parallelism across all the cores of any single node of an HPC cluster could be accomplished without MPI by using the `multprocessing` package -- Unity nodes currently have up to 128 cores.

#### Installing OpenMPI<a id="install-openmpi"></a>

- The most popular open-source MPI packages seems to be **OpenMPI** and **MPICH**.  Of these, OpenMPI seems a bit more recommended/supported by the Unity HPC cluster (maybe), so for now **only OpenMPI is discussed in this document**.

- On Unity as of 1/25 the available OpenMPI modules on Unity HPC were Open MPI  4.1.6 and 5.0.3, so decided to use these same versions of OpenMPI on my PCs.

- Following commands worked  1/25 to create a conda environment **`ompi5`** on my PCs with Python 3.11.11, OpenMPI 5.0.3, Numpy 1.26.4, SciPy 1.15.1, Matplotlib 3.10.0 (but trying to use Python 3.12 or above did not work).  It also worked to use these same commands but specifying openmpi=4.1.6 to make an environment **`ompi4`**.
  
  ```
  $ conda update conda
  $ conda create -n ompi5 python=3.11
  $ conda activate ompi5
  (opmi5)..$ python --version
  3.11.11
  (ompi5)..$ conda install -c conda-forge openmpi=4.1.6 mpi4py
  (ompi5)..$ conda install numpy scipy matplotlib
  ```
  
  To make `mpirun` (and presumably other OpenMPI commands) usable must do
  
  ```
  $ sudo apt install openmpi-bin
  ```

#### Simple MPI test programs: `mpi_hw.py` and `osu_bw.py` <a id="mpi-testprogs"></a>

- **`mpi_hw.py`** tests...
  
  ```
  (ompi5)..$ mpirun -n 6 python mpi_hw.py
  Hello world from rank 0 of 6 on candela-20 running Open MPI v4.1.6
  Hello world from rank 3 of 6 on candela-20 running Open MPI v4.1.6
  Hello world from rank 5 of 6 on candela-20 running Open MPI v4.1.6
  Hello world from rank 2 of 6 on candela-20 running Open MPI v4.1.6
  Hello world from rank 4 of 6 on candela-20 running Open MPI v4.1.6
  Hello world from rank 1 of 6 on candela-20 running Open MPI v4.1.6
  ```
  
  (above could omit -n 6 and would use all 6 cores, -n 3 eg to use 3 cores, -n 7 fails.  `mpirun --use-hwthread-cpus python mpi_hw.py` will make two ranks per core using hyperthreading.

below need -n 2)

- **`osu_bw.py`** tests...
  
  ```
  (ompi5)..$ mpirun -n 2 python osu_bw.py
  2
  2
  # MPI Bandwidth Test
  # Size [B]    Bandwidth [MB/s]
           1                2.89
           2                5.50
           4               11.31
           8               23.19
          16               46.33
          32               92.53
          64              171.07
         128              316.73
         256              597.47
         512            1,297.02
       1,024            2,479.73
       2,048            4,631.86
       4,096            2,245.38
       8,192            3,837.26
      16,384            5,731.62
      32,768            7,912.04
      65,536            9,754.16
     131,072           10,643.03
     262,144            9,186.16
     524,288            8,955.39
   1,048,576            9,121.18
   2,097,152            9,196.89
   4,194,304            9,130.77
   8,388,608            8,922.07
  16,777,216            7,397.23
  ```
  
  #### A more elaborate MPI program: `boxpct.py` with the `dem21` package<a id="boxpct-dem21"></a>

TODO MPI stuff from 9/22 cheat-sheet

**TODO** here and in the HPC sections: When does an MPI program that uses eg NumPy multithread?  How can this be controlled?

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
    
    Next run the install PyTorch using the appropriate command from the [PyTorch Getting Started page](https://pytorch.org/get-started/locally/). The installation command depends on which version of CUDA is installed, if any -- since CUDA 12.2 was installed I selected the nearest version no later than 12.2 which was 12.1:
    
    ```
    (pyt-gmem)..$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    
    To check that PyTorch is usable, run this code in a JN which creates a small tensor filled with random numbers:
    
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
  
  - **Using the GPU in PyTorch.** The fundamental objects in PyTorch are **tensors**, which are NumPy arrays with lots of additional features added: the ability to store gradients, participate in backpropagation, etc.  A tensor can exist on either the CPU or the GPU, and you cannot do operations between tensors in two different places (e.g. multiply a tensor on the CPU by a tensor on the GPU).  PyTorch makes it easy to move tensors to the GPU if it exists, otherwise leave them on the CPU as this example (run in a JN) shows.
    
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
    
    This is as far as we will go with PyTorch in this document. 

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

NVIDA has made many different GPUs. This table shows includes the relatively small GPUs in my PCs, a somewhat bigger GPU available for free on Google Colab, and a few more powerful GPUs available on the Unity HPC cluster.

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
| float32 CPU /GPU:                  |                        | **0.66 /1.41 GF**       | 0.22 / 3.82 GF           | 0.23 / 13.8 GF     |                    |                  |
| **0.43 / 55 GF**                   |                        |                         |                          |                    |                    |                  |

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
  
  - **Comparison of available GPUs** using **float32** and **candela-21 GPU** as reference:
    
    - **candela-20** is **1.5 times slower for dense and sparse operations**.
    - **T4** (best GPU available free on Colab as of 11/23) is **1.9 times faster for dense operations, 1.3-3.2 times faster for sparse operations**. But **CPU operations were much slower on Colab+T4 than on candela-21.**
    - **V100** (good GPU sometimes available on Unity as of 11/23) is **3 times faster for dense operations, 4-10 times faster for sparse operations**. But again **CPU operations were slower than on candela-21.**
    - **A100** (very good GPU sometimes available on Unity as of 11/23) is **5 times faster for dense operations, 20-40 times faster for sparse operations. CPU operations were similar** to candela-21.

- **Running CuPy in a Google Colab notebook.** This was quite simple, as it seems compatible Numpy, CUDA and CuPy are installed by default:
  
  - Paste Python code that imports CuPy into a code cell (for example the GPU test program `testgpu.py` can simply be pasted into a cell).
  - Select a GPU runtime.  As of 10/23, the only GPU type available for free on Colab was the Tesla T4. Tesla V100 and Tesla A100 GPUs were available but only on a paid tier.
  - Hit run.

### Using Apptainer on a Linux PC<a id="apptainer-pc"></a>

#### Why do it<a id="why-apptainer-pc"></a>

Code that is containerized using Apptainer should be usable on various PCs without setting up environments with the correct packages (with compatible versions) installed.  However, this does require Apptainer itself to be installed on the PCs, which is not necessarily trivial or commonly done.

Probably the best reason for containerizing code is to make it easy to run the code on an HPC cluster, which is likely to have Apptainer pre-installed and ready to use (as Unity does).  In the examples below, containers developed and usable on a PC were also usable without modification on Unity.

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
  be omitted.

- Build the container image: In the directory that contains the definition file `os-only.def` do the following command. **Root privilege is necessary to build a container image, but not to use it.** (According to the Apptainer docs it should be possible to build a container without root privilege by using the `--fakeroot` option, but I haven’t tried this.)
  
  ```
  $ sudo apptainer build "$SIFS"/os-only.sif os-only.def
  ```

- This created a 30 MB image file **`os-only.sif`** ` in the directory $SIFS`. We can open a shell running inside the container using the `apptainer shell` command.  This gives a prompt `Apptainer>` from which we can issue shell commands that will run inside the container:
  
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
    
    I believe these two things: (a) the ability to run a container as a non-superuser, and (b) the ability to access files outside the container are among the primary differences between Apptainer and Docker containers, making Apptainer more suitable for use on a shared HPC system.  I think Docker containers are most frequently run in cloud-based virtual machines for which the user will have superuser access.
    
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

TODO

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

#### Why do it<a id="why-hpc"></a>

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

#### Logging in<a id="unity-login"></a>

- **Logging with ssh.** To login to Unity from a terminal program on a remote PC, **ssh keys must be set up** - here are the [instructions in the Unity docs](https://docs.unity.rc.umass.edu/documentation/connecting/ssh/).  While a bit of a pain to set up, ssh is convenient to use and is necessary to enable usage of the `scp` and `rsync` file transfer commands described below.
  
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
    
    - ssh will set up and maintain the file `~/.ssh/known_hosts`
    
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
  
  To fix this I did the `sh-keygen -f '/home/dc/.ssh/known_hosts' -R ...` command suggested in the error message. Then I followed the instructions above to generate a new key, saving it in same file used previously (overwriting previous file), and re-did the `chmod 600 ... command`.

- **Logging in with Unity OnDemand.**  The **Shell, >_Unity Shell Access** menu item of [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard) opens a login shell (in a browser window, not a terminal) without using ssh (Unity OnDemand is accessed with `netid` and `pw`). This is reasonably convenient for all platforms and also allows logging into Unity from Windows without setting up ssh keys and without software beyond a browser.  But there seem to be some limitations on what can be done from this browser-window shell.

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
    $ scp <localfile> unity:~/<subdirec>                  # copy under directory /home/<userc
    $ scp <localfile> unity:/work/pi_<direc>/<subdirec>   # copy under group working directory
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

#### Slurm on Unity<a id="unity-slurm"></a>

- Some Slurm resources:
  
  - [Quick Start User Guide](https://slurm.schedmd.com/quickstart.html) in the [Slurm docs](https://slurm.schedmd.com/documentation.html).
  - [Overview of threads, cores, and sockets in Slurm](https://docs.unity.rc.umass.edu/documentation/get-started/hpc-theory/threads-cores-processes-sockets/) in the [Unity docs](https://docs.unity.rc.umass.edu/documentation/).
  - Stanford tutorial [SLURM Basics](https://stanford-rc.github.io/docs-earth/docs/slurm-basics).
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
        #SBATCH -p=cpu,cpu-prempt
        ```
  
  - Jobs submitted to the `gpu` or `gpu-preempt` partitions will be rejected if they do not request GPUs using e.g. `#SBATCH -G=..`.

- Some useful Slurm commands:
  
  ```
  $ squeue --me                    # show info on my jobs
  $ sacct -j <jobid>               # show more detailed info on a specific job
  $ seff <jobid>                   # show utilization efficiency of a completed job
  $ sinfo -p cpu -r -l             # show status of nodes in partition cpu
  $ scontrol show partition cpu    # detailed info on partition cpu
  $ scontrol show node cpu029      # detailed info on node cpu029
  $ scontrol show config           # show slurm configuration including default values
  ```

- Some useful scripts to see current usage/availablilty of resources on Unity, [more are here](https://docs.unity.rc.umass.edu/documentation/jobs/helper_scripts/).
  
  ```
  $ unity-slurm-partition-usage  # show how many idle cores and gpus in each partition
  $ unity-slurm-node-usage       # for each node show idle cores and gpus, partition is in
  $ unity-slurm-account-usage    # show cores and gpus my group is currently using
  $ unity-slurm-find-nodes a100  # show nodes with specified constraint (here A100 GPUs)
  $ unity-slurm-find-nodes a100 | unity-slurm-node-usage    # show idleness of nodes with
                                                            # specified constraint
  ```

#### Running jobs interactively: `salloc` or `unity-compute`<a id="run-interactive"></a>

- You are not supposed to run interactive jobs on a login node.  Instead, from the login-node shell, issue a command like
  
  ```
  $ salloc -c 6 -p cpu
  ```
  
  to allocate 6 cores on a node in the `cpu` partition and start an interactive shell on that node -- you will see the node name in the prompt. Similarly to allocate 6 cores and one GPU on a node in the `gpu` partition do
  
  ```
  $ salloc -c 6 -G 1 -p gpu
  ```
  
  In either case use `ctrl-d` to exit back to the login-node shell.

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
  
  - You must start by loading the **Conda module**.  Then `conda` commands can be used to create and activate Conda environments. Both `conda install` and `pip install` can be used in a Conda environment to install packages in that environment (see [Pip and Conda](#pip-conda) above). 
  
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

- **Installing a local package on Unity.**
  
  - This is done much the same way as installing a local package on a PC, as [shown above](#local-package).
  
  - Following the same example as in that section, the repository for the **`dcfuncs`** package has been copied to a directory `work/pi_<userc>...dcfuncs` on Unity. Then a Conda environment **`dfs`** is created and NumPy and  `dcfuncs` are installed in that environment. 
    
    ```
    $ unity-compute                    # get shell on a compute node
    $ module load conda/latest
    $ conda create -n dfs python=3.12
    $ conda activate dfs
    (dfs)...$ python --version
    Python 3.12.3
    (dfs)..$ conda install numpy       # needed by dcfuncs
    (dfs)...$ cd ...dcfuncs            # go to directory where dcfuncs repo was copied
    (dfs)...dcfuncs$ ls
    LICENSE  README.md  pyproject.toml  setup.py  src  test
    (dfs)...dcfuncs$ pip install -e .  # install dcfuncs to current environment
    (dfs)...dcfuncs$ pip list
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
    $ unity-compute                     # get shell on a compute node
    $ module load conda/latest
    $ conda activate dfs
    (dfs)...$ cd ...tests               # go to directory test code for dcfuncs is located
    (dfs)...test$ ls
    test-configs.py  test-util.ipynb  test-util.py  test0.yaml  test1.yaml
    (dfs)...test$ python test-util.py   # we can run this program that imports dcfuncs
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
  ##SBATCH -p cpu-preempt        # submit job to partition cpu-preempt (this is commented out)
  module purge                  # unload all modules
  module load python/3.12       # load version of Python needed
  python myscript.py > output   # run myscript.py sending its output to a file
  ```
  
    Notes on script:
  
  - The first line `#!/bin/bash` indicates Bash should be used to interpret the file (you could use a different shell).
  
  - The sbatch script is a regular shell file except that lines that **start with `#SBATCH`** (exactly like this, in all caps) are interpreted specially by the `sbatch` command.  This means `#SBATCH` lines can be commented out by doubling the initial `#`, as shown above.
  
  - Next should come all the `#SBATCH` lines, which give information to Slurm on how to schedule the job.  Any `#SBATCH` lines after other shell commands other than comments are ignored.
  
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
    
    Using `srun` establishes a **job step** and can also launch **multiple copies of the program** as separate tasks if `#SBATCH -n=..` was used to specify more than one task – see [this page](https://groups.oist.jp/scs/advanced-slurm) for more info.
  
  - MPI programs can be run using `srun` or `mpirun`. This starts 10 copies of `myscript.py` as separate tasks:
    
    ```
    mpirun -n 10 python myscript.py > output
    ```
    
    while this sets the number of copies (MPI tasks) according to the `#SBATCH -n=...` setting:
    
    ```
    mpirun python myscript.py > output
    ```
    
    The second form is probably preferable -- I'm not sure what happens if the `#SBATCH -n=...` and `mpirun -n ..` settings disagree (probably nothing good).

- Some resources for writing sbatch scripts, especially on choosing and setting #SBATCH parameters:
  
  - In Unity docs: [Introduction to batch jobs](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/),  [Overview of threads, cores and sockets](https://docs.unity.rc.umass.edu/documentation/get-started/hpc-theory/threads-cores-processes-sockets/), and [Slurm cheat sheet](https://docs.unity.rc.umass.edu/documentation/jobs/slurm/).
  - In official Slurm docs: [sbatch options](https://slurm.schedmd.com/archive/slurm-23.11.6/sbatch.html#SECTION_OPTIONS) (there are many options).
  - A [quick-start guide](https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-quick-start-guide) from Lawrence Livermore National Lab. Warning: some things here are particular to LLNL, won’t apply to Unity.
  - [Examples of sbatch scripts for different kinds](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/) of jobs from Berkeley (same warning).
  - Some [more advanced Slurm topics](https://groups.oist.jp/scs/advanced-slurm) (array jobs, using srun in sbatch scripts to run multiple copies of one or more programs…) from Okinawa Institute of Science and Technology.

- In Slurm, a **task** can use one or more **cores** (which are called cpu’s in #SBATCH settings), but a task always lives on a single **node**, which typically means a single "computer" -- one or two CPU chips on a board with shared RAM and other resources like GPUs.  On the other hand, a single node can run multiple tasks if it has enough cores. 
  
  - For an **MPI job** (for which Slurm was originally conceived) there will be **more than one task** and **each task corresponds to an MPI rank, running an independent copy of the code.**  
  - For a **non-MPI job** there might be a **single task** (possibly multi-core), which therefore uses part or all of a **single node**.  These match the Slurm defaults: `#SBATCH n=1` gives one task, `#SBATCH N=1` gives one node.
  - [This page](https://groups.oist.jp/scs/advanced-slurm) shows how **`srun`** can be used to run multiple copies of a program that don't communicate using MPI -- thus **more than one task** (I haven't tried this).

- Here are some #SBATCH settings useful for both single-task (non-MPI) and multi-task (MPI) jobs:   
  
  ```
  #SBATCH -J=<name>           # set a name for job, otherwise will be script filename
  #SBATCH --job-name=<name>   # “ “
  
  #SBATCH -o=<ofname>         # set filename for output, otherwise will be slurm-<jobid>.out
  #SBATCH --output=<ofname>   # “ “
  #SBATCH -e=<efname>         # set filename for error output, otherwise will go to output file
  #SBATCH --error=<name>      # “ “
  
  #SBATCH –-mail-type=END     # send email to submitting user when job ends
  #SBATCH –-mail-type=ALL     # send email when job starts, ends, or fails
  
  #SBATCH -p=<pname>          # run the job on the nodes in partition <pname> 
  #SBATCH --partition=<pname> # “ “
  #SBATCH -p=<pname1>,<pname2>  # use nodes in either of two partitions
  
  #SBATCH -t=10               # set wall-time limit for job to complete to 10 minutes
  #SBATCH --time=10           # “ “
  #SBATCH -t=3:30:00          # set wall-time limit to 3.5 hours
  #SBATCH -t=2-3              # set wall-time limit to 2 days + 3 hours
  
  #SBATCH -c=6                # allocate 6 cores (not cpus!) per task
  #SBATCH --cpus-per-task=6   # “ “
  
  #SBATCH -G=1                # allocate one GPU for the whole job
  #SBATCH --gpus=1            # “ “
  
  #SBATCH --mem-per-cpu=500M  # allocate 500 MB of memory per core (not cpu!)
  
  #SBATCH -q=<qos>            # request quality of service <qos>
  #SBATCH –-qos=<qos>         # “ “
  #SBATCH -q=long             # on unity allow time limit up to 14 days
  #SBATCH -q=short            # on unity get higher priority for a single job per user
                              # on <=2 nodes with time limit <=4 hours
  
  #SBATCH -C=”<cons>”         # only use nodes that have constraint <cons>, quotes may be unnec.
  #SBATCH –-constraint=”<cons”> # “ “
  #SBATCH -C=”<cons1>&<cons2>”  # only use nodes that match both constraints
  #SBATCH -C=”v100|a100”      # on unity only use nodes that have V100 or A100 GPUs
  ```
  
  Notes on these: 
  
  - For all of the Unity general-access partitions: The **default time limit is one hour**, when `-t=..` is not used. **Jobs submitted to preempt partitions can be killed after two hours**.
  - The **maximum time limit** that can be set using `-t=..` is **two days** unless `-q=long` is used in which case it is **14 days (336 hours)**.
  - By running `scontrol config` it is seen that on Unity the **default memory per core is 1024 MB**.
  - [This page](https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/) shows the **constraints** available on Unity.
  - To set multiple constraints I think a single `#SBATCH C=..` line using the `&` or `|` operators is needed as shown above.

- Here are some #SBATCH settings more useful for **multi-task (e.g. MPI) jobs**:
  
  ```
  #SBATCH -n=100            # allocate resources for 100 tasks (100 MPI ranks)
  #SBATCH --ntasks=100      # “ “
  #SBATCH -C=ib             # only use nodes with InfiniBand networking
  #SBATCH --gpus-per-task=1 # allocate one GPU per task
  
  # To use following probably need to set node constraints:
  #SBATCH -N=10            # run the job on 10 nodes
  #SBATCH --nodes=10        # “ “
  #SBATCH --exclusive       # use entire nodes (don’t share nodes with other jobs)
  #SBATCH --mem=5G          # allocate 5 GB of memory per node
  #SBATCH –-mem=0           # allocate all available memory on nodes use
  ```
  
  The nodes on Unity are very heterogeneous, with between 12 and 192 cores per node, so I don’t think it makes sense to use -N,--nodes or --exclusive unless constraints are used to match the type of nodes used to the size of the job (tasks or cores used).  Similarly --mem sets the memory per node and I’m not sure what this means unless full nodes are used.

- **Example of a  simple batch job** not using MPI, or a GPU, or Apptainer.
  
  - As a container is not being used, a Conda environment must be set up on Unity with the needed packages. This example uses the environment **`npsp`** [set up above](#conda-hpc) with Python, Numpy, and Scipy, but not CuPy.
  
  - The program `gputest.py` described above (which won't try to use a GPU if CuPy cannot be imported) was put in a directory `/work/.../try-gputest` along with an sbatch script **`noapp-nogpu.sh`** with these contents:
    
    ```
    #!/bin/bash
    # noapp-nogpu.sh 1/16/14 D.C.
    # Sample one-task sbatch script using neither Apptainer nor GPU
    #SBATCH -c 6                        # use 6 CPU cores
    #SBATCH -p cpu                      # submit to partition cpu
    
    module purge                         # unload all modules
    module load conda/latest
    conda activate npsp                  # environment with NumPy and SciPy but not CuPy
    
    python gputest.py > npapp-nogpu.out  # run gputest.py sending its output to a file
    ```
    
    With this way of running, `sbatch` is run from the directory `try-gputest` and the output will go there as well -- this is why we are using a directory under /work. The script gputest.py will not try to use a GPU because CuPy cannot be imported in this environment.

#### Using MPI on Unity (without Apptainer)<a id="unity-mpi"></a>

TODO copy from earlier UMSC cheat sheet and update for Unity

#### Using a GPU on Unity (without Apptainer)<a id="unity-gpu"></a>

- A **Conda environment `gpu` capable of using a GPU** was created on Unity as follows (as of 1/25 it seemed the current version of Python, 3.13, was incompatible with CuPy - hence the specification here python=3.12):
  
  ```
  $ module load conda/latest
  $ conda create -n gpu python=3.12
  $ conda activate gpu
  (gpu)$ python –version
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

- **Run gputest.py on Unity interactively.**
  
  - Here we get an interactive shell with 6 cores and one GPU on a compute node in the `gpu` partition, and load a CUDA module (although CUDA typically seems to be loaded already on GPU nodes). Then we run `nvidia-smi` to check that the GPU and CUDA are available and get info on them (not sure why CUDA version reported by `nvidia-smi` doesn’t match module loaded):
    
    ```
    $ salloc -c 6 -G 1 -p gpu
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
    ```

- **A batch job using a GPU.**
  
  - As in the non-GPU background job example above, here we again run `gputest.py` in the directory `/work/...test_gpu` but now we activate the Conda evironment `gpu` with does include CuPy, so `gputest.py` will try to use a GPU.  We will also need to ensure the CUDA module is loaded, request a GPU, and run the job in a GPU partition.  So we will use an sbatch script **`noapp-gpu.sh`** with these contents:
    
    ```
    #!/bin/bash
    # noapp-gpu.sh 1/16/24 D.C.
    # Sample one-task sbatch script using a GPU but not Apptainer
    #SBATCH -c 6                  # use 6 CPU cores
    #SBATCH -G 1                  # use one GPU
    #SBATCH -p cpu                # submit to partition gpu
    
    module purge                  # unload all modules
    module load conda/latest
    module load cuda/12.6         # need CUDA to use a GPU
    conda activate gpu            # environment with NumPy, SciPy, and CuPy
    
    python gputest.py > npapp-nogpu.out   # run gputest.py sending its output to a file
    ```
    
    - The job is submitted by doing
      
      ```
      (base) try-gputest$ sbatch noapp-gpu.sh
      ```
      
      which will create the output file try-gputest/noapp-gpu.out.

### Using Apptainer on the Unity HPC cluster<a id="unity-apptainer"></a>

#### Getting container images on the cluster<a id="images-to-unity"></a>

To run on Unity, a suitable container image (`.sif` file) must be present in a Unity job I/O location under `/work/pi_<userc>`.  Note `.sif` files are typically one to several GB in size.

- An image can be built on a Linux PC as described in [Using Apptainer on a Linux PC](#apptainer-pc) above, then [tranferred to Unity](#unity-file-transfer) using `scp` (the graphical [**Unity OnDemand**](https://ood.unity.rc.umass.edu/pun/sys/dashboard) does not seem able to transfer files this big).
- It may be possible to build an image directly on Unity using the **`--fakeroot`** option to `apptainer build`, I haven’t tried this.

#### Running a container interactively or in a batch job<a id="unity-run-container"></a>

**TODO** see what files can be accessed inside a container in unity - maybe not directories in work above starting directory?

This section describes how to run a container that **does not use MPI or a GPU** -- the additional steps needed for those things are in separate sections below.

- **Running the container interactively:** Obtain a shell on a compute node, and in this shell load the Apptainer module (in fact, Apptainer typically is already loaded on Unity).  For many purposes it should not be necessary to load other modules, etc.:
  
  - Python and packages typically loaded with Conda like NumPy and SciPy should be pre-loaded in the container, all in the desired versions.
  
  - User packages installed locally should also be pre-loaded in the container.
  
  - It should not be necessary to set a Conda environment before running the container, unless this is required for code running outside the container.
    
    ```
    $ salloc -c 6 -p cpu    # Get 6 cores on a compute node in the cpu partition
    $ module load apptainer/latest
    ```
    
    Here we have made a directory on Unity and copied into it:
    
    - The container **`dsf.sif`** that was built in the section [A container with a local Python package installed](#local-package-container) that can run programs that import the **`dcfuncs`** package.
    
    - The short program `np-version.py` that imports Numpy and prints its version number.
    
    - The program `test-util.py` that imports the `dcfuncs` package and tests that it can be run.
      First we check the version of Python loaded on the Unity node we are using:
      
      ```
      $ ls
      dsf.sif  np-version.py  test-util.py
      $ python --version
      Python 3.12.3
      ```
      
      Next we run the container, which executes the commands in the `%runscript` section of the container definition file:
      
      ```
      $ chmod +x dsf.sif    # only needed if transferring container made it non-executable
      $ ./dsf.sif
      foo!
      ```
      
      Shelling into the container we see the version of Python installed inside when it was built:
      
      ```
      $ apptainer shell dsf.sif
      Apptainer> python --version
      Python 3.12.8
      Apptainer>             # ctrl-d to get out of container
      ```
      
      Finally we use Python inside the container to run the scripts `np-version.py` and `test-util.py` that are outside the container.  The first script `np-version.py` uses NumPy installed in the container when it was built, independent of what Numpy if any exists outside the container.  The second script `test-util.py` imports and uses the package `dcfuncs`, which was installed locally inside the container when it was built:
      
      ```
      $ apptainer exec dsf.sif python np-version.py
      numpy version = 2.1.3
      $ apptainer exec dsf.sif python test-util.py
      This is: dutil.py 8/19/24 D.C.
      Using: util.py 8/18/24 D.C.
          ...
      ```

- **Running a batch job using the container.**
  
  - For this purpose we have copied the python script `gputest.py` to the Unity directory that holds **`dsf.sif`**.  Because this container does not contain CuPy, if we use it to run `gputest.py` only the CPU will be used.  Here is an sbatch script called **`app-nogpu.sh`**:
    
    ```
    #!/bin/bash
    # app-nogpu.sh 1/16/24 D.C.
    # Sample one-task sbatch script using a container, but not a GPU
    #SBATCH -c 6                  # use 6 CPU cores
    #SBATCH -p cpu                # submit to partition cpu
    
    module purge                  # unload all modules
    module load apptainer/latest
    
    # run gputest.py in a container without CuPy, sending its output to a file
    apptainer exec dsf.sif python gputest.py > app-nogpu.out
    ```
    
    Notice the only module we need to load is Apptainer, and we do not need to set a Conda environment. To run the job:
    
    ```
    (base) $ sbatch app-gpu.sh    # run in directory containing dsf.sif and gputest.py
    ```

#### Running a container that uses MPI<a id="unity-mpi-container"></a>

TODO haven't tried this yet

#### Running a container the uses a GPU<a id="unity-gpu-container"></a>

- This is very similar to running a non-GPU container as described above, with these differences:
  
  - Obviously this must be done on a node with GPU(s), with a GPU allocated to the job by SLURM.
  - Both CUDA and Apptainer modules should be loaded (although both packages seem to be pre-loaded on GPU nodes).
  - The container (`.sif file`) must have been built with CUDA libraries. Installing CuPy in the container build definition seems to accomplish this.
  - Apptainer commands running the container must have the --nv flag to make the external CUDA libraries available.

- Here is an example of these things in action for **interactive use of a GPU with a container**:
  
  - An interactive shell is allocated on a compute node with 6 cores and one GPU, then CUDA and Apptainer modules are loaded (`nvidia-smi` checks that the GPU is available but is not necessary here):
    
    ```
    $ salloc -c 6 -G 1 -p gpu
    $ module load cuda/12.6
    $ module load apptainer/latest
    $ nvidia-smi
    Tue Jan 14 16:57:25 2025
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4
                                 ...
    ```
  
  - The container **`gpu.sif`** (built as described in [A container that can use a GPU](#gpu-container) above and the Python script `gputest.py` are put in a directory on Unity.  Then `apptainer exec` with the flag `--nv` is able to use the GPU:
    
    ```
    $ ls
    gpu.sif gputest.py
    $ apptainer exec --nv gpu.sif python gputest.py
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
  
  - This is similar to the non-GPU container job shown earlier, with these differences:
    
    - We request a GPU, and submit to a GPU partition.
    - In addition to Apptainer, we need to load the module for CUDA.
    - We use the container `gpu.sif` that was built including Cupy.
    - We need the `–-nv` flag on apptainer exec.
  
  - Here is an sbatch script **`app-gpu.sh`** that incorporates these changes:
    
    ```
    #!/bin/bash
    # app-gpu.sh 1/16/24 D.C
    # Sample one-task sbatch script using a container and a GPU
    
    #SBATCH -c 6                  # use 6 CPU cores
    #SBATCH -G 1                  # use one GPU
    #SBATCH -p gpu                # submit to partition gpu
    
    module purge                  # unload all modules
    module load apptainer/latest
    module load cuda/12.6         # need CUDA to use a GPU
    
    # run gputest.py in a container with CuPy, sending its output to a file
    apptainer exec --nv gpu.sif python gputest.py > app-gpu.out
    ```
    
    To run the job:
    
    ```
    (base) try-gputest$ sbatch app-gpu.sh # run in a directory containing gpu.sif and gputest.py
    ```

## Random notes on parallel speedup<a id="speedup-notes"></a>

### Wall time and CPU time<a id="wall-cpu-time"></a>

### Factors other than parallelism affecting execution speed<a id="other-speed-factors"></a>

### Strong and weak scaling<a id="strong-weak-scaling"></a>

Estimating MPI communication overhead<a id="estimate-mpi-overhead"></a>

```

```

```

```
