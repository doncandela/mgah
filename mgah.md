# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   1/20/25

- [Introduction](#intro)  
  
  - [What is this?](#what-is)
  - [Parallel computing in Python](#parcomp-python)
  - [Hardware used](#hardware)
    - [PCs](#pcs)
    - [GPUs](#gpus)
    - [Unity HPC cluster](#unity-intro)
  - [Test code used](#test-code)
    - [Pip and Conda](#pip-conda)
    - [Test programs and Conda environments used](#testprogs-conda)

- [Part 1: MPI, GPU, and Apptainer on a Linux PC](#on-linux-pc)
  
  - [MPI on a Linux PC](#mpi-pc)
    - [Why do it](#why-mpi-pc)    
    - [Installing OpenMPI](#install-openmpi)
    - [Simple MPI test programs: `mpi_hw.py` and `osu_bw.py`](#mpi-testprogs)
    - [A more elaborate MPI program: `boxpct.py` with the `dem21` package](#boxpct-dem21)
  - [Using an NVIDIA GPU on a Linux PC](#gpu-pc)
    - [Why do it](#why-gpu-pc)
    - [Installing NVIDIA drivers](#nvidia-drivers)
    - [Installing CUDA-aware Python packages: PyTorch, CuPy...](#cupy-pytorch)
    - [Testing the GPU and comparing its speed to that of the CPU](#test-gpu)
    - [A few of NVDIA's many GPUS, with test results](#gpu-list)
  - [Using Apptainer on a Linux PC](#apptainer-pc)
    - [Why do it](#why-apptainer-pc)
    - [Apptainer history](#apptainer-history)
    - [Installing Apptainer](#install-apptainer)
    - [Testing the install: An OS-only container](#os-only-container)
    - [A container including chosen Python packages](#packages-container)
    - [A container with a local Python package included and installed](#local-package-container)
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
    - [Using a GPU](#unity-gpu")
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

- A **GPU** installed in a single computer can carry out highly parallel computations, so it offers an alternative to "MPI on a cluster of computers"" for parallelizing code - but the degree of parallel operation is limited by the model of GPU that is available (unless multiple GPUs and/or GPUs on multiple MPI-connected computers are used, things not discussed in this document).

- [**Apptainer**](https://apptainer.org/) is a **container** system that allows user code and most of its dependencies (OS version, packages like NumPy) to be packaged together into a single large "image" file, which should then be usable  without modification or detailed environment configuration on many different computer systems from a Linux PC to a large cluster.

- High-performance Computing (**HPC**) typically refers to using a large cluster of connected computers assembled and maintained by Universities and other organizations for the use of their communities.  This document only discusses an HPC cluster running Linux and managed by  [**Slurm**](https://slurm.schedmd.com/overview.html) scheduling software, with  the the [**UMass Unity cluster**](https://unity.rc.umass.edu/index.php) as the specific HPC system used here.

Why Python?  Why Linux? Because those are what I use, and this is my cheat sheet.

Although there may be some information useful for the following topics, this document **does not cover:**

- Other than brief mentions, the use of OpenMP (a multithreading package not to be confused with OpenMPI) and/or the Python Mutiproccessing package for parallelization on the cores of a single computer.

- Operating systems other than Linux (Windows, macOS...).

- Computer languages other than Python such as C++.

- Direct, low-level programming of GPUs in CUDA-C++  (as opposed to the use of GPU-aware Python packages like CuPy and PyTorch, which are briefly covered).

- "Higher level" (than MPI) packages for using computer clusters such as Spark, Dask, Charm4Py/Charm++...).

- Cloud computing (Amazon Web Services, Microsoft Azure...). 

- The Docker container system, other than as a source for building Apptainer containers.

- The Kubernetes scheduling/management software typically used rather than Slurm in commercial settings, particularly with Docker.

### Parallel computing in Python<a id="parcomp-python"></a>

Python is a semi-interpreted language (compiled to a byte code, like Java) and so is much more slowly executed than a fully compiled language like C++, unless an add-on like [Numba](http://numba.pydata.org/) or [Cython](https://cython.org/) is used (neither of these is discussed further in this document, although they may certainly be useful).

Therefore good performance on large tasks is often achieved by using **packages** (typically written by others in a compiled language like C++) like [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [CuPy](https://cupy.dev/), and [PyTorch](https://pytorch.org/), to carry out the time-consuming **inner loops** of algorithms. The same is true of other high-level languages like [MATLAB](https://www.mathworks.com/products/matlab.html) and [Mathematica](https://www.mathematica.org/).   While some think Python is inherently slower than C++, if the time limiting factor is, for example, a large linear algebra operation then in either language it will likely be carried out by the same highly-optimized [BLAS](https://www.netlib.org/blas/) function on a CPU (via NumPy, for Python), or the corresponding [cuBLAS](https://developer.nvidia.com/cublas) function on a GPU (via CuPy).

There are however some murky intermediate situations. For example [Numpy advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html) allows many complicated operations (e.g. operations on elements meeting complicated conditions) on arrays to be carried out much faster than if they were coded directly in Python -- but maybe slower than would be possible in C++.

Be that as it may, the premise of this document is **speeding up Python code** by using one or the other of the following strategies (or potentially both together, although that is not discussed in detail):

(a)  **by running many copies of the same Python code at the same time** on the multiple cores of one or more CPUs, or

(b) **by using a GPU** which is a highly-parallel computational device which however does not directly run Python code (or C++ code, for that matter, although a specialized hybrid language called [CUDA C++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) is often used to program GPUs).

For case (a) the individual, simultaneously-executing copies of a Python program can each take advantage of packages like Numpy and SciPy, providing parallel speedup in addition to that provided by such packages. Conversely for case (b) packages like Numpy and SciPy are **replaced** by GPU-aware packages like CuPy and PyTorch.

it is important to distinguish between **multithreading** and **multiprocessing**:

- A **process** is an independently-running program with its own memory space and other resources. Each process can run an independent Python program. Each core of a CPU can run multiple processes, but only one at a time (i.e. serially) - running multiple processes in parallel requires multiple cores.

- A **thread** is part of a process, that can sometimes use multiple cores to run in parallel with other threads in the same process. For example BLAS which is called by Numpy to do linear algebra can use **multithreading** to run faster if multiple cores are available to the process.

- Although a Python program can call packages like Numpy/BLAS that are sped up by doing multithreading on multiple cores, only one Python interpreter at a time can run in a process (for now - there is a [proposal](https://peps.python.org/pep-0703/) to relax this). Thus to carry out parallel *Python* operations **multiprocessing** is required. This can take several different forms:
  
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

The PCs commands and test programs shown in this document were tested on one or more of these PCs (only two of these have GPUs):

- **candela-20:** Assembled from parts 9/19, 3.6 GHz AMD Ryzen 5 3600 6-core CPU, 32 GB RAM, EVGA GPU similar to NVIDIA GeForce GTX 1050 Ti, running Ubuntu 24.04.
- **candela-21:** Assembled from parts 11/22, 3.4 GHz AMD Ryzen 9 5950X 16-core CPU, 64 GB RAM, EVGA GPU similar to NVIDIA GeForce GTX 1060 6 GB VRAM, running Ubuntu 24.04.
- **hoffice:** Lenovo A10 510-23 purch 11/17, 2.4 GHz Intel Core i5 4-core CPU, 8 GB RAM, no NVIDIA-type GPU, running Ubuntu 24.04.

#### GPUs<a id="gpus"></a>

A more detailed version of this table is in the section [A few of NVDIA's many GPUS, with test results](#gpu-list) below. The GPUs in candela-20 and candela-21 were actually EVGA models equivalent to the NVIDIA models listed here.

| NVIDIA Model: | GeForce GTX 1050 Ti | GeForce GTX 1660 | Tesla T4                     | Tesla V100 DGXS    | Tesla A100 SMX4    | Hopper H100 SXM5 |
| ------------- | ------------------- | ---------------- | ---------------------------- | ------------------ | ------------------ | ---------------- |
| Where:        | candela-20 GPU      | candela-21 GPU   | Best free GPU on Colab 10/23 | Available on Unity | Available on Unity | Exists on Unity  |
| Price:        | $250 9/19           | $297 11/22       | $2000 10/23                  | $1500 10/23        | $15,00 10/23       | $30,000 10/23    |

#### Unity HPC cluster<a id="unity-intro"></a>

The HPC commands shown in this document were tested on the [Unity cluster](https://unity.rc.umass.edu/index.php) at UMass, Amherst. Unity runs the [Slurm](https://slurm.schedmd.com/overview.html) job scheduling sytem and as of 1/25 had roughly:

- 350 general-access nodes plus another 350 “preempt” nodes belonging to groups but available for general access when not otherwise being used (also additional nodes never available for general access).
- 20,000 total cores in the general-access and preempt nodes, with individual nodes mostly having two CPUs and between 24 and 192 cores.
- GPUs on half of the nodes, with individual nodes having between 2 and 8 GPUs giving about 1000 total GPUs.  The GPU models (thus capabilities) vary widely, with the newest/best GPUs on the preempt modes as might be expected.
  Detailed information on using Unity is in the section [Unity cluster at UMass, Amherst](#unity-cluster) below.

### Test code used<a id="test-code"></a>

#### Pip and Conda<a id="pip-conda"></a>

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
  
  - If a **virtual environment** has been created and activated by **venv** or by **conda**, pip will install Python packages “in that environment”, i.e. they will only be visible and importable when that environment is activated.
  
  - You often see `python -m pip install package` used rather than the simpler commands listed above. I haven’t found this necessary when using pip inside a virtual environment; a discussion is [here](https://snarky.ca/why-you-should-use-python-m-pip/).

- [**Conda**](https://anaconda.org/anaconda/conda) combines and extends the package-management functions of **pip** and the environment-management functions of **venv**. Conda can install packages and libraries for any language, not just Python, which means Conda can install Python itself.
  
  - **Anaconda** and **Miniconda** are **distributions** of packages (many, and not so many respectively).
  
  - It seems preferable to use Conda to install packages when they are available as Conda packages, but many packages (and more recent versions of packages) are not available as Conda packages and can only be installed using pip. Here is an [article on using Conda and pip together](https://www.anaconda.com/blog/using-pip-in-a-conda-environment); it says **pip should be used *after* Conda**.
  
  - Some Conda commands:
    
    ```
    $ conda update conda                 # good to do this before other conda commands
    $ conda create -n p39 python=3.9.12  # create environment p39 with specific Python version
    $ conda activate p39                 # activate environment p39, will change prompt
    (p39)$ conda install matplotlib numba numpy pyyaml scipy  # install packages to current
                                                              # environment p39
    (p39)$ conda install spyder=5.2.2 # install Spyder IDE to this evironment
    (p39)$ conda install -c conda-forge quaternion            # install a package from the
                                                              # Conda-Forge repository
    (p39)$ conda search packagename     # search for available packages with specified name
    $ conda env list                    # list all defined environments
    $ conda create -n enew –clone eold  # create environment enew by cloning existing eold
    $ conda env remove -n p39           # get rid of environment p39
    ```
  
  - Sometimes `conda create` or `conda install` will fail with the message `Solving environment: failed`.  Tips to avoid this situation:
    
    - Include all of the packages needed in the initial `conda create` command, rather than adding them later with `conda install` (or at least all of the packages that seem to be interacting).
    - Let Conda choose the version numbers rather than specifying them.
    - If you do specify a Python version, use an earlier version more likely to be compatible with the available Conda versions of the other packages you need.
    - An IDE like Spyder has many complex dependencies. But when used only to edit files (as opposed to running them) Spyder can be run from the base or no environment, so there is no need to install it in your environments.

#### Test programs and Conda environments used<a id="testprogs-conda"></a>

- TODO add here as added below

## Part 1: MPI, GPU, and Apptainer on a Linux PC<a id="on-linux-pc"></a>

### MPI on a Linux PC<a id="mpi-pc"></a>

#### Why do it<a id="why-mpi-pc"></a>

Using MPI, multiple copies of a Python program can run in parallel on the cores of a PC, but the same thing can be accomplished with the [Python `multiprocessing` package](https://docs.python.org/3/library/multiprocessing.html), probably more easily (I haven't tried `multiproccesing`).

What MPI can do (and `multiprocessing` cannot do) is increase the parallelism to copies of Python running on **multiple computers connected by a network** - i.e. multiple nodes of an HPC cluster. Therefore a possible reason for developing MPI-parallel code on a PC is to enable eventual expansion to a higher degree of parallelism on an HPC cluster.

Note, however, that parallelism across all the cores of any single node of an HPC cluster could be accomplished without MPI by using the `multprocessing` package.  (Unity nodes currently have up to 128 cores.)

#### Installing OpenMPI<a id="install-openmpi"></a>

TODO MPI stuff from 9/22 cheat-sheet

#### Simple MPI test programs: `mpi_hw.py` and `osu_bw.py` <a id="mpi-testprogs"></a>

TODO MPI stuff from 9/22 cheat-sheet

#### A more elaborate MPI program: `boxpct.py` with the `dem21` package<a id="boxpct-dem21"></a>

TODO MPI stuff from 9/22 cheat-sheet

**TODO** here and in the HPC sections: When does an MPI program that uses eg Numpy multithread?  How can this be controlled?

### Using an NVIDIA GPU on a Linux PC<a id="gpu-pc"></a>

#### Why do it<a id="why-gpu-pc"></a>

A relatively inexpensive GPU can offer significant speedups. For example in [test results](#gpu-list) on the candela-21 PC assembled in 2022 the \$300 GPU was about four times faster than the \$550 16-core CPU chip for operations on large dense and sparse matrices.

If the code might eventually be transferred to an HPC cluster, the more capable GPUs on the HPC nodes should offer even greater speed-ups than this.

#### Installing NVIDIA drivers<a id="nvidia-drivers"></a>

These steps are only need once on a given PC, unless updating to newer versions.

- Find out if (and which) NVIDIA GPU hardware is installed:
  
  ```
  $ sudo lshw -C display
  ```

- By default Ubuntu will use the open-source X driver for NVIDIA hardware, but this seems not to be appropriate when using the GPU for computation, and sometimes even just for graphics (e.g. Mayavi did not function correctly on candela-21 with the X driver).

- It’s possible to [download and install drivers from NVIDIA](https://www.nvidia.com/download/index.aspx) but this will require stopping the X driver, etc (didn’t try).

- Instead it’s easier to install from Linux channels following e.g. [instructions here](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/). This can be done in Terminal, but easiest to use the Ubuntu graphical **Software and Updates** app: The **Additional Drivers** tab should show the GPU and available drivers (from NVIDIA as well as X) – pick most recent (proprietary, tested) NVIDIA driver, Apply Changes, then when completed reboot system.

- Now you can use the text-based and/or graphical apps supplied by NVIDIA to check that driver is functioning and get info on the GPU:
  
  ```
  $ nvidia-smi         # text-based
  $ nvidia-settings    # graphical
  ```
  
  Note the **CUDA Version** from nvida-smi (was 12.0 in 10/23)

#### Installing  CUDA-aware Python packages: PyTorch, CuPy...<a id="pytorch-cupy"></a>

#### Testing the GPU and comparing its speed to that of the CPU<a id="test-gpu"></a>

#### A few of NVDIA's many GPUS, with test results<a id="gpu-list"></a>

### Using Apptainer on a Linux PC<a id="apptainer-pc"></a>

#### Why do it<a id="why-apptainer-pc"></a>

Code that is containerized using Apptainer should be usable on various PCs without setting up environments with the correct packages (with compatible versions) installed.  However, this does require Apptainer itself to be installed on the PCs, which is not necessarily trivial or commonly done.

Probably the best reason for containerizing code is to make it easy to run the code on an HPC cluster, which is likely to have Apptainer pre-installed and ready to use (as Unity does).  In the examples below, containers developed and usable on a PC were also usable without modification on Unity.

#### Apptainer history<a id="apptainer-history"></a>

#### Installing Apptainer<a id="install-apptainer"></a>

#### Testing the install: An OS-only container<a id="os-only-container"></a>

#### A container including chosen Python packages<a id="packages-container"></a>

#### A container with a local Python package included and installed<a id="local-package-container"></a>

#### A container that can use MPI<a id="mpi-container"></a>

#### A container that can use a GPU<a id="gpu-container"></a>

## Part 2: Moving code to a Slurm HPC cluster<a id="move-to-hpc"></a>

#### Why do it<a id="why-hpc"></a>

In general one uses an HPC cluster to get more computational power (e.g. if doing simulations carry out bigger and/or more simulations) than might be otherwise possible, due to one or more of the following factors:

- For code can only run on one computer but can use multiple cores, an HPC cluster typically will have computers with high core count (up to 128 cores per node on Unity).
- For code that can use MPI to run on more than one networked computer, an HPC cluster can offer even larger core counts since by a cluster consists of many computers networked together.
- For code that can use a GPU to speed up computations, an HPC cluster may include higher-performance models of GPU than otherwise available.
- For code that requires a lot of memory, HPC clusters are typically configured with considerable memory.
- Even if none of the situations above is true, multiple jobs can be run simultaneously on an HPC cluster.

However, the individual CPUs in an HPC cluster are typically no faster than those in a desktop PC (sometimes slower, as HPC computers are optimized for reliability) -- so there may be little advantage to running code that is not parallelized with either MPI or utilization of a GPU on an HPC cluster, apart from the higher core count of individual computers and the possibility of running multiple jobs simultaneously.

Finally, the computational resources of an HPC cluster are only useful if available within reasonably short times waiting in queue. Jobs requiring many computers or the fastest GPUs may sit a long time before starting.

### Unity cluster at UMass, Amherst<a id="unity-cluster"></a>

#### History<a id="unity-history"></a>

#### Logging in<a id="unity-login"></a>

#### Storage<a id="unity-storage"></a>

#### Transferring files to/from Unity <a id="unity-file-transfer"></a>

#### Slurm on Unity<a id="unity-slurm"></a>

#### Running jobs interactively: `salloc` or `unity-compute`<a id="run-interactive"></a>

#### Using `.bashrc` and `.bash_aliases`<a id="rc-files"></a>

#### Using modules and Conda<a id="unity-modules-conda"></a>

#### Running batch jobs: `sbatch`<a id="run-batch"></a>

#### Using MPI<a id="unity-mpi"></a>

#### Using a GPU<a id="unity-gpu"></a>

### Using Apptainer on the Unity HPC cluster<a id="unity-apptainer"></a>

#### Getting container images on the cluster<a id="images-to-unity"></a>

#### Running a container interactively or in a batch job<a id="unity-run-container"></a>

#### Running a container that uses MPI<a id="unity-mpi-container"></a>

#### Running a container the uses a GPU<a id="unity-gpu-container"></a>

## Random notes on parallel speedup<a id="speedup-notes"></a>

### Wall time and CPU time<a id="wall-cpu-time"></a>

### Factors other than parallelism affecting execution speed<a id="other-speed-factors"></a>

### Strong and weak scaling<a id="strong-weak-scaling"></a>

### Estimating MPI communication overhead<a id="estimate-mpi-overhead"></a>
