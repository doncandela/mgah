# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   1/19/25

- [Introduction](#intro)
  
  - [What is this?](#what-is)
  - [Types of parallel computing](#parcomp)
  - [Hardware and test code](#hardware-test)
    - [PCs used](#pcs)
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
    - [Installing a CUDA-aware Python package: CuPy, PyTorch...](#cupy-pytorch)
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

### Factors other than parallelism affecting execution speed<a id="other-speed-factors"></a>

### Strong and weak scaling<a id="strong-weak-scaling"></a>

### Estimating MPI communication overhead<a id="estimate-mpi-overhead"></a>

## Introduction <a id="intro"></a>

### What is this?<a id="what-is"></a>

This the cheat sheet I recorded as I learned to combine several tools for **parallel computing** in **Python** on various **Linux** computer systems:

- **MPI** allows multiple instances of Python to operate in parallel and communicate with each other, in the cores of a single computer or a cluster of connected computers. Code written to parallelize using MPI can utilize all the cores of a desktop computer and also scale to a larger number of cores in an HPC computer cluster.

- A **GPU** installed in a single computer can carry out highly parallel computations, so it offers an alternative to "MPI on a cluster of computers"" for parallelizing code - but the degree of parallel operation is limited by the model of GPU that is available (unless multiple GPUs and/or GPUs on multiple MPI-connected computers are used, things not discussed in this document).

- **Apptainer** is a **container** system that allows user code and most of its dependencies (OS version, packages like NumPy) to be packaged together into a single large "image" file, which should then be usable  without modification or detailed environment configuration on many different computer systems from a Linux PC to a large cluster.

- High-performance Computing (**HPC**) typically refers to using a large cluster of connected computers assembled and maintained by Universities and other organizations for the use of their communities.  This document only discusses an HPC cluster running Linux and managed by  **Slurm** scheduling software, with  the the **UMass Unity cluster** as the specific HPC system described here.

Although there may be some information useful for the following topics, this document **does not cover:**

- Other than brief mentions, the use of OpenMP (a multithreading package not to be confused with OpenMPI) and/or the Python Mutiproccessing package for parallelization on the cores of a single computer.

- Operating systems other than Linux (Windows, macOS...).

- Computer languages other than Python such as C++.

- Direct, low-level programming of GPUs in CUDA-C++  (as opposed to the use of GPU-aware Python packages like CuPy and PyTorch, which are briefly covered).

- "Higher level" (than MPI) packages for using computer clusters such as Spark, Dask, Charm4Py/Charm++...).

- Cloud computing (Amazon Web Services, Microsoft Azure...). 

- The Docker container system, other than as a source for building Apptainer containers.

- The Kubernetes scheduling/management software typically used rather than Slurm in commercial settings, particularly with Docker.

### Types of parallel computing<a id="parcomp"></a>

### Hardware and test code<a id="hardware-test"></a>

#### PCs used<a id="pcs"></a>

#### Pip and Conda<a id="pip-conda"></a>

#### Test programs and Conda environments used<a id="testprogs-conda"></a>

## Part 1: MPI, GPU, and Apptainer on a Linux PC<a id="on-linux-pc"></a>

### MPI on a Linux PC<a id="mpi-pc"></a>

#### Why do it<a id="why-mpi-pc"></a>

Using MPI, multiple copies of a Python program can run in parallel on the cores of a PC, but the same thing can be accomplished with the [Python `multiprocessing` package](https://docs.python.org/3/library/multiprocessing.html), probably more easily (I haven't tried `multiproccesing`).

What MPI can do (and `multiprocessing` cannot do) is increase the parallelism to copies of Python running on **multiple computers connected by a network** - i.e. multiple nodes of an HPC cluster. Therefore a possible reason for developing MPI-parallel code on a PC is to enable eventual expansion to a higher degree of parallelism on an HPC cluster.

Note, however, that parallelism across all the cores of any single node of an HPC cluster can be accomplished without MPI. (Unity nodes currently have up to 128 cores.)

#### Installing OpenMPI<a id="install-openmpi"></a>

#### Simple MPI test programs: `mpi_hw.py` and `osu_bw.py` <a id="mpi-testprogs"></a>

#### A more elaborate MPI program: `boxpct.py` with the `dem21` package<a id="boxpct-dem21"></a>

### Using an NVIDIA GPU on a Linux PC<a id="gpu-pc"></a>

#### Why do it<a id="why-gpu-pc"></a>

A relatively inexpensive GPU can offer significant speedups. For example in [test results](#gpu-list) on a PC assembled in 2022 the \$300 GPU was about four times faster than the \$550 16-core CPU chip for operations on large dense and sparse matrices.

If the code might eventually be transferred to an HPC cluster, the more capable GPUs on the HPC nodes should offer greater speed-ups than this.

#### Installing NVIDIA drivers<a id="nvidia-drivers"></a>

#### Installing a CUDA-aware Python package: CuPy, PyTorch...<a id="cupy-pytorch"></a>

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

However, the individual CPUs in an HPC cluster are typically no faster than those in a desktop PC (sometimes slower, as HPC computers are optimized for reliability) -- so there may be little advantage to running code that is not parallelized with either MPI or utilization of a GPU on an HPC cluster, apart from the possibility of running multiple jobs simultaneously.

FInally, the computational resources of an HPC cluster are only useful if available within reasonably short times waiting in queue. Jobs requiring many computers or the fastest GPUs may sit a long time before starting.

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
