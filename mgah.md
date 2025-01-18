# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   1/18/25

- [Introduction](#intro)
  - [What is this?](#what-is)
  - [Types of parallel computing](#parcomp)
  - [PCs and test programs used](#pcs-test)
- [Part 1: MPI, GPU, and Apptainer on a Linux PC](#on-pc)
- [Part 2: Moving code to a Slurm HPC cluster (Unity)](#on-hpc)

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

### Hardware and test code<a id="pcs-test"></a>

#### PCs used<a id="pcs-test"></a>

#### Pip and Conda<a id="pip-conda"></a>

#### Test programs and Conda environments<a id="pcs-test"></a>

[A comment?] : #

## Part 1: MPI, GPU, and Apptainer on a Linux PC<a id="on-pc"></a>

### MPI on a PC<a id="mpi-pc"></a>

#### Install OpenMPI on a PC<a id="install-openmpi"></a>

#### Simple MPI test programs: `mpi_hw.py` and `osu_bw.py` <a id="pcs-test"></a>

#### A more elaborate MPI program: `boxpct.py` with the `dem21` package<a id="pcs-test"></a>

## Part 2: Moving code to a Slurm HPC cluster (Unity)<a id="on-hpc"></a>

Part 1: foo
