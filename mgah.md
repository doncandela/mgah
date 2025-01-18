# My cheat sheet for MPI, GPU, Apptainer, and HPC

mgah.md  D. Candela   1/17a/25

- [Introduction](#intro)
  - [What is this?](#whatis)
  - [What are these things?](#whatare)
  - [Abbreviations, etc. in this document](#abbrev)
- [Part 1: Git+Github without the CLI](#part1)

## Introduction <a id="intro"></a>

### What is this?<a id="whatis"></a>

This the cheat sheet I recorded as I learned to combine several tools for **parallel computing** in **Python** on various **Linux** computer systems:

- **MPI** allows multiple instances of Python to operate in parallel and communicate with each other, in the cores of a single computer or a cluster of connected computers. Code written to parallelize using MPI can utilize all the cores of a desktop computer and also scale to a larger number of cores in an HPC computer cluster.

- A **GPU** installed in a single computer can carry out highly parallel computations, so it offers an alternative to "MPI on a cluster of computers"" for parallelizing code - but the degree of parallel operation is limited by the model of GPU that is available (unless multiple GPUs and/or GPUs on multiple MPI-connected computers are used, things not discussed in this document).

- **Apptainer** is a **container** system that allows user code and most of its dependencies (OS version, packages like NumPy) to be packaged together into a single large "image" file, which should then be usable  without modification or detailed environment configuration on many different computer systems from a Linux PC to a large cluster.

- High-performance Computing (**HPC**) typically refers to using a large cluster of connected computers assembled and maintained by Universities and other organizations for the use of their communities.  This document only discusses an HPC cluster running Linux and managed by  **Slurm** scheduling software, with  the the **UMass Unity cluster** as the specific HPC system described here.

Although there may be some information useful for the following topics, this document **does not cover:**

- Other than brief mentions, the use of OpenMP (a multithreading package not to be confused with OpenMPI) and/or the Python Mutiproccessing package for parallelization on the cores of a single computer.

- Operating systems other than Linux (Windows, MacOS...).

- Computer languages other than Python.

- Direct, low-level programming of GPUs in CUDA  (as opposed to the use of GPU-aware Python packages like CuPy and PyTorch, which is covered).

- "Higher level" packages for using computer clusters such as Spark, Dask, Charm4Py/Charm++...).

- Cloud computing (Amazon Web Services, Microsoft Azure...). 

- The Docker container system, other than as a source for building Apptainer containers.

- The Kubernetes scheduling/management software typically used rather than Slurm in commercial settings, particularly with Docker.

## Part 1: foo<a id="part1"></a>
