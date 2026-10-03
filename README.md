# PDC

## Repository

Private Git repository:

https://github.com/phanhuynhhagiang0508123456789-code/PDC.git


# Parallel N-Body Simulation

## Overview

This project implements a two-dimensional N-body simulation using both
MPI for distributed-memory parallelism and OpenMP for shared-memory
parallelism.

The project is divided into the following implementations:

### Part 1 — MPI

- **Part 1A:** Replaces the baseline timestep `MPI_Allgather` with explicit
  point-to-point ring communication.
- **Part 1B:** Extends the ring design to reduce per-process memory usage so
  that each MPI process permanently stores only the body data that it owns.

### Part 2 — OpenMP

- **Part 2A.1 – Critical:** Parallelises the shared-force implementation and
  protects conflicting force updates using an OpenMP `critical` region.
- **Part 2A.2 – Locks:** Protects conflicting force updates using OpenMP
  locks.
- **Part 2B – Basic:** Basic OpenMP N-body implementation.
- **Part 2B – Reduced Default:** Reduced-force implementation using default
  OpenMP scheduling.
- **Part 2B – Reduced Forces Cyclic:** Reduced-force implementation using
  cyclic scheduling for the force-calculation loop.
- **Part 2B – Reduced All Cyclic:** Reduced-force implementation using cyclic
  scheduling for all relevant parallel loops.

The simulation calculates gravitational forces between particles and updates
their positions and velocities using Euler's method.


# Part 1 — MPI

## Requirements

The MPI programs require:

- A C compiler
- An MPI implementation such as Open MPI
- `mpicc`
- `mpiexec`

On macOS with Homebrew, Open MPI can be installed using:

```bash
brew install open-mpi
```

The installation can be checked using:

```bash
mpicc --version
mpiexec --version
```


## MPI Compilation

### Compile the Baseline

```bash
mpicc -g -Wall -o mpi_nbody_basic mpi_nbody_basic.c -lm
```

### Compile Part 1A

```bash
mpicc -g -Wall -o part1a part1a.c -lm
```

### Compile Part 1B

```bash
mpicc -g -Wall -o part1b part1b.c -lm
```

The `-lm` option links the C mathematics library required by the
gravitational force calculations.


## Running the MPI Programs

The command-line format is:

```bash
mpiexec -n <processes> ./<program> <particles> <timesteps> <timestep_size> <output_frequency> <g|i>
```

The arguments are:

| Argument | Description |
|---|---|
| `processes` | Number of MPI processes |
| `particles` | Total number of particles |
| `timesteps` | Number of simulation timesteps |
| `timestep_size` | Size of each timestep |
| `output_frequency` | Number of timesteps between printed states |
| `g` | Generate initial conditions |
| `i` | Read initial conditions from standard input |

For the MPI implementations, the total number of particles must be evenly
divisible by the number of MPI processes.

For example:

```bash
mpiexec -n 4 ./part1b 8 10 0.01 1 g
```

This runs:

```text
4 MPI processes
8 particles
10 timesteps
0.01 timestep size
output every timestep
generated initial conditions
```


## Disabling MPI Output for Performance Testing

Simulation output can be disabled by defining `NO_OUTPUT` during compilation.

For Part 1A:

```bash
mpicc -g -Wall -DNO_OUTPUT -o part1a part1a.c -lm
```

For Part 1B:

```bash
mpicc -g -Wall -DNO_OUTPUT -o part1b part1b.c -lm
```

The program can then be executed normally:

```bash
mpiexec -n 4 ./part1b 8 10 0.01 1 g
```

Only the elapsed execution time will be reported.


# Part 2 — OpenMP

## Requirements

The OpenMP programs require an OpenMP-compatible C compiler.

On macOS, the implementations were compiled using Homebrew GCC 16.

The compiler can be checked using:

```bash
gcc-16 --version
```


## OpenMP Compilation

### Part 2A.1 — Critical

```bash
gcc-16 -g -Wall -fopenmp -o part2a_critical part2a_critical.c -lm
```

### Part 2A.2 — Locks

```bash
gcc-16 -g -Wall -fopenmp -o part2a_locks part2a_locks.c -lm
```

### Part 2B — Basic

```bash
gcc-16 -g -Wall -fopenmp -o omp_nbody_basic omp_nbody_basic.c -lm
```

### Part 2B — Reduced Default

```bash
gcc-16 -g -Wall -fopenmp -o omp_nbody_red_default omp_nbody_red_default.c -lm
```

### Part 2B — Reduced Forces Cyclic

```bash
gcc-16 -g -Wall -fopenmp -o omp_nbody_red_forces_cyclic omp_nbody_red_forces_cyclic.c -lm
```

### Part 2B — Reduced All Cyclic

```bash
gcc-16 -g -Wall -fopenmp -o omp_nbody_red_all_cyclic omp_nbody_red_all_cyclic.c -lm
```


## Running the OpenMP Programs

The OpenMP command-line format is:

```bash
./<program> <threads> <particles> <timesteps> <timestep_size> <output_frequency> <g|i>
```

The arguments are:

| Argument | Description |
|---|---|
| `threads` | Number of OpenMP threads |
| `particles` | Total number of particles |
| `timesteps` | Number of simulation timesteps |
| `timestep_size` | Size of each timestep |
| `output_frequency` | Number of timesteps between printed states |
| `g` | Generate initial conditions |
| `i` | Read initial conditions from standard input |

For example, the lock implementation can be run using:

```bash
./part2a_locks 4 16 5 0.01 1 g
```

This runs:

```text
4 OpenMP threads
16 particles
5 timesteps
0.01 timestep size
output every timestep
generated initial conditions
```


## Disabling OpenMP Output for Performance Testing

Output can be disabled by defining `NO_OUTPUT` during compilation.

For example, Part 2A.2 can be compiled using:

```bash
gcc-16 -g -Wall -fopenmp -DNO_OUTPUT \
    -o part2a_locks_no_output part2a_locks.c -lm
```

It can then be executed normally:

```bash
./part2a_locks_no_output 4 16 5 0.01 1 g
```

Only the elapsed execution time will be reported.


# Part 2B Performance Testing

For performance testing, the Part 2B implementations can be compiled with
optimisation and simulation output disabled.

### Basic

```bash
gcc-16 -O2 -Wall -fopenmp -DNO_OUTPUT \
    -o bench_basic omp_nbody_basic.c -lm
```

### Reduced Default

```bash
gcc-16 -O2 -Wall -fopenmp -DNO_OUTPUT \
    -o bench_red_default omp_nbody_red_default.c -lm
```

### Reduced Forces Cyclic

```bash
gcc-16 -O2 -Wall -fopenmp -DNO_OUTPUT \
    -o bench_red_forces_cyclic omp_nbody_red_forces_cyclic.c -lm
```

### Reduced All Cyclic

```bash
gcc-16 -O2 -Wall -fopenmp -DNO_OUTPUT \
    -o bench_red_all_cyclic omp_nbody_red_all_cyclic.c -lm
```


## Running the Part 2B Benchmark

The benchmark script can be made executable and run using:

```bash
chmod +x run_part2b_benchmarks.sh
./run_part2b_benchmarks.sh
```

The benchmark tests the OpenMP implementations using multiple thread counts
and repeated runs.

The measured execution times are written to:

```text
part2b_results.csv
```


# Implementation Summary

## Part 1A

Part 1A replaces the timestep `MPI_Allgather` with explicit point-to-point
ring communication.

Each process sends position blocks around a logical ring using point-to-point
MPI communication. The original owner of each received block is tracked so
that the received positions can be placed correctly.

## Part 1B

Part 1B redesigns the MPI program so that each process permanently stores only
its locally owned masses, positions, velocities, and forces.

Remote masses and positions circulate around the MPI ring during force
calculation. Each process accumulates the contributions from these blocks into
the forces acting on its local particles.

This reduces per-process memory usage from storage that includes global arrays
proportional to `n` to storage primarily proportional to `n/P`, at the cost of
communicating masses in addition to positions.

Complete simulation output is still supported by temporarily gathering
positions and velocities on process 0 when output is required.

## Part 2A

Part 2A parallelises the shared-force N-body implementation using OpenMP.

The critical implementation protects conflicting updates to the shared force
array using an OpenMP `critical` region.

The lock implementation instead uses OpenMP locks to protect shared force
updates while allowing updates to different particle force entries to be
synchronised independently.

## Part 2B

Part 2B compares the basic OpenMP implementation with reduced-force
implementations using different scheduling strategies.

The implementations tested are:

- Basic
- Reduced Default
- Reduced Forces Cyclic
- Reduced All Cyclic

Performance results are collected across multiple thread counts and repeated
runs so that runtime, speedup, and parallel efficiency can be compared.
