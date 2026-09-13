# PDC
## Repository

Private Git repository:

https://github.com/phanhuynhhagiang0508123456789-code/PDC.git


# MPI N-Body Simulation

## Overview

This project implements a two-dimensional N-body simulation using MPI.

The project is divided into two implementations:

- **Part 1A:** Replaces the baseline timestep `MPI_Allgather` with explicit point-to-point ring communication.
- **Part 1B:** Extends the ring design to reduce per-process memory usage so that each MPI process permanently stores only the body data that it owns.

The simulation calculates the gravitational forces between particles and updates their positions and velocities using Euler's method.


# Compilation and Execution

## Requirements

The program requires:

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

---

## Compile the Baseline

```bash
mpicc -g -Wall -o mpi_nbody_basic mpi_nbody_basic.c -lm
```

---

## Compile Part 1A

```bash
mpicc -g -Wall -o part1a part1a.c -lm
```

---

## Compile Part 1B

```bash
mpicc -g -Wall -o part1b part1b.c -lm
```

The `-lm` option links the C mathematics library required by the gravitational force calculations.

---

# Running the Programs

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

The total number of particles must be evenly divisible by the number of MPI processes.

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


# Disabling Output for Performance Testing

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

---


# Summary

Part 1A replaces the timestep `MPI_Allgather` with explicit point-to-point ring communication.

Part 1B further redesigns the program so that each MPI process permanently stores only its locally owned masses, positions, velocities, and forces.

Remote masses and positions circulate around the MPI ring during force calculation. Each process accumulates the contributions from these blocks into the forces acting on its local particles.

This design reduces per-process memory usage from storage that includes global arrays proportional to `n` to storage primarily proportional to `n/P`, at the cost of communicating masses in addition to positions.

Complete simulation output is still supported by temporarily gathering positions and velocities on process 0 when output is required.
