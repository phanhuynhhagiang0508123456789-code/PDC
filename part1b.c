/* File:     mpi_nbody_basic.c
 * Purpose:  Implement a 2-dimensional n-body solver that uses the 
 *           basic algorithm.  This version uses an in-place Allgather
 *
 * Compile:  mpicc -g -Wall -o mpi_nbody_basic mpi_nbody_basic.c -lm
 *           To turn off output (e.g., when timing), define NO_OUTPUT
 *           To get verbose output, define DEBUG
 *
 * Run:      mpiexec -n <number of processes> ./mpi_nbody_basic
 *              <number of particles> <number of timesteps>  <size of timestep> 
 *              <output frequency> <g|i>
 *              'g': generate initial conditions using a random number
 *                   generator
 *              'i': read initial conditions from stdin
 *              number of particles should be evenly divisible by the number
 *                 of MPI processes
 *           A stepsize of 0.01 seems to work well with automatically
 *           generated data.
 *
 * Input:    If 'g' is specified on the command line, none.  
 *           If 'i', mass, initial position and initial velocity of 
 *              each particle
 * Output:   If the output frequency is k, then position and velocity of 
 *              each particle at every kth timestep.  This value is
 *              ignored (but still necessary) if NO_OUTPUT is defined
 *
 *    for each timestep t {
 *       for each particle i I own
 *          compute F(i), the total force on i
 *       for each particle i I own
 *          update position and velocity of i using F(i) = ma
 *       Allgather positions
 *       if (output step) {
 *          Allgather velocities
 *          Output new positions and velocities
 *       }
 *    }
 *
 * Force:    The force on particle i due to particle k is given by
 *
 *    -G m_i m_k (s_i - s_k)/|s_i - s_k|^3
 *
 * Here, m_j is the mass of particle j, s_j is its position vector
 * (at time t), and G is the gravitational constant (see below).  
 *
 * Note that the force on particle k due to particle i is 
 * -(force on i due to k).  So we could approximately halve the number 
 * of force computations.  This version of the program does not
 * exploit this.
 *
 * Integration:  We use Euler's method:
 *
 *    v_i(t+1) = v_i(t) + h v'_i(t)
 *    s_i(t+1) = s_i(t) + h v_i(t)
 *
 * Here, v_i(u) is the velocity of the ith particle at time u and
 * s_i(u) is its position.
 *
 * Notes:
 * 1.  Each process stores the masses of all the particles:  the
 *     masses array has dimension n = number of particles.
 *
 * IPP:  Section 6.1.9 (pp. 290 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

typedef double vect_t[DIM];  /* Vector type for position, etc. */

/* Global variables.  Except or vel all are unchanged after being set */
const double G = 6.673e-11;  /* Gravitational constant. */
                             /* Units are m^3/(kg*s^2)  */
int my_rank, comm_sz;
MPI_Comm comm;
MPI_Datatype vect_mpi_t;

/* Scratch array used by process 0 for global velocity I/O */
vect_t *vel = NULL;

void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(double loc_masses[], vect_t loc_pos[],
      vect_t loc_vel[], int n, int loc_n);
void Gen_init_cond(double loc_masses[], vect_t loc_pos[],
      vect_t loc_vel[], int n, int loc_n);
void Output_state(double time, double masses[], vect_t pos[],
      vect_t loc_vel[], int n, int loc_n);
void Compute_force(int loc_part, double masses[], vect_t loc_forces[], 
      vect_t pos[], int n, int loc_n);
void Accumulate_force_block(double loc_masses[], vect_t loc_pos[],
      vect_t loc_forces[], double block_masses[], vect_t block_pos[],
      int loc_n, int owner) {
   int loc_part, k;
   int target_global, source_global;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

   for (loc_part = 0; loc_part < loc_n; loc_part++) {
      target_global = my_rank * loc_n + loc_part;

      for (k = 0; k < loc_n; k++) {
         source_global = owner * loc_n + k;

         /* Avoid self-interaction */
         if (target_global != source_global) {
            f_part_k[X] = loc_pos[loc_part][X] - block_pos[k][X];
            f_part_k[Y] = loc_pos[loc_part][Y] - block_pos[k][Y];

            len = sqrt(f_part_k[X] * f_part_k[X]
                     + f_part_k[Y] * f_part_k[Y]);
            len_3 = len * len * len;

            mg = -G * loc_masses[loc_part] * block_masses[k];
            fact = mg / len_3;

            loc_forces[loc_part][X] += f_part_k[X] * fact;
            loc_forces[loc_part][Y] += f_part_k[Y] * fact;
         }
      }
   }
}
void Update_part(int loc_part, double masses[], vect_t loc_forces[], 
      vect_t loc_pos[], vect_t loc_vel[], int n, int loc_n, double delta_t);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int n;                      /* Total number of particles  */
   int loc_n;                  /* Number of my particles     */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int loc_part;               /* Current local particle     */
   int next, previous;         /* Neighbouring ranks in ring */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   double t;                   /* Current Time               */
   double* loc_masses;         /* Masses of my particles     */
   vect_t* loc_pos;            /* Positions of my particles  */
   vect_t* loc_vel;            /* Velocities of my particles */
   vect_t* loc_forces;         /* Forces on my particles     */

   vect_t* send_block;         /* Current block sent around ring */
   vect_t* recv_block;         /* Block received from previous rank */
   double* send_masses;
   double* recv_masses;
   int owner;                  /* Original owner of current block */
   int pass;

   char g_i;                   /*_G_en or _i_nput init conds */
   double start, finish;       /* For timings                */

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   /* Identify neighbouring MPI ranks for ring communication */
   next = (my_rank + 1) % comm_sz;
   previous = (my_rank - 1 + comm_sz) % comm_sz;   

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   loc_n = n/comm_sz;  /* n should be evenly divisible by comm_sz */
   loc_masses = malloc(loc_n * sizeof(double));
   loc_pos = malloc(loc_n * sizeof(vect_t));
   loc_forces = malloc(loc_n * sizeof(vect_t));
   loc_vel = malloc(loc_n * sizeof(vect_t));

   send_block = malloc(loc_n * sizeof(vect_t));
   recv_block = malloc(loc_n * sizeof(vect_t));
   send_masses = malloc(loc_n * sizeof(double));
   recv_masses = malloc(loc_n * sizeof(double));

   if (my_rank == 0) vel = malloc(n*sizeof(vect_t));
   MPI_Type_contiguous(DIM, MPI_DOUBLE, &vect_mpi_t);
   MPI_Type_commit(&vect_mpi_t);

   if (g_i == 'i')
      Get_init_cond(loc_masses, loc_pos, loc_vel, n, loc_n);
   else
      Gen_init_cond(loc_masses, loc_pos, loc_vel, n, loc_n);

   start = MPI_Wtime();
#  ifndef NO_OUTPUT
   Output_state(0.0, loc_masses, loc_pos, loc_vel, n, loc_n);
#  endif
   for (step = 1; step <= n_steps; step++) {
      t = step * delta_t;

      /* Reset total forces */
      for (loc_part = 0; loc_part < loc_n; loc_part++) {
         loc_forces[loc_part][X] = 0.0;
         loc_forces[loc_part][Y] = 0.0;
      }

      /* Force contribution from this rank's own particles */
      Accumulate_force_block(loc_masses, loc_pos, loc_forces,
            loc_masses, loc_pos, loc_n, my_rank);

      /* Prepare local mass and position block for circulation */
      memcpy(send_masses, loc_masses, loc_n * sizeof(double));
      memcpy(send_block, loc_pos, loc_n * sizeof(vect_t));
      owner = my_rank;

      /* Circulate remote blocks and accumulate their forces */
      for (pass = 0; pass < comm_sz - 1; pass++) {

         MPI_Sendrecv(send_masses, loc_n, MPI_DOUBLE, next, 0,
                     recv_masses, loc_n, MPI_DOUBLE, previous, 0,
                     comm, MPI_STATUS_IGNORE);

         MPI_Sendrecv(send_block, loc_n, vect_mpi_t, next, 1,
                     recv_block, loc_n, vect_mpi_t, previous, 1,
                     comm, MPI_STATUS_IGNORE);

         owner = (owner - 1 + comm_sz) % comm_sz;

         /* Add force contribution from received block */
         Accumulate_force_block(loc_masses, loc_pos, loc_forces,
               recv_masses, recv_block, loc_n, owner);

         /* Forward received block next round */
         vect_t* temp = send_block;
         send_block = recv_block;
         recv_block = temp;

         double* temp_masses = send_masses;
         send_masses = recv_masses;
         recv_masses = temp_masses;
      }

      /* Update only after the full force has been calculated */
      for (loc_part = 0; loc_part < loc_n; loc_part++)
         Update_part(loc_part, loc_masses, loc_forces,
               loc_pos, loc_vel, n, loc_n, delta_t);

   #  ifndef NO_OUTPUT
      if (step % output_freq == 0)
         Output_state(t, loc_masses, loc_pos, loc_vel, n, loc_n);
   #  endif
   }
   
   finish = MPI_Wtime();
   if (my_rank == 0)
      printf("Elapsed time = %e seconds\n", finish-start);

   MPI_Type_free(&vect_mpi_t);
   free(loc_masses);
   free(loc_pos);
   free(loc_forces);
   free(loc_vel);
   if (my_rank == 0) free(vel);

   free(send_block);
   free(recv_block);
   free(send_masses);
   free(recv_masses);

   MPI_Finalize();

   return 0;
}  /* main */


/*---------------------------------------------------------------------
 * Function: Usage
 * Purpose:  Print instructions for command-line and exit
 * In arg:   
 *    prog_name:  the name of the program as typed on the command-line
 */
void Usage(char* prog_name) {
   
   fprintf(stderr, "usage: mpiexec -n <number of processes> %s\n", prog_name);
   fprintf(stderr, "   <number of particles> <number of timesteps>\n");
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");
    
   exit(0);
}  /* Usage */


/*---------------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get command line args
 * In args:
 *    argc:            number of command line args
 *    argv:            command line args
 * Out args:
 *    n_p:             pointer to n, the number of particles
 *    n_steps_p:       pointer to n_steps, the number of timesteps
 *    delta_t_p:       pointer to delta_t, the size of each timestep
 *    output_freq_p:   pointer to output_freq, which is the number of
 *                     timesteps between steps whose output is printed
 *    g_i_p:           pointer to char which is 'g' if the init conds
 *                     should be generated by the program and 'i' if
 *                     they should be read from stdin
 */
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p) {
   if (my_rank == 0) {
      if (argc != 6) Usage(argv[0]);
      *n_p = strtol(argv[1], NULL, 10);
      *n_steps_p = strtol(argv[2], NULL, 10);
      *delta_t_p = strtod(argv[3], NULL);
      *output_freq_p = strtol(argv[4], NULL, 10);
      *g_i_p = argv[5][0];
   }
   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
   MPI_Bcast(n_steps_p, 1, MPI_INT, 0, comm);
   MPI_Bcast(delta_t_p, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(output_freq_p, 1, MPI_INT, 0, comm);
   MPI_Bcast(g_i_p, 1, MPI_CHAR, 0, comm);

   if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0) {
      if (my_rank == 0) Usage(argv[0]);
      MPI_Finalize();
      exit(0);
   }
   if (*g_i_p != 'g' && *g_i_p != 'i') {
      if (my_rank == 0) Usage(argv[0]);
      MPI_Finalize();
      exit(0);
   }
#  ifdef DEBUG
   if (my_rank == 0) {
      printf("n = %d\n", *n_p);
      printf("n_steps = %d\n", *n_steps_p);
      printf("delta_t = %e\n", *delta_t_p);
      printf("output_freq = %d\n", *output_freq_p);
      printf("g_i = %c\n", *g_i_p);
   }
#  endif
}  /* Get_args */


/*---------------------------------------------------------------------
 * Function:   Get_init_cond
 * Purpose:    Read in initial conditions:  mass, position and velocity
 *             for each particle
 * In args:  
 *    n:       total number of particles
 *    loc_n:   number of particles assigned to this process
 * Out args:
 *    masses:  global array of the masses of the particles
 *    pos:     global array of positions
 *    loc_vel: local array of velocities assigned to this process.
 *
 * Global var:
 *    vel:     Scratch.  Used by process 0 for global velocities
 */
void Get_init_cond(double loc_masses[], vect_t loc_pos[],
      vect_t loc_vel[], int n, int loc_n) {
   int part;
   double* all_masses = NULL;
   vect_t* all_pos = NULL;

   if (my_rank == 0) {
      all_masses = malloc(n * sizeof(double));
      all_pos = malloc(n * sizeof(vect_t));

      printf("For each particle, enter (in order):\n");
      printf("   its mass, its x-coord, its y-coord, ");
      printf("its x-velocity, its y-velocity\n");

      for (part = 0; part < n; part++) {
         scanf("%lf", &all_masses[part]);
         scanf("%lf", &all_pos[part][X]);
         scanf("%lf", &all_pos[part][Y]);
         scanf("%lf", &vel[part][X]);
         scanf("%lf", &vel[part][Y]);
      }
   }

   MPI_Scatter(all_masses, loc_n, MPI_DOUBLE,
         loc_masses, loc_n, MPI_DOUBLE, 0, comm);

   MPI_Scatter(all_pos, loc_n, vect_mpi_t,
         loc_pos, loc_n, vect_mpi_t, 0, comm);

   MPI_Scatter(vel, loc_n, vect_mpi_t,
         loc_vel, loc_n, vect_mpi_t, 0, comm);

   if (my_rank == 0) {
      free(all_masses);
      free(all_pos);
   }
} /* Get_init_cond */

/*---------------------------------------------------------------------
 * Function:  Gen_init_cond
 * Purpose:   Generate initial conditions:  mass, position and velocity
 *            for each particle
 * In args:  
 *    n:       total number of particles
 *    loc_n:   number of particles assigned to this process
 * Out args:
 *    masses:  global array of the masses of the particles
 *    pos:     global array of positions
 *    loc_vel: local array of velocities assigned to this process.
 * Global var:
 *    vel:     Scratch.  Used by process 0 for global velocities
 *
 * Note:      The initial conditions place all particles at
 *            equal intervals on the nonnegative x-axis with 
 *            identical masses, and identical initial speeds
 *            parallel to the y-axis.  However, some of the
 *            velocities are in the positive y-direction and
 *            some are negative.
 */
void Gen_init_cond(double loc_masses[], vect_t loc_pos[],
      vect_t loc_vel[], int n, int loc_n) {
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   double* all_masses = NULL;
   vect_t* all_pos = NULL;

   if (my_rank == 0) {
      all_masses = malloc(n * sizeof(double));
      all_pos = malloc(n * sizeof(vect_t));

      for (part = 0; part < n; part++) {
         all_masses[part] = mass;
         all_pos[part][X] = part * gap;
         all_pos[part][Y] = 0.0;

         vel[part][X] = 0.0;

         if (part % 2 == 0)
            vel[part][Y] = speed;
         else
            vel[part][Y] = -speed;
      }
   }

   MPI_Scatter(all_masses, loc_n, MPI_DOUBLE,
         loc_masses, loc_n, MPI_DOUBLE, 0, comm);

   MPI_Scatter(all_pos, loc_n, vect_mpi_t,
         loc_pos, loc_n, vect_mpi_t, 0, comm);

   MPI_Scatter(vel, loc_n, vect_mpi_t,
         loc_vel, loc_n, vect_mpi_t, 0, comm);

   if (my_rank == 0) {
      free(all_masses);
      free(all_pos);
   }
}  /* Gen_init_cond */


/*---------------------------------------------------------------------
 * Function:   Output_state
 * Purpose:    Print the current state of the system
 * In args:
 *    time:    current time
 *    masses:  global array of particle masses
 *    pos:     global array of particle positions
 *    loc_vel: local array of my particle velocities
 *    n:       total number of particles
 *    loc_n:   number of my particles
 */
void Output_state(double time, double masses[], vect_t pos[],
      vect_t loc_vel[], int n, int loc_n) {
   int part;

   MPI_Gather(loc_vel, loc_n, vect_mpi_t, vel, loc_n, vect_mpi_t, 
         0, comm);
   if (my_rank == 0) {
      printf("%.2f\n", time);
      for (part = 0; part < n; part++) {
//       printf("%.3f ", masses[part]);
         printf("%3d %10.3e ", part, pos[part][X]);
         printf("  %10.3e ", pos[part][Y]);
         printf("  %10.3e ", vel[part][X]);
         printf("  %10.3e\n", vel[part][Y]);
      }
      printf("\n");
   }
}  /* Output_state */


/*---------------------------------------------------------------------
 * Function:       Compute_force
 * Purpose:        Compute the total force on particle loc_part.  Don't 
 *                 exploit the symmetry (force on particle i due to 
 *                 particle k) = -(force on particle k due to particle i) 
 * In args:   
 *    loc_part:    the particle (local index) on which we're computing 
 *                 the total force
 *    masses:      global array of particle masses
 *    pos:         global array of particle positions
 *    n:           total number of particles
 *    loc_n:       number of my particles
 * Out arg:
 *    loc_forces:  array of total forces acting on my particles
 *
 * Note: This function uses the force due to gravitation.  So 
 * the force on particle i due to particle k is given by
 *
 *    m_i m_k (s_k - s_i)/|s_k - s_i|^2
 *
 * Here, m_k is the mass of particle k and s_k is its position vector
 * (at time t). 
 */
void Compute_force(int loc_part, double masses[], vect_t loc_forces[], 
      vect_t pos[], int n, int loc_n) {
   int k, part;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;

   /* Global index corresponding to loc_part */
   part = my_rank*loc_n + loc_part;
   loc_forces[loc_part][X] = loc_forces[loc_part][Y] = 0.0;
#  ifdef DEBUG
   printf("Proc %d > Current total force on part %d = (%.3e, %.3e)\n",
         my_rank, part, loc_forces[loc_part][X], 
         loc_forces[loc_part][Y]);
#  endif
   for (k = 0; k < n; k++) {
      if (k != part) {
         /* Compute force on part due to k */
         f_part_k[X] = pos[part][X] - pos[k][X];
         f_part_k[Y] = pos[part][Y] - pos[k][Y];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*masses[part]*masses[k];
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;
#        ifdef DEBUG
         printf("Proc %d > Force on part %d due to part %d = (%.3e, %.3e)\n",
               my_rank, part, k, f_part_k[X], f_part_k[Y]);
#        endif
   
         /* Add force in to total forces */
         loc_forces[loc_part][X] += f_part_k[X];
         loc_forces[loc_part][Y] += f_part_k[Y];
      }
   }
}  /* Compute_force */


/*---------------------------------------------------------------------
 * Function:  Update_part
 * Purpose:   Update the velocity and position for particle loc_part
 * In args:
 *    loc_part:    local index of the particle we're updating
 *    masses:      global array of particle masses
 *    loc_forces:  local array of total forces
 *    n:           total number of particles
 *    loc_n:       number of particles assigned to this process
 *    delta_t:     step size
 *
 * In/out args:
 *    loc_pos:     local array of positions
 *    loc_vel:     local array of velocities
 *
 * Note:  This version uses Euler's method to update both the velocity
 *    and the position.
 */
void Update_part(int loc_part, double masses[], vect_t loc_forces[], 
      vect_t loc_pos[], vect_t loc_vel[], int n, int loc_n, 
      double delta_t) {
   int part;
   double fact;

   part = my_rank*loc_n + loc_part;
   fact = delta_t/loc_masses[loc_part];
#  ifdef DEBUG
   printf("Proc %d > Before update of %d:\n", my_rank, part);
   printf("   Position  = (%.3e, %.3e)\n", 
         loc_pos[loc_part][X], loc_pos[loc_part][Y]);
   printf("   Velocity  = (%.3e, %.3e)\n", 
         loc_vel[loc_part][X], loc_vel[loc_part][Y]);
   printf("   Net force = (%.3e, %.3e)\n", 
         loc_forces[loc_part][X], loc_forces[loc_part][Y]);
#  endif
   loc_pos[loc_part][X] += delta_t * loc_vel[loc_part][X];
   loc_pos[loc_part][Y] += delta_t * loc_vel[loc_part][Y];
   loc_vel[loc_part][X] += fact * loc_forces[loc_part][X];
   loc_vel[loc_part][Y] += fact * loc_forces[loc_part][Y];
#  ifdef DEBUG
   printf("Proc %d > Position of %d = (%.3e, %.3e), Velocity = (%.3e,%.3e)\n",
         my_rank, part, loc_pos[loc_part][X], loc_pos[loc_part][Y],
               loc_vel[loc_part][X], loc_vel[loc_part][Y]);
#  endif
}  /* Update_part */
