/******************************************************************************
* FILE: omp_mm.c
* DESCRIPTION:  
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. Threads share row iterations
*   according to a predefined chunk size.
* AUTHOR: Blaise Barney
* LAST REVISED: 06/28/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 15872                 /* number of rows in matrix A */
#define NCA 3840                 /* number of columns in matrix A */
#define NCB 1792                  /* number of columns in matrix B */

int main (int argc, char *argv[]) 
{
double itime, ftime, exec_time;
itime = omp_get_wtime();

int	tid, nthreads, i, j, k, chunk, i1, j1, k1;
static double	a[NRA][NCA],           /* matrix A to be multiplied */
	        b[NCA][NCB],           /* matrix B to be multiplied */
	        c[NRA][NCB];           /* result matrix C */

chunk = 10;                    /* set loop iteration chunk size */

/*** Spawn a parallel region explicitly scoping all variables ***/
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
  {
  int block_size = 4;
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Starting matrix multiple example with %d threads\n",nthreads);
    printf("Initializing matrices...\n");
    }
  /*** Initialize matrices ***/
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++)
      a[i][j]= i+j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= i*j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCB; j++)
      c[i][j]= 0;

  /*** Do matrix multiply sharing iterations on outer loop ***/
  /*** Display who does which iterations for demonstration purposes ***/
  //printf("Thread %d starting matrix multiply...\n",tid);
  //#pragma omp for schedule (static, chunk) collapse(3)
  for (i=0; i<NRA; i+=block_size)    
    {
    //printf("Thread=%d did row=%d // time=%f\n",tid,i,omp_get_wtime());
    for(j=0; j<NCB; j+=block_size)       
      for (k=0; k<NCA; k+=block_size)
	#pragma omp for schedule (static, chunk) collapse(3)
	for (i1=0; i1<block_size; i1++)
	  for (j1=0; j1<block_size; j1++)
	    {
	     int sum = 0;
	     for (k1=0; k1<block_size; k1++)
	       {
		 #pragma omp critical
	         sum += a[i1][k1] * b[k1][j1];
	       }
	     c[i1][j1] += sum;
	    }
    }
  }   /*** End of parallel region ***/

ftime = omp_get_wtime();
exec_time = ftime - itime;

/*** Print results ***/
/***
printf("******************************************************\n");
printf("Result Matrix:\n");
for (i=0; i<NRA; i++)
  {
  for (j=0; j<NCB; j++) 
    printf("%6.2f   ", c[i][j]);
  printf("\n"); 
  }
printf("******************************************************\n");
***/
printf ("Done.\n");
printf("\nTime taken is %f\n", exec_time);
}
