/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D square array of dimension NxN stored as 1D 
   index(i, j, N) means access element at row i, column j, and N is the dimension which is NxN */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);


/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU version\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 120;
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 130;
  for(i = 1; i < N-1; i++)
    playground[index(i,0,N)] = 70;
  for(i = 1; i < N-1; i++)
    playground[index(i,N-1,N)] = 70;
  

  switch(type_of_device)
  {
	case 0: printf("CPU sequential version:\n");
			start = clock();
			seq_heat_dist(playground, N, iterations);
			end = clock();
			break;
		
	case 1: printf("GPU version:\n");
			start = clock();
			gpu_heat_dist(playground, N, iterations); 
			cudaDeviceSynchronize();
			end = clock();  
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken = %lf\n", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/

__global__ void heat_dist_kernel(float* d_new, const float* d_old, unsigned int N) {
    // Calculate the global row and column index for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within the NxN grid
    if (i < N && j < N) {
        // The calculation only applies to interior points 
        if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
            // Apply the heat distribution formula 
            d_new[index(i, j, N)] = (d_old[index(i - 1, j, N)] +
                                     d_old[index(i + 1, j, N)] +
                                     d_old[index(i, j - 1, N)] +
                                     d_old[index(i, j + 1, N)]) / 4.0f;
        }
        // Note: Edge points retain their values. Since the host `playground` array
        // is copied to both d_old and d_new initially, the boundary values
        // are already correct in d_new and do not need to be touched.
    }
}
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
    // Device pointers for our double-buffering strategy
    float *d_old, *d_new;
    
    // Calculate the total size of the grid in bytes
    size_t size = N * N * sizeof(float);
  
    // 1. Allocate memory on the GPU for two grids
    cudaMalloc(&d_old, size);
    cudaMalloc(&d_new, size);
    
    // 2. Copy the initial grid data from the host (CPU) to the device (GPU)
    // Both device arrays are initialized with the same starting data.
    cudaMemcpy(d_old, playground, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_new, playground, size, cudaMemcpyHostToDevice);
    
    // 3. Set up the execution configuration (Grid and Block dimensions)
    // As per the lab instructions, you should experiment to find the best values.
    // A 16x16 block size (256 threads) is a good starting point.
    dim3 threadsPerBlock(16, 16);
    
    // Calculate the number of blocks needed to cover the entire grid
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    // 4. Start the iterative computation
    for (int k = 0; k < iterations; k++) {
        // Launch the kernel to compute one iteration.
        // It reads from d_old and writes the new values into d_new.
        heat_dist_kernel<<<numBlocks, threadsPerBlock>>>(d_new, d_old, N);
        
        // Swap the pointers for the next iteration (double buffering).
        // The newly computed grid (d_new) becomes the old grid (d_old) for the next step.
        // This is much faster than copying the entire array.
        float* temp = d_old;
        d_old = d_new;
        d_new = temp;
    }
    
    // 5. Copy the final result back from the device to the host
    // After the loop, d_old holds the data from the final completed iteration.
    cudaMemcpy(playground, d_old, size, cudaMemcpyDeviceToHost);
    
    // 6. Free the allocated memory on the device
    cudaFree(d_old);
    cudaFree(d_new);
}


