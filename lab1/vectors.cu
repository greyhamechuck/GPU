#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h> // Required for fabs()

#define RANGE 17.78
#define BLOCK_NUM 8         // Default grid size for final submission
#define THREADS_NUM 500     // Default block size for final submission

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__ void vecGPU(float *ad, float *bd, float *cd, int n);
/**** end of the kernel declaration ***/


int main(int argc, char *argv[]){

	int n = 0; //number of elements in the arrays
	int i;  //loop index
	float *a, *b, *c; // The arrays that will be processed in the host.
	float *temp;  //array in host used in the sequential code.
	float *ad, *bd, *cd; //The arrays that will be processed in the device.
	clock_t start, end; // to meaure the time taken by a specific part of code
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
		}
		
	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);
	
	
	//Allocating the arrays in the host
	size_t size = n * sizeof(float);
	
	if( !(a = (float *)malloc(size)) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	
	if( !(b = (float *)malloc(size)) )
	{
	   printf("Error allocating array b\n");
	   exit(1);
	}
	
	if( !(c = (float *)malloc(size)) )
	{
	   printf("Error allocating array c\n");
	   exit(1);
	}
	
	if( !(temp = (float *)malloc(size)) )
	{
	   printf("Error allocating array temp\n");
	   exit(1);
	}
	
    //Fill out the arrays with random numbers between 0 and RANGE;
    srand((unsigned int)time(NULL));
    for (i = 0; i < n;  i++){
        a[i] = ( (float) rand() / (float) (RAND_MAX) ) * RANGE;
        b[i] = ( (float) rand() / (float) (RAND_MAX) ) * RANGE;
        c[i] = ( (float) rand() / (float) (RAND_MAX) ) * RANGE;
        temp[i] = c[i]; //temp is just another copy of C
    }
    
    //The sequential part
	start = clock();
	for(i = 0; i < n; i++)
		temp[i] += a[i] * b[i];
	end = clock();
	printf("Total time taken by the sequential part = %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    /****************** The start GPU part: Do not modify anything in main() above this line  ************/
	//The GPU part
	
    // Create CUDA events for accurate timing
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_elapsed_time;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

	/* TODO: in this part you need to do the following:
		1. allocate ad, bd, and cd in the device
		2. send a, b, and c to the device  
	*/
    cudaMalloc((void **)&ad, size);
    cudaMalloc((void **)&bd, size);
    cudaMalloc((void **)&cd, size);

    // Record the start event
    cudaEventRecord(gpu_start);

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
		
	/* TODO: 	
		3. write the kernel, call it: vecGPU
		4. call the kernel (the kernel itself will be written at the comment at the end of this file), 
		   you need to decide about the number of threads, blocks, etc and their geometry.
	*/
    vecGPU<<<BLOCK_NUM, THREADS_NUM>>>(ad, bd, cd, n);
		
	/* TODO: 
		5. bring the cd array back from the device and store it in c array (declared earlier in main)
		6. free ad, bd, and cd
	*/
	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

    // Record the stop event and synchronize
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    // Calculate the elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
	
	printf("Total time taken by the GPU part = %f seconds\n", gpu_elapsed_time / 1000.0f); // Time is in ms
	/****************** The end of the GPU part: Do not modify anything in main() below this line  ************/
	
	//checking the correctness of the GPU part
	for(i = 0; i < n; i++)
	  if( fabs(temp[i] - c[i]) >= 0.009) //compare up to the second degit in floating point
		printf("Element %d in the result array does not match the sequential version\n", i);
		
	// Free the arrays in the host
	free(a); free(b); free(c); free(temp);
    
    // Destroy the events
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

	return 0;
}


/**** TODO: Write the kernel itself below this line *****/
__global__ void vecGPU(float *ad, float *bd, float *cd, int n)
{
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    int stride = gridDim.x * blockDim.x;

    // Use a grid-stride loop to process all elements
    for (int i = index; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];
    }
}

