#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCK_NUM 8
#define BLOCK_SIZE 500

#define RANGE 19.87

/*** Declaration of the kernel function below this line ***/

__global__ void vecGPU(float *a, float *b, float *c, int n, int stride);

/**** end of the kernel declaration ***/

int main(int argc, char *argv[])
{

	int n = 0;									// number of elements in the arrays
	int i;											// loop index
	float *a, *b, *c;						// The arrays that will be processed in the host.
	float *temp;								// array in host used in the sequential code.
	float *ad, *bd, *cd;				// The arrays that will be processed in the device.
	struct timespec start, end; // to meaure the time taken by a specific part of code

	if (argc != 2)
	{
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
	}

	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);

	// Allocating the arrays in the host

	if (!(a = (float *)malloc(n * sizeof(float))))
	{
		printf("Error allocating array a\n");
		exit(1);
	}

	if (!(b = (float *)malloc(n * sizeof(float))))
	{
		printf("Error allocating array b\n");
		exit(1);
	}

	if (!(c = (float *)malloc(n * sizeof(float))))
	{
		printf("Error allocating array c\n");
		exit(1);
	}

	if (!(temp = (float *)malloc(n * sizeof(float))))
	{
		printf("Error allocating array temp\n");
		exit(1);
	}

	// Fill out the arrays with random numbers between 0 and RANGE;
	srand((unsigned int)time(NULL));
	for (i = 0; i < n; i++)
	{
		a[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
		b[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
		c[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
		temp[i] = c[i]; // temp is just another copy of C
	}

	// The sequential part

	clock_gettime(CLOCK_REALTIME, &start);

	for (i = 0; i < n; i++)
		temp[i] += a[i] * b[i];

	clock_gettime(CLOCK_REALTIME, &end);

	printf("Total time taken by the sequential part = %lf\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1000000000);

	/******************  The start GPU part: Do not modify anything in main() above this line  ************/
	// The GPU part

	// Allocate the arrays in the device
	cudaMalloc((void **)&ad, n * sizeof(float));
	cudaMalloc((void **)&bd, n * sizeof(float));
	cudaMalloc((void **)&cd, n * sizeof(float));

	// Copy the arrays from the host to the device
	cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cd, c, n * sizeof(float), cudaMemcpyHostToDevice);

	clock_gettime(CLOCK_REALTIME, &start);

	// Call the kernel function
	dim3 grid_size(BLOCK_NUM, 1, 1);
	dim3 block_size(BLOCK_SIZE, 1, 1);
	int stride = ceil(n / (float)(grid_size.x * block_size.x));
	vecGPU<<<grid_size, block_size>>>(ad, bd, cd, n, stride);

	// Force host to wait on the completion of the kernel
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_REALTIME, &end);

	// Copy the result from the device to the host
	cudaMemcpy(c, cd, n * sizeof(float), cudaMemcpyDeviceToHost);

	// Free the memory allocated in the device
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);

	printf("Total time taken by the GPU part = %lf\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1000000000);
	/******************  The end of the GPU part: Do not modify anything in main() below this line  ************/

	// Checking the correctness of the GPU part
	for (i = 0; i < n; i++)
		if (fabs(temp[i] - c[i]) >= 0.009) // compare up to the second degit in floating point
			printf("Element %d in the result array does not match the sequential version\n", i);

	// Free the arrays in the host
	free(a);
	free(b);
	free(c);
	free(temp);

	return 0;
}

/**** Write the kernel itself below this line *****/

__global__ void vecGPU(float *a, float *b, float *c, int n, int stride)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = i * stride; j < (i + 1) * stride && j < n; j++)
	{
		c[j] += a[j] * b[j];
	}
}
