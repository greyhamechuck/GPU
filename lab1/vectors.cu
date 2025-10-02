#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h> // Required for fabs()

#define RANGE 17.78

// Kernel function declaration
__global__ void vecGPU(float *ad, float *bd, float *cd, int n);


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
    
    // Define the configurations to test
    int block_configs[] = {4, 8, 16, 4, 8, 16};
    int thread_configs[] = {500, 500, 500, 250, 250, 250};
    int num_configs = sizeof(block_configs) / sizeof(int);

    // Make a backup of the original 'c' array to reset for each test
    float *c_original = (float *)malloc(size);
    if (!c_original) {
        printf("Error allocating array c_original\n");
        exit(1);
    }
    // memcpy(destination, source, size)
    for(i = 0; i < n; i++) {
        c_original[i] = c[i];
    }

    // Loop through each configuration
    for (int j = 0; j < num_configs; j++) {
        int current_blocks = block_configs[j];
        int current_threads = thread_configs[j];

        printf("\n--- Testing: %d blocks, %d threads ---\n", current_blocks, current_threads);

        // Reset c to its original state before this GPU run
        for(i = 0; i < n; i++) {
            c[i] = c_original[i];
        }

        //The GPU part
        
        // 1. Allocate memory for arrays on the GPU device
        cudaMalloc((void **)&ad, size);
        cudaMalloc((void **)&bd, size);
        cudaMalloc((void **)&cd, size);

        // Start the timer for the GPU computation
        start = clock();

        // 2. Send a, b, and c to the device
        cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
        cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
            
        // 3. & 4. Launch the kernel
        vecGPU<<<current_blocks, current_threads>>>(ad, bd, cd, n);
            
        // 5. Bring the cd array back from the device
        cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

        // Block host execution until the device has completed all preceding tasks
        cudaDeviceSynchronize();

        // Stop the timer
        end = clock();

        // 6. Free the device memory
        cudaFree(ad);
        cudaFree(bd);
        cudaFree(cd);
        
        // Print the GPU time using clock()
        printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
        
        //checking the correctness of the GPU part for this run
        int error_count = 0;
        for(i = 0; i < n; i++) {
            if( fabs(temp[i] - c[i]) >= 0.009) {
                error_count++;
            }
        }
        if (error_count == 0) {
            printf("Correctness check PASSED\n");
        } else {
            printf("Correctness check FAILED with %d mismatches\n", error_count);
        }
    }
    /****************** The end of the GPU part: Do not modify anything in main() below this line  ************/
    
    // Free the arrays in the host
    free(a); free(b); free(c); free(temp); free(c_original);
    
    return 0;
}


// Kernel function definition
__global__ void vecGPU(float *ad, float *bd, float *cd, int n)
{
    // Calculate the global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid
    int stride = gridDim.x * blockDim.x;

    // Use a grid-stride loop to process all elements, making the kernel more robust
    for (int i = index; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];
    }
}

