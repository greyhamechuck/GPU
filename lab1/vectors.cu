#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> // Use cuda_runtime.h for CUDA API functions
#include <math.h>         // For fabs()

// --- Configuration for Experiments ---
// Change these values to run the different tests required for Experiment 1
#define BLOCK_NUM 8         // The number of blocks in the grid
#define THREADS_NUM 500     // The number of threads in each block
// ------------------------------------

#define RANGE 19.87

// Kernel function declaration
__global__ void vecGPU(float *ad, float *bd, float *cd, int n);

int main(int argc, char *argv[]) {

    int n = 0; // Number of elements in the arrays
    int i;     // Loop index
    float *a, *b, *c; // Host arrays
    float *temp;      // Host array for sequential result validation
    float *ad, *bd, *cd; // Device arrays

    // Use struct timespec for high-precision CPU timing
    struct timespec start_cpu, end_cpu;
    double cpu_time_used;

    // Use CUDA Events for accurate GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time_used;

    if (argc != 2) {
        printf("Usage: ./vectorprog <number_of_elements>\n");
        exit(1);
    }

    n = atoi(argv[1]);
    if (n <= 0) {
        printf("Number of elements must be a positive integer.\n");
        exit(1);
    }
    printf("Each vector will have %d elements\n", n);

    // Allocate arrays on the host
    size_t size = n * sizeof(float);
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    temp = (float *)malloc(size);

    if (a == NULL || b == NULL || c == NULL || temp == NULL) {
        printf("Error: Failed to allocate host memory.\n");
        exit(1);
    }

    // Initialize host arrays with random numbers
    srand((unsigned int)time(NULL));
    for (i = 0; i < n; i++) {
        a[i] = ((float)rand() / (float)RAND_MAX) * RANGE;
        b[i] = ((float)rand() / (float)RAND_MAX) * RANGE;
        c[i] = ((float)rand() / (float)RAND_MAX) * RANGE;
        temp[i] = c[i]; // temp is a copy of c for sequential calculation
    }

    // --- The Sequential (CPU) Part ---
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);

    for (i = 0; i < n; i++) {
        temp[i] += a[i] * b[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    cpu_time_used = (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e9;
    printf("Total time taken by the sequential part = %f seconds\n", cpu_time_used);

    /****************** The GPU Part ******************/

    // Create CUDA events
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // 1. Allocate memory for ad, bd, and cd on the device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&ad, size);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector a: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMalloc((void **)&bd, size);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector b: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMalloc((void **)&cd, size);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to allocate device vector c: %s\n", cudaGetErrorString(err)); exit(1); }


    // --- Start GPU Timer ---
    // Record start event. All operations after this are timed.
    cudaEventRecord(start_gpu);

    // 2. Copy data from host (a, b, c) to device (ad, bd, cd)
    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);

    // 4. Call the kernel
    vecGPU<<<BLOCK_NUM, THREADS_NUM>>>(ad, bd, cd, n);

    // 5. Bring the result (cd) back from the device to the host (c)
    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

    // --- Stop GPU Timer ---
    // Record stop event and wait for the GPU to finish all preceding tasks.
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    // Calculate elapsed time
    cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
    printf("Total time taken by the GPU part = %f seconds\n", gpu_time_used / 1000.0f); // Convert ms to s

    // 6. Free device memory
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    // Destroy CUDA events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    /****************** End of the GPU Part ******************/

    // Checking the correctness of the GPU part
    int error_count = 0;
    for (i = 0; i < n; i++) {
        if (fabs(temp[i] - c[i]) >= 0.01) { // Use a reasonable tolerance for floating point
            error_count++;
        }
    }
    if (error_count > 0) {
        printf("Verification FAILED: %d elements do not match.\n", error_count);
    } else {
        printf("Verification PASSED: All elements match.\n");
    }

    // Free the arrays in the host
    free(a);
    free(b);
    free(c);
    free(temp);

    return 0;
}

// 3. The kernel function itself
// This kernel uses a "grid-stride loop" to process the vectors.
// This is a robust pattern that works for any vector size 'n'.
__global__ void vecGPU(float *ad, float *bd, float *cd, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // Total number of threads in the grid

    for (int i = index; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];
    }
}

