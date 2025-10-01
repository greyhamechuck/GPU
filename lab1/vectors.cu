// vectors.cu  —  follows the original lab skeleton & print style

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h> // for fabsf

#define RANGE 17.78
#define BLOCK_NUM   8      // final submission requires 8 blocks
#define THREADS_NUM 500    // final submission requires 500 threads per block

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__ void vecGPU(const float *ad, const float *bd, float *cd, int n);
/**** end of the kernel declaration ***/

int main(int argc, char *argv[]) {
    int i, n;
    float *a, *b, *c, *temp;
    clock_t start, end;
    double time_seq = 0.0, time_gpu = 0.0;

    if (argc < 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    n = atoi(argv[1]);
    if (n <= 0) {
        printf("n must be positive.\n");
        return 1;
    }

    printf("Each vector will have %d elements\n", n);

    // Host allocations
    a    = (float*) malloc(n * sizeof(float));
    b    = (float*) malloc(n * sizeof(float));
    c    = (float*) malloc(n * sizeof(float));
    temp = (float*) malloc(n * sizeof(float));

    if (!a || !b || !c || !temp) {
        printf("Host allocation failed.\n");
        return 1;
    }

    // Initialize host arrays with random numbers in [0, RANGE)
    srand((unsigned int) time(NULL));
    for (i = 0; i < n; ++i) {
        a[i]    = ((float) rand() / (float) RAND_MAX) * RANGE;
        b[i]    = ((float) rand() / (float) RAND_MAX) * RANGE;
        c[i]    = ((float) rand() / (float) RAND_MAX) * RANGE;
        temp[i] = c[i];   // temp is copy of c for CPU result
    }

    // ===== Sequential part (CPU) =====
    start = clock();
    for (i = 0; i < n; ++i) {
        temp[i] += a[i] * b[i];
    }
    end = clock();
    time_seq = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by the sequential part = %.6f seconds\n", time_seq);

    // ===== GPU part (H2D + kernel + D2H as one measured block) =====

    // Device pointers
    float *ad = NULL, *bd = NULL, *cd = NULL;

    // Start timing the whole GPU “part” (matches your print style)
    start = clock();

    /*** TODO: allocate memory on the device for ad, bd, and cd ***/
    cudaError_t err;
    err = cudaMalloc((void**)&ad, n * sizeof(float)); if (err != cudaSuccess) { printf("cudaMalloc ad failed\n"); return 1; }
    err = cudaMalloc((void**)&bd, n * sizeof(float)); if (err != cudaSuccess) { printf("cudaMalloc bd failed\n"); return 1; }
    err = cudaMalloc((void**)&cd, n * sizeof(float)); if (err != cudaSuccess) { printf("cudaMalloc cd failed\n"); return 1; }
    /**** end of memory allocation on device ***/

    /*** TODO: copy a, b, c from host to device ***/
    err = cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice); if (err != cudaSuccess) { printf("H2D a failed\n"); return 1; }
    err = cudaMemcpy(bd, b, n * sizeof(float), cudaMemcpyHostToDevice); if (err != cudaSuccess) { printf("H2D b failed\n"); return 1; }
    err = cudaMemcpy(cd, c, n * sizeof(float), cudaMemcpyHostToDevice); if (err != cudaSuccess) { printf("H2D c failed\n"); return 1; }
    /**** end of copy to device ***/

    /*** TODO: launch the kernel with BLOCK_NUM and THREADS_NUM ***/
    vecGPU<<<BLOCK_NUM, THREADS_NUM>>>(ad, bd, cd, n);
    // Make sure the kernel is finished before we stop the timer
    cudaDeviceSynchronize();
    /**** end of kernel launch ***/

    /*** TODO: copy the result vector back from device to host (into c) ***/
    err = cudaMemcpy(c, cd, n * sizeof(float), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { printf("D2H c failed\n"); return 1; }
    /**** end of copy back to host ***/

    // Stop timing the whole GPU part (memcpy + kernel + memcpy)
    end = clock();
    time_gpu = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by the GPU part = %.6f seconds\n", time_gpu);

    // ===== Correctness check (GPU vs CPU result) =====
    // Per handout: compare up to the second digit using 0.009 tolerance.
    int mismatch = 0;
    for (i = 0; i < n; ++i) {
        if (fabsf(temp[i] - c[i]) >= 0.009f) {
            mismatch = 1;
            break;
        }
    }
    if (mismatch)
        printf("Results do not match within tolerance.\n");
    else
        printf("Results match within tolerance.\n");

    // Cleanup
    if (ad) cudaFree(ad);
    if (bd) cudaFree(bd);
    if (cd) cudaFree(cd);
    free(a); free(b); free(c); free(temp);

    return 0;
}

/**** TODO: Write the kernel itself below this line *****/
__global__ void vecGPU(const float *ad, const float *bd, float *cd, int n)
{
    // Global thread id
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    // Total stride over the grid
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop covers any n and any launch size
    for (int i = index; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];
    }
}
/**** end of kernel *****/
