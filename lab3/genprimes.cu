#include <stdio.h>
#include <stdlib.h>
#include <math.h>   
#include <cuda_runtime.h>

// check cuda errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- CUDA Kernels ---

// generate numbers from 0 to N and initialize the is_prime array
__global__ void init_kernel(char* is_prime, long N) {
    long global_id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = global_id; i <= N; i += stride) {
        if (i == 0 || i == 1) {
            is_prime[i] = 0; // 0 and 1 are not prime
        } else {
            is_prime[i] = 1; // all numbers are prime, lets assume
        }
    }
}

// cross out the prime numbers' multiples
__global__ void sieve_kernel(char* is_prime, long N, int p) {
    long global_id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    long start_index = (long)p * (global_id + 2);
    long step = (long)p * stride;
    for (long i = start_index; i <= N; i += step) {
        is_prime[i] = 0; // "Cross out" the number
    }
}

// Output files generator
void write_file(const char* h_is_prime, long N) {
    char filename[256];
    sprintf(filename, "%ld.txt", N);
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Error opening output file!\n");
        return;
    }
    // Loop and write all prime numbers
    for (long i = 2; i <= N; i++) {
        if (h_is_prime[i] == 1) {
            fprintf(f, "%ld ", i);
        }
    }
    fclose(f);
}


int main(int argc, char** argv) {
    
    if (argc != 2) {
        return 1;
    }
    long N = atol(argv[1]); // the upper limit
    if (N <= 1) {
        fprintf(stderr, "N must be > 1\n");
        return 1;
    }

    clock_t start = clock();

    // Allocate GPU Memory
    char* d_is_prime; // Device pointer
    size_t array_size = (N + 1) * sizeof(char);
    CHECK_CUDA(cudaMalloc(&d_is_prime, array_size));

    // Set Kernel Launch Parameters 
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024; // Use a large, fixed number

    // Launch init_kernel
    init_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N);
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for init to finish

    // The Main Sieve Loop (The Bottleneck) 
    for (long p = 2; p <= (N + 1) / 2; p++) {
        
        char p_is_prime;
        // Copy 1 byte from Device -> Host
        CHECK_CUDA(cudaMemcpy(&p_is_prime, d_is_prime + p, 1, cudaMemcpyDeviceToHost));

        if (p_is_prime == 1) {
            // Launch the sieve kernel for this prime 'p'
            sieve_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N, p);
        }
    }
    // Wait for all queued sieve_kernels to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Stop Timer and Allocate Host Memory ---
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    char* h_is_prime = (char*)malloc(array_size);
    if (h_is_prime == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_is_prime, d_is_prime, array_size, cudaMemcpyDeviceToHost));

    //Write Output and Clean Up
    write_file(h_is_prime, N);
    free(h_is_prime);
    CHECK_CUDA(cudaFree(d_is_prime));

    printf("Total GPU time (real): %f seconds\n", cpu_time_used);

    return 0;
}