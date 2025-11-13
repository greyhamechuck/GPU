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

// Vertical processing kernel: each thread handles one potential prime p
// Checks if p is prime on GPU (by reading from device memory) and marks its multiples
// This eliminates device-to-host transfers - everything happens on the GPU
// The "vertical" approach means processing by potential primes (columns) in parallel
// rather than processing one prime at a time sequentially (rows)
__global__ void vertical_process_kernel(char* is_prime, long N, long start_p, long end_p) {
    long p = start_p + (long)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p >= start_p && p <= end_p && p <= (N + 1) / 2) {
        // Check if p is still marked as prime (read directly from device memory)
        // This avoids the device-to-host transfer used in genprimes.cu
        if (is_prime[p] == 1) {
            // Mark all multiples of p starting from 2*p
            for (long i = 2 * p; i <= N; i += p) {
                is_prime[i] = 0; // "Cross out" the number
            }
        }
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
    
    // Find first prime for proper formatting
    long first_p = 2;
    while (first_p <= N && h_is_prime[first_p] == 0) {
        first_p++;
    }
    
    if (first_p <= N) {
        fprintf(f, "%ld", first_p);
    }
    
    // Print remaining primes with leading space
    for (long i = first_p + 1; i <= N; i++) {
        if (h_is_prime[i] == 1) {
            fprintf(f, " %ld", i);
        }
    }
    fprintf(f, "\n");
    fclose(f);
}

int main(int argc, char** argv) {
    
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }
    long N = atol(argv[1]); // the upper limit
    if (N <= 1) {
        fprintf(stderr, "N must be > 1\n");
        return 1;
    }

    clock_t start = clock();

    // Allocate GPU Memory
    char* d_is_prime; 
    size_t array_size = (N + 1) * sizeof(char);
    CHECK_CUDA(cudaMalloc(&d_is_prime, array_size));

    // Set Kernel Launch Parameters 
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;

    // Launch init_kernel
    init_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Vertical approach: process all potential primes in parallel batches
    // Each thread handles one potential prime p, checks if it's prime on GPU,
    // and marks its multiples - all without device-to-host transfers
    // This is the "vertical" optimization: processing by columns (potential primes)
    // rather than rows (one prime at a time with host coordination)
    
    // Process in batches to balance parallelism and memory coherence
    long batch_size = 100000; // Process 100000 numbers at a time
    long sieve_limit = (N + 1) / 2;
    
    for (long batch_start = 2; batch_start <= sieve_limit; batch_start += batch_size) {
        long batch_end = (batch_start + batch_size - 1 < sieve_limit) ? 
                         batch_start + batch_size - 1 : sieve_limit;
        
        long batch_range = batch_end - batch_start + 1;
        int blocks_for_batch = (batch_range + threadsPerBlock - 1) / threadsPerBlock;
        
        if (blocks_for_batch > 0) {
            // Launch kernel where each thread processes one potential prime
            // Threads check if their number is prime (reading from device memory)
            // and mark multiples - all on GPU, no host involvement
            vertical_process_kernel<<<blocks_for_batch, threadsPerBlock>>>(
                d_is_prime, N, batch_start, batch_end);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Stop Timer
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Allocate Host Memory and Copy Results
    char* h_is_prime = (char*)malloc(array_size);
    if (h_is_prime == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_is_prime, d_is_prime, array_size, cudaMemcpyDeviceToHost));

    // Write Output and Clean Up
    write_file(h_is_prime, N);
    free(h_is_prime);
    CHECK_CUDA(cudaFree(d_is_prime));

    printf("Total GPU time (real): %f seconds\n", cpu_time_used);

    return 0;
}

