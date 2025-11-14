/*
 * newgen.cu
 * Lab 3: Parallel Prime Sieve (CUDA)
 *
 * Implements the Sieve of Eratosthenes algorithm as specified in the PDF:
 * 1. Generate all numbers from 2 to N
 * 2. Start with 2, cross out multiples of 2 (4, 6, 8, ... N)
 * 3. Move to next unmarked number (3), cross out multiples of 3 (9, 15, ...)
 * 4. Continue sequentially until floor((N+1)/2)
 * 5. Remaining numbers are primes
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Error Checking Utility
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- CUDA Kernels ---

/**
 * Initialize the sieve array: mark all numbers from 2 to N as prime (1)
 * Mark 0 and 1 as not prime (0)
 */
__global__ void initKernel(char *d_is_prime, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    for (unsigned int i = idx; i <= N; i += stride) {
        d_is_prime[i] = (i >= 2) ? 1 : 0;
    }
}

/**
 * Cross out multiples of prime p starting from p*p
 * (Smaller multiples already crossed by smaller primes)
 * Each thread processes different multiples in parallel
 */
__global__ void sieveKernel(char *d_is_prime, unsigned int p, unsigned int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    // Start from p*p (optimization: smaller multiples already crossed)
    unsigned long long start = (unsigned long long)p * p;
    
    // If p*p > N, nothing to do
    if (start > N) return;
    
    // Each thread handles multiples with stride - parallel processing
    for (unsigned long long multiple = start + tid * p; 
         multiple <= N; 
         multiple += stride * p) {
        d_is_prime[multiple] = 0; // Cross out the number
    }
}

// --- Host Code (main) ---

int main(int argc, char *argv[]) {
    
    // --- 1. Process Input ---
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }
    
    unsigned int N = (unsigned int)strtoul(argv[1], NULL, 10);
    if (N <= 2) {
        fprintf(stderr, "Error: N must be bigger than 2.\n");
        return 1;
    }
    
    // --- 2. Setup Memory ---
    size_t bytes = (N + 1) * sizeof(char);
    
    char *d_is_prime;
    CHECK_CUDA(cudaMalloc(&d_is_prime, bytes));
    
    // --- 3. Initialize Sieve Array (GPU) ---
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- 4. Implement the Sieve Algorithm (Sequential as per PDF) ---
    // Algorithm: Start with 2, cross out multiples, then move to next unmarked (3), etc.
    // Stop at floor((N+1)/2)
    
    unsigned int sieve_limit = (N + 1) / 2;
    
    // Use pinned memory for faster single-byte transfers
    char *h_prime_check;
    CHECK_CUDA(cudaMallocHost(&h_prime_check, sizeof(char)));
    
    // Process each number from 2 to sieve_limit sequentially
    for (unsigned int p = 2; p <= sieve_limit; p++) {
        
        // Check if p is still marked as prime (device-to-host transfer)
        CHECK_CUDA(cudaMemcpy(h_prime_check, &d_is_prime[p], 
                             sizeof(char), cudaMemcpyDeviceToHost));
        
        if (*h_prime_check == 1) {
            // p is prime, so cross out all its multiples
            // Start from p*p (smaller multiples already crossed by smaller primes)
            unsigned long long start_multiple = (unsigned long long)p * p;
            
            if (start_multiple <= N) {
                // Calculate number of multiples to process
                unsigned int numMultiples = (N - start_multiple) / p + 1;
                
                // Launch kernel to cross out multiples in parallel
                unsigned int threads_sieve = 256;
                unsigned int blocks_sieve = (numMultiples + threads_sieve - 1) / threads_sieve;
                
                sieveKernel<<<blocks_sieve, threads_sieve>>>(d_is_prime, p, N);
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }
    
    // Final synchronization
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- 5. Copy Final Result Back to Host ---
    char *h_is_prime = (char *)malloc(bytes);
    if (h_is_prime == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        cudaFreeHost(h_prime_check);
        cudaFree(d_is_prime);
        return 1;
    }
    
    CHECK_CUDA(cudaMemcpy(h_is_prime, d_is_prime, bytes, cudaMemcpyDeviceToHost));
    
    // --- 6. Write Output to File ---
    char filename[256];
    sprintf(filename, "%u.txt", N);
    
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        free(h_is_prime);
        cudaFreeHost(h_prime_check);
        cudaFree(d_is_prime);
        return 1;
    }
    
    // Handle output spacing: first prime without leading space, rest with space
    unsigned int first_p = 2;
    while (first_p <= N && h_is_prime[first_p] == 0) {
        first_p++;
    }
    
    if (first_p <= N) {
        fprintf(fp, "%u", first_p);
    }
    
    // Print remaining primes with leading space
    for (unsigned int i = first_p + 1; i <= N; i++) {
        if (h_is_prime[i] == 1) {
            fprintf(fp, " %u", i);
        }
    }
    fprintf(fp, "\n");
    
    // --- 7. Cleanup ---
    fclose(fp);
    free(h_is_prime);
    cudaFreeHost(h_prime_check);
    cudaFree(d_is_prime);
    
    return 0;
}

