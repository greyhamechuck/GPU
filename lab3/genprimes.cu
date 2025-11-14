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

// Kernel to find the next prime number starting from 'start'
// Uses atomicMin to ensure we get the smallest prime found by any thread
__global__ void find_next_prime_kernel(char* is_prime, long start, long end, unsigned long long* next_prime) {
    long global_id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    
    // Initialize to a large value
    unsigned long long local_min = (unsigned long long)(end + 1);
    
    // Each thread finds the minimum prime in its range
    for (long i = start + global_id; i <= end; i += stride) {
        if (i > 1 && is_prime[i] == 1) {
            unsigned long long val = (unsigned long long)i;
            if (val < local_min) {
                local_min = val;
            }
        }
    }
    
    // Use atomicMin to find the global minimum across all threads
    if (local_min <= (unsigned long long)end) {
        atomicMin(next_prime, local_min);
    }
}

// Kernel to mark all multiples of prime p (vertical processing of multiples)
// Each thread handles different multiples in parallel - this is the "vertical" optimization
__global__ void mark_multiples_kernel(char* is_prime, long p, long N) {
    long global_id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    
    // Start from p*p (optimization: smaller multiples already marked by smaller primes)
    long start = p * p;
    if (start > N) return;
    
    // Each thread marks multiples with stride - vertical parallel processing
    for (long multiple = start + global_id * p; multiple <= N; multiple += stride * p) {
        is_prime[multiple] = 0; // "Cross out" the number
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

    // Vertical approach: Follow Sieve of Eratosthenes sequentially
    // Start with 2, cross out its multiples, then move to next unmarked number (3), etc.
    // The "vertical" optimization: find next prime on GPU (no device-to-host transfer)
    // and process multiples in parallel (vertical processing of multiples)
    
    long sieve_limit = (N + 1) / 2;
    
    // Allocate device memory for finding next prime (use unsigned long long for atomicMin)
    unsigned long long* d_next_prime;
    CHECK_CUDA(cudaMalloc(&d_next_prime, sizeof(unsigned long long)));
    
    long current_p = 1; // Start before 2
    
    while (current_p < sieve_limit) {
        // Find next prime starting from current_p + 1
        long search_start = current_p + 1;
        
        // Initialize to a large value (larger than any possible prime)
        unsigned long long init_value = (unsigned long long)(sieve_limit + 1);
        CHECK_CUDA(cudaMemcpy(d_next_prime, &init_value, sizeof(unsigned long long), cudaMemcpyHostToDevice));
        
        // Launch kernel to find next prime on GPU
        // This finds the smallest prime >= search_start
        find_next_prime_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_is_prime, search_start, sieve_limit, d_next_prime);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy result back
        unsigned long long h_next_prime_ull = 0;
        CHECK_CUDA(cudaMemcpy(&h_next_prime_ull, d_next_prime, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        long h_next_prime = (long)h_next_prime_ull;
        
        // Check if we found a valid prime
        if (h_next_prime > sieve_limit || h_next_prime <= current_p) {
            break; // No more primes found
        }
        
        current_p = h_next_prime;
        
        // Mark all multiples of current_p in parallel (vertical processing)
        // Calculate number of multiples to process
        long start_multiple = current_p * current_p;
        if (start_multiple <= N) {
            long num_multiples = (N - start_multiple) / current_p + 1;
            int blocks_for_multiples = (num_multiples + threadsPerBlock - 1) / threadsPerBlock;
            if (blocks_for_multiples > 0) {
                mark_multiples_kernel<<<blocks_for_multiples, threadsPerBlock>>>(
                    d_is_prime, current_p, N);
            }
        }
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Clean up device memory
    CHECK_CUDA(cudaFree(d_next_prime));
    
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

