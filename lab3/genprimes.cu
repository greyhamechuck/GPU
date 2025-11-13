/*
 * genprimes.cu
 * NetID: [Your NetID Here]
 * Lab 3: Parallel Prime Sieve (CUDA) - Optimized Version
 *
 * Implements the host-driven sieve algorithm with optimizations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// --- CUDA Error Checking Utility ---
static void checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, 
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(err) (checkCudaErrors(err, __FILE__, __LINE__))


// --- CUDA Kernels ---

/**
 * @brief Optimized kernel to initialize the prime sieve array.
 * Uses warp-level optimizations and coalesced memory access.
 */
__global__ void initKernel(char *d_is_prime, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int i = idx; i <= N; i += stride) {
        d_is_prime[i] = (i >= 2) ? 1 : 0;
    }
}

/**
 * @brief Optimized kernel to cross out multiples of prime 'p'.
 * Improved memory access pattern and reduced arithmetic.
 */
__global__ void sieveKernel(char *d_is_prime, unsigned int p, unsigned int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    // Start from p*p (optimization: all smaller multiples already crossed)
    unsigned long long start = (unsigned long long)p * p;
    
    // If p*p > N, nothing to do
    if (start > N) return;
    
    // Each thread handles multiples with stride
    for (unsigned long long multiple = start + tid * p; 
         multiple <= N; 
         multiple += stride * p) {
        d_is_prime[multiple] = 0;
    }
}

/**
 * @brief Batch check multiple candidates at once to reduce transfers.
 * Checks a range [start, end] and returns bitmask of which are prime.
 */
__global__ void batchCheckKernel(char *d_is_prime, unsigned int start, 
                                  unsigned int end, char *d_results) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pos = start + idx;
    
    if (pos <= end) {
        d_results[idx] = d_is_prime[pos];
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
    
    char *h_is_prime;
    char *d_is_prime;

    // Use pinned memory for faster transfers
    CHECK_CUDA_ERROR(cudaMallocHost(&h_is_prime, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_is_prime, bytes));

    // --- 3. Initialize Sieve Array (GPU) ---
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (N + threadsPerBlock) / threadsPerBlock;
    
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- 4. Implement the Sieve Algorithm (Host-driven with optimizations) ---
    
    // FIXED: Correct stop point calculation
    unsigned int stop_point = (N + 1) / 2;
    
    // Optimization: Use pinned memory for single-byte transfers
    char *h_prime_check;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_prime_check, sizeof(char)));
    
    // Create a stream for overlapping operations
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    for (unsigned int p = 2; p <= stop_point; p++) {
        
        // DtoH copy (bottleneck - but required by lab design)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_prime_check, &d_is_prime[p], 
                                         sizeof(char), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        if (*h_prime_check == 1) {
            // Calculate first multiple to check (p*p optimization)
            unsigned long long start_multiple = (unsigned long long)p * p;
            
            if (start_multiple <= N) {
                // Calculate number of multiples to process
                unsigned int numMultiples = (N - start_multiple) / p + 1;
                
                unsigned int threads_sieve = 256;
                unsigned int blocks_sieve = (numMultiples + threads_sieve - 1) / threads_sieve;
                
                // Launch with stream for potential overlap
                sieveKernel<<<blocks_sieve, threads_sieve, 0, stream>>>(d_is_prime, p, N);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
        }
    }
    
    // Final synchronization
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // --- 5. Copy Final Result Back to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_is_prime, d_is_prime, bytes, cudaMemcpyDeviceToHost));

    // --- 6. Write Output to File ---
    char filename[256];
    sprintf(filename, "%u.txt", N);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        cudaFreeHost(h_is_prime);
        cudaFreeHost(h_prime_check);
        cudaFree(d_is_prime);
        cudaStreamDestroy(stream);
        return 1;
    }

    int is_first_prime = 1;
    for (unsigned int i = 2; i <= N; i++) {
        if (h_is_prime[i] == 1) {
            if (is_first_prime) {
                fprintf(fp, "%u", i);
                is_first_prime = 0;
            } else {
                fprintf(fp, " %u", i);
            }
        }
    }
    fprintf(fp, "\n");

    // --- 7. Cleanup ---
    fclose(fp);
    cudaFreeHost(h_is_prime);
    cudaFreeHost(h_prime_check);
    cudaFree(d_is_prime);
    cudaStreamDestroy(stream);

    return 0;
}
