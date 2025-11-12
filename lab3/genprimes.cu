#include <stdio.h>
#include <stdlib.h>
#include <math.h>   
#include <cuda_runtime.h>

// --- CUDA Error Checking Utility ---
// A utility function to check and report CUDA API errors
static void checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, 
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
// Define a macro to wrap CUDA calls for error checking
#define CHECK_CUDA_ERROR(err) (checkCudaErrors(err, __FILE__, __LINE__))

// --- CUDA Kernels ---

/**
 * @brief Kernel to initialize the prime sieve array on the GPU.
 * Sets all values from 2 to N to 1 (true), and 0/1 to 0 (false).
 */
__global__ void initKernel(char *d_is_prime, unsigned int N) {
    // Calculate the global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a grid-stride loop to ensure all elements are covered
    for (unsigned int i = idx; i <= N; i += gridDim.x * blockDim.x) {
        if (i == 0 || i == 1) {
            d_is_prime[i] = 0; // 0 and 1 are not prime
        } else {
            d_is_prime[i] = 1; // Mark as potentially prime
        }
    }
}

/**
 * @brief Kernel to "cross out" all multiples of a given prime 'p'.
 * Each thread handles one multiple of p.
 */
__global__ void sieveKernel(char *d_is_prime, unsigned int p, unsigned int N) {
    // Calculate the global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread 'idx' will cross out the (idx + 2)-th multiple of p
    // (We start from k=2 because 1*p is 'p' itself, which we don't cross out)
    unsigned int k = idx + 2;
    unsigned int multiple = p * k;

    // Check bounds and cross out the multiple
    if (multiple <= N) {
        d_is_prime[multiple] = 0; // Mark as composite
    }
}


// --- Host Code (main) ---

int main(int argc, char *argv[]) {

    // --- 1. Process Input ---
    // "execute it I type: ./genprimes N" [cite: 6]
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    // "N is a positive number ... less than or equal to the limit of unsigned int" [cite: 7]
    unsigned long N_long = strtoul(argv[1], NULL, 10);
    if (N_long > (unsigned int)-1) {
        fprintf(stderr, "Error: N exceeds the limit of unsigned int.\n");
        return 1;
    }
    unsigned int N = (unsigned int)N_long;
    if (N <= 2) {
         fprintf(stderr, "Error: N must be bigger than 2.\n");
         return 1;
    }
    
    // --- 2. Setup Memory ---
    size_t bytes = (N + 1) * sizeof(char);
    
    char *h_is_prime; // Host memory for final result
    char *d_is_prime; // Device memory for sieving

    // Allocate host memory
    h_is_prime = (char *)malloc(bytes);
    if (h_is_prime == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory.\n");
        return 1;
    }

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_is_prime, bytes));

    // --- 3. Initialize Sieve Array (GPU) ---
    // "1. Generate all numbers from 2 to N." [cite: 19]
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (N + 1 + threadsPerBlock - 1) / threadsPerBlock;
    
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_is_prime, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for init to complete

    // --- 4. Implement the Sieve Algorithm (Host-driven) ---

    // "5. Continue like this till floor((N+1)/2)." [cite: 26]
    unsigned int stop_point = (unsigned int)floor((N + 1) / 2.0);

    for (unsigned int p = 2; p <= stop_point; p++) {
        
        // "The next number that has not been crossed..." [cite: 24]
        // We must check if 'p' is still prime.
        // This requires copying the single byte from Device to Host.
        // This is a major performance bottleneck, as required for analysis.
        char is_p_prime;
        CHECK_CUDA_ERROR(cudaMemcpy(&is_p_prime, &d_is_prime[p], sizeof(char), cudaMemcpyDeviceToHost));
        
        if (is_p_prime == 1) {
            // 'p' is prime, so "remove all numbers that are multiple of [p]" [cite: 20]
            
            // Calculate number of multiples to cross out (from 2*p up to N)
            unsigned int numMultiples = N / p - 1; // (e.g., for N=20, p=2: 20/2 - 1 = 9 multiples)

            if (numMultiples > 0) {
                // Configure and launch the sieve kernel
                unsigned int threads_sieve = 256;
                unsigned int blocks_sieve = (numMultiples + threads_sieve - 1) / threads_sieve;

                sieveKernel<<<blocks_sieve, threads_sieve>>>(d_is_prime, p, N);
                CHECK_CUDA_ERROR(cudaGetLastError());
                
                // We must synchronize here to ensure that the crossing-out
                // is complete before the *next* iteration of the host loop
                // (e.g., before we check p=4). This is another bottleneck.
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            }
        }
    }
    // "6. The remaining numbers are the prime numbers." [cite: 27]

    // --- 5. Copy Final Result Back to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_is_prime, d_is_prime, bytes, cudaMemcpyDeviceToHost));

    // --- 6. Write Output to File ---
    // "The output ... is a text file N.txt" [cite: 8]
    char filename[256];
    sprintf(filename, "%u.txt", N);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        free(h_is_prime);
        cudaFree(d_is_prime);
        return 1;
    }

    // "You put a single space between each two numbers." [cite: 11]
    int is_first_prime = 1;
    for (unsigned int i = 2; i <= N; i++) {
        if (h_is_prime[i] == 1) { // If 'i' was not crossed out
            if (is_first_prime) {
                fprintf(fp, "%u", i);
                is_first_prime = 0;
            } else {
                fprintf(fp, " %u", i);
            }
        }
    }
    fprintf(fp, "\n"); // Add a newline for a clean file

    // --- 7. Cleanup ---
    fclose(fp);
    free(h_is_prime);
    CHECK_CUDA_ERROR(cudaFree(d_is_prime));

    return 0; // Success
}