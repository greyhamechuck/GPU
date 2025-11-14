// genprimes.cu
//
// Usage: ./genprimes N
// Output: N.txt containing all primes in [2, N], separated by spaces.
//
// Algorithm: sieve-style "cross multiples" on the GPU.
// Host controls p, GPU marks multiples of p as composite.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple CUDA error check macro (optional but useful)
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


// Kernel: each thread marks one multiple of p as composite.
__global__
void markMultiples(unsigned char* isComposite,
                   unsigned int N,
                   unsigned int p)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Start at 2*p to follow the simple "cross multiples" description.
    // (You could use p*p as an optimization if desired.)
    unsigned int start = 2u * p;
    unsigned int j = start + tid * p;

    if (j <= N) {
        isComposite[j] = 1;
    }
}


int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse N as 64-bit first, then clamp to 32-bit for indexing
    unsigned long long N64 = strtoull(argv[1], nullptr, 10);
    if (N64 < 2 || N64 > 0xFFFFFFFFull) {
        fprintf(stderr, "N must be in [2, 2^32-1]\n");
        return EXIT_FAILURE;
    }
    unsigned int N = static_cast<unsigned int>(N64);

    // Host array for final composite flags
    unsigned char* h_isComposite =
        (unsigned char*)malloc((size_t)(N + 1) * sizeof(unsigned char));
    if (!h_isComposite) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // Device array
    unsigned char* d_isComposite = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_isComposite,
                          (size_t)(N + 1) * sizeof(unsigned char)));

    // Initialize all entries to 0 = "not composite"
    CHECK_CUDA(cudaMemset(d_isComposite, 0,
                          (size_t)(N + 1) * sizeof(unsigned char)));

    // Mark 0 and 1 as composite explicitly on device
    unsigned char ones[2] = {1, 1};
    CHECK_CUDA(cudaMemcpy(d_isComposite,
                          ones,
                          2 * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Outer loop over p on the host:
    //
    // Lab handout version: p <= (N + 1) / 2  (strict reading)
    // Classical sieve optimization: p <= floor(sqrt(N)).
    //
    // To keep very close to the "cross multiples" description,
    // use (N + 1) / 2 here.
    unsigned int limit = (N + 1u) / 2u;

    const int THREADS_PER_BLOCK = 256;

    for (unsigned int p = 2; p <= limit; ++p) {
        // Check if p is already marked composite on device
        unsigned char flag = 0;
        CHECK_CUDA(cudaMemcpy(&flag,
                              d_isComposite + p,
                              sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));

        if (flag != 0) {
            // p is composite -> skip
            continue;
        }

        // p is prime -> launch kernel to cross its multiples
        unsigned int start = 2u * p;
        if (start > N) {
            // No multiples of p in [2, N] beyond p itself
            continue;
        }

        // Number of multiples of p in [start, N]
        unsigned int numMultiples = ((N - start) / p) + 1u;

        unsigned int blocks =
            (numMultiples + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;

        markMultiples<<<blocks, THREADS_PER_BLOCK>>>(d_isComposite, N, p);

        // Check async kernel launch
        CHECK_CUDA(cudaGetLastError());

        // In the default stream, the next cudaMemcpy will
        // implicitly synchronize with this kernel, so an
        // explicit cudaDeviceSynchronize() is not strictly required
        // here unless you want to measure time around this loop.
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Copy composite flags back to host
    CHECK_CUDA(cudaMemcpy(h_isComposite,
                          d_isComposite,
                          (size_t)(N + 1) * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_isComposite));

    // Open output file "<N>.txt"
    char filename[64];
    std::snprintf(filename, sizeof(filename), "%u.txt", N);
    FILE* f = std::fopen(filename, "w");
    if (!f) {
        perror("fopen");
        free(h_isComposite);
        return EXIT_FAILURE;
    }

    // Write primes in [2, N] where isComposite[i] == 0
    bool first = true;
    for (unsigned int i = 2; i <= N; ++i) {
        if (!h_isComposite[i]) {
            if (!first) {
                std::fputc(' ', f);
            }
            std::fprintf(f, "%u", i);
            first = false;
        }
    }

    std::fclose(f);
    free(h_isComposite);

    return EXIT_SUCCESS;
}
