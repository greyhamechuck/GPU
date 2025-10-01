#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE   19.87
#define BILLION 1000000000.00  // (minor cleanup: no trailing semicolon)

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__
void vecGPU(float* ad, float* bd, float* cd, int n);
/**** end of the kernel declaration ***/

int main(int argc, char *argv[]){

    int n = 0; // number of elements in the arrays
    int i;     // loop index
    float *a, *b, *c;   // host arrays
    float *temp;        // host reference (sequential) result
    float *ad, *bd, *cd; // device arrays

    struct timespec start2, end2; // to measure elapsed time
    double accum;

    if(argc != 2){
        printf("usage:  ./vectorprog n\n");
        printf("n = number of elements in each vector\n");
        exit(1);
    }

    n = atoi(argv[1]);
    printf("Each vector will have %d elements\n", n);

    // Host allocations
    if( !(a = (float *)malloc(n*sizeof(float))) ){ printf("Error allocating array a\n"); exit(1); }
    if( !(b = (float *)malloc(n*sizeof(float))) ){ printf("Error allocating array b\n"); exit(1); }
    if( !(c = (float *)malloc(n*sizeof(float))) ){ printf("Error allocating array c\n"); exit(1); }
    if( !(temp = (float *)malloc(n*sizeof(float))) ){ printf("Error allocating array temp\n"); exit(1); }

    // Initialize host arrays with random numbers in [0, RANGE)
    srand((unsigned int)time(NULL));
    for (i = 0; i < n;  i++){
        a[i]    = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        b[i]    = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        c[i]    = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        temp[i] = c[i]; // temp is copy of c for CPU baseline
    }

    // ===== Sequential part (CPU) =====
    clock_gettime(CLOCK_MONOTONIC, &start2);
    for(i = 0; i < n; i++)
        temp[i] += a[i] * b[i];
    clock_gettime(CLOCK_MONOTONIC, &end2);

    accum = ( end2.tv_sec - start2.tv_sec )
          + ( end2.tv_nsec - start2.tv_nsec ) / BILLION;
    printf("Total time taken by the sequential part = %lf seconds\n", accum);

    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
    // ===== GPU part =====

    /* TODO: in this part you need to do the following:
        1. allocate ad, bd, and cd in the device
        2. send a, b, and c to the device
        3. write the kernel, call it: vecGPU
        4. call the kernel (decide blocks/threads)
        5. bring cd back into c (host)
        6. free ad, bd, cd
    */

    int size = n * sizeof(float);
    cudaError_t err = cudaSuccess;

    // 1) device allocations
    err = cudaMalloc((void **)&ad, size);
    if(err != cudaSuccess){ fprintf(stderr, "Error allocating array ad on device: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMalloc((void **)&bd, size);
    if(err != cudaSuccess){ fprintf(stderr, "Error allocating array bd on device: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMalloc((void **)&cd, size);
    if(err != cudaSuccess){ fprintf(stderr, "Error allocating array cd on device: %s\n", cudaGetErrorString(err)); exit(1); }

    // —— START GPU TIMING BLOCK ——
    // We time: H2D copies + kernel + D2H copy (this matches the lab’s “GPU part”)
    clock_gettime(CLOCK_MONOTONIC, &start2);

    // 2) H2D copies (in timed region)
    err = cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ fprintf(stderr, "H2D a failed: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ fprintf(stderr, "H2D b failed: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ fprintf(stderr, "H2D c failed: %s\n", cudaGetErrorString(err)); exit(1); }

    // 4) launch kernel (E1/E2 configs: change these two lines only)
    int blocksPerGrid   = 8;    // use 4/8/16 and 250/500 per experiment
    int threadsPerBlock = 500;  // final submission requires 8 x 500
    vecGPU<<<blocksPerGrid, threadsPerBlock>>>(ad, bd, cd, n);

    // Always check and synchronize so kernel completion is inside the timed region
    err = cudaGetLastError();
    if(err != cudaSuccess){ fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); exit(1); }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err)); exit(1); }

    // 5) D2H copy (still in timed region)
    err = cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ fprintf(stderr, "D2H c failed: %s\n", cudaGetErrorString(err)); exit(1); }

    clock_gettime(CLOCK_MONOTONIC, &end2);
    // —— END GPU TIMING BLOCK ——

    accum = ( end2.tv_sec - start2.tv_sec )
          + ( end2.tv_nsec - start2.tv_nsec ) / BILLION;
    printf("Total time taken by the GPU part = %lf seconds\n", accum);

    // 6) device cleanup
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    /******************  The end of the GPU part: Do not modify anything in main() below this line  ************/

    // Correctness check (compare to CPU result with lab’s tolerance 0.009)
    for(i = 0; i < n; i++){
        if( fabsf(temp[i] - c[i]) >= 0.009f){ // compare up to second digit
            printf("Element %d in the result array does not match the sequential version\n", i);
            // You can break early or count mismatches; lab only requires detection.
            break;
        }
    }

    // Free host memory
    free(a); free(b); free(c); free(temp);

    return 0;
}

/**** TODO: Write the kernel itself below this line *****/

__global__
void vecGPU(float* ad, float* bd, float* cd, int n)
{
    // Grid-stride loop (clearer and equivalent to your manual i-loop)
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        cd[i] += ad[i] * bd[i];
    }
}
