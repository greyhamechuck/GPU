/*
 * seqprimes.c
 *
 * More explicit, "lower-level" style.
 * This version avoids using memset() and floor() by
 * implementing that logic with explicit loops and integer arithmetic.
 */

#include <stdio.h>  // For file I/O (fopen, fprintf, etc.) and stderr
#include <stdlib.h> // For malloc(), free(), strtoul(), and exit()

int main(int argc, char *argv[]) {

    // --- 1. Process Input ---
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1; // Exit with an error code
    }

    unsigned long N_long = strtoul(argv[1], NULL, 10);
    
    if (N_long > (unsigned int)-1 || N_long <= 2) {
        fprintf(stderr, "Error: N must be > 2 and within the unsigned int limit.\n");
        return 1;
    }
    unsigned int N = (unsigned int)N_long;

    // --- 2. Allocate Memory for Sieve ---
    // We must use malloc (heap allocation) because N can be
    // billions, which would cause a stack overflow.
    char *is_prime = (char *)malloc((N + 1) * sizeof(char));
    
    // Always check if malloc succeeded
    if (is_prime == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for %u numbers.\n", N);
        return 1;
    }

    // --- 3. Implement the Sieve Algorithm ---
    
    // "1. Generate all numbers from 2 to N."
    // This loop explicitly replaces the function call memset()
    for (unsigned int i = 2; i <= N; i++) {
        is_prime[i] = 1; // Set all as potentially prime
    }
    is_prime[0] = 0; // 0 is not prime
    is_prime[1] = 0; // 1 is not prime

    // Calculate the stopping point using integer division
    // This replaces floor((N + 1) / 2.0)
    unsigned int stop_point = (N + 1) / 2;

    // Start sieving from p = 2
    for (unsigned int p = 2; p <= stop_point; p++) {
        
        // If 'p' is still marked as prime (1)...
        if (is_prime[p] == 1) {
            
            // ...then cross out all of its multiples.
            // Start crossing from p*p, or 2*p (as per lab)
            // Note: 2*p is simpler to follow from the lab text
            for (unsigned int i = 2 * p; i <= N; i += p) {
                is_prime[i] = 0; // Mark as composite
            }
        }
    }
    // "6. The remaining numbers are the prime numbers."


    // --- 4. Write Output to File ---
    char filename[256];
    sprintf(filename, "%u.txt", N);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        free(is_prime); // Don't forget to free memory on error
        return 1;
    }

    // Iterate and write all primes to the file
    int is_first_prime = 1;
    for (unsigned int i = 2; i <= N; i++) {
        if (is_prime[i] == 1) { // If 'i' was not crossed out
            if (is_first_prime) {
                fprintf(fp, "%u", i);
                is_first_prime = 0;
            } else {
                fprintf(fp, " %u", i);
            }
        }
    }
    fprintf(fp, "\n"); // Add a final newline

    // --- 5. Cleanup ---
    fclose(fp);
    free(is_prime); // Free the memory we allocated

    return 0; // Success
}