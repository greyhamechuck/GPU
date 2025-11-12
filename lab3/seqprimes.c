/*
 * seqprimes.c
 * NetID: [Your NetID Here]
 * Lab 3: Sequential Prime Sieve
 */

#include <stdio.h>
#include <stdlib.h> // For malloc, strtoul, free

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    // Parse N from command line
    unsigned int N = (unsigned int)strtoul(argv[1], NULL, 10);

    // Allocate memory for the sieve. 1 = prime, 0 = composite.
    // We use (N + 1) bytes to go from index 0 to N.
    char *sieve = (char *)malloc(N + 1);
    if (sieve == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    // Initialize the sieve array
    // We replace memset() with a simple loop
    for (unsigned int i = 2; i <= N; i++) {
        sieve[i] = 1; // Assume all are prime to start
    }
    sieve[0] = 0;
    sieve[1] = 0;

    // Calculate the limit for the outer loop
    // (N+1)/2 is integer division, which is the same as floor()
    unsigned int sieve_limit = (N + 1) / 2;

    // Run the sieve
    for (unsigned int p = 2; p <= sieve_limit; p++) {
        // If p is still marked as prime...
        if (sieve[p] == 1) {
            // ...cross out all of its multiples
            for (unsigned int i = 2 * p; i <= N; i += p) {
                sieve[i] = 0;
            }
        }
    }

    // --- Write output to file ---

    char filename[256];
    sprintf(filename, "%u.txt", N);

    FILE *outfile = fopen(filename, "w");
    if (outfile == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        free(sieve);
        return 1;
    }

    // Handle output spacing
    // Find the first prime and print it without a leading space
    unsigned int first_p = 2;
    while (first_p <= N && sieve[first_p] == 0) {
        first_p++;
    }
    
    // Check if any prime was found at all
    if (first_p <= N) {
        fprintf(outfile, "%u", first_p);
    }

    // Now, print all *other* primes, each with a leading space
    for (unsigned int i = first_p + 1; i <= N; i++) {
        if (sieve[i] == 1) {
            fprintf(outfile, " %u", i);
        }
    }

    // Cleanup
    fprintf(outfile, "\n");
    fclose(outfile);
    free(sieve);

    return 0;
}