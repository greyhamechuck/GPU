/*
 * seqprimes.c
 *
 * Sequential prime number generator for CSCI-GA.3033-025, Lab 3.
 * Implements the sieve algorithm as described in the lab handout.
 *
 * Usage: ./seqprimes N
 * Input: N (a positive number > 2)
 * Output: A file named N.txt (e.g., 10.txt) containing all prime
 * numbers from 2 to N, separated by single spaces.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Using string.h for memset
#include <math.h>   // Using math.h for floor

int main(int argc, char *argv[]) {

    // --- 1. Process Input ---
    // Check for correct number of arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    // Parse N from command line argument
    // Use unsigned long for parsing to handle large inputs safely
    unsigned long N_long = strtoul(argv[1], NULL, 10);
    
    // Check if N fits in unsigned int, as specified by the lab
    if (N_long > (unsigned int)-1 || N_long <= 2) {
        fprintf(stderr, "Error: N must be a number > 2 and within the unsigned int limit.\n");
        return 1;
    }
    unsigned int N = (unsigned int)N_long;

    // --- 2. Allocate Memory for Sieve ---
    // We need a boolean array from 0 to N.
    // Using char as a boolean: 1 = prime (uncrossed), 0 = composite (crossed)
    // We allocate N+1 elements to use 1-based indexing (e.g., index `i` maps to number `i`)
    char *is_prime = (char *)malloc((N + 1) * sizeof(char));
    if (is_prime == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for %u numbers.\n", N);
        return 1;
    }

    // --- 3. Implement the Sieve Algorithm ---
    
    // "1. Generate all numbers from 2 to N."
    // Initialize all numbers from 2 to N as potentially prime (1).
    memset(is_prime, 1, (N + 1) * sizeof(char));
    is_prime[0] = 0; // 0 is not prime
    is_prime[1] = 0; // 1 is not prime

    // Calculate the stopping point for the outer loop
    // "5. Continue like this till floor((N+1)/2)"
    unsigned int stop_point = (unsigned int)floor((N + 1) / 2.0);

    // Start sieving from p = 2
    for (unsigned int p = 2; p <= stop_point; p++) {
        
        // Find the next number 'p' that has not been crossed out
        if (is_prime[p] == 1) {
            
            // 'p' is prime. Now, cross out all its multiples.
            // "remove all numbers that are multiple of [p]... till you reach N"
            // Start crossing from 2*p
            for (unsigned int i = 2 * p; i <= N; i += p) {
                is_prime[i] = 0; // Cross it out (mark as composite)
            }
        }
    }
    // "6. The remaining numbers are the prime numbers."


    // --- 4. Write Output to File ---
    // "The output of your program is a text file N.txt"
    char filename[256];
    sprintf(filename, "%u.txt", N);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        free(is_prime);
        return 1;
    }

    // "You put a single space between each two numbers."
    // Use a flag to avoid a leading or trailing space.
    int is_first_prime = 1;
    for (unsigned int i = 2; i <= N; i++) {
        if (is_prime[i] == 1) { // If 'i' was not crossed out
            if (is_first_prime) {
                fprintf(fp, "%u", i);
                is_first_prime = 0; // All subsequent primes will get a space
            } else {
                fprintf(fp, " %u", i);
            }
        }
    }
    // Add a newline for a clean file, though not explicitly required
    fprintf(fp, "\n");

    // --- 5. Cleanup ---
    fclose(fp);
    free(is_prime);

    return 0; // Success
}