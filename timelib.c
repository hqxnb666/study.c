/*******************************************************************************
 * Time Functions Library (implementation)
 *
 * Description:
 *     A library to handle various time-related functions and operations.
 *
 * Author:
 *     Renato Mancuso <rmancuso@bu.edu>
 *
 * Affiliation:
 *     Boston University
 *
 * Creation Date:
 *     September 10, 2023
 *
 * Last Changes:
 *     September 21, 2024
 *
 * Notes:
 *     Ensure to link against the necessary dependencies when compiling and
 *     using this library. Modifications or improvements are welcome. Please
 *     refer to the accompanying documentation for detailed usage instructions.
 *
 *******************************************************************************/

#include "timelib.h"

/* Macro wrapper for RDTSC instruction */
#define get_clocks(clocks)						\
    do {										\
        unsigned int __a, __d;					\
        asm volatile("rdtsc" : "=a" (__a), "=d" (__d)); \
        (clocks) = ((unsigned long long)__a) | (((unsigned long long)__d) << 32); \
    } while(0)

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using sleeping functions */
uint64_t get_elapsed_sleep(long sec, long nsec)
{
    uint64_t start, end;
    struct timespec wait_time = {sec, nsec};

    get_clocks(start);
    nanosleep(&wait_time, NULL);
    get_clocks(end);

    return end - start;
}

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using busy-waiting functions */
uint64_t get_elapsed_busywait(long sec, long nsec)
{
    uint64_t start, end;
    struct timespec start_time, current_time, wait_time = {sec, nsec};

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    get_clocks(start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
    } while (timespec_cmp(&current_time, &start_time) < 0 ||
             (current_time.tv_sec - start_time.tv_sec < wait_time.tv_sec) ||
             (current_time.tv_sec - start_time.tv_sec == wait_time.tv_sec &&
              current_time.tv_nsec - start_time.tv_nsec < wait_time.tv_nsec));

    get_clocks(end);

    return end - start;
}

/* Busywait for the amount of time described via the delay
 * parameter */
uint64_t busywait_timespec(struct timespec delay)
{
    uint64_t start, end;
    struct timespec start_time, current_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    get_clocks(start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
    } while (timespec_cmp(&current_time, &start_time) < 0 ||
             (current_time.tv_sec - start_time.tv_sec < delay.tv_sec) ||
             (current_time.tv_sec - start_time.tv_sec == delay.tv_sec &&
              current_time.tv_nsec - start_time.tv_nsec < delay.tv_nsec));

    get_clocks(end);

    return end - start;
}

/* Utility function to add two timespec structures together. The input
 * parameter a is updated with the result of the sum. */
void timespec_add (struct timespec * a, struct timespec * b)
{
    /* Try to add up the nsec and see if we spill over into the
     * seconds */
    time_t addl_seconds = b->tv_sec;
    a->tv_nsec += b->tv_nsec;
    if (a->tv_nsec >= NANO_IN_SEC) {
        addl_seconds += a->tv_nsec / NANO_IN_SEC;
        a->tv_nsec = a->tv_nsec % NANO_IN_SEC;
    }
    a->tv_sec += addl_seconds;
}

/* Utility function to compare two timespec structures. It returns 1
 * if a is in the future compared to b; -1 if b is in the future
 * compared to a; 0 if they are identical. */
int timespec_cmp(struct timespec *a, struct timespec *b)
{
    if(a->tv_sec == b->tv_sec && a->tv_nsec == b->tv_nsec) {
        return 0;
    } else if((a->tv_sec > b->tv_sec) ||
          (a->tv_sec == b->tv_sec && a->tv_nsec > b->tv_nsec)) {
        return 1;
    } else {
        return -1;
    }
}

/* Busywait for the amount of time described via the delay
 * parameter */
uint64_t busywait_timespec(struct timespec delay)
{
    uint64_t start, end;
    struct timespec start_time, current_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    get_clocks(start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
    } while (timespec_cmp(&current_time, &start_time) < 0 ||
             (current_time.tv_sec - start_time.tv_sec < delay.tv_sec) ||
             (current_time.tv_sec - start_time.tv_sec == delay.tv_sec &&
              current_time.tv_nsec - start_time.tv_nsec < delay.tv_nsec));

    get_clocks(end);

    return end - start;
}
