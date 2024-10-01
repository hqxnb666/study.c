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
* Last Update:
*     September 9, 2024
*
* Notes:
*     Ensure to link against the necessary dependencies when compiling and
*     using this library. Modifications or improvements are welcome. Please
*     refer to the accompanying documentation for detailed usage instructions.
*
*******************************************************************************/

#include "timelib.h"
#include <time.h>
#include <stdint.h>

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using sleeping functions */
uint64_t get_elapsed_sleep(long sec, long nsec)
    
{
    
	/* IMPLEMENT ME! */
	
	
	uint64_t before, after;
	
	struct timespec time = {sec, nsec};

    // Snapshot TSC before sleep
    get_clocks(before);

    // Sleep for the specified time
    nanosleep(&time, NULL);

    // Snapshot TSC after sleep
    get_clocks(after);

    return after - before;
}

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using busy-waiting functions */
uint64_t get_elapsed_busywait(long sec, long nsec)
{
	/* IMPLEMENT ME! */
	
    struct timespec begin_timestamp, current_timestamp;
    uint64_t before, after;
    long elapsed_sec, elapsed_nsec;

    // Get the starting system time
    clock_gettime(CLOCK_MONOTONIC, &begin_timestamp);

    // Snapshot TSC before busy waiting
    get_clocks(before);

    do{
        // Continuously get the current time
        clock_gettime(CLOCK_MONOTONIC, &current_timestamp);

        // Calculate elapsed time in seconds and nanoseconds
        elapsed_sec = current_timestamp.tv_sec - begin_timestamp.tv_sec;
        elapsed_nsec = current_timestamp.tv_nsec - begin_timestamp.tv_nsec;
        if (elapsed_nsec < 0) {
            elapsed_sec -= 1;               
            elapsed_nsec += 1000000000;  
        } 
    } while(elapsed_sec< sec || (elapsed_sec == sec && elapsed_nsec < nsec));


    // Snapshot TSC after busy wait
    get_clocks(after);

    return after - before;
  
}

/* Utility function to add two timespec structures together. The input
 * parameter a is updated with the result of the sum. */
void timespec_add (struct timespec * a, struct timespec * b)
{
	/* Try to add up the nsec and see if we spill over into the
	 * seconds */
	time_t addl_seconds = b->tv_sec;
	a->tv_nsec += b->tv_nsec;
	if (a->tv_nsec > NANO_IN_SEC) {
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
    struct timespec start, current;
    uint64_t before, after;

    // 获取开始时间和时钟周期数
    clock_gettime(CLOCK_MONOTONIC, &start);
    get_clocks(before);

    do {
        clock_gettime(CLOCK_MONOTONIC, &current);
    } while ((current.tv_sec - start.tv_sec) < delay.tv_sec ||
             ((current.tv_sec - start.tv_sec) == delay.tv_sec &&
              current.tv_nsec - start.tv_nsec < delay.tv_nsec));

    // 获取结束时钟周期数
    get_clocks(after);

    return after - before;
}

