/*******************************************************************************
* Time Functions Library (header)
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
* Notes:
*     Ensure to link against the necessary dependencies when compiling and
*     using this library. Modifications or improvements are welcome. Please
*     refer to the accompanying documentation for detailed usage instructions.
*
*******************************************************************************/
#ifndef __TIMELIB_H__
#define __TIMELIB_H__
/* DO NOT WRITE ANY CODE ABOVE THIS LINE*/

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

/* How many nanoseconds in a second */
#define NANO_IN_SEC (1000*1000*1000)

/* Macro wrapper for RDTSC instruction */
#define get_clocks(clocks)						\
	do {								\
		uint32_t __clocks_hi, __clocks_lo;			\
		__asm__ __volatile__("rdtsc" :				\
				     "=a" (__clocks_lo),		\
				     "=d" (__clocks_hi)			\
			);						\
		clocks = (((uint64_t)__clocks_hi) << 32) |		\
			((uint64_t)__clocks_lo);			\
	} while (0)

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using sleeping functions */
uint64_t get_elapsed_sleep(long sec, long nsec);

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using busy-waiting functions */
uint64_t get_elapsed_busywait(long sec, long nsec);

/* Busywait for the amount of time described via the delay
 * parameter */
uint64_t busywait_timespec(struct timespec delay);

/* Add two timespec structures together */
void timespec_add (struct timespec *, struct timespec *);

/* Compare two timespec structures with one another */
int timespec_cmp(struct timespec *, struct timespec *);

/* Translate a double timestamp into a valid timespec */
struct timespec dtotspec(double timestamp);


/* DO NOT WRITE ANY CODE BEYOND THIS LINE*/
#endif
