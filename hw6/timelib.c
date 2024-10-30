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
* Notes:
*     Ensure to link against the necessary dependencies when compiling and
*     using this library. Modifications or improvements are welcome. Please
*     refer to the accompanying documentation for detailed usage instructions.
*
*******************************************************************************/

#include "timelib.h"

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using sleeping functions */
uint64_t get_elapsed_sleep(long sec, long nsec)
{
	uint64_t start, end;
	struct timespec wait_time;
	wait_time.tv_sec = sec;
	wait_time.tv_nsec = nsec;

	/* Get the start timestamp */
	get_clocks(start);

	/* Sleep X seconds */
	nanosleep(&wait_time, NULL);

	/* Get end timestamp */
	get_clocks(end);

	return (end - start);
}

/* Return the number of clock cycles elapsed when waiting for
 * wait_time seconds using busy-waiting functions */
uint64_t get_elapsed_busywait(long sec, long nsec)
{
	uint64_t start, end;
	struct timespec now;
	struct timespec time_end;

	/* Measure the current system time */
	clock_gettime(CLOCK_MONOTONIC, &now);
	time_end.tv_sec = sec;
	time_end.tv_nsec = nsec;
	timespec_add(&time_end, &now);

	/* Get the start timestamp */
	get_clocks(start);

	/* Busy wait until enough time has elapsed */
	do {
		clock_gettime(CLOCK_MONOTONIC, &now);
	} while (time_end.tv_sec > now.tv_sec || time_end.tv_nsec > now.tv_nsec);

	/* Get end timestamp */
	get_clocks(end);

	return (end - start);
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
	uint64_t start, end;
	struct timespec now;

	/* Measure the current system time */
	clock_gettime(CLOCK_MONOTONIC, &now);
	timespec_add(&delay, &now);

	/* Get the start timestamp */
	get_clocks(start);

	/* Busy wait until enough time has elapsed */
	do {
		clock_gettime(CLOCK_MONOTONIC, &now);
	} while (delay.tv_sec > now.tv_sec || delay.tv_nsec > now.tv_nsec);

	/* Get end timestamp */
	get_clocks(end);

	return (end - start);
}

/* Translate a double timestamp into a valid timespec */
inline struct timespec dtotspec(double timestamp)
{
	/* Timestamp assumed is in seconds, so fill timespec
	 * accordingly */
	struct timespec retval;
	retval.tv_sec = (long)timestamp;
	retval.tv_nsec = (long)(timestamp * NANO_IN_SEC) % NANO_IN_SEC;
	return retval;
}
