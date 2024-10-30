/*
  Common definitions and includes between server and client.
*/

#ifndef __COMMON_H__
#define __COMMON_H__
/* DO NOT WRITE ANY CODE ABOVE THIS LINE*/

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <math.h>

/* Includes that are specific for TCP/IP */
#include <netinet/in.h>
#include <netinet/ip.h> /* superset of previous */

/* Includes that are specific for TCP/IP */
#include <netinet/in.h>
#include <netinet/ip.h> /* superset of previous */

/* Include our own timelib */
#include "timelib.h"

/* Include our own imglib */
#include "imglib.h"

/* Include our own md5lib */
#include "md5sum.h"

/* Define the value of 0 for a positive acknowledgement and 1 for
 * negative acknowledgement (rejected request) */
#define RESP_COMPLETED  0
#define RESP_REJECTED   1

/* This is a handy definition to print out runtime errors that report
 * the file and line number where the error was encountered. */
#define ERROR_INFO()							\
	do {								\
		fprintf(stdout, "Runtime error at %s:%d\n", __FILE__, __LINE__); \
		fflush(stdout);						\
	} while(0)

/* A simple macro to convert between a struct timespec and a double
 * representation of a timestamp. */
#define TSPEC_TO_DOUBLE(spec)				\
    ((double)(spec.tv_sec) + (double)(spec.tv_nsec)/NANO_IN_SEC)

/* Image operation opcodes */
enum img_opcode {
    IMG_UNUSED = 0,
    IMG_REGISTER,
    IMG_ROT90CLKW,
    IMG_BLUR,
    IMG_SHARPEN,
    IMG_VERTEDGES,
    IMG_HORIZEDGES,
    IMG_RETRIEVE
};

/* String version of the opcodes */
const char * __opcode_strings [] = {
    "IMG_UNUSED",
    "IMG_REGISTER",
    "IMG_ROT90CLKW",
    "IMG_BLUR",
    "IMG_SHARPEN",
    "IMG_VERTEDGES",
    "IMG_HORIZEDGES",
    "IMG_RETRIEVE"
};

/* Handy macro to render an opcode as a string */
#define OPCODE_TO_STRING(opcode)		\
    (__opcode_strings[opcode])

/* Request payload as sent by the client and received by the
 * server. */
struct request {
	uint64_t req_id;
	struct timespec req_timestamp;
	union {
		struct timespec req_length;
		struct {
			uint8_t  img_op;
			uint8_t  overwrite;
			uint64_t img_id;
		};
	};
};

/* Response payload as sent by the server and received by the
 * client. */
struct response {
	uint64_t req_id;
	uint64_t img_id;
	uint8_t  ack;
};

/* DO NOT WRITE ANY CODE BEYOND THIS LINE*/
#endif
