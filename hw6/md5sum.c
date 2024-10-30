/*******************************************************************************
* MD5 Cryptographic Hash Computation Library (implementation)
*
* Description: A simple library to handle the calculation of MD5
*     hashes. The code has been adapted from the NetLib project
*     (https://www.netlib.org/). It has been modified to export the
*     md5sum function without a main function. The original comment is
*     kept below to provide appropriate credit to the original
*     creator.
*
* Author:
*     Renato Mancuso <rmancuso@bu.edu>
*
* Affiliation:
*     Boston University
*
* Creation Date:
*     October 30, 2023
*
* Notes:
*     Ensure to link against the necessary dependencies when compiling and
*     using this library. Modifications or improvements are welcome. Please
*     refer to the accompanying documentation for detailed usage instructions.
*
*******************************************************************************/

/*  * Comment block in original code. See https://www.netlib.org/crc/md5sum.c *
 *  rfc1321 requires that I include this.  The code is new.  The constants
 *  all come from the rfc (hence the copyright).  We trade a table for the
 *  macros in rfc.  The total size is a lot less.
 *  presotto@plan9.bell-labs.com  (with tweaks by ehg@netlib.bell-labs.com)
 *
 *	Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
 *	rights reserved.
 *
 *	License to copy and use this software is granted provided that it
 *	is identified as the "RSA Data Security, Inc. MD5 Message-Digest
 *	Algorithm" in all material mentioning or referencing this software
 *	or this function.
 *
 *	License is also granted to make and use derivative works provided
 *	that such works are identified as "derived from the RSA Data
 *	Security, Inc. MD5 Message-Digest Algorithm" in all material
 *	mentioning or referencing the derived work.
 *
 *	RSA Data Security, Inc. makes no representations concerning either
 *	the merchantability of this software or the suitability of this
 *	software forany particular purpose. It is provided "as is"
 *	without express or implied warranty of any kind.
 *	These notices must be retained in any copies of any part of this
 *	documentation and/or software.
 */

#include "md5sum.h"

typedef struct MD5state
{
	uint len;
	uint state[4];
} MD5state;

/*
 *	Rotate amounts used in the algorithm
 */
enum
{
	S11=	7,
	S12=	12,
	S13=	17,
	S14=	22,

	S21=	5,
	S22=	9,
	S23=	14,
	S24=	20,

	S31=	4,
	S32=	11,
	S33=	16,
	S34=	23,

	S41=	6,
	S42=	10,
	S43=	15,
	S44=	21
};

typedef struct Table
{
	uint	sin;	/* integer part of 4294967296 times abs(sin(i)) */
	byte	x;	/* index into data block */
	byte	rot;	/* amount to rotate left by */
} Table;

Table tab[] =
{
	/* round 1 */
	{ 0xd76aa478, 0, S11},
	{ 0xe8c7b756, 1, S12},
	{ 0x242070db, 2, S13},
	{ 0xc1bdceee, 3, S14},
	{ 0xf57c0faf, 4, S11},
	{ 0x4787c62a, 5, S12},
	{ 0xa8304613, 6, S13},
	{ 0xfd469501, 7, S14},
	{ 0x698098d8, 8, S11},
	{ 0x8b44f7af, 9, S12},
	{ 0xffff5bb1, 10, S13},
	{ 0x895cd7be, 11, S14},
	{ 0x6b901122, 12, S11},
	{ 0xfd987193, 13, S12},
	{ 0xa679438e, 14, S13},
	{ 0x49b40821, 15, S14},

	/* round 2 */
	{ 0xf61e2562, 1, S21},
	{ 0xc040b340, 6, S22},
	{ 0x265e5a51, 11, S23},
	{ 0xe9b6c7aa, 0, S24},
	{ 0xd62f105d, 5, S21},
	{  0x2441453, 10, S22},
	{ 0xd8a1e681, 15, S23},
	{ 0xe7d3fbc8, 4, S24},
	{ 0x21e1cde6, 9, S21},
	{ 0xc33707d6, 14, S22},
	{ 0xf4d50d87, 3, S23},
	{ 0x455a14ed, 8, S24},
	{ 0xa9e3e905, 13, S21},
	{ 0xfcefa3f8, 2, S22},
	{ 0x676f02d9, 7, S23},
	{ 0x8d2a4c8a, 12, S24},

	/* round 3 */
	{ 0xfffa3942, 5, S31},
	{ 0x8771f681, 8, S32},
	{ 0x6d9d6122, 11, S33},
	{ 0xfde5380c, 14, S34},
	{ 0xa4beea44, 1, S31},
	{ 0x4bdecfa9, 4, S32},
	{ 0xf6bb4b60, 7, S33},
	{ 0xbebfbc70, 10, S34},
	{ 0x289b7ec6, 13, S31},
	{ 0xeaa127fa, 0, S32},
	{ 0xd4ef3085, 3, S33},
	{  0x4881d05, 6, S34},
	{ 0xd9d4d039, 9, S31},
	{ 0xe6db99e5, 12, S32},
	{ 0x1fa27cf8, 15, S33},
	{ 0xc4ac5665, 2, S34},

	/* round 4 */
	{ 0xf4292244, 0, S41},
	{ 0x432aff97, 7, S42},
	{ 0xab9423a7, 14, S43},
	{ 0xfc93a039, 5, S44},
	{ 0x655b59c3, 12, S41},
	{ 0x8f0ccc92, 3, S42},
	{ 0xffeff47d, 10, S43},
	{ 0x85845dd1, 1, S44},
	{ 0x6fa87e4f, 8, S41},
	{ 0xfe2ce6e0, 15, S42},
	{ 0xa3014314, 6, S43},
	{ 0x4e0811a1, 13, S44},
	{ 0xf7537e82, 4, S41},
	{ 0xbd3af235, 11, S42},
	{ 0x2ad7d2bb, 2, S43},
	{ 0xeb86d391, 9, S44},
};

int test_main(int argc, char **argv)
{
	int c;
	struct md5digest d;

	for(c = 1; c < argc; c++){
		d = file_md5sum(argv[c]);
		print_digest(d);
	}

	return EXIT_SUCCESS;
}

/*
 *	encodes input (uint) into output (byte). Assumes len is
 *	a multiple of 4.
 */
static void encode(byte *output, uint *input, uint len)
{
	uint x;
	byte *e;

	for(e = output + len; output < e;) {
		x = *input++;
		*output++ = x;
		*output++ = x >> 8;
		*output++ = x >> 16;
		*output++ = x >> 24;
	}
}

/*
 *	decodes input (byte) into output (uint). Assumes len is
 *	a multiple of 4.
 */
static void decode(uint *output, byte *input, uint len)
{
	byte *e;

	for(e = input+len; input < e; input += 4)
		*output++ = input[0] | (input[1] << 8) |
			(input[2] << 16) | (input[3] << 24);
}

/*
 *  I require len to be a multiple of 64 for all but
 *  the last call
 */
static MD5state* md5(byte *p, uint len, byte *digest, MD5state *s)
{
	uint a, b, c, d, tmp;
	uint i, done;
	Table *t;
	byte *end;
	uint x[16];

	if(s == NULL){
		s = calloc(sizeof(*s),1);
		if(s == NULL)
			return NULL;

		/* seed the state, these constants would look nicer big-endian */
		s->state[0] = 0x67452301;
		s->state[1] = 0xefcdab89;
		s->state[2] = 0x98badcfe;
		s->state[3] = 0x10325476;
	}
	s->len += len;

	i = len & 0x3f;
	if(i || len == 0){
		done = 1;

		/* pad the input, assume there's room */
		if(i < 56)
			i = 56 - i;
		else
			i = 120 - i;
		if(i > 0){
			memset(p + len, 0, i);
			p[len] = 0x80;
		}
		len += i;

		/* append the count */
		x[0] = s->len<<3;
		x[1] = s->len>>29;
		encode(p+len, x, 8);
	} else
		done = 0;

	for(end = p+len; p < end; p += 64){
		a = s->state[0];
		b = s->state[1];
		c = s->state[2];
		d = s->state[3];

		decode(x, p, 64);

		for(i = 0; i < 64; i++){
			t = tab + i;
			switch(i>>4){
			case 0:
				a += (b & c) | (~b & d);
				break;
			case 1:
				a += (b & d) | (c & ~d);
				break;
			case 2:
				a += b ^ c ^ d;
				break;
			case 3:
				a += c ^ (b | ~d);
				break;
			}
			a += x[t->x] + t->sin;
			a = (a << t->rot) | (a >> (32 - t->rot));
			a += b;

			/* rotate variables */
			tmp = d;
			d = c;
			c = b;
			b = a;
			a = tmp;
		}

		s->state[0] += a;
		s->state[1] += b;
		s->state[2] += c;
		s->state[3] += d;
	}

	/* return result */
	if(done){
		encode(digest, s->state, 16);
		free(s);
		return NULL;
	}
	return s;
}

struct md5digest buf_md5sum(const char * orig_buf, size_t len)
{
	struct md5digest digest;
	MD5state * s = NULL;

	byte * buf = calloc(1, len + 256);
	memcpy(buf, orig_buf, len);

	/* For the first call, n must be a multiple of 64 */
	size_t n = (len >> 6) << 6;
	s = md5(buf, n, NULL, s);

	/* Set n to the remainder, potentially 0 */
	md5(buf+n, len % 64, digest.__digest, s);

	free(buf);
	return digest;
}

struct md5digest file_md5sum(const char *name)
{
	byte *buf;
	struct md5digest digest;
	int i, n;
	MD5state *s;

	int fd = open(name, O_RDONLY);
	if(fd < 0){
		memset(&digest, 0, sizeof(struct md5digest));
		fprintf(stderr, "md5sum: can't open %s\n", name);
		return digest;
	}

	s = NULL;
	n = 0;
	buf = calloc(256,64);
	for(;;){
		i = read(fd, buf+n, 128*64-n);
		if(i <= 0)
			break;
		n += i;
		if(n & 0x3f)
			continue;
		s = md5(buf, n, NULL, s);
		n = 0;
	}
	md5(buf, n, digest.__digest, s);
	free(buf);
	close(fd);
	return digest;
}

