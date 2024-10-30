/*******************************************************************************
* Image Manipulation Library (header)
*
* Description:
*     A library to handle various image-related functions and operations.
*
* Author:
*     Renato Mancuso <rmancuso@bu.edu>
*     Koneshka Bandyopadhyay <kon1402@bu.edu>
*
* Affiliation:
*     Boston University
*
* Creation Date:
*     October 23, 2023
*
* Update Date:
*     October 24, 2024
*
* Notes:
*     Ensure to link against the necessary dependencies when compiling and
*     using this library. Modifications or improvements are welcome. Please
*     refer to the accompanying documentation for detailed usage instructions.
*
*******************************************************************************/

#ifndef __IMGLIB_H__
#define __IMGLIB_H__
/* DO NOT WRITE ANY CODE ABOVE THIS LINE */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

struct image {
	uint32_t width; /* The width of the image */
	uint32_t height; /* The height of the image */
	uint32_t * pixels; /* Array of pixel values in x-y order */
};

#pragma pack(push, 1)  // Ensure structure is packed

typedef struct {
    uint16_t type;              // Magic identifier: 0x4d42
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;              // Offset to image data in bytes
} BMPHeader;

typedef struct {
    uint32_t size;                // Header size in bytes
    uint32_t width, height;       // Width and height of image
    uint16_t planes;            // Number of color planes
    uint16_t bits;              // Bits per pixel
    uint32_t compression;         // Compression type
    uint32_t imagesize;           // Image size in bytes
    uint32_t xresolution, yresolution;
    uint32_t ncolors;             // Number of colors
    uint32_t importantcolors;     // Important colors
} BMPInfoHeader;

#pragma pack(pop)  // End packed structure

/* Allocate and initialize the memory and metadata for a new
 * <width>x<height> pixels. */
struct image * createImage(uint32_t width, uint32_t height);

/* Deallocate all the memory for a given image. */
void deleteImage(struct image * img);

/* Set a specific pixel at position (<x>,<y>) in the image <img> to a
 * specific <value>. The function returns 0 if the operation is
 * successful and 1 in case of error. */
uint8_t setPixel(struct image * img, uint32_t x, uint32_t y, uint32_t value);

/*
 * Get the value of a specific pixel at position (<x>,<y>) in the
 * image <img>.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, 0 is returned by the function.
*/
uint32_t getPixel(const struct image * img, uint32_t x, uint32_t y, uint8_t * err);

/* Creates a new image with the same dimensions as the original one
 * and copies its content over, effectively cloning the input
 * image. If successful, the function returns a pointer to the newly
 * created image.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
*/
struct image * cloneImage(const struct image * src, uint8_t * err);

/* Creates a new image by rotating the input image by 90 degreees
 * clockwise. NOTE: the original image must be manually deallocated if
 * not needed. If successful, the function returns a pointer to the
 * new image.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
*/
struct image * rotate90Clockwise(const struct image * img, uint8_t * err);

/**
 * @brief Blur an image using a 3x3 averaging kernel.
 *
 * This function applies a 3x3 averaging kernel to each pixel in the image, resulting in 
 * a blurred effect. Edge pixels are not blurred to keep the implementation simple.
 *
 * @param img The original image to be blurred.
 * @return A new image structure containing the blurred image. The original image remains 
 *         unchanged.
 *
 * If @err is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
 *
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* blurImage(const struct image* img, uint8_t * err);

/**
 * @brief Sharpen an image using a 3x3 sharpening kernel.
 *
 * This function applies a 3x3 sharpening kernel to each pixel in the image to enhance its
 * details. Edge pixels are not sharpened to keep the implementation simple.
 *
 * @param img The original image to be sharpened.
 * @return A new image structure containing the sharpened image. The original image remains 
 *         unchanged.
 *
 * If @err is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
 *
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* sharpenImage(const struct image* img, uint8_t * err);

/**
 * @brief Detect vertical edges in an image using the Sobel operator.
 *
 * This function applies the Sobel vertical operator to each pixel in the image to detect
 * vertical edges. Edge pixels are not processed to keep the implementation simple.
 *
 * @param img The original image.
 * @return A new image structure containing the edge detected image. The original image remains 
 *         unchanged.
 *
 * If @err is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
 *
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* detectVerticalEdges(const struct image* img, uint8_t * err);


/**
 * @brief Detect horizontal edges in an image using the Sobel operator.
 *
 * This function applies the Sobel horizontal operator to each pixel in the image to detect
 * horizontal edges. Edge pixels are not processed to keep the implementation simple.
 *
 * @param img The original image.
 * @return A new image structure containing the edge detected image. The original image remains 
 *         unchanged.
 *
 * If @err is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
 *
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* detectHorizontalEdges(const struct image* img, uint8_t * err);

/**
 * @brief Load a BMP image from a file.
 *
 * This function loads a 24-bit BMP image from the specified file and returns a pointer to 
 * an image structure. The image is represented as a 2D array of uint32_t values where 
 * each entry corresponds to an RGB pixel.
 *
 * @param filename The path to the BMP file to be loaded.
 * @return A pointer to a struct image containing the image data. Returns NULL if the 
 *         file couldn't be opened or if the file is not a valid 24-bit BMP image.
 *
 * If @err is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
 *
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* loadBMP(const char* filename);

/**
 * @brief Save an image to a BMP file.
 *
 * This function saves the provided image to a 24-bit BMP file. The image is represented 
 * as a 2D array of uint32_t values where each entry corresponds to an RGB pixel.
 *
 * @param filename The path where the BMP file should be saved.
 * @param img A pointer to the struct image containing the image data.
 * @return 0 if the image was saved successfully, 1 otherwise.
 *
 * Note: This function overwrites the file if it already exists. Ensure to have necessary 
 *       backup or checks in place if overwriting is not desired.
 */
uint8_t saveBMP(const char* filename, const struct image* img);


/**
 * sendImage - Serialize and send an image structure over a given socket.
 *
 * This function takes in an image and a connected socket descriptor. It sends the image
 * data over the socket with the following format:
 *   - First 3 bytes: The magic identifier "IMG".
 *   - 4 bytes: Image width.
 *   - 4 bytes: Image height.
 *   - Width x Height x 4 bytes: Pixel data (in rows then columns).
 *
 * @param img Pointer to the image structure to be sent.
 * @param sockfd The socket descriptor to send data over.
 * @return 0 on success, 1 on error.
 */
uint8_t sendImage(struct image* img, int sockfd);

/**
 * recvImage - Deserialize and receive an image structure over a given socket.
 *
 * This function retrieves serialized image data from a socket, expecting the following format:
 *   - First 3 bytes: The magic identifier "IMG".
 *   - 4 bytes: Image width.
 *   - 4 bytes: Image height.
 *   - Width x Height x 4 bytes: Pixel data (in rows then columns).
 *
 * @param img Pointer to the image structure to be filled.
 * @param sockfd The socket descriptor to receive data from.
 * @return a valid image pointer on success, NULL on error.
 */
struct image * recvImage(int sockfd);

/* DO NOT WRITE ANY CODE BEYOND THIS LINE*/
#endif
