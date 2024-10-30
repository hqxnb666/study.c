/*******************************************************************************
* Image Manipulation Library (implementation)
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

#include "imglib.h"

#define pix(img, x, y)				\
	img->pixels[((y) * img->width) + (x)]

/* Allocate and initialize the memory and metadata for a new
 * <width>x<height> pixels. */
struct image * createImage(uint32_t width, uint32_t height)
{
	uint64_t img_bytes = height * width * sizeof(uint32_t);
	struct image * img = (struct image*)malloc(sizeof(struct image));
	img->width = width;
	img->height = height;
	img->pixels = (uint32_t * )malloc(img_bytes);

	/* Reset all the pixels to 0 for an all-black image */
	memset(img->pixels, 0, img_bytes);

	return img;
}

/* Deallocate all the memory for a given image. */
void deleteImage(struct image * img)
{
	/* Remove image payload, if any. */
	if (img && img->pixels) {
		free(img->pixels);
		img->pixels = NULL;
	}

	/* Deallocate image metadata */
	if (img) {
		free(img);
	}
}

/* Set a specific pixel at position (<x>,<y>) in the image <img> to a
 * specific <value>. The function returns 0 if the operation is
 * successful and 1 in case of error. */
uint8_t setPixel(struct image * img, uint32_t x, uint32_t y, uint32_t value) {
	if (!img || !img->pixels) {
		return 1;
	}

	if (x < img->width && y < img->height) {
		pix(img, x, y) = value;
		return 0;
	}

	return 1;
}

/*
 * Get the value of a specific pixel at position (<x>,<y>) in the
 * image <img>.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, 0 is returned by the function.
*/
uint32_t getPixel(const struct image * img, uint32_t x, uint32_t y, uint8_t * err) {
	if (!img || !img->pixels) {
		if (err) {
			*err = 1;
		}
		return 0;
	}

	if (x < img->width && y < img->height) {
		if (err) {
			*err = 0;
		}
		return pix(img, x, y);
	}

	if (err) {
		*err = 1;
	}

	return 0;
}

/* Creates a new image with the same dimensions as the original one
 * and copies its content over, effectively cloning the input
 * image. If successful, the function returns a pointer to the newly
 * created image.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
*/
struct image * cloneImage(const struct image * src, uint8_t * err) {

	if(!src || !src->pixels) {
		if (err) {
			*err = 1;
		}
		return NULL;
	}

	/* Create an empty destination image */
	uint64_t img_bytes = src->height * src->width * sizeof(uint32_t);
	struct image * dest = createImage(src->width, src->height);

	if(!dest || !dest->pixels) {
		if (err) {
			*err = 1;
		}
		return NULL;
	}

	/* Copy over all the content from the source image */
	memcpy(dest->pixels, src->pixels, img_bytes);

	if(err) {
		*err = 0;
	}

	return dest;
}

/* Creates a new image by rotating the input image by 90 degreees
 * clockwise. NOTE: the original image must be manually deallocated if
 * not needed. If successful, the function returns a pointer to the
 * new image.
 *
 * If <err> is not NULL, the function sets 0 in the err parameter if
 * retrieval of the selected pixel is successful, and 1 if an error
 * has occurred. In case of error, NULL is returned by the function.
*/
struct image * rotate90Clockwise(const struct image * img, uint8_t * err) {
    struct image * rotated;
    uint32_t y, x;

    if (!img || !img->pixels) {
	    if (err) {
		    *err = 1;
	    }
	    return NULL;
    }

    rotated = createImage(img->height, img->width);

    for (y = 0; y < img->height; y++) {
        for (x = 0; x < img->width; x++) {
            uint32_t newX = y;
            uint32_t newY = img->width - x - 1;
            pix(rotated, newX, newY) = pix(img, x, y);
        }
    }

    if (err) {
	    *err = 0;
    }

    return rotated;
}

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
struct image* blurImage(const struct image* img, uint8_t * err) {
    struct image* blurredImg;
    uint32_t x, y;

    if (!img || !img->pixels) {
	    if (err) {
		    *err = 1;
	    }
	    return NULL;
    }

    blurredImg = createImage(img->width, img->height);

    for (y = 0; y < img->height; y++) {
        for (x = 0; x < img->width; x++) {
            // For simplicity, edge pixels are not blurred
            if (y == 0 || x == 0 || y == img->height - 1 || x == img->width - 1) {
		    pix(blurredImg, x, y) = pix(img, x, y);
            } else {
                uint32_t sumR = 0, sumG = 0, sumB = 0;

                // Convolve with the kernel
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
			    uint32_t pixel = pix(img, x + kx, y + ky);
			    sumR += (pixel >> 16) & 0xFF;
			    sumG += (pixel >> 8) & 0xFF;
			    sumB += pixel & 0xFF;
                    }
                }

                // Average the sums to get the new pixel value
                sumR /= 9;
                sumG /= 9;
                sumB /= 9;

                pix(blurredImg, x, y) = (sumR << 16) | (sumG << 8) | sumB;
            }
        }
    }

    if (err) {
	    *err = 0;
    }

    return blurredImg;
}

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
struct image* sharpenImage(const struct image* img, uint8_t * err) {
    struct image* sharpenedImg;
    uint32_t x, y;

    if (!img || !img->pixels) {
	    if (err) {
		    *err = 1;
	    }
	    return NULL;
    }

    sharpenedImg = createImage(img->width, img->height);

    for (y = 0; y < img->height; y++) {
        for (x = 0; x < img->width; x++) {
            // For simplicity, edge pixels are not sharpened
            if (y == 0 || x == 0 || y == img->height - 1 || x == img->width - 1) {
		    pix(sharpenedImg, x, y) = pix(img, x, y);
            } else {
                int sumR = 0, sumG = 0, sumB = 0;

                /* Convolve with the kernel that emphasizes center pixel */
                int kernel[3][3] = { {-1, -1, -1},
                                     {-1,  9, -1},
                                     {-1, -1, -1} };

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
			    uint32_t pixel = pix(img, x + kx, y + ky);
			    sumR += ((pixel >> 16) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumG += ((pixel >> 8) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumB += (pixel & 0xFF) * kernel[ky + 1][kx + 1];
                    }
                }

                /* Clip the values to [0, 255] */
                sumR = (sumR > 255) ? 255 : (sumR < 0) ? 0 : sumR;
                sumG = (sumG > 255) ? 255 : (sumG < 0) ? 0 : sumG;
                sumB = (sumB > 255) ? 255 : (sumB < 0) ? 0 : sumB;

                pix(sharpenedImg, x, y) = (sumR << 16) | (sumG << 8) | sumB;
            }
        }
    }

    if (err) {
	    *err = 0;
    }

    return sharpenedImg;
}

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
struct image* detectVerticalEdges(const struct image* img, uint8_t * err) {
    struct image* edgeImg;
    uint32_t x, y;

    int kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    if (!img || !img->pixels) {
	    if (err) {
		    *err = 1;
	    }
	    return NULL;
    }

    edgeImg = createImage(img->width, img->height);

    for (y = 0; y < img->height; y++) {
        for (x = 0; x < img->width; x++) {
            // For simplicity, edge pixels are not processed
            if (y == 0 || x == 0 || y == img->height - 1 || x == img->width - 1) {
		    pix(edgeImg, x, y) = 0; // Set to black for edge pixels
            } else {
                int sumR = 0, sumG = 0, sumB = 0;

                // Convolve with the kernel
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
			    uint32_t pixel = pix(img, x + kx, y + ky);
			    sumR += ((pixel >> 16) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumG += ((pixel >> 8) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumB += (pixel & 0xFF) * kernel[ky + 1][kx + 1];
                    }
                }

                // Clip the values to [0, 255]
                sumR = (sumR > 255) ? 255 : (sumR < 0) ? 0 : sumR;
                sumG = (sumG > 255) ? 255 : (sumG < 0) ? 0 : sumG;
                sumB = (sumB > 255) ? 255 : (sumB < 0) ? 0 : sumB;

                pix(edgeImg, x, y) = (sumR << 16) | (sumG << 8) | sumB;
            }
        }
    }

    if (err) {
	    *err = 0;
    }

    return edgeImg;
}

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
struct image* detectHorizontalEdges(const struct image* img, uint8_t * err) {
    struct image* edgeImg;
    uint32_t x, y;

    int kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if (!img || !img->pixels) {
	    if (err) {
		    *err = 1;
	    }
	    return NULL;
    }

    edgeImg = createImage(img->width, img->height);

    for (y = 0; y < img->height; y++) {
        for (x = 0; x < img->width; x++) {
            // For simplicity, edge pixels are not processed
            if (y == 0 || x == 0 || y == img->height - 1 || x == img->width - 1) {
		    pix(edgeImg, x, y) = 0; // Set to black for edge pixels
            } else {
                int sumR = 0, sumG = 0, sumB = 0;

                // Convolve with the kernel
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
			    uint32_t pixel = pix(img, x + kx, y + ky);
			    sumR += ((pixel >> 16) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumG += ((pixel >> 8) & 0xFF) * kernel[ky + 1][kx + 1];
			    sumB += (pixel & 0xFF) * kernel[ky + 1][kx + 1];
                    }
                }

                // Clip the values to [0, 255]
                sumR = (sumR > 255) ? 255 : (sumR < 0) ? 0 : sumR;
                sumG = (sumG > 255) ? 255 : (sumG < 0) ? 0 : sumG;
                sumB = (sumB > 255) ? 255 : (sumB < 0) ? 0 : sumB;

                pix(edgeImg, x, y) = (sumR << 16) | (sumG << 8) | sumB;
            }
        }
    }

    if (err) {
	    *err = 0;
    }

    return edgeImg;
}

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
 * Note: The returned image structure should be freed using the deleteImage function 
 *       to avoid memory leaks.
 */
struct image* loadBMP(const char* filename) {
	int fd = open(filename, O_RDONLY);
	uint32_t x, y;
	BMPHeader header;
	BMPInfoHeader infoHeader;

	if (fd == -1) return NULL;

	header.type = 0;

	read(fd, &header, sizeof(BMPHeader));
	read(fd, &infoHeader, sizeof(BMPInfoHeader));

	if (header.type != 0x4D42 || infoHeader.bits != 24) {
		close(fd);
		return NULL;
	}

	//printf("IMG: %d x %d x %d\n", infoHeader.width, infoHeader.height, infoHeader.bits);
	struct image* img = createImage(infoHeader.width, infoHeader.height);
	int padding = (4 - (infoHeader.width * 3) % 4) % 4;

	lseek(fd, header.offset, SEEK_SET);

	/* Start from the last row */
	y = infoHeader.height - 1;

	do {
		for (x = 0; x < infoHeader.width; x++) {
			unsigned char color[3];
			read(fd, color, sizeof(unsigned char) * 3);
			pix(img, x, y) = (color[2] << 16) | (color[1] << 8) | color[0];
		}
		lseek(fd, padding, SEEK_CUR);
	} while (y-- > 0); /* The post-increment here is important not
			    * to miss the last row. */

	close(fd);
	return img;
}

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
uint8_t saveBMP(const char* filename, const struct image* img) {
	/* Create if the file does not exist, overwrite otherwise. Set
	 * file permissions: 0644 */
	int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC,
		      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
	uint32_t x, y;

	if (fd == -1) return 1;

	BMPHeader header = { 0x4D42, 54 + img->width * img->height * 3, 0, 0, 54 };
	BMPInfoHeader infoHeader = { 40, img->width, img->height, 1, 24, 0,
				     img->width * img->height * 3, 0, 0, 0, 0 };

	write(fd, &header, sizeof(BMPHeader));
	write(fd, &infoHeader, sizeof(BMPInfoHeader));

	int padding = (4 - (img->width * 3) % 4) % 4;

	/* Start by serializing the last row. */
	y = img->height - 1;
	do {
		for (x = 0; x < img->width; x++) {
			uint32_t pixel = pix(img, x, y);
			unsigned char color[3] = { pixel & 0xFF,
						   (pixel >> 8) & 0xFF,
						   (pixel >> 16) & 0xFF };
			write(fd, color, sizeof(unsigned char) * 3);
		}
		for (int i = 0; i < padding; i++) {
			unsigned char pad = 0;
			write(fd, &pad, 1);
		}
	} while ( y-- > 0); /* The post-increment here is important
			    * not to miss the last row. */

	close(fd);
	return 0;
}

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
uint8_t sendImage(struct image* img, int sockfd) {
    char magic[3] = {'I', 'M', 'G'};
    size_t to_send = img->width * img->height * sizeof(uint32_t);
    char * bufptr = (char *)(img->pixels);

    /* Send the magic bytes */
    if (send(sockfd, magic, 3, 0) != 3) {
        return 1;
    }

    /* Send the width and height */
    if (send(sockfd, &(img->width), sizeof(img->width), 0) != sizeof(img->width) ||
        send(sockfd, &(img->height), sizeof(img->height), 0) != sizeof(img->height)) {
        return 1;
    }

    /* Send all the pixel data on the socket */
    while (to_send) {
	    size_t cur = send(sockfd, bufptr, to_send, 0);
	    if (cur <= 0) {
		    perror("Unable to send image on socket");
		    return 1;
	    }
	    bufptr += cur;
	    to_send -= cur;
    }

    return 0;
}

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
struct image * recvImage(int sockfd) {
	char magic[3];
	size_t to_recv;
	char * bufptr;
	uint32_t width, height;
	struct image * img = NULL;

	/* Receive the magic bytes */
	if (recv(sockfd, magic, 3, 0) != 3 || strncmp(magic, "IMG", 3) != 0) {
		return NULL;
	}

	/* Receive the width and height */
	if (recv(sockfd, &(width), sizeof(uint32_t), 0) != sizeof(uint32_t) ||
	    recv(sockfd, &(height), sizeof(uint32_t), 0) != sizeof(uint32_t)) {
		return NULL;
	}

	/* Create a new image to fill up */
	img = createImage(width, height);
	to_recv = img->width * img->height * sizeof(uint32_t);
	bufptr = (char *)(img->pixels);

	/* Receive all the pixel bytes on the socket */
	while(to_recv) {
		size_t cur = recv(sockfd, bufptr, to_recv, 0);
		if (cur <= 0) {
			deleteImage(img);
			return NULL;
		}
		bufptr += cur;
		to_recv -= cur;
	}

	return img;
}
