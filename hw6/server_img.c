/*******************************************************************************
 * Single-Threaded FIFO Image Server Implementation w/ Queue Limit
 *
 * Description:
 *     A server implementation designed to process client
 *     requests for image processing in First In, First Out (FIFO)
 *     order. The server binds to the specified port number provided as
 *     a parameter upon launch. It launches a secondary thread to
 *     process incoming requests and allows to specify a maximum queue
 *     size.
 *
 * Usage:
 *     <build directory>/server -q <queue_size> -w <workers> -p <policy> <port_number>
 *
 * Parameters:
 *     port_number - The port number to bind the server to.
 *     queue_size  - The maximum number of queued requests.
 *     workers     - The number of parallel threads to process requests.
 *     policy      - The queue policy to use for request dispatching.
 *
 * Author:
 *     Renato Mancuso
 *
 * Affiliation:
 *     Boston University
 *
 * Creation Date:
 *     October 31, 2023
 *
 * Notes:
 *     Ensure to have proper permissions and available port before running the
 *     server. The server relies on a FIFO mechanism to handle requests, thus
 *     guaranteeing the order of processing. If the queue is full at the time a
 *     new request is received, the request is rejected with a negative ack.
 *
 *******************************************************************************/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>
#include <signal.h>
#include <pthread.h>

/* Needed for wait(...) */
#include <sys/types.h>
#include <sys/wait.h>

/* Needed for semaphores */
#include <semaphore.h>

/* Include struct definitions and other libraries that need to be
 * included by both client and server */
#include "common.h"
#include "imglib.h"
#include "timelib.h"
#include "md5sum.h"
#include <inttypes.h>

#define BACKLOG_COUNT 100
#define USAGE_STRING				\
	"Missing parameter. Exiting.\n"		\
	"Usage: %s -q <queue size> "		\
	"-w <workers: 1> "			\
	"-p <policy: FIFO> "			\
	"<port_number>\n"

/* 4KB of stack for the worker thread */
#define STACK_SIZE (4096)

/* Mutex needed to protect the threaded printf. DO NOT TOUCH */
sem_t * printf_mutex;


/* Synchronized printf for multi-threaded operation */
#define sync_printf(...)			\
	do {					\
		sem_wait(printf_mutex);		\
		printf(__VA_ARGS__);		\
		sem_post(printf_mutex);		\
	} while (0)

/* START - Variables needed to protect the shared queue. DO NOT TOUCH */
sem_t * queue_mutex;
sem_t * queue_notify;
/* END - Variables needed to protect the shared queue. DO NOT TOUCH */

struct request_meta {
	struct request request;
	struct timespec receipt_timestamp;
	struct timespec start_timestamp;
	struct timespec completion_timestamp;
};

enum queue_policy {
	QUEUE_FIFO,
	QUEUE_SJN
};

struct queue {
	size_t wr_pos;
	size_t rd_pos;
	size_t max_size;
	size_t available;
	enum queue_policy policy;
	struct request_meta * requests;
};

struct connection_params {
	size_t queue_size;
	size_t workers;
	enum queue_policy queue_policy;
};

struct worker_params {
	int conn_socket;
	int worker_done;
	struct queue * the_queue;
	int worker_id;
};

enum worker_command {
	WORKERS_START,
	WORKERS_STOP
};


/* Global counter for generating unique image IDs */
static uint64_t next_img_id = 1;

/* Struct to represent each stored image entry */
struct image_entry {
    uint64_t img_id;
    struct image *img;
    struct image_entry *next;
};

struct image_entry *image_storage_head = NULL;
static pthread_mutex_t image_storage_lock = PTHREAD_MUTEX_INITIALIZER;


/* Image Storage Management */
void add_image_to_storage(uint64_t img_id, struct image *img) {
    pthread_mutex_lock(&image_storage_lock);
    struct image_entry *entry_node = malloc(sizeof(struct image_entry));
    entry_node->img_id = img_id;
    entry_node->img = img;
    entry_node->next = image_storage_head;
    image_storage_head = entry_node;
    pthread_mutex_unlock(&image_storage_lock);
}

void remove_image_from_storage(uint64_t img_id) {
    pthread_mutex_lock(&image_storage_lock);
    struct image_entry **current_ptr = &image_storage_head;
    while (*current_ptr) {
        if ((*current_ptr)->img_id == img_id) {
            struct image_entry *entry_to_remove = *current_ptr;
            *current_ptr = entry_to_remove->next;
            deleteImage(entry_to_remove->img);
            free(entry_to_remove);
            break;
        }
        current_ptr = &(*current_ptr)->next;
    }
    pthread_mutex_unlock(&image_storage_lock);
}

void queue_init(struct queue * the_queue, size_t queue_size, enum queue_policy policy)
{
	the_queue->rd_pos = 0;
	the_queue->wr_pos = 0;
	the_queue->max_size = queue_size;
	the_queue->requests = (struct request_meta *)malloc(sizeof(struct request_meta)
						     * the_queue->max_size);
	the_queue->available = queue_size;
	the_queue->policy = policy;
}

/* Add a new request <request> to the shared queue <the_queue> */
int add_to_queue(struct request_meta to_add, struct queue * the_queue)
{
	int result_code = 0;
	/* QUEUE PROTECTION INTRO START --- DO NOT TOUCH */
	sem_wait(queue_mutex);
	/* QUEUE PROTECTION INTRO END --- DO NOT TOUCH */

	/* MAKE SURE NOT TO RETURN WITHOUT GOING THROUGH THE OUTRO CODE! */

	/* Make sure that the queue is not full */
	if (the_queue->available == 0) {
		result_code = 1;
	} else {
		/* If all good, add the item in the queue */
		the_queue->requests[the_queue->wr_pos] = to_add;
		the_queue->wr_pos = (the_queue->wr_pos + 1) % the_queue->max_size;
		the_queue->available--;
		/* QUEUE SIGNALING FOR CONSUMER --- DO NOT TOUCH */
		sem_post(queue_notify);
	}

	/* QUEUE PROTECTION OUTRO START --- DO NOT TOUCH */
	sem_post(queue_mutex);
	/* QUEUE PROTECTION OUTRO END --- DO NOT TOUCH */
	return result_code;
}

/* Add a new request <request> to the shared queue <the_queue> */
struct request_meta get_from_queue(struct queue * the_queue)
{
	struct request_meta request_result;
	/* QUEUE PROTECTION INTRO START --- DO NOT TOUCH */
	sem_wait(queue_notify);
	sem_wait(queue_mutex);
	/* QUEUE PROTECTION INTRO END --- DO NOT TOUCH */

	/* MAKE SURE NOT TO RETURN WITHOUT GOING THROUGH THE OUTRO CODE! */
	request_result = the_queue->requests[the_queue->rd_pos];
	the_queue->rd_pos = (the_queue->rd_pos + 1) % the_queue->max_size;
	the_queue->available++;

	/* QUEUE PROTECTION OUTRO START --- DO NOT TOUCH */
	sem_post(queue_mutex);
	/* QUEUE PROTECTION OUTRO END --- DO NOT TOUCH */
	return request_result;
}

void dump_queue_status(struct queue * the_queue)
{
	size_t index, count;
	/* QUEUE PROTECTION INTRO START --- DO NOT TOUCH */
	sem_wait(queue_mutex);
	/* QUEUE PROTECTION INTRO END --- DO NOT TOUCH */

	/* MAKE SURE NOT TO RETURN WITHOUT GOING THROUGH THE OUTRO CODE! */
	sync_printf("Q:[");

	for (index = the_queue->rd_pos, count = 0; count < the_queue->max_size - the_queue->available;
	     index = (index + 1) % the_queue->max_size, ++count)
	{
		sync_printf("R%ld%s", the_queue->requests[index].request.req_id,
		       ((count+1 != the_queue->max_size - the_queue->available)?",":""));
	}

	sync_printf("]\n");

	/* QUEUE PROTECTION OUTRO START --- DO NOT TOUCH */
	sem_post(queue_mutex);
	/* QUEUE PROTECTION OUTRO END --- DO NOT TOUCH */
}

/* Main logic of the worker thread */
void * worker_main (void * arg)
{
	struct timespec current_time;
	struct worker_params * worker_params_ptr = (struct worker_params *)arg;

	/* Print the first alive message. */
	clock_gettime(CLOCK_MONOTONIC, &current_time);
	sync_printf("[#WORKER#] %lf Worker Thread Alive!\n", TSPEC_TO_DOUBLE(current_time));

	/* Okay, now execute the main logic. */
	while (!worker_params_ptr->worker_done) {

		struct request_meta request_data;
		struct response response_data;
		request_data = get_from_queue(worker_params_ptr->the_queue);

		/* Detect wakeup after termination asserted */
		if (worker_params_ptr->worker_done)
			break;

		clock_gettime(CLOCK_MONOTONIC, &request_data.start_timestamp);

		uint8_t has_error = 0;
        struct image *source_image = NULL;
        struct image *temp_image = NULL;
        uint64_t image_id_result = 0;

		/* Image Retrieval*/
		pthread_mutex_lock(&image_storage_lock);
		struct image_entry *image_node = image_storage_head;
		while (image_node != NULL) {
    		if (image_node->img_id == request_data.request.img_id) {
        		source_image = image_node->img;
        		break;
    		}
   			image_node = image_node->next;
		}
		pthread_mutex_unlock(&image_storage_lock);

		/*Reject if image not found*/
		if (source_image == NULL) {
			response_data.req_id = request_data.request.req_id;
            response_data.img_id = 0;
            response_data.ack = RESP_REJECTED;
            send(worker_params_ptr->conn_socket, &response_data, sizeof(struct response), 0);

            /* Log the rejection with timing details */
            clock_gettime(CLOCK_MONOTONIC, &request_data.completion_timestamp);
			sync_printf("T%d R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
				worker_params_ptr->worker_id,
				request_data.request.req_id,
				TSPEC_TO_DOUBLE(request_data.request.req_timestamp),
				OPCODE_TO_STRING(request_data.request.img_op),
				request_data.request.overwrite,
				request_data.request.img_id,
				0UL,
				TSPEC_TO_DOUBLE(request_data.receipt_timestamp),
				TSPEC_TO_DOUBLE(request_data.start_timestamp),
				TSPEC_TO_DOUBLE(request_data.completion_timestamp)
			);
            continue;
        }

		/*Overwrite*/
		if (request_data.request.overwrite) {
            image_id_result = request_data.request.img_id;
            temp_image = source_image;  // Process in-place
        } else {
            temp_image = cloneImage(source_image, &has_error);
            if (has_error) {
                /* Send rejection if cloning fails */
				response_data.req_id = request_data.request.req_id;
                response_data.img_id = 0;
                response_data.ack = RESP_REJECTED;
                send(worker_params_ptr->conn_socket, &response_data, sizeof(struct response), 0);

				clock_gettime(CLOCK_MONOTONIC, &request_data.completion_timestamp);
				sync_printf("T%d R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
					worker_params_ptr->worker_id,
					request_data.request.req_id,
					TSPEC_TO_DOUBLE(request_data.request.req_timestamp),
					OPCODE_TO_STRING(request_data.request.img_op),
					request_data.request.overwrite,
					request_data.request.img_id,
					0UL,
					TSPEC_TO_DOUBLE(request_data.receipt_timestamp),
					TSPEC_TO_DOUBLE(request_data.start_timestamp),
					TSPEC_TO_DOUBLE(request_data.completion_timestamp)
				);
			continue;

            }
			/* Assign a new image ID for the cloned image */
            image_id_result = next_img_id++;
            add_image_to_storage(image_id_result, temp_image);
        }

		/* Perform the requested image operation */
        struct image *final_image = NULL;
        switch (request_data.request.img_op) {
            case IMG_ROT90CLKW:
                final_image = rotate90Clockwise(temp_image, &has_error);
                break;
            case IMG_BLUR:
                final_image = blurImage(temp_image, &has_error);
                break;
            case IMG_SHARPEN:
                final_image = sharpenImage(temp_image, &has_error);
                break;
            case IMG_VERTEDGES:
                final_image = detectVerticalEdges(temp_image, &has_error);
                break;
            case IMG_HORIZEDGES:
                final_image = detectHorizontalEdges(temp_image, &has_error);
                break;
            case IMG_RETRIEVE:
			{
                
                if(saveBMP("retrieved_image.bmp", source_image)) {
                    sync_printf("INFO: The image has been successfully saved as retrieved_image.bmp\n");
                } else {
                    sync_printf("ERROR: The image saving failed.\n");
                }

              
                response_data.req_id = request_data.request.req_id;
				response_data.img_id = request_data.request.img_id;
                response_data.ack = RESP_COMPLETED;
                send(worker_params_ptr->conn_socket, &response_data, sizeof(struct response), 0);
                sendImage(source_image, worker_params_ptr->conn_socket);
				clock_gettime(CLOCK_MONOTONIC, &request_data.completion_timestamp);

				sync_printf("T%d R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
					worker_params_ptr->worker_id,
					request_data.request.req_id,
					TSPEC_TO_DOUBLE(request_data.request.req_timestamp),
					OPCODE_TO_STRING(request_data.request.img_op),
					request_data.request.overwrite,
					request_data.request.img_id,
					request_data.request.img_id,
					TSPEC_TO_DOUBLE(request_data.receipt_timestamp),
					TSPEC_TO_DOUBLE(request_data.start_timestamp),
					TSPEC_TO_DOUBLE(request_data.completion_timestamp)
				);

				dump_queue_status(worker_params_ptr->the_queue);
                continue;
			}
            default:
                has_error = 1;  // Unknown operation
                break;
        }

		/* Handle any errors in processing */
        if (has_error || final_image == NULL) {
			response_data.req_id = request_data.request.req_id;
            response_data.img_id = 0;
            response_data.ack = RESP_REJECTED;
            send(worker_params_ptr->conn_socket, &response_data, sizeof(struct response), 0);

			/* Log the rejection */
			clock_gettime(CLOCK_MONOTONIC, &request_data.completion_timestamp);
			sync_printf("T%d R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
				worker_params_ptr->worker_id,
				request_data.request.req_id,
				TSPEC_TO_DOUBLE(request_data.request.req_timestamp),
				OPCODE_TO_STRING(request_data.request.img_op),
				request_data.request.overwrite,
				request_data.request.img_id,
				0UL,
				TSPEC_TO_DOUBLE(request_data.receipt_timestamp),
				TSPEC_TO_DOUBLE(request_data.start_timestamp),
				TSPEC_TO_DOUBLE(request_data.completion_timestamp)
			);

            /* Clean up temporary storage if needed */
            if (!request_data.request.overwrite && temp_image) {
                remove_image_from_storage(image_id_result);
            }
            continue;
        }

		/* Update or store the processed image */
        pthread_mutex_lock(&image_storage_lock);
        image_node = image_storage_head;
        while (image_node != NULL) {
            if (image_node->img_id == image_id_result) {
                if (request_data.request.overwrite) {
                    deleteImage(image_node->img);
                }
                image_node->img = final_image;
                break;
            }
            image_node = image_node->next;
        }
        pthread_mutex_unlock(&image_storage_lock);


		/* Now provide a response! */
		response_data.req_id = request_data.request.req_id;
		response_data.img_id = image_id_result;
		response_data.ack = RESP_COMPLETED;
		send(worker_params_ptr->conn_socket, &response_data, sizeof(struct response), 0);
		clock_gettime(CLOCK_MONOTONIC, &request_data.completion_timestamp);

		sync_printf("T%d R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
    		worker_params_ptr->worker_id,
    		request_data.request.req_id,
    		TSPEC_TO_DOUBLE(request_data.request.req_timestamp),
    		OPCODE_TO_STRING(request_data.request.img_op),
    		request_data.request.overwrite,
    		request_data.request.img_id,
    		image_id_result,
    		TSPEC_TO_DOUBLE(request_data.receipt_timestamp),
    		TSPEC_TO_DOUBLE(request_data.start_timestamp),
    		TSPEC_TO_DOUBLE(request_data.completion_timestamp)
		);

		dump_queue_status(worker_params_ptr->the_queue);
	}

	return NULL;
}





/* This function will start/stop all the worker threads wrapping
 * around the pthread_join/create() function calls */

int control_workers(enum worker_command cmd, size_t worker_count,
		    struct worker_params * common_params)
{
	/* Anything we allocate should we kept as static for easy
	 * deallocation when the STOP command is issued */
	static pthread_t * worker_pthreads = NULL;
	static struct worker_params ** worker_params_array = NULL;
	static int * worker_ids_array = NULL;


	/* Start all the workers */
	if (cmd == WORKERS_START) {
		size_t index;
		/* Allocate all structs and parameters */
		worker_pthreads = (pthread_t *)malloc(worker_count * sizeof(pthread_t));
		worker_params_array = (struct worker_params **)
		malloc(worker_count * sizeof(struct worker_params *));
		worker_ids_array = (int *)malloc(worker_count * sizeof(int));


		if (!worker_pthreads || !worker_params_array || !worker_ids_array) {
			ERROR_INFO();
			perror("Unable to allocate arrays for threads.");
			return EXIT_FAILURE;
		}


		/* Allocate and initialize as needed */
		for (index = 0; index < worker_count; ++index) {
			worker_ids_array[index] = -1;


			worker_params_array[index] = (struct worker_params *)
				malloc(sizeof(struct worker_params));


			if (!worker_params_array[index]) {
				ERROR_INFO();
				perror("Unable to allocate memory for thread.");
				return EXIT_FAILURE;
			}


			worker_params_array[index]->conn_socket = common_params->conn_socket;
			worker_params_array[index]->the_queue = common_params->the_queue;
			worker_params_array[index]->worker_done = 0;
			worker_params_array[index]->worker_id = index;
		}


		/* All the allocations and initialization seem okay,
		 * let's start the threads */
		for (index = 0; index < worker_count; ++index) {
			worker_ids_array[index] = pthread_create(&worker_pthreads[index], NULL, worker_main, worker_params_array[index]);


			if (worker_ids_array[index] < 0) {
				ERROR_INFO();
				perror("Unable to start thread.");
				return EXIT_FAILURE;
			} else {
				printf("INFO: Worker thread %ld (TID = %d) started!\n",
				       index, worker_ids_array[index]);
			}
		}
	}


	else if (cmd == WORKERS_STOP) {
		size_t index;


		/* Command to stop the threads issues without a start
		 * command? */
		if (!worker_pthreads || !worker_params_array || !worker_ids_array) {
			return EXIT_FAILURE;
		}


		/* First, assert all the termination flags */
		for (index = 0; index < worker_count; ++index) {
			if (worker_ids_array[index] < 0) {
				continue;
			}


			/* Request thread termination */
			worker_params_array[index]->worker_done = 1;
		}


		/* Next, unblock threads and wait for completion */
		for (index = 0; index < worker_count; ++index) {
			if (worker_ids_array[index] < 0) {
				continue;
			}


			sem_post(queue_notify);
		}


        for (index = 0; index < worker_count; ++index) {
            pthread_join(worker_pthreads[index],NULL);
            printf("INFO: Worker thread exited.\n");
        }


		/* Finally, do a round of deallocations */
		for (index = 0; index < worker_count; ++index) {
			free(worker_params_array[index]);
		}


		free(worker_pthreads);
		worker_pthreads = NULL;


		free(worker_params_array);
		worker_params_array = NULL;


		free(worker_ids_array);
		worker_ids_array = NULL;
	}


	else {
		ERROR_INFO();
		perror("Invalid thread control command.");
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}
/* Main function to handle connection with the client. This function
 * takes in input conn_socket and returns only when the connection
 * with the client is interrupted. */
void handle_connection(int conn_socket, struct connection_params conn_params)
{
	struct request_meta * request_meta_ptr;
	struct queue * queue_ptr;
	size_t received_bytes;

	/* The connection with the client is alive here. Let's start
	 * the worker thread. */
	struct worker_params worker_params_common;
	int result_status;
	struct response response_info;

	/* Now handle queue allocation and initialization */
	queue_ptr = (struct queue *)malloc(sizeof(struct queue));
	queue_init(queue_ptr, conn_params.queue_size, conn_params.queue_policy);

	worker_params_common.conn_socket = conn_socket;
	worker_params_common.the_queue = queue_ptr;
	result_status = control_workers(WORKERS_START, conn_params.workers, &worker_params_common);

	/* Do not continue if there has been a problem while starting
	 * the workers. */
	if (result_status != EXIT_SUCCESS) {
		free(queue_ptr);

		/* Stop any worker that was successfully started */
		control_workers(WORKERS_STOP, conn_params.workers, NULL);
		return;
	}

	/* We are ready to proceed with the rest of the request
	 * handling logic. */

	request_meta_ptr = (struct request_meta *)malloc(sizeof(struct request_meta));

	do {
		received_bytes = recv(conn_socket, &request_meta_ptr->request, sizeof(struct request), 0);
		clock_gettime(CLOCK_MONOTONIC, &request_meta_ptr->receipt_timestamp);

		/* Don't just return if in_bytes is 0 or -1. Instead
		 * skip the response and break out of the loop in an
		 * orderly fashion so that we can de-allocate the req
		 * and resp variables, and shutdown the socket. */
		if (received_bytes > 0) {

			/* IMPLEMENT ME! Check right away if the
			 * request has img_op set to IMG_REGISTER. If
			 * so, handle the operation right away,
			 * reading in the full image payload, replying
			 * to the server, and bypassing the queue.
			 (
				Don't forget to send a response back to the client after
			  registering an image :) 
			 )
			  */
			if (request_meta_ptr->request.img_op == IMG_REGISTER) {
				/* Handle image registration */
				struct image *registered_image = recvImage(conn_socket);
        
        		/* Error handling if image reception fails */
        		if (registered_image == NULL) {
            		response_info.req_id = request_meta_ptr->request.req_id;
            		response_info.img_id = 0;
            		response_info.ack = RESP_REJECTED;
            		send(conn_socket, &response_info, sizeof(struct response), 0);
            		continue;
        		}

				/* Assign a unique img_id */
				uint64_t assigned_img_id = next_img_id++;

				/* Store the image */
				add_image_to_storage(assigned_img_id, registered_image);

				/* Prepare and send the response */
				response_info.req_id = request_meta_ptr->request.req_id;
				response_info.img_id = assigned_img_id;
				response_info.ack = RESP_COMPLETED;
				send(conn_socket, &response_info, sizeof(struct response), 0);

				/* Log the operation */
				clock_gettime(CLOCK_MONOTONIC, &request_meta_ptr->start_timestamp);
				clock_gettime(CLOCK_MONOTONIC, &request_meta_ptr->completion_timestamp);

				sync_printf("T0 R%ld:%lf,%s,%d,%ld,%ld,%lf,%lf,%lf\n",
					request_meta_ptr->request.req_id,
					TSPEC_TO_DOUBLE(request_meta_ptr->request.req_timestamp),
					OPCODE_TO_STRING(request_meta_ptr->request.img_op),
					request_meta_ptr->request.overwrite,
					0UL,
					assigned_img_id,
					TSPEC_TO_DOUBLE(request_meta_ptr->receipt_timestamp),
					TSPEC_TO_DOUBLE(request_meta_ptr->start_timestamp),
					TSPEC_TO_DOUBLE(request_meta_ptr->completion_timestamp)
				);

				continue; /* Skip adding to queue */
			}


			result_status = add_to_queue(*request_meta_ptr, queue_ptr);

			/* The queue is full if the return value is 1 */
			if (result_status) {
				/* Now provide a response! */
				response_info.req_id = request_meta_ptr->request.req_id;
				response_info.ack = RESP_REJECTED;
				send(conn_socket, &response_info, sizeof(struct response), 0);

				sync_printf("X%ld:%lf,%lf,%lf\n", request_meta_ptr->request.req_id,
				       TSPEC_TO_DOUBLE(request_meta_ptr->request.req_timestamp),
				       TSPEC_TO_DOUBLE(request_meta_ptr->request.req_length),
				       TSPEC_TO_DOUBLE(request_meta_ptr->receipt_timestamp)
					);
			}
		}
	} while (received_bytes > 0);


	/* Stop all the worker threads. */
	control_workers(WORKERS_STOP, conn_params.workers, NULL);

	free(request_meta_ptr);
	shutdown(conn_socket, SHUT_RDWR);
	close(conn_socket);
	printf("INFO: Client disconnected.\n");
}


/* Template implementation of the main function for the FIFO
 * server. The server must accept in input a command line parameter
 * with the <port number> to bind the server to. */
int main (int argc, char ** argv) {
	int server_socket, bind_status, client_socket, reuse_option, option_char;
	in_port_t port_number;
	struct sockaddr_in server_address, client_address;
	struct in_addr any_addr;
	socklen_t client_address_len;
	struct connection_params connection_settings;
	connection_settings.queue_size = 0;
	connection_settings.queue_policy = QUEUE_FIFO;
	connection_settings.workers = 1;

	/* Parse all the command line arguments */
	while((option_char = getopt(argc, argv, "q:w:p:")) != -1) {
		switch (option_char) {
		case 'q':
			connection_settings.queue_size = strtol(optarg, NULL, 10);
			printf("INFO: setting queue size = %ld\n", connection_settings.queue_size);
			break;
		case 'w':
			connection_settings.workers = strtol(optarg, NULL, 10);
			printf("INFO: setting worker count = %ld\n", connection_settings.workers);
			if (connection_settings.workers != 1) {
				ERROR_INFO();
				fprintf(stderr, "Only 1 worker is supported in this implementation!\n" USAGE_STRING, argv[0]);
				return EXIT_FAILURE;
			}
			break;
		case 'p':
			if (!strcmp(optarg, "FIFO")) {
				connection_settings.queue_policy = QUEUE_FIFO;
			} else {
				ERROR_INFO();
				fprintf(stderr, "Invalid queue policy.\n" USAGE_STRING, argv[0]);
				return EXIT_FAILURE;
			}
			printf("INFO: setting queue policy = %s\n", optarg);
			break;
		default: /* '?' */
			fprintf(stderr, USAGE_STRING, argv[0]);
		}
	}

	if (!connection_settings.queue_size) {
		ERROR_INFO();
		fprintf(stderr, USAGE_STRING, argv[0]);
		return EXIT_FAILURE;
	}

	if (optind < argc) {
		port_number = strtol(argv[optind], NULL, 10);
		printf("INFO: setting server port as: %d\n", port_number);
	} else {
		ERROR_INFO();
		fprintf(stderr, USAGE_STRING, argv[0]);
		return EXIT_FAILURE;
	}

	/* Now onward to create the right type of socket */
	server_socket = socket(AF_INET, SOCK_STREAM, 0);

	if (server_socket < 0) {
		ERROR_INFO();
		perror("Unable to create socket");
		return EXIT_FAILURE;
	}

	/* Before moving forward, set socket to reuse address */
	reuse_option = 1;
	setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (void *)&reuse_option, sizeof(reuse_option));

	/* Convert INADDR_ANY into network byte order */
	any_addr.s_addr = htonl(INADDR_ANY);

	/* Time to bind the socket to the right port  */
	server_address.sin_family = AF_INET;
	server_address.sin_port = htons(port_number);
	server_address.sin_addr = any_addr;

	/* Attempt to bind the socket with the given parameters */
	bind_status = bind(server_socket, (struct sockaddr *)&server_address, sizeof(struct sockaddr_in));

	if (bind_status < 0) {
		ERROR_INFO();
		perror("Unable to bind socket");
		return EXIT_FAILURE;
	}

	/* Let us now proceed to set the server to listen on the selected port */
	bind_status = listen(server_socket, BACKLOG_COUNT);

	if (bind_status < 0) {
		ERROR_INFO();
		perror("Unable to listen on socket");
		return EXIT_FAILURE;
	}

	/* Ready to accept connections! */
	printf("INFO: Waiting for incoming connection...\n");
	client_address_len = sizeof(struct sockaddr_in);
	client_socket = accept(server_socket, (struct sockaddr *)&client_address, &client_address_len);

	if (client_socket == -1) {
		ERROR_INFO();
		perror("Unable to accept connections");
		return EXIT_FAILURE;
	}

	/* Initialize threaded printf mutex */
	printf_mutex = (sem_t *)malloc(sizeof(sem_t));
	bind_status = sem_init(printf_mutex, 0, 1);
	if (bind_status < 0) {
		ERROR_INFO();
		perror("Unable to initialize printf mutex");
		return EXIT_FAILURE;
	}

	/* Initialize queue protection variables. DO NOT TOUCH. */
	queue_mutex = (sem_t *)malloc(sizeof(sem_t));
	queue_notify = (sem_t *)malloc(sizeof(sem_t));
	bind_status = sem_init(queue_mutex, 0, 1);
	if (bind_status < 0) {
		ERROR_INFO();
		perror("Unable to initialize queue mutex");
		return EXIT_FAILURE;
	}
	bind_status = sem_init(queue_notify, 0, 0);
	if (bind_status < 0) {
		ERROR_INFO();
		perror("Unable to initialize queue notify");
		return EXIT_FAILURE;
	}
	/* DONE - Initialize queue protection variables */

	/* Ready to handle the new connection with the client. */
	handle_connection(client_socket, connection_settings);

	free(queue_mutex);
	free(queue_notify);

	close(server_socket);
	return EXIT_SUCCESS;
}
