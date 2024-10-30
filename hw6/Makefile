###############################################################################
# Makefile for Compiling TimeLib, ImgLib, MD5Lib, and Server Modules
#
# Description:
#     This Makefile is designed to compile various components, including:
#     - TimeLib: A library for time-related operations
#     - ImageLib: A library for image manipulation
#     - MD5Lib: A library to compute MD5 hashes for images and memory buffers
#     - Server: Processes client image manipulation requests in FIFO order
#
# Targets:
#     - all: Compiles all modules
#     - server_img: Compiles the server executable
#     - clean: Removes compiled binaries and intermediate files
#
# Usage:
#     make <target_name>
#     NOTE: all the binaries will be created in the build/ subfolder
#
# Author:
#     Renato Mancuso
#
# Affiliation:
#     Boston University
#
# Creation Date:
#     October 31, 2023
#
# Notes:
#     Ensure you have the necessary dependencies and permissions before
#     compiling. Modify the Makefile as necessary if directory structures
#     change or if additional modules are added in the future.
#
###############################################################################


TARGETS = server_img
LIBS = timelib imglib md5sum
LDFLAGS = -lm -lpthread -O0
BUILDDIR = build
BUILD_TARGETS = $(addprefix $(BUILDDIR)/,$(TARGETS))
OBJS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(TARGETS) $(LIBS)))
LIBOBJS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(LIBS)))

all: $(BUILD_TARGETS)

$(BUILD_TARGETS): $(BUILDDIR) $(OBJS)
	gcc -o $@ $@.o $(LIBOBJS) $(LDFLAGS) -W -Wall

$(BUILDDIR):
	mkdir $(BUILDDIR)

$(BUILDDIR)/%.o: %.c
	gcc -o $@ -c $< -W -Wall

clean:
	rm *~ -rf $(BUILDDIR)
