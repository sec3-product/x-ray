# the compiler to use
CC = clang

# compiler flags:
#   -g        adds debugging information to the executable file
#   -Wall     turns on most, but not all, compiler warnings
CFLAGS = -g -Wall -I.

# the name to use for both the target source file, and the output file:
TARGET = hello

all: $(TARGET)

$(TARGET): hello.o world.o 
	$(CC) -o $(TARGET) hello.o world.o 
