CFLAGS = -fopenmp -g

all: atomic1 atomic2 atomic3 atomic4 atomic5

atomic1:
	clang $(CFLAGS) -o atomic1 atomic1.c

atomic2:
	clang $(CFLAGS) -o atomic2 atomic2.c

atomic3:
	clang $(CFLAGS) -o atomic3 atomic3.c

atomic4:
	clang $(CFLAGS) -o atomic4 atomic4.c

atomic5:
	clang $(CFLAGS) -o atomic5 atomic5.c

clean:
	$(RM) atomic1
	$(RM) atomic2
	$(RM) atomic3
	$(RM) atomic4
	$(RM) atomic5
