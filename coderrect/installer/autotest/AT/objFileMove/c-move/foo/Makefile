LIB         :=  lib

.PHONY : clean

default: liba

liba: $(LIB)
	gcc -c -Wall -Werror -fpic foo.c
	#ar -rc lib/libfoo.a foo.o

libso:
	gcc -c -Wall -Werror -fpic foo.c
	gcc -shared -o lib/libfoo.so foo.o

$(LIB):
	mkdir $@

clean:
	rm -rf *.o $(objects) *.so *.a
	rm -rf lib
