LIB         :=  lib
# OBJDIR      := obj
# SOURCES := $(wildcard *.c)
# OBJECTS := $(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(SOURCES)))

# all: $(OBJDIR) $(OBJECTS)
all: liba

# $(OBJDIR)/%.o: %.c
# 	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

# $(OBJDIR)/%.a: %.o
# 	ar -rc lib/libbar.a foo.o

.PHONY : clean, liba, libso, $(LIB)

liba: $(LIB)
	gcc -c -Wall -Werror -fpic foo.c -o output/foo.o
	gcc -c -Wall -Werror -fpic foo.c -o foo.o
	ar -rc lib/libbar.a foo.o

# libso:
# 	gcc -c -Wall -Werror -fpic foo.c
# 	gcc -shared -o lib/libbar.so foo.o

$(LIB):
	mkdir output
	mkdir $@

clean:
	rm -rf *.o output *.so *.a
	rm -rf lib