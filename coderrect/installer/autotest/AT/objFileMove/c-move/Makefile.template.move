BIN         :=  bin

#SUBDIRS := bar foo 
SUBDIRS := DIR_HOLD_PLACE

.PHONY: all $(SUBDIRS) 

#all: $(BIN) $(SUBDIRS) main test_mvs
all: $(BIN) $(SUBDIRS) TARGET_HOLD_PLACE
$(SUBDIRS):
	$(MAKE) -C $@

default: $(BIN) main

main:
	mv foo/foo.o bin/tmp
	ar -rc bin/libfoo.a bin/tmp/foo.o
	gcc -Lbin/ -Wall -o bin/main main.c -lfoo -lpthread

test_mvs:
	mv bar/output bin/tmp
	ar -rc bin/libmvs.a bin/tmp/output/foo.o
	gcc -Lbin/ -Wall -o bin/main_mvs main.c -lmvs -lpthread

$(BIN):
	mkdir $@
	mkdir bin/tmp

clean:
	$(MAKE) -C foo clean
	$(MAKE) -C bar clean
	rm -rf *.o *.out
	rm -rf bin/
	rm -rf .coderrect
