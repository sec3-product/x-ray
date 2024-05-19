TESTS=$(wildcard *.cpp)
TARGETS=$(TESTS:.cpp=.ll)

all: $(TARGETS)

.PHONY: run clean

%.ll: %.cpp
	clang++ -S -emit-llvm -g -fno-discard-value-names -O1 $<

run: $(TARGETS)
	@for ll in $(TARGETS); do \
		echo $$ll; \
		racedetect $$ll; \
	done

clean:
	@rm -rf $(TARGETS)