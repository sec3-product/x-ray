
TESTS = thread threadjoin threadpassref threadpasssmrtptr unique_lock unique_lock_nodeadlock unique_lock_race static_mutex_lock
LLS = $(TESTS:=.ll)
.PHONY: clean

all: $(TESTS)

lls: $(LLS)

%: %.cpp
	clang++ $< -o $@ -lpthread

%.ll: %.cpp 
	clang++ -S -emit-llvm -g -fno-discard-value-names $< -o $@ -lpthread

clean:
	@rm -f $(TESTS) $(LLS) log.current