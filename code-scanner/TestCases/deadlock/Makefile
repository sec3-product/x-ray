CXXFLAGS = -lpthread -g

all:
	make deadlock1
	make deadlock2
	make deadlock3

deadlock1:
	clang++ $(CXXFLAGS) -o deadlock1 deadlock1.cpp

deadlock2:
	clang++ $(CXXFLAGS) -o deadlock2 deadlock2.cpp

deadlock3:
	clang++ $(CXXFLAGS) -o deadlock3 deadlock3.cpp

clean:
	$(RM) deadlock1
	$(RM) deadlock2
	$(RM) deadlock3
