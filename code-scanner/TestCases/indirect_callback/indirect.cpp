#include <iostream>
#include <cstdlib>
#include <pthread.h>

using namespace std;

long counter = 0;

void *PrintHello(void *arg) {
	cout << "HELLO: Thread ID " << ++counter << endl;
	return 0;
}

int main () {
	pthread_t thread;
	void *(*callback) (void *) = PrintHello;
	int rc = pthread_create(&thread, NULL, callback, NULL);
	if (rc) {
		cout << "Error:unable to create thread," << rc << endl;
		exit(-1);
	}
	pthread_join(thread, 0);
	return 0;
}
