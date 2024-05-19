#include <iostream>
#include <cstdlib>
#include <pthread.h>
 
using namespace std;

long counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
 
void *PrintHello(void *arg) {
	cout << "HELLO: Thread ID " << ++counter << endl;
	if (counter < 10) {
		pthread_t thread;
		pthread_create(&thread, NULL, PrintHello, NULL);
		pthread_join(thread, 0);
	}
	//pthread_exit(NULL);
	return 0;
}
 
void load_data_in_thread() {
	pthread_t thread;
	int rc = pthread_create(&thread, NULL, PrintHello, NULL);
	if (rc) {
		cout << "Error:unable to create thread," << rc << endl;
		exit(-1);
	}
	pthread_join(thread, 0);
}

int main () {
        load_data_in_thread();      
	return 0;
}
