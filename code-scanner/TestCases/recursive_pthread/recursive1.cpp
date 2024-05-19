#include <iostream>
#include <cstdlib>
#include <pthread.h>
 
using namespace std;

long counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
 
void *PrintHello(void *threadid) {
	long tid = *(long *)threadid;
	cout << "Thread ID " << tid << endl;
	pthread_exit(NULL);
}
 
void load_data_in_thread() {
	if (counter < 10) {
		counter++;
		pthread_t thread;
		long arg = counter;
		int rc = pthread_create(&thread, NULL, PrintHello, &arg);
		load_data_in_thread();
		if (rc) {
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}
}

int main () {
        load_data_in_thread();      
 
        //pthread_join(thread2,0);
}
