// @purpose create three threads in main to load data
// @dataRaces 0

#include <iostream>
#include <pthread.h>
using namespace std;

int x = 0;

void* PrintHello(void* id) {
    long tid = (long)id;
    x++;
    return nullptr;
}
pthread_t load_data_in_thread(long id){

	pthread_t thread;
	void *arg = (void*)id;
	int rc = pthread_create(&thread,NULL,PrintHello,arg);
	if(rc) {
		cout << "Error: unable to create thread, "<<rc<<endl;
		exit(-1);
	}
	return thread;
}

int main(int argc, char** argv) {
        pthread_t thread1, thread2;
	if(argc==1)
        	pthread_create(&thread1, NULL, PrintHello, (void *)1);
	else if(argc==2)
        	pthread_create(&thread2, NULL, PrintHello, (void *)1);

        pthread_join(thread1, nullptr);
        pthread_join(thread2, nullptr);

    cout << "x = " << x << endl;
    return 0;
}
