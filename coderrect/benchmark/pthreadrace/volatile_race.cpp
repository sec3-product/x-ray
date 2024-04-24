// @purpose two threads increase number in one volatile variable without lock protected.
// @dataRaces 1

#include <pthread.h>
#include <iostream>
using namespace std;

volatile int num = 0;

void* add(void* id) {
	long tid = (long)id;
	for (int i=0; i<100000; i++) {
		num++;
	}
	return nullptr;
}

pthread_t generate_thread(long id){

        pthread_t thread;
        void *arg = (void*)id;
        int rc = pthread_create(&thread,NULL,add,arg);
        if(rc) {
                cout << "Error: unable to create thread, "<<rc<<endl;
                exit(-1);
        }
        return thread;
}

int main(int argc, char** argv) {
        pthread_t thread1, thread2;
        thread1 = generate_thread(1);
        thread2 = generate_thread(2);

        pthread_join(thread1, nullptr);
        pthread_join(thread2, nullptr);

	cout << "num = " << num << endl;
	return 0;
}

