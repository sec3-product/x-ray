#include <stdio.h>
#include "micthread.h"

#define NUM_THREADS 2

int total = 0;
pthread_mutex_t mtx;

void* counter_1(void* args) {
	for (int i = 0; i < 100000; i++) {
		total = total + 1;
	}
	return 0;
}

void* counter_2(void* args) {
	for (int i = 0; i < 50000; i++) {
		total = total + 2;
	}
	return 0;
}

int main() {
	pthread_t tids[NUM_THREADS];
	int index[NUM_THREADS];

	int ret = micthread_create(&tids[0], NULL, counter_1, NULL);
	if (ret != 0) {
		printf("pthread_create error: error code: %d", ret);
	}
	
	ret = micthread_create(&tids[0], NULL, counter_2, NULL);
	if (ret != 0) {
		printf("pthread_create error: error code: %d", ret);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		micthread_join(tids[i], NULL);
	}

	micthread_exit(NULL);
}
