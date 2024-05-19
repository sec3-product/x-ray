#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int i;

void *myThreadFun(void *vargp)
{
	pthread_mutex_lock(&lock);
	i++;
	printf("Printing GeeksQuiz from Thread \n");
	pthread_mutex_unlock(&lock);
	return NULL;
}

void *myThreadFun2(void *vargp)
{
	// pthread_mutex_lock(&lock);
	i++;
	printf("Printing GeeksQuiz from Thread \n");
	pthread_mutex_unlock(&lock);
	return NULL;
}

void foo(void)
{
	puts("Hello, I am foo");

	pthread_t thread_id, tid;
	printf("Before Thread\n");
	pthread_create(&thread_id, NULL, myThreadFun, NULL);
	pthread_create(&tid, NULL, myThreadFun2, NULL);

	pthread_join(thread_id, NULL);
	pthread_join(tid, NULL);
}
