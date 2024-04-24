// @purpose C program to demonstrate working of Semaphores
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <stdio.h> 
#include <pthread.h> 
#include <semaphore.h> 
#include <unistd.h> 

sem_t mutex; 
int x =0;

void* thread(void* arg) 
{ 
	//wait 
	sem_wait(&mutex); 
	printf("\nEntered..\n"); 

	//critical section 
	sleep(4); 
	x++;
	//signal 
	printf("\nJust Exiting...\n"); 
	sem_post(&mutex); 
} 


int main() 
{ 
	sem_init(&mutex, 0, 1); 
	pthread_t t1,t2; 
	pthread_create(&t1,NULL,thread,NULL); 
	sleep(2); 
	pthread_create(&t2,NULL,thread,NULL); 
	pthread_join(t1,NULL); 
	pthread_join(t2,NULL); 
	sem_destroy(&mutex); 
	return 0; 
} 

