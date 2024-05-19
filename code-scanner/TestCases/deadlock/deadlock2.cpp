
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex3 = PTHREAD_MUTEX_INITIALIZER;


// These two functions will run concurrently.
void* print_i(void *ptr) {
  pthread_mutex_lock(&mutex1);
  pthread_mutex_lock(&mutex2);
  printf("I am in i");
  pthread_mutex_unlock(&mutex2);
  pthread_mutex_unlock(&mutex1);
}

void* print_j(void *ptr) {
  pthread_mutex_lock(&mutex2);
  pthread_mutex_lock(&mutex3);
  printf("I am in j");
  pthread_mutex_unlock(&mutex3);
  pthread_mutex_unlock(&mutex2);
}

void* print_k(void *ptr) {
  pthread_mutex_lock(&mutex3);
  pthread_mutex_lock(&mutex1);
  printf("I am in k");
  pthread_mutex_unlock(&mutex1);
  pthread_mutex_unlock(&mutex3);
}

int main() {
  pthread_t t1, t2, t3;
  int iret1 = pthread_create(&t1, NULL, print_i, NULL);
  int iret2 = pthread_create(&t2, NULL, print_j, NULL);
  int iret3 = pthread_create(&t3, NULL, print_k, NULL);

  while(1){}
  exit(0); //never reached.
}
