// @purpose thread reentrant a pthread mutex
// @dataRaces 0
// @misMatchedAPI 2 

#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>

using namespace std;

int v = 0;

pthread_mutex_t _lock1;

void reentrant()
{
	pthread_mutex_lock(&_lock1);
	v++;
	pthread_mutex_unlock(&_lock1);
}

void *f1(void *arg)
{
	(void)arg;

	pthread_mutex_lock(&_lock1);
	reentrant();
	pthread_mutex_unlock(&_lock1);

	return NULL;
}


int main () {
   	pthread_t t1, t2;

	pthread_mutex_init(&_lock1, NULL);

	pthread_create(&t1, NULL, f1, NULL);
	pthread_create(&t2, NULL, f1, NULL);


	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
   
    cout << "value of v=" << v << endl;
}


