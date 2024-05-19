// @purpose two threads access two different elements of an array
// @dataRaces 0

#include <iostream>
#include <cstdlib>

using namespace std;

int v[2];


typedef void(*CallbackFn)();

void *f1(void *arg)
{
	int* p = &v[0];
	*p = 34;

	return NULL;
}

void *f2(void *arg)
{
	(void)arg;

	int* p = &v[1];
	*p = 12;

	return NULL;
}

int main () {
   	pthread_t t1, t2;

	pthread_create(&t1, NULL, f1, NULL);
	pthread_create(&t2, NULL, f2, NULL);

	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
   
    cout << "value of v=" << v << endl;
}


