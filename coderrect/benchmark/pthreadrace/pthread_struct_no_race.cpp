// @purpose - two threads access different fields of a struct
// @dataRaces 0

#include <iostream>
#include <cstdlib>

using namespace std;

struct X
{
	char a;
	char b;
} v;

void *f1(void *arg)
{
	(void) arg;

	char* p = &v.a;
	*p = 12;

	return NULL;
}

void *f2(void *arg)
{
	(void) arg;

	char* p = &v.b;
	// p--;
	*p = 34;

	return NULL;
}

int main () {
   	pthread_t t1, t2;
	

	pthread_create(&t1, NULL, f1, NULL);
	pthread_create(&t2, NULL, f2, NULL);

	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
}


