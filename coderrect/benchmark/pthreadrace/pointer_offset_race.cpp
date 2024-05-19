// @purpose pointer calculation may cause race
// @dataRaces 1

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
    int offset = (long)arg;

    char* p = &v.a;
    p += offset;
    cout << "f1, ptr=0x" << hex << (long)p << endl;

    *p = 12;
    return NULL;
}

void *f2(void *arg)
{
    int offset = (long)arg;

    char* p = &v.a;
    p += offset;
    cout << "f2, ptr=0x" << hex << (long)p << endl;

    *p = 34;
    return NULL;
}

int main () {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, f1, (void*)0);
    pthread_create(&t2, NULL, f2, (void*)0);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    cout << "v.a=" << dec << (int)v.a << ", v.b=" << (int)v.b << endl;
}

