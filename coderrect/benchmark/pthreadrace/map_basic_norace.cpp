// @purpose shows a basic map case
// @dataRaces 0


#include <map>
#include <iostream>


void *update(void *v) {
    int *p = (int *)v;
    *p = rand();
    std::cout << *p << "\n";
}


int main() {
    std::map<int, int*> m{};
    int i = rand(),
            j = rand(),
            k = rand();

    m[1] = &i;
    m[2] = &j;
    m[3] = &k;

    pthread_t th1, th2;
    pthread_create(&th1, nullptr, update, m[1]);
    pthread_create(&th2, nullptr, update, m[2]);
    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}