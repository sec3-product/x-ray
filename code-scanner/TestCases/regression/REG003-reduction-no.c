#include <assert.h>

int main() {
    const int N = 100;
    int counter = 0;

    #pragma omp parallel for reduction(+:counter)
    for (int i = 0; i < N; i++) {
        counter++;
    }

    assert(counter == N);
}