#include <assert.h>

int main() {
    const int N = 100;
    int x = 0;
    #pragma omp parallel for lastprivate(x)
    for (int i = 1; i < N; i++) {
        // Each thread works on a local x
        x = i;
    }
    // Only the last iteration writes back to the shared x
    assert(x == N-1);
}