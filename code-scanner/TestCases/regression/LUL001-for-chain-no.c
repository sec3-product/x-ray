// This test is based on LULESH
// lulesh.cc:2329

int main() {
    const int N = 100;
    int A[N+1];
    int B[N+1];
    int C[N+1];

    int min = 7;
    int max = 12;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            A[i] = B[i];
        }

        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            if (A[i] < min) {
                A[i] = min;
            }
        }

        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            if (A[i] > max) {
                A[i] = max;
            }
        }


    }
}