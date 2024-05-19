
int main() {

    const int N = 10;

    int A[N];

    #pragma omp parallel
    {
        #pragma omp for
        for (int  i = 0; i < N; i++) {
            A[i] = i;
        }

        #pragma omp for
        for (int  i = 0; i < N/2; i++) {
            A[i] += i;
        }
    }
}