int main() {
    const int N = 100;
    int x = 0;
    #pragma omp parallel for firstprivate(x)
    for (int i = 1; i < N; i++) {
        // Each thread writes to its own x
        x++;
    }
}