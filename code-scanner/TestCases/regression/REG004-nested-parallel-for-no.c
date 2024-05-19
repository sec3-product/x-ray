int main() {
    const int N = 10;
    #pragma omp parallel
    {
        // Every thread creates a local x
        int x = 0;

        #pragma omp for
        for (int i = 0; i < N; i++) { 
            // Each thread writes to ots own local x
            x++;
        }
    }
}