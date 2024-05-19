int main() {
    int x = 0;
#pragma omp parallel
    {
#pragma omp master
        { x++; }

#pragma omp single
        { x--; }
    }
}