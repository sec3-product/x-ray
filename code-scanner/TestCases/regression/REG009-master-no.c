int main() {
    int x = 0;

#pragma omp parallel
    {
#pragma omp master
        { x = 1; }
    }

#pragma omp master
    {
#pragma omp master
        { x = 0; }
    }
}