int main() {
    int x = 0;

#pragma omp parallel
    {
#pragma omp single
        { x++; }

#pragma omp single
        { x--; }
    }

    assert(x == 0);
}