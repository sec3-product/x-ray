int main() {

    int x;

    #pragma omp parallel
    {
        #pragma omp single
        {
            x++;
        }
    }
}