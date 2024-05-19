
int global;

int main() {

    int local;

    #pragma omp parallel
    {
        local++; // Race on local
        global++; // Race on global
    }
}