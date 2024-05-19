int main ()
{
    const int N = 10;
    int A[N];
    int task_pending = 0;

// both static scheduling and dynamic scheduling
// should have no race in this test case
#pragma omp parallel for schedule(dynamic, 1)
    for (int i =0; i< N; ++i) 
    {
        A[i] = task_pending;
    }

    return  0;
}
