// Based on lulesh.cc:282

// Race on 17:31

#include <vector>

struct Object {
    std::vector<float> P;
    std::vector<float> Q;

    Object() = delete;
    Object(int N) {
        P.resize(N);
        Q.resize(N);
    }

    float& p(int i) { return P[i]; }
    float& q(int i) { return Q[i]; }
};

void foo(Object &obj, float *A, float *B, float *C, int N) {
    #pragma omp parallel for firstprivate(N)
    for (int i = 0 ; i < N ; ++i){
        A[i] = B[i] = C[i] =  - obj.p(i) - obj.q(i) ;
        obj.p(i+1) = A[i];
    }
}

int main() {
    const int N = 100;
    Object o(N);

    auto A = new float[N];
    auto B = new float[N];
    auto C = new float[N];

    foo(o, A, B, C, N);
}