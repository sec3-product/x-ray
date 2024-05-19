// Based on lulesh.cc:1510

// Potential race on Domain::out at line 54 and line 62

#include <vector>

class Domain {
    std::vector<float> P;
    std::vector<float> Q;
    std::vector<float> O;
    std::vector<int> LIST;

public:
    Domain() = delete;
    Domain(int N) {
        P.resize(N);
        Q.resize(N);
    }

    float& p(int i) { return P[i]; }
    float& q(int i) { return Q[i]; }
    float& out(int i) { return O[i]; }

    int* list(int i) { return &LIST[8*i]; }

};

void inner(Domain &d, const int* elem, int X[8]) {
    int nd0i = elem[0] ;
    int nd1i = elem[1] ;
    int nd2i = elem[2] ;
    int nd3i = elem[3] ;
    int nd4i = elem[4] ;
    int nd5i = elem[5] ;
    int nd6i = elem[6] ;
    int nd7i = elem[7] ;

    X[0] = d.p(nd0i);
    X[1] = d.p(nd1i);
    X[2] = d.p(nd2i);
    X[3] = d.p(nd3i);
    X[4] = d.p(nd4i);
    X[5] = d.p(nd5i);
    X[6] = d.p(nd6i);
    X[7] = d.out(nd7i); // Read
}

void foo(Domain &d, float *A, float *B, int N) {
    #pragma omp parallel for firstprivate(N)
    for (int i = 0; i < N; i++) {
        int X[8];
        int local = 0;

        const int* const elem = d.list(i);
        inner(d, elem, X);


        for (int i = 0; i < 8; i++) {
            local += d.p(i) * d.q(i);
        }

        d.out(i) = local; // Write
    }
}


int main() {
    const int N = 1024;
    Domain d(N);
    auto A = new float[N];
    auto B = new float[N];

    foo(d, A, B, N);
    foo(d, A, B, N);
}