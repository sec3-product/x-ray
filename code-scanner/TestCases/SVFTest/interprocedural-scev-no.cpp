int get1(int *base, int index) {
  return base[index];
}

int get(int* base, int index) {
  return get1(base, index);
}

int main() {
  const int N = 100;
  int A[N];
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    A[i] = get(A, i);
  }
}
