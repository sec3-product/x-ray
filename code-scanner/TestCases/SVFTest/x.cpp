#include <stdio.h>

int get1(int index) {
	int x = index-10;
	printf("get1: %d\n",x);
      	return x;
}

int get(int index) {
   int x = 2*index;
   printf("get1: %d\n",x);
  return get1(x);
}

int main() {
  const int N = 100;
  int A[N];
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    A[i+1]=i;
    int index = get(i);
    int x = A[i]+99;
    A[index]=x;

    printf("main: %d\n",x);
  }
}
