#include <stdio.h>
/*
 * Test if atomic can be recognized properly. No data races.
 * */
int main (void)
{
  int a=0;
#pragma omp parallel 
  {
    int x;
#pragma omp atomic write
    a = 5;
#pragma omp atomic read
    x = a;
  }
  printf ("a=%d\n",a);
  return 0;
}

