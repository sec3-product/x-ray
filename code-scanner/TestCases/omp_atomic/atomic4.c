#include <stdio.h>
/*
 * Test if atomic can be recognized properly. No data races.
 * */
int main (void)
{
  int a=0;
#pragma omp parallel 
  {
#pragma omp atomic update
    a++;
  }
  printf ("a=%d\n",a);
  return 0;
}

