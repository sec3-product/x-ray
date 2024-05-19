#include <stdio.h>
/*
 * Test if atomic can be recognized properly. No data races.
 * */

#define expr (1 * v + 10 / v)
#define binop *

// See: https://www.openmp.org/spec-html/5.0/openmpsu95.html#x126-4840002.17.7
int main(void) {
    int x = 0;



#pragma omp parallel
{
    int v;
    // Read
    #pragma omp atomic read
    v = x; 

    // Write
    #pragma omp atomic write
    x = expr;

    // Update
    #pragma omp atomic update
    x++;
    #pragma omp atomic update
    x--;
    #pragma omp atomic update
    ++x;
    #pragma omp atomic update
    --x;
    #pragma omp atomic update
    x += expr;
    #pragma omp atomic update
    x = x binop expr;
    #pragma omp atomic update
    x = expr binop x;

    // capture
    #pragma omp atomic capture
    v = x++;
    #pragma omp atomic capture
    v = x--;
    #pragma omp atomic capture
    v = ++x;
    #pragma omp atomic capture
    v = --x;
    #pragma omp atomic capture
    v = x *= expr;
    #pragma omp atomic capture
    v = x = x binop expr;
    #pragma omp atomic capture
    v = x = expr binop x;

    // structured block
    #pragma omp atomic capture
    { v = x; x *= expr; }
    #pragma omp atomic capture
    { x *= expr; v = x; }
    #pragma omp atomic capture
    { v = x; x = x binop expr; }
    #pragma omp atomic capture
    { v = x; x = expr binop x; }
    #pragma omp atomic capture
    { x = x binop expr; v = x; }
    #pragma omp atomic capture
    { x = expr binop x; v = x; }
    #pragma omp atomic capture
    { v = x; x = expr; }
    #pragma omp atomic capture
    { v = x; x++; }
    #pragma omp atomic capture
    { v = x; ++x; }
    #pragma omp atomic capture
    { ++x; v = x; }
    #pragma omp atomic capture
    { x++; v = x; }
    #pragma omp atomic capture
    { v = x; x--; }
    #pragma omp atomic capture
    { v = x; --x; }
    #pragma omp atomic capture
    { --x; v = x; }
    #pragma omp atomic capture
    { x--; v = x; }
    }

    return 0;
}
