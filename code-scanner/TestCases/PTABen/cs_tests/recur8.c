#include "aliascheck.h"
int z1,z2;
void foo(int **p);
void bar(int **a){
	int *c, b;
	*a = &b;
	c = *a;
	MUSTALIAS(c,&b);
	NOALIAS(c,&z1);
	MAYALIAS(c,&z2);
	foo(a);
}


void foo(int** p){

	p = malloc(10);
	*p = &z2;
	bar(p);
}

int main(){

	int **x, *y;
	x = &y;
	y = &z1;
	foo(x);
}
