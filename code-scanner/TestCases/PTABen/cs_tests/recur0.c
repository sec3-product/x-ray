#include "aliascheck.h"
int *x, y, z;

void f() {
  if (1) {
         x = &y;
	MUSTALIAS(x,&y);
          f();
         x = &z;
	MUSTALIAS(x,&z);
	NOALIAS(x,&y);
          f();
  }
}


int main()
{
           f();
	   return 0;
}
