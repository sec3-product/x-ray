#include <stdio.h>
#include "foo/foo.h"

int main(void)
{
	puts("This is a shared library test...");
	foo();
	return 0;
}
