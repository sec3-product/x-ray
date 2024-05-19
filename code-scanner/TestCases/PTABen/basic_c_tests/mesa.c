/*
 * Testcases from 177.mesa.
 * Author: Sen Ye
 * Date: 12/05/2014
 */
#include <stdio.h>
#include <stdlib.h>
#include "aliascheck.h"

void gl_Accum(int* x, unsigned y, float z)
{}

struct gl_api_table {
	void (*Accum)(int*, unsigned, float);
};


void check_pointers(struct gl_api_table* table)
{
	void **entry;
	int numentries = sizeof( struct gl_api_table ) / sizeof(void*);
	int i;

	entry = (void **) table;
	MAYALIAS(*entry, gl_Accum);
	for (i=0;i<numentries;i++) {
	    MAYALIAS(entry[i], gl_Accum);
		if (!entry[i])
			printf("found uninitialized function pointer at %d\n", i );
	}
}


void init_exec_pointers( struct gl_api_table *table )
{
   table->Accum = gl_Accum;
}

int main()
{
	struct gl_api_table* table = (struct gl_api_table*)calloc(1, sizeof(struct gl_api_table));
	init_exec_pointers( table );
	check_pointers( table );
	return 0;
}
