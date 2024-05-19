/*
 * Struct with array of structs.
 * Author: Sen Ye
 * Date: 28/04/2014
 */
#include "aliascheck.h"

struct InnerArrayStruct {
	int in1[10];
	char in2;
	double in3;
};

struct MidArrayStruct {
	char mid1;
	struct InnerArrayStruct mid2[5];
	double mid3[20];
};

struct ArrayStruct {
	int out1;
	char out2;
	struct MidArrayStruct out3;
	int out4;
};

int main() {
	struct ArrayStruct* p;
	struct ArrayStruct s;

	p = &s;

	MUSTALIAS(&p->out4, &s.out4);

	MAYALIAS(&p->out3.mid2[1].in1[3], &s.out3.mid2[2]);
	MAYALIAS(&p->out3.mid2[1], &p->out3.mid2[2]);
	NOALIAS(&p->out3.mid2[3].in3, &s.out3.mid3[2]);
	NOALIAS(&p->out3.mid2[0], &s.out4);
	
	return 0;
}
