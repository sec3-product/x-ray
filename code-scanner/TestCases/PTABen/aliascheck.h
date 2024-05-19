#include <stdio.h>
#include <stddef.h>

extern void __aser_alias__(void *p, void *q);
extern void __aser_no_alias__(void *p, void *q);

#define MUSTALIAS(p, q) __aser_alias__((void *)p, (void *)q)
#define PARTAILALIAS(p, q) __aser_alias__((void *)p, (void *)q)
#define MAYALIAS(p, q) __aser_alias__((void *)p, (void *)q)
#define NOALIAS(p, q) __aser_no_alias__((void *)p, (void *)q)
