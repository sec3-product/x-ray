#include "aliascheck.h"
#include <iostream>
#include <vector>

using namespace std;

int global_obj;
int *global_ptr = &global_obj;

class A {
  public:
    virtual void f(int *i, A* self) const {
      MUSTALIAS(global_ptr, i);
      MUSTALIAS(this, self);
    }
};

int main(int argc, char **argv)
{
  int *ptr = &global_obj;

  vector<const A*> vec;
  A b;
  A a;
  vec.push_back(&a);
  vec.push_back(&b);
  	
  vector<const A*>::const_iterator it = vec.begin();
  const A *aptr = *it;
  aptr->f(ptr, &b);

  return 0;
}
