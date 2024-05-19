// @purpose exam PTA on resolving virtual functions
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <thread>
#include <iostream>

using namespace std;

int shared = 0;

class Base {
public:
    Base() {
        cout << "calling base ctor\n";
    }
    virtual void foo() {
        shared ++;
    }
};

class Child : public Base {
   int non_shared;
public:
    Child() : Base(), non_shared(0) {
        cout << "calling child ctor\n";
    }

    void foo() override {
        this->non_shared ++;
    }
};


void *thread_entry(Base *ptr) {
    ptr->foo();
    return nullptr;
}


int main() {
    Base A;
    Child B;

    thread T1(&thread_entry, &A);
    thread T2(&thread_entry, &B);
}