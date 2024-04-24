// @purpose demo how to use lockunlockFunctions to define customized lock functions
// @dataRaces 1
// @configuratin lockunlockfunc.json


// @purpose demo how lockUnlockFunctions handles homemade lock functions
// @configuration lockunlockfunc.json

#include <iostream>       // std::cout
#include <atomic>         // std::atomic, std::memory_order_relaxed
#include <thread>         // std::thread


struct my_spinlock_t {
    std::atomic<bool> a;

    my_spinlock_t() : a(false) {}
};


void MySpinLock(my_spinlock_t* lock) {
    bool expected = false;
    while (!lock->a.compare_exchange_strong(expected, true))
        ;
}


void MySpinUnlock(my_spinlock_t* lock) {
    lock->a = false;
}


my_spinlock_t lock1;
my_spinlock_t lock2;
int x;


void thread_function_1() {
    MySpinLock(&lock1);
    x++;
    std::cout << x << "\n";
    MySpinUnlock(&lock1);
}


void thread_function_2() {
    MySpinLock(&lock2);
    x++;
    std::cout << x << "\n";
    MySpinUnlock(&lock2);
}


int main ()
{
    std::thread first (thread_function_1);
    std::thread second (thread_function_2);
    first.join();
    second.join();
    return 0;
}

