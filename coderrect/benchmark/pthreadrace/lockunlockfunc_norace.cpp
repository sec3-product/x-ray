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


my_spinlock_t lock;
int x;


void thread_function() {
    MySpinLock(&lock);
    x++;
    std::cout << x << "\n";
    MySpinUnlock(&lock);
}


int main ()
{
    std::thread first (thread_function);
    std::thread second (thread_function);
    first.join();
    second.join();
    return 0;
}
