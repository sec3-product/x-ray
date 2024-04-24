#include <iostream>
#include <mutex>
#include <thread>

struct Counter {
  std::mutex mut;
  size_t count;
  void add() {
    // This access is guarded by lock
    std::unique_lock<std::mutex> lock(mut);
    count++;
  }

  Counter() : count(0) {}
};

void write(Counter *counter) {
  counter->add();
  // This access is not guarded by lock
  counter->count++;
}

int main() {
  Counter counter;

  std::thread t1(write, &counter);
  std::thread t2(write, &counter);
  t1.join();
  t2.join();

  std::cout << "Expected: 4\t Got: " << counter.count << "\n";
}