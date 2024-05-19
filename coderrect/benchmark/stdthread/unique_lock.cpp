#include <iostream>
#include <mutex>
#include <thread>

struct Counter {
  std::mutex mut;
  size_t count;
  void add() { count++; }

  Counter() : count(0) {}
};

void write(Counter *counter) {
  // Unique lock holds mutex until its destructor is called
  std::unique_lock<std::mutex> lock(counter->mut);
  counter->add();
}

int main() {
  Counter counter;

  std::thread t1(write, &counter);
  std::thread t2(write, &counter);
  t1.join();
  t2.join();

  std::cout << "Expected: 2\t Got: " << counter.count << "\n";
}