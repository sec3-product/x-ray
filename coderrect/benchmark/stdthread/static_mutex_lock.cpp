#include <iostream>
#include <mutex>
#include <thread>

void add(size_t &counter) {
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  counter++;
}

int main() {
  size_t counter = 0;

  // Race on counter
  std::thread t1(add, std::ref(counter));
  std::thread t2(add, std::ref(counter));

  t1.join();
  t2.join();

  // Bonus race on this read
  std::cout << "Expected: 2\t Got: " << counter << "\n";
}