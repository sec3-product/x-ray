#include <iostream>
#include <thread>

void write(size_t &counter) { counter++; }

int main() {
  size_t counter = 0;

  // Race on counter
  std::thread t1(write, std::ref(counter));
  std::thread t2(write, std::ref(counter));
  t1.join();
  t2.join();

  std::cout << "Expected: 2\t Got: " << counter << "\n";
}