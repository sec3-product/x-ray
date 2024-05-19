#include <iostream>
#include <thread>

void write(size_t *counter) { (*counter)++; }

int main() {
  size_t counter = 0;

  // Race on counter
  std::thread t1(write, &counter);
  std::thread t2(write, &counter);

  t1.join();
  t2.join();

  std::cout << "Expected: 2\t Got: " << counter << "\n";
}