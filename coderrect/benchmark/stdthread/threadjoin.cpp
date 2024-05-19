#include <iostream>
#include <thread>

void write(size_t *counter) { (*counter)++; }

int main() {
  size_t counter = 0;

  // Joins prevent race
  std::thread t1(write, &counter);
  t1.join();
  std::thread t2(write, &counter);
  t2.join();

  std::cout << "Expected: 2\t Got: " << counter << "\n";
}