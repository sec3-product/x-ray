#include <iostream>
#include <thread>

void write(std::shared_ptr<size_t> pcounter) {
  size_t *counter = pcounter.get();
  (*counter)++;
}

int main() {
  auto counter = std::make_shared<size_t>(0);

  // Race on counter
  std::thread t1(write, counter);
  std::thread t2(write, counter);

  t1.join();
  t2.join();

  std::cout << "Expected: 2\t Got: " << *counter << "\n";
}