#include <iostream>
#include <mutex>
#include <thread>

// Based on example from:
// https://en.cppreference.com/w/cpp/thread/unique_lock

struct Account {
  std::mutex mut;
  size_t balance;  // this bank doesn't allow debt

  Account(size_t startingBalance) : balance(startingBalance) {}
};

void transfer(Account &from, Account &to, size_t amount) {
  std::unique_lock<std::mutex> lockFrom(from.mut, std::defer_lock);
  std::unique_lock<std::mutex> lockTo(to.mut, std::defer_lock);

  // No deadlock
  std::lock(lockFrom, lockTo);

  from.balance -= amount;
  to.balance += amount;

  // locks released when unique_lock destructor is called
}

int main() {
  Account brad(100);
  Account yanze(100000);

  // Race on counter
  std::thread t1(transfer, std::ref(brad), std::ref(yanze), 50);
  std::thread t2(transfer, std::ref(yanze), std::ref(brad), 50000);
  t1.join();
  t2.join();

  std::cout << "Yanze: " << yanze.balance << "\tBrad: " << brad.balance << "\n";
}