//
// Created by peiming on 9/14/20.
//
// @purpose boost::thread API support
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <queue>
#include <boost/thread.hpp>
using namespace std;

int main() {
  int c = 0;
  bool done = false;
  queue<int> goods;

  boost::thread producer([&]() {
    for (int i = 0; i < 500; ++i) {
      goods.push(i);
      c++;
    }

    done = true;
  });

  boost::thread consumer([&]() {
    while (!done) {
      while (!goods.empty()) {
        goods.pop();
        c--;
      }
    }
  });

  producer.join();
  consumer.join();
  cout << "Net: " << c << endl;
}