//
// Created by peiming on 9/14/20.
//
// @purpose boost::thread API support
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <boost/thread.hpp>

using namespace boost;

constexpr size_t SIZE = 1024;


void sum(uint32_t *buf) {
  uint32_t sum = 0;
  for (int i = 0; i < SIZE; i++) {
    sum += buf[i];
  }

  std::cout << sum << "\n";
}


void randomize(uint32_t *buf) {
  for (int i = 0; i < SIZE; i++) {
    buf[i] = rand();
  }
}


int main(int argc, char**argv) {
  uint32_t *buf = new uint32_t[SIZE];

  thread sumth(&sum, buf);
  thread randomizeth(&randomize, buf);

  sumth.join();
  randomizeth.join();

  delete []buf;
  return 0;
}