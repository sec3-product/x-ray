// @purpose a simple single-threaded hello-world program without any race
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>


int main(int argc, char**argv) {
    std::cout << "hello world\n";
}