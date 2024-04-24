// @purpose support std::async
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
using namespace std;

int accum = 0;

void square(int x)
{
    accum += x * x;
}

int main() {
    for (int i = 1; i <= 20; i++) {
        async(&square, i);
    }

    cout << "accum = " << accum << endl;
    return 0;
}