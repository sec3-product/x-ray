// @purpose std::atomic support
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0


#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
using namespace std;

atomic<int> accum(0);

void square(int x)
{
    accum += x * x;
}

int main() {
    vector<thread> ths;
    for (int i = 1; i <= 20; i++) {
        ths.push_back(thread(&square, i));
    }

    for (auto& th : ths) {
        th.join();
    }
    cout << "accum = " << accum << endl;
    return 0;
}