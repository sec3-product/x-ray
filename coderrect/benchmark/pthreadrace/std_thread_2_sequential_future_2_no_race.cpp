// @purpose create two concurrent std::future that update a global variable sequentially that is guaranteed by checking future.valid()
// @dataRaces 0

#include <thread>
#include <future>
#include <iostream>


int g_x;


int main() {
    auto f1 = std::async(std::launch::async, []() {
        g_x ++;
        std::cout << g_x << "\n";
    });

    while (!f1.valid()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << g_x << "\n";

    auto f2 = std::async(std::launch::async, [](){
        g_x --;
        std::cout << g_x << "\n";
    });

    f2.get();

    std::cout << g_x << "\n";
}