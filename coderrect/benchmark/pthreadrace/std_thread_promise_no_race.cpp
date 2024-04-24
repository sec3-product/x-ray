// @purpose uses promise and future to coordinate two threads
// @dataRaces 0

#include <iostream>
#include <thread>
#include <future>


int g_x;


int main()
{
    std::promise<int> p1;

    std::async(std::launch::async, [](std::promise<int>* p) {
        g_x ++;
        std::cout << g_x << "\n";

        p->set_value(g_x);
    }, &p1);

    auto f1 = p1.get_future();
    std::cout << f1.get() << "\n";
    g_x --;

    std::cout << g_x << "\n";
}