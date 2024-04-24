// @purpose demostrates the access to vector by multiple access
// @dataRaces 1

#include <thread>
#include <vector>
#include <iostream>

void func(std::vector<int *> *vec) {
    for (auto ptr : *vec) {
        *ptr = 1;  // write to vec
    }
}

void func2(std::vector<int *> *vec) {
    for (auto ptr : *vec) {
        std::cout << *ptr;  // read to vec
    }
}

int main() {
    int o1;

    std::vector<int *> vec1;
    std::vector<int *> vec2;

    vec1.push_back(&o1);
    vec2.push_back(&o1);

    std::thread producerThrObj(&func, &vec1);
    std::thread consumerThrObj(&func2, &vec2);

    producerThrObj.join();
    consumerThrObj.join();
}

