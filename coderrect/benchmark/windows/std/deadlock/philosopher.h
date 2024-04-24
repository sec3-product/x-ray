#pragma once
// philosophers-deadlock.cpp
// compile with: /EHsc
#include <agents.h>
#include <string>
#include <array>
#include <iostream>
#include <algorithm>
#include <random>

using namespace concurrency;
using namespace std;

// Defines a single chopstick.
typedef int chopstick;

// The total number of philosophers.
const int philosopher_count = 5;

// The number of times each philosopher should eat.
const int eat_count = 50;

// Implements the logic for a single dining philosopher.
class Philosopher : public agent
{
public:
    Philosopher(chopstick& left, chopstick& right, const wstring& name)
        : _left(left)
        , _right(right)
        , _name(name)
        , _random_generator(time(0))
    {
        send(_times_eaten, 0);
    }

    Philosopher(const Philosopher& p)
        : Philosopher(p._left, p._right, p._name)
    {
        _random_generator = p._random_generator;
    }

    // Retrieves the number of times the philosopher has eaten.
    int times_eaten()
    {
        return receive(_times_eaten);
    }

    // Retrieves the name of the philosopher.
    wstring name() const
    {
        return _name;
    }

    void pickup_chopsticks();
    void putdown_chopsticks();

protected:
    // Performs the main logic of the dining philosopher algorithm.
    void run()
    {
        // Repeat the thinks/eat cycle a set number of times.
        for (int n = 0; n < eat_count; ++n)
        {
            think();
            pickup_chopsticks();
            eat();
            send(_times_eaten, n + 1);
            putdown_chopsticks();
        }

        done();
    }

    // Simulates thinking for a brief period of time.
    void think()
    {
        random_wait(100);
    }

    // Simulates eating for a brief period of time.
    void eat()
    {
        random_wait(100);
    }

private:
    // Index of the left chopstick in the chopstick array.
    chopstick& _left;
    // Index of the right chopstick in the chopstick array.
    chopstick& _right;

    // The name of the philosopher.
    wstring _name;
    // Stores the number of times the philosopher has eaten.
    overwrite_buffer<int> _times_eaten;

    // A random number generator.
    mt19937 _random_generator;

    // Yields the current context for a random period of time.
    void random_wait(unsigned int max)
    {
        concurrency::wait(_random_generator() % max);
    }
    
};