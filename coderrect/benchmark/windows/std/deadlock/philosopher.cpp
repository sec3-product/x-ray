#include "philosopher.h"

// A shared array of critical sections. Each critical section 
// guards access to a single chopstick.
critical_section locks[philosopher_count];

// Gains access to the chopsticks.
void Philosopher::pickup_chopsticks()
{
    // Deadlock occurs here if each philosopher gains access to one
    // of the chopsticks and mutually waits for another to release
    // the other chopstick.
    locks[_left].lock();
    locks[_right].lock();
}

// Releases the chopsticks for others.
void Philosopher::putdown_chopsticks()
{
    locks[_right].unlock();
    locks[_left].unlock();
}
