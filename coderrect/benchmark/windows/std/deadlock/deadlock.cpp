// deadlock.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "philosopher.h"


int main()
{
	// Create an array of index values for the chopsticks.
	array<chopstick, philosopher_count> chopsticks = { 0, 1, 2, 3, 4 };

	// Create an array of philosophers. Each pair of neighboring 
	// philosophers shares one of the chopsticks.
	array<Philosopher, 5> philosophers = {
		   Philosopher(chopsticks[0], chopsticks[1], L"aristotle"),
		   Philosopher(chopsticks[1], chopsticks[2], L"descartes"),
		   Philosopher(chopsticks[2], chopsticks[3], L"hobbes"),
		   Philosopher(chopsticks[3], chopsticks[4], L"socrates"),
		   Philosopher(chopsticks[4], chopsticks[0], L"plato"),
	};

	// Begin the simulation.
	for_each(begin(philosophers), end(philosophers), [](Philosopher& p) {
		p.start();
		});

	// Wait for each philosopher to finish and print his name and the number
	// of times he has eaten.
	for_each(begin(philosophers), end(philosophers), [](Philosopher& p) {
		agent::wait(&p);
		wcout << p.name() << L" ate " << p.times_eaten() << L" times." << endl;
		});

}
