# Recursive pthread tiny benchmarks

Both test cases involve recursive pthread creation that our tool cannot handle correctly for now.

## recursive1.cpp

Calling `pthread_create()` in a recusive function. Since our race detector does not handle recursive function, the race detector only sees one thread, therefore missing data races.

## recursive2.cpp

Calling `pthread_create()` inside a pthread callback function. Our race detector will go into a endless recursion.
