#include <iostream>
#include <string>
#include <cstdlib>
#include <pthread.h>
 
using namespace std;

string shared_str = "This is a shared string";
 
void *WriteToString(void *) {
  shared_str = "Shared string changed by WriteToString";
  pthread_exit(NULL);
}

void *ReadFromString(void *) {
  string local_str = shared_str;
  cout << "Read from shared_str: " << local_str << "\n";
  pthread_exit(NULL);
}
 
int main () {
    pthread_t thread1;
    pthread_t thread2;
    int rc = pthread_create(&thread1, NULL, WriteToString, NULL);
    if (rc) {
      cout << "Error:unable to create thread," << rc << endl;
      exit(-1);
    }
    rc = pthread_create(&thread2, NULL, ReadFromString, NULL);
    if (rc) {
      cout << "Error:unable to create thread," << rc << endl;
      exit(-1);
    }
 
    pthread_join(thread1,0);
    pthread_join(thread2,0);
}
