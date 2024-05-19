// @purpose demo the ability to identify which field causes the race
// @dataRaces 1


#include <pthread.h>
#include <iostream>


typedef struct Client {
    int cx;
    int cy;
    int cz;
} Client;


typedef struct Server {
    int x;
    int y;
    int z;
    Client cli;
} Server;


Server server;


void *worker(void *unused) {
    // server.x = rand();
    std::cout << server.x << "\n";
}


void *workercli(void *unused) {
    server.cli.cx = rand();
    std::cout << server.cli.cx << "\n";
}


int main() {
    pthread_t th1, th2;
    pthread_create(&th1, nullptr, worker, nullptr);
    pthread_create(&th2, nullptr, worker, nullptr);
    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);

    pthread_create(&th1, nullptr, workercli, nullptr);
    pthread_create(&th2, nullptr, workercli, nullptr);
    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}
