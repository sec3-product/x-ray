// @purpose a state-machine
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

/**
    {
      "access1": {
        "col": 9,
        "dir": "/home/jsong/source/memcached",
        "filename": "items.c",
        "line": 1762,
        "snippet": " 1760|    if (*tail == it) {\n 1761|        assert(it->next == 0);\n>1762|        *tail = it->prev;\n 1763|    }\n 1764|    assert(it->ne
xt != it);\n",
        "sourceLine": " 1762|        *tail = it->prev;\n",
        "stacktrace": [
          "pthread_create [crawler.c:505]",
          "item_crawler_thread [crawler.c:505]",
          "lru_crawler_class_done [crawler.c:378]",
          "do_item_unlinktail_q [crawler.c:345]"
        ]
      },
      "access2": {
        "col": 14,
        "dir": "/home/jsong/source/memcached",
        "filename": "items.c",
        "line": 1098,
        "snippet": " 1096|    id |= cur_lru;\n 1097|    pthread_mutex_lock(&lru_locks[id]);\n>1098|    search = tails[id];\n 1099|    \n 1100|    for (; tries > 0 && search != NULL; tries--, search=next_it) {\n",
"sourceLine": " 1098|    search = tails[id];\n",
"stacktrace": [
"pthread_create [storage.c:226]",
"storage_write_thread [storage.c:226]",
"storage_write [storage.c:169]",
"lru_pull_tail [storage.c:25]"
]
},
"isOmpRace": false,
"priority": 3,
"sharedObj": {
"dir": "/home/jsong/source/memcached",
"filename": "items.c",
"line": 56,
"name": "tails",
"sourceLine": " 56|static item *tails[LARGEST_ID];\n"
}
},
*/


#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <unistd.h>


/**
 * A non-trivial program
 */

static pthread_mutex_t mutex;
static uint32_t *buf;


static void *SumThread(void *unused) {
    uint32_t sum = 0;
    bool start = false;
    int processed = 0;

    /**
     * I mimic a state machine here so that the lock-update-unlock isn't
     * linear from the perspective of source code.
     *
     * Every "if" is a state and "while(true)" drive the state machine
     */
    while (true) {
        if (start && processed != 1024) {
            for (int i = processed; i < 256; i++, processed++) {
                sum += buf[processed];
            }
            continue;
        }

        if (processed == 1024) {
            start = false;
            processed = 0;
            pthread_mutex_unlock(&mutex);
            sleep(5);
            continue;
        }

        if (processed == 0) {
            pthread_mutex_lock(&mutex);
            start = true;
        }

        std::cout << sum;
    }
}


int main(int argc, char**argv) {
    pthread_t thr_sum;

    pthread_mutex_init(&mutex, nullptr);
    buf = new uint32_t[1024];

    pthread_create(&thr_sum, nullptr, SumThread, nullptr);

    while (true) {
        pthread_mutex_lock(&mutex);
        delete buf;
        buf = new uint32_t[1024];
        pthread_mutex_unlock(&mutex);
    }
}