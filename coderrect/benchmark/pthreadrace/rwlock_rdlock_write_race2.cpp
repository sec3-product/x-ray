// @purpose a thread hold the read lock and update the shared data
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <pthread.h>
#include <unistd.h>


namespace pthreadrace {


    class BulletinRWLock {
    private:
        pthread_rwlock_t rwlock_;
        std::string message_;
        int count_;

    public:
        BulletinRWLock() : message_("no message"), count_(0) {
            pthread_rwlock_init(&rwlock_, NULL);
        }

        ~BulletinRWLock() {}


        std::string Read() {
            pthread_rwlock_rdlock(&rwlock_);
            std::string msg = message_;
            count_++;
            pthread_rwlock_unlock(&rwlock_);
            return msg;
        }

        void Update(const std::string message) {
            pthread_rwlock_wrlock(&rwlock_);
            message_ = message;
            count_++;
            pthread_rwlock_unlock(&rwlock_);
        }

    };


}   // namespace pthreadrace



static void *Editor(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::BulletinRWLock*)rbuf;
    int i = 0;
    while (true) {
        rbuf_ptr->Update("hell world ");
        i++;
        sleep(1);
    }
}


static void *Reader(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::BulletinRWLock*)rbuf;
    int i = 0;
    while (true) {
        std::string msg = rbuf_ptr->Read();
        std::cout << msg << std::endl;
    }
}


int main() {
    pthreadrace::BulletinRWLock bulletin;
    int rc;
    void *res;
    pthread_t thr_editor, thr_reader, thr_reader2;

    rc = pthread_create(&thr_editor, NULL, Editor, (void *)&bulletin);
    if (rc) {
        perror("Failed to create the producer thread");
        exit(-1);
    }

    rc = pthread_create(&thr_reader, NULL, Reader, (void *)&bulletin);
    if (rc) {
        perror("Failed to create the consumer thread");
        exit(-1);
    }
    rc = pthread_create(&thr_reader2, NULL, Reader, (void *)&bulletin);
    if (rc) {
        perror("Failed to create the consumer thread");
        exit(-1);
    }

    pthread_join(thr_editor, &res);
    pthread_join(thr_reader, &res);
    pthread_join(thr_reader2, &res);
}
