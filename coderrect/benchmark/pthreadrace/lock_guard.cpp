// @purpose std::lock_guard support
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cstddef>
#include <stdexcept>
#include <mutex>

namespace pthreadrace {


class RingBufferLockGuard {
    private:
        std::mutex critical_section_;
        int *buffer_;
        size_t write_pos_;
        size_t available_;
        size_t capacity_;

    public:
        RingBufferLockGuard(size_t capacity) : capacity_(capacity) {
            if (capacity == 0)
                throw std::invalid_argument("capacity must be greater than 0");

            buffer_ = new int[capacity];
            available_ = 0;
            write_pos_ = 0;
        }


        ~RingBufferLockGuard() {
            if (buffer_ != nullptr)
                delete [] buffer_;
        }


        bool Publish(int value) {
            std::lock_guard<std::mutex> lg(critical_section_);

            if(available_ < capacity_){
                if(write_pos_ >= capacity_){
                    write_pos_ = 0;
                }
                buffer_[write_pos_] = value;
                write_pos_++;
                available_++;
                return true;
            }

            return false;
        }


        bool Consume(int *r) {
            std::lock_guard<std::mutex> lg(critical_section_);

            if(available_ == 0){
                return false;
            }
            int next_slot = write_pos_ - available_;
            if(next_slot < 0){
                next_slot += capacity_;
            }
            *r = buffer_[next_slot];
            available_--;
            return true;
        }

    };


}  // namespace pthreadrace


static void *Producer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferLockGuard*)rbuf;
    int i = 0;
    while (true) {
        if (rbuf_ptr->Publish(i))
            i++;
        sleep(1);
    }
}


static void *Consumer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferLockGuard*)rbuf;
    int i = 0;
    while (true) {
        if (rbuf_ptr->Consume(&i)) {
            std::cout << i << std::endl;
        }
    }
}


int main() {
    pthreadrace::RingBufferLockGuard rbuf(1024);
    int rc;
    pthread_t thr_producer, thr_consumer;

    rc = pthread_create(&thr_producer, NULL, Producer, (void *)&rbuf);
    if (rc) {
        perror("Failed to create the producer thread");
        exit(-1);
    }

    rc = pthread_create(&thr_consumer, NULL, Consumer, (void *)&rbuf);
    if (rc) {
        perror("Failed to create the consumer thread");
        exit(-1);
    }

    pthread_exit(nullptr);
}
