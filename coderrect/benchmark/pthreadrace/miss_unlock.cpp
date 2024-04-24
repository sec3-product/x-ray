// @purpose missing unlock
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 1

#include <iostream>
#include <pthread.h>
#include <unistd.h>


namespace pthreadrace {


class RingBufferSafeMissUnlock {
    private:
        pthread_mutex_t critical_section_;
        int *buffer_;
        size_t write_pos_;
        size_t available_;
        size_t capacity_;

    public:
        RingBufferSafeMissUnlock(size_t capacity) : capacity_(capacity) {
            if (capacity == 0)
                throw std::invalid_argument("capacity must be greater than 0");

            buffer_ = new int[capacity];
            available_ = 0;
            write_pos_ = 0;

            pthread_mutex_init(&critical_section_, NULL);
        }


        ~RingBufferSafeMissUnlock() {
            if (buffer_ != nullptr)
                delete [] buffer_;
        }


        bool Publish(int value) {
            pthread_mutex_lock(&critical_section_);

            if(available_ < capacity_){
                if(write_pos_ >= capacity_){
                    write_pos_ = 0;
                }
                buffer_[write_pos_] = value;
                write_pos_++;
                available_++;
                // pthread_mutex_unlock(&critical_section_);
                return true;
            }

            pthread_mutex_unlock(&critical_section_);
            return false;
        }


        bool Consume(int *r) {
            pthread_mutex_lock(&critical_section_);

            if(available_ == 0){
                // pthread_mutex_unlock(&critical_section_);
                return false;
            }
            int next_slot = write_pos_ - available_;
            if(next_slot < 0){
                next_slot += capacity_;
            }
            *r = buffer_[next_slot];
            available_--;

            //pthread_mutex_unlock(&critical_section_);
            return true;
        }
    };


}  // namespace pthreadrace


static void *Producer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferSafeMissUnlock*)rbuf;
    int i = 0;
    while (true) {
        if (rbuf_ptr->Publish(i))
            i++;
        sleep(1);
    }
}


static void *Consumer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferSafeMissUnlock*)rbuf;
    int i = 0;
    while (true) {
        if (rbuf_ptr->Consume(&i)) {
            std::cout << i << std::endl;
        }
    }
}


int main() {
    pthreadrace::RingBufferSafeMissUnlock rbuf(1024);
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
