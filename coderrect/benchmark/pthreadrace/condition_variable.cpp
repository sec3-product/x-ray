// @purpose condition variable support
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <pthread.h>
#include <unistd.h>


namespace pthreadrace {


    class RingBufferFinerGrain {
    private:
        pthread_mutex_t mutex_;
        pthread_cond_t cond_;

        int *buffer_;
        size_t write_pos_;
        size_t available_;
        size_t capacity_;

    public:
        RingBufferFinerGrain(size_t capacity) : capacity_(capacity) {
            if (capacity <= 0)
                throw std::invalid_argument("capacity must be greater than 0");

            buffer_ = new int[capacity];
            available_ = 0;
            write_pos_ = 0;

            pthread_mutex_init(&mutex_, NULL);
            pthread_cond_init(&cond_, NULL);
        }


        ~RingBufferFinerGrain() {
            if (buffer_ != nullptr)
                delete [] buffer_;
        }


        void Publish(int value) {
            pthread_mutex_lock(&mutex_);

            while (available_ == capacity_)
                pthread_cond_wait(&cond_, &mutex_);

            if(write_pos_ >= capacity_){
                write_pos_ = 0;
            }
            buffer_[write_pos_] = value;
            write_pos_++;
            available_++;

            pthread_mutex_unlock(&mutex_);
        }


        bool Consume(int *r) {
            pthread_mutex_lock(&mutex_);

            if(available_ == 0){
                pthread_mutex_unlock(&mutex_);
                return false;
            }

            int next_slot = write_pos_ - available_;
            if(next_slot < 0){
                next_slot += capacity_;
            }
            *r = buffer_[next_slot];
            available_--;
            pthread_cond_signal(&cond_);
            pthread_mutex_unlock(&mutex_);
            return true;
        }

    };


}  // namespace pthreadrace




static void *Producer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferFinerGrain*)rbuf;
    int i = 0;
    while (true) {
        rbuf_ptr->Publish(i);
        i++;
    }
}


static void *Consumer(void *rbuf) {
    auto *rbuf_ptr = (pthreadrace::RingBufferFinerGrain*)rbuf;
    int i = 0;
    while (true) {
        if (rbuf_ptr->Consume(&i)) {
            std::cout << i << std::endl;
        }
    }
}


int main() {
    pthreadrace::RingBufferFinerGrain rbuf(1024);
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

