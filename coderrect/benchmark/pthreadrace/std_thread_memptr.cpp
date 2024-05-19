// @purpose std::thread API support
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <thread>
#include <unistd.h>


namespace pthreadrace {


    class RingBuffer {
    private:
        int *buffer_;
        size_t write_pos_;
        size_t available_;
        size_t capacity_;

    public:
        RingBuffer(size_t capacity) : capacity_(capacity) {
            if (capacity == 0)
                throw std::invalid_argument("capacity must be greater than 0");

            buffer_ = new int[capacity];
            available_ = 0;
            write_pos_ = 0;
        }


        ~RingBuffer() {
            if (buffer_ != nullptr)
                delete [] buffer_;
        }


        bool Publish(int value) {
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


class ProducerThread {
private:
    pthreadrace::RingBuffer *rbuf_;

public:
    explicit ProducerThread(pthreadrace::RingBuffer* rbuf) : rbuf_(rbuf) {}

    void Produce() {
        int i = 0;
        while (true) {
            if (rbuf_->Publish(i))
                i++;
            sleep(1);
        }
    }
};


class ConsumerThread {
private:
    pthreadrace::RingBuffer *rbuf_;

public:
    explicit ConsumerThread(pthreadrace::RingBuffer* rbuf) : rbuf_(rbuf) {}

    void Consume() {
        int i = 0;
        while (true) {
            if (rbuf_->Consume(&i)) {
                std::cout << i << std::endl;
            }
        }
    }
};

int main() {
    pthreadrace::RingBuffer rbuf(1024);

    ProducerThread t1(&rbuf);
    ConsumerThread t2(&rbuf);

    std::thread producerThrObj(&ProducerThread::Produce, &t1);
    std::thread consumerThrObj(&ConsumerThread::Consume, &t2);

    producerThrObj.join();
    consumerThrObj.join();
}
