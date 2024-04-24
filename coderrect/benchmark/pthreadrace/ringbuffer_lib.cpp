// @purpose a static library to implement ringbuffer without any lock
// @dataRaces 3 
// @toctou 0
// @lib libringbuffer.a
// @configuration libringbuffer.json

#include "ringbuffer_lib.h"


namespace pthreadrace {


    RingBuffer::RingBuffer(size_t capacity) : capacity_(capacity) {
        if (capacity == 0)
            throw std::invalid_argument("capacity must be greater than 0");

        buffer_ = new int[capacity];
        available_ = 0;
        write_pos_ = 0;
    }


    RingBuffer::~RingBuffer() {
        if (buffer_ != nullptr)
            delete[] buffer_;
    }


    bool RingBuffer::Publish(int value) {
        if (available_ < capacity_) {
            if (write_pos_ >= capacity_) {
                write_pos_ = 0;
            }
            buffer_[write_pos_] = value;
            write_pos_++;
            available_++;
            return true;
        }

        return false;
    }


    bool RingBuffer::Consume(int *r) {
        if (available_ == 0) {
            return false;
        }
        int next_slot = write_pos_ - available_;
        if (next_slot < 0) {
            next_slot += capacity_;
        }
        *r = buffer_[next_slot];
        available_--;
        return true;
    }


}   // namespace pthreadrace
