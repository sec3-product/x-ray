#ifndef PTHREADRACE_RINGBUFFER_LIB_H
#define PTHREADRACE_RINGBUFFER_LIB_H


#include <cstddef>
#include <stdexcept>

namespace pthreadrace {


    class RingBuffer {
    private:
        int *buffer_;
        size_t write_pos_;
        size_t available_;
        size_t capacity_;

    public:
        RingBuffer(size_t capacity);
        ~RingBuffer();

        bool Publish(int value);
        bool Consume(int *r);
    };


}  // namespace pthreadrace



#endif //PTHREADRACE_RINGBUFFER_LIB_H
