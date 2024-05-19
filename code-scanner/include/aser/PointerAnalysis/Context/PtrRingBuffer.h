//
// Created by peiming on 11/19/19.
//

#ifndef ASER_PTA_PTRRINGBUFFER_H
#define ASER_PTA_PTRRINGBUFFER_H

namespace aser {

// forward declaration
template <typename PtrT, uint32_t N>
class PtrRingBuffer;

template <typename PtrT, uint32_t N>
class RingBufferIterator {
    const PtrRingBuffer<PtrT, N> *buffer;
    uint32_t index;

public:
    RingBufferIterator(const PtrRingBuffer<PtrT, N> *buffer, uint32_t index) : buffer(buffer), index(index){};

    RingBufferIterator<PtrT, N> &operator++() {
        if ((index + 1) % N == buffer->last) {
            this->index = N;
        } else {
            this->index = (this->index + 1) % N;
        }
        return *this;
    }

    RingBufferIterator<PtrT, N> operator++(int) {
        uint32_t i;
        if ((index + 1) % N == buffer->last) {
            i = N;
        } else {
            i = (this->index + 1) % N;
        }
        this->index = i;
        return RingBufferIterator(this->buffer, i);
    }

    PtrT *operator*() const { return (*buffer)[index]; }

    bool operator==(RingBufferIterator<PtrT, N> &rhs) { return this->index == rhs.index && this->buffer == rhs.buffer; }

    bool operator!=(RingBufferIterator<PtrT, N> &rhs) { return !this->operator==(rhs); }
};

// ring buffer that stores pointers with fixed N size
template <typename PtrT, uint32_t N>
class PtrRingBuffer {
private:
    std::array<PtrT *, N> buffer;
    uint32_t last;

    PtrT *operator[](uint32_t n) const noexcept { return buffer[n]; }

public:
    using iterator = RingBufferIterator<PtrT, N>;

    PtrRingBuffer() noexcept : buffer(), last(0) {
        buffer.fill(nullptr);
    }
    PtrRingBuffer(const PtrRingBuffer &r) : buffer(r.buffer), last(r.last) {}
    PtrRingBuffer &operator=(const PtrRingBuffer &r) {
        if (this == &r) {
            return *this;
        }

        this->buffer = r.buffer;
        this->last = r.last;
        return *this;
    };

    // return poped item
    PtrT *push(PtrT *value) {
        PtrT *poped = buffer[last];
        buffer[last] = value;
        last = (last + 1) % N;

        return poped;
    }

    iterator begin() const { return RingBufferIterator<PtrT, N>(this, last); }

    iterator end() const { return RingBufferIterator<PtrT, N>(this, N); }

    PtrRingBuffer(PtrRingBuffer &&) = delete;
    PtrRingBuffer &operator=(PtrRingBuffer &&) = delete;

    friend iterator;
};

}  // namespace aser

#endif  // ASER_PTA_PTRRINGBUFFER_H
