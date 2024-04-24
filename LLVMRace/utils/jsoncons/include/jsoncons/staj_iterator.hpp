// Copyright 2018 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_STAJ_ITERATOR_HPP
#define JSONCONS_STAJ_ITERATOR_HPP

#include <new> // placement new
#include <memory>
#include <string>
#include <stdexcept>
#include <system_error>
#include <ios>
#include <iterator> // std::input_iterator_tag
#include <jsoncons/json_exception.hpp>
#include <jsoncons/staj_reader.hpp>

namespace jsoncons {

template<class Json, class T=Json>
class staj_array_iterator
{
    typedef typename Json::char_type char_type;

    basic_staj_reader<char_type>* reader_;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;
    T* valuep_;
public:
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::input_iterator_tag iterator_category;

    staj_array_iterator() noexcept
        : reader_(nullptr), valuep_(nullptr)
    {
    }

    staj_array_iterator(basic_staj_reader<char_type>& reader)
        : reader_(std::addressof(reader)), valuep_(nullptr)
    {
        if (reader_->current().event_type() == staj_event_type::begin_array)
        {
            next();
        }
        else
        {
            reader_ = nullptr;
        }
    }

    staj_array_iterator(basic_staj_reader<char_type>& reader,
                        std::error_code& ec)
        : reader_(std::addressof(reader)), valuep_(nullptr)
    {
        if (reader_->current().event_type() == staj_event_type::begin_array)
        {
            next(ec);
            if (ec)
            {
                reader_ = nullptr;
            }
        }
        else
        {
            reader_ = nullptr;
        }
    }

    staj_array_iterator(const staj_array_iterator& other)
        : reader_(other.reader_), valuep_(nullptr)
    {
        if (other.valuep_)
        {
            valuep_ = ::new(&storage_)T(*other.valuep_);
        }
    }

    staj_array_iterator(staj_array_iterator&& other) noexcept
        : reader_(nullptr), valuep_(nullptr)
    {
        std::swap(reader_,other.reader_);
        std::swap(valuep_,other.valuep_);
    }

    ~staj_array_iterator()
    {
        if (valuep_)
        {
            valuep_->~T();
        }
    }

    staj_array_iterator& operator=(const staj_array_iterator& other)
    {
        reader_ = other.reader_;
        if (valuep_)
        {
            valuep_->~T();
            valuep_ = nullptr;
        }
        if (other.valuep_)
        {
            valuep_ = ::new(&storage_)T(*other.valuep_);
        }
        return *this;
    }

    staj_array_iterator& operator=(staj_array_iterator&& other) noexcept
    {
        std::swap(reader_,other.reader_);
        std::swap(valuep_,other.valuep_);
        return *this;
    }

    const T& operator*() const
    {
        return *valuep_;
    }

    const T* operator->() const
    {
        return valuep_;
    }

    staj_array_iterator& operator++()
    {
        next();
        return *this;
    }

    staj_array_iterator& increment(std::error_code& ec)
    {
        next(ec);
        if (ec)
        {
            reader_ = nullptr;
        }
        return *this;
    }

    staj_array_iterator operator++(int) // postfix increment
    {
        staj_array_iterator temp(*this);
        next();
        return temp;
    }

    friend bool operator==(const staj_array_iterator<Json,T>& a, const staj_array_iterator<Json,T>& b)
    {
        return (!a.reader_ && !b.reader_)
            || (!a.reader_ && b.done())
            || (!b.reader_ && a.done());
    }

    friend bool operator!=(const staj_array_iterator<Json,T>& a, const staj_array_iterator<Json,T>& b)
    {
        return !(a == b);
    }

private:

    bool done() const
    {
        return reader_->done() || reader_->current().event_type() == staj_event_type::end_array;
    }

    void next();

    void next(std::error_code& ec);
};

template <class Json, class T>
staj_array_iterator<Json,T> begin(staj_array_iterator<Json,T> iter) noexcept
{
    return iter;
}

template <class Json, class T>
staj_array_iterator<Json,T> end(const staj_array_iterator<Json,T>&) noexcept
{
    return staj_array_iterator<Json,T>();
}

template <class Json, class T=Json>
class staj_object_iterator
{
public:
    typedef typename Json::char_type char_type;
    typedef std::basic_string<char_type> key_type;
    typedef std::pair<key_type,T> value_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type* pointer;
    typedef value_type& reference;
    typedef std::input_iterator_tag iterator_category;

private:
    typename std::aligned_storage<sizeof(value_type), alignof(value_type)>::type storage_;
    basic_staj_reader<char_type>* reader_;
    value_type* kvp_;
public:

    staj_object_iterator() noexcept
        : reader_(nullptr), kvp_(nullptr)
    {
    }

    staj_object_iterator(basic_staj_reader<char_type>& reader)
        : reader_(std::addressof(reader)), kvp_(nullptr)
    {
        if (reader_->current().event_type() == staj_event_type::begin_object)
        {
            next();
        }
        else
        {
            reader_ = nullptr;
        }
    }

    staj_object_iterator(basic_staj_reader<char_type>& reader, 
                         std::error_code& ec)
        : reader_(std::addressof(reader)), kvp_(nullptr)
    {
        if (reader_->current().event_type() == staj_event_type::begin_object)
        {
            next(ec);
            if (ec)
            {
                reader_ = nullptr;
            }
        }
        else
        {
            reader_ = nullptr;
        }
    }

    staj_object_iterator(const staj_object_iterator& other)
        : reader_(other.reader_), kvp_(nullptr)
    {
        if (other.kvp_)
        {
            kvp_ = ::new(&storage_)value_type(*other.kvp_);
        }
    }

    staj_object_iterator(staj_object_iterator&& other) noexcept
        : reader_(other.reader_), kvp_(nullptr)
    {
        if (other.kvp_)
        {
            kvp_ = ::new(&storage_)value_type(std::move(*other.kvp_));
        }
    }

    ~staj_object_iterator()
    {
        if (kvp_)
        {
            kvp_->~value_type();
        }
    }

    staj_object_iterator& operator=(const staj_object_iterator& other)
    {
        reader_ = other.reader_;
        if (kvp_)
        {
            kvp_->~T();
            kvp_ = nullptr;
        }
        if (other.kvp_)
        {
            kvp_ = ::new(&storage_)value_type(*other.kvp_);
        }
        return *this;
    }

    staj_object_iterator& operator=(staj_object_iterator&& other) noexcept
    {
        std::swap(reader_,other.reader_);
        std::swap(kvp_,other.kvp_);
        return *this;
    }

    const value_type& operator*() const
    {
        return *kvp_;
    }

    const value_type* operator->() const
    {
        return kvp_;
    }

    staj_object_iterator& operator++()
    {
        next();
        return *this;
    }

    staj_object_iterator& increment(std::error_code& ec)
    {
        next(ec);
        if (ec)
        {
            reader_ = nullptr;
        }
        return *this;
    }

    staj_object_iterator operator++(int) // postfix increment
    {
        staj_object_iterator temp(*this);
        next();
        return temp;
    }

    friend bool operator==(const staj_object_iterator<Json,T>& a, const staj_object_iterator<Json,T>& b)
    {
        return (!a.reader_ && !b.reader_)
               || (!a.reader_ && b.done())
               || (!b.reader_ && a.done());
    }

    friend bool operator!=(const staj_object_iterator<Json,T>& a, const staj_object_iterator<Json,T>& b)
    {
        return !(a == b);
    }

private:

    bool done() const
    {
        return reader_->done() || reader_->current().event_type() == staj_event_type::end_object;
    }


    void next();

    void next(std::error_code& ec);

};

template<class Json, class T>
staj_object_iterator<Json,T> begin(staj_object_iterator<Json,T> iter) noexcept
{
    return iter;
}

template<class Json, class T>
staj_object_iterator<Json,T> end(const staj_object_iterator<Json,T>&) noexcept
{
    return staj_object_iterator<Json,T>();
}

}

#include <jsoncons/ser_traits.hpp>

namespace jsoncons {

template <class Json, class T>
void staj_array_iterator<Json,T>::next()
{
    if (!done())
    {
        reader_->next();
        if (!done())
        {
            if (valuep_)
            {
                valuep_->~T();
            }
            std::error_code ec;
            valuep_ = ::new(&storage_)T(ser_traits<T>::decode(*reader_, Json(), ec));
            if (ec)
            {
                JSONCONS_THROW(ser_error(ec, reader_->context().line(), reader_->context().column()));
            }
        }
    }
}

template<class Json, class T>
void staj_array_iterator<Json,T>::next(std::error_code& ec)
{
    if (!done())
    {
        reader_->next(ec);
        if (ec)
        {
            return;
        }
        if (!done())
        {
            if (valuep_)
            {
                valuep_->~T();
            }
            valuep_ = ::new(&storage_)T(ser_traits<T>::decode(*reader_, Json(), ec));
        }
    }
}

template<class Json, class T>
void staj_object_iterator<Json,T>::next()
{
    reader_->next();
    if (!done())
    {
        JSONCONS_ASSERT(reader_->current().event_type() == staj_event_type::name);
        key_type key = reader_->current(). template get<key_type>();
        reader_->next();
        if (!done())
        {
            if (kvp_)
            {
                kvp_->~value_type();
            }
            std::error_code ec;
            kvp_ = ::new(&storage_)value_type(std::move(key),ser_traits<T>::decode(*reader_, Json(), ec));
            if (ec)
            {
                JSONCONS_THROW(ser_error(ec, reader_->context().line(), reader_->context().column()));
            }
        }
    }
}

template<class Json, class T>
void staj_object_iterator<Json,T>::next(std::error_code& ec)
{
    reader_->next(ec);
    if (ec)
    {
        return;
    }
    if (!done())
    {
        JSONCONS_ASSERT(reader_->current().event_type() == staj_event_type::name);
        auto key = reader_->current(). template get<key_type>();
        reader_->next(ec);
        if (ec)
        {
            return;
        }
        if (!done())
        {
            if (kvp_)
            {
                kvp_->~value_type();
            }
            kvp_ = ::new(&storage_)value_type(std::move(key),ser_traits<T>::decode(*reader_, Json(), ec));
        }
    }
}

template <class T, class CharT>
typename std::enable_if<is_basic_json_class<T>::value,staj_array_iterator<T,T>>::type
make_array_iterator(basic_staj_reader<CharT>& reader)
{
    return staj_array_iterator<T,T>(reader);
}

template <class T, class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,staj_array_iterator<basic_json<CharT>,T>>::type
make_array_iterator(basic_staj_reader<CharT>& reader)
{
    return staj_array_iterator<basic_json<CharT>,T>(reader);
}

template <class T, class CharT>
typename std::enable_if<is_basic_json_class<T>::value,staj_array_iterator<T,T>>::type
make_array_iterator(basic_staj_reader<CharT>& reader, std::error_code& ec)
{
    return staj_array_iterator<T,T>(reader, ec);
}

template <class T, class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,staj_array_iterator<basic_json<CharT>,T>>::type
make_array_iterator(basic_staj_reader<CharT>& reader, std::error_code& ec)
{
    return staj_array_iterator<basic_json<CharT>,T>(reader, ec);
}

template <class T, class CharT>
typename std::enable_if<is_basic_json_class<T>::value,staj_object_iterator<T,T>>::type
make_object_iterator(basic_staj_reader<CharT>& reader)
{
    return staj_object_iterator<T,T>(reader);
}

template <class T, class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,staj_object_iterator<basic_json<CharT>,T>>::type
make_object_iterator(basic_staj_reader<CharT>& reader)
{
    return staj_object_iterator<basic_json<CharT>,T>(reader);
}

template <class T, class CharT>
typename std::enable_if<is_basic_json_class<T>::value,staj_object_iterator<T,T>>::type
make_object_iterator(basic_staj_reader<CharT>& reader, std::error_code& ec)
{
    return staj_object_iterator<T,T>(reader, ec);
}

template <class T, class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,staj_object_iterator<basic_json<CharT>,T>>::type
make_object_iterator(basic_staj_reader<CharT>& reader, std::error_code& ec)
{
    return staj_object_iterator<basic_json<CharT>,T>(reader, ec);
}

}

#endif

