// Copyright 2013 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_BYTE_STRING_HPP
#define JSONCONS_BYTE_STRING_HPP

#include <sstream>
#include <vector>
#include <ostream>
#include <cmath>
#include <cstring> // std::memcmp
#include <memory> // std::allocator
#include <iterator>
#include <exception>
#include <iomanip> // std::setw
#include <initializer_list>
#include <utility> // std::move
#include <jsoncons/config/jsoncons_config.hpp>
#include <jsoncons/json_exception.hpp>

namespace jsoncons {

// Algorithms

namespace detail {
template <class InputIt, class Container>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type,uint8_t>::value,size_t>::type
encode_base64_generic(InputIt first, InputIt last, const char alphabet[65], Container& result)
{
    size_t count = 0;
    unsigned char a3[3];
    unsigned char a4[4];
    unsigned char fill = alphabet[64];
    int i = 0;
    int j = 0;

    while (first != last)
    {
        a3[i++] = *first++;
        if (i == 3)
        {
            a4[0] = (a3[0] & 0xfc) >> 2;
            a4[1] = ((a3[0] & 0x03) << 4) + ((a3[1] & 0xf0) >> 4);
            a4[2] = ((a3[1] & 0x0f) << 2) + ((a3[2] & 0xc0) >> 6);
            a4[3] = a3[2] & 0x3f;

            for (i = 0; i < 4; i++) 
            {
                result.push_back(alphabet[a4[i]]);
                ++count;
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 3; ++j) 
        {
            a3[j] = 0;
        }

        a4[0] = (a3[0] & 0xfc) >> 2;
        a4[1] = ((a3[0] & 0x03) << 4) + ((a3[1] & 0xf0) >> 4);
        a4[2] = ((a3[1] & 0x0f) << 2) + ((a3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; ++j) 
        {
            result.push_back(alphabet[a4[j]]);
            ++count;
        }

        if (fill != 0)
        {
            while (i++ < 3) 
            {
                result.push_back(fill);
                ++count;
            }
        }
    }

    return count;
}

// Hack to support vs2015
template <class InputIt, class F, class Container>
typename std::enable_if<std::true_type::value && sizeof(typename Container::value_type) == sizeof(uint8_t),void>::type 
decode_base64_generic(InputIt first, InputIt last, 
                      const uint8_t reverse_alphabet[256],
                      F f,
                      Container& result)
{
    uint8_t a4[4], a3[3];
    uint8_t i = 0;
    uint8_t j = 0;

    while (first != last && *first != '=')
    {
        if (!f(*first))
        {
            JSONCONS_THROW(json_runtime_error<std::invalid_argument>("Cannot decode encoded byte string"));
        }

        a4[i++] = *first++; 
        if (i == 4)
        {
            for (i = 0; i < 4; ++i) 
            {
                a4[i] = reverse_alphabet[a4[i]];
            }

            a3[0] = (a4[0] << 2) + ((a4[1] & 0x30) >> 4);
            a3[1] = ((a4[1] & 0xf) << 4) + ((a4[2] & 0x3c) >> 2);
            a3[2] = ((a4[2] & 0x3) << 6) +   a4[3];

            for (i = 0; i < 3; i++) 
            {
                result.push_back(a3[i]);
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = 0; j < i; ++j) 
        {
            a4[j] = reverse_alphabet[a4[j]];
        }

        a3[0] = (a4[0] << 2) + ((a4[1] & 0x30) >> 4);
        a3[1] = ((a4[1] & 0xf) << 4) + ((a4[2] & 0x3c) >> 2);

        for (j = 0; j < i - 1; ++j) 
        {
            result.push_back(a3[j]);
        }
    }
}

}

template <class InputIt, class Container>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type,uint8_t>::value,size_t>::type
encode_base16(InputIt first, InputIt last, Container& result)
{
    static constexpr char characters[] = "0123456789ABCDEF";

    for (auto it = first; it != last; ++it)
    {
        uint8_t c = *it;
        result.push_back(characters[c >> 4]);
        result.push_back(characters[c & 0xf]);
    }
    return (last-first)*2;
}

template <class InputIt, class Container>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type,uint8_t>::value,size_t>::type
encode_base64url(InputIt first, InputIt last, Container& result)
{
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                  "abcdefghijklmnopqrstuvwxyz"
                                                  "0123456789-_"
                                                  "\0";
    return detail::encode_base64_generic(first, last, alphabet, result);
}

template <class InputIt, class Container>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type,uint8_t>::value,size_t>::type
encode_base64(InputIt first, InputIt last, Container& result)
{
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                               "abcdefghijklmnopqrstuvwxyz"
                                               "0123456789+/"
                                               "=";
    return detail::encode_base64_generic(first, last, alphabet, result);
}

template <class Char>
bool is_base64(Char c) 
{
    return (c >= 0 && c < 128) && (isalnum((int)c) || c == '+' || c == '/');
}

template <class Char>
bool is_base64url(Char c) 
{
    return (c >= 0 && c < 128) && (isalnum((int)c) || c == '-' || c == '_');
}

inline 
static bool is_base64url(int c) 
{
    return isalnum(c) || c == '-' || c == '_';
}

// decode

// Hack to support vs2015
template <class InputIt, class Container>
typename std::enable_if<std::true_type::value && sizeof(typename Container::value_type) == sizeof(uint8_t),void>::type 
decode_base64url(InputIt first, InputIt last, Container& result)
{
    static constexpr uint8_t reverse_alphabet[256] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 62,   0xff, 0xff,
        52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,
        15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   0xff, 0xff, 0xff, 0xff, 63,
        0xff, 26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,
        41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };

    
    jsoncons::detail::decode_base64_generic(first, last, reverse_alphabet, 
                                            is_base64url<typename std::iterator_traits<InputIt>::value_type>, 
                                            result);
}

// Hack to support vs2015
template <class InputIt, class Container>
typename std::enable_if<std::true_type::value && sizeof(typename Container::value_type) == sizeof(uint8_t),void>::type 
decode_base64(InputIt first, InputIt last, Container& result)
{
    static constexpr uint8_t reverse_alphabet[256] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 62,   0xff, 0xff, 0xff, 63,
        52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,
        15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,
        41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    jsoncons::detail::decode_base64_generic(first, last, reverse_alphabet, 
                                            is_base64<typename std::iterator_traits<InputIt>::value_type>, 
                                            result);
}

// Hack to support vs2015
template <class InputIt,class Container>
typename std::enable_if<std::true_type::value && sizeof(typename Container::value_type) == sizeof(uint8_t),void>::type 
decode_base16(InputIt first, InputIt last, Container& result)
{
    size_t len = std::distance(first,last);
    if (len & 1) 
    {
        JSONCONS_THROW(json_runtime_error<std::invalid_argument>("Cannot decode encoded base16 string - odd length"));
    }

    InputIt it = first;
    while (it != last)
    {
        uint8_t val;
        auto a = *it++;
        if (a >= '0' && a <= '9') 
        {
            val = (a - '0') << 4;
        } 
        else if ((a | 0x20) >= 'a' && (a | 0x20) <= 'f') 
        {
            val = ((a | 0x20) - 'a' + 10) << 4;
        } 
        else 
        {
            JSONCONS_THROW(json_runtime_error<std::invalid_argument>("Not a hex digit. Cannot decode encoded base16 string"));
        }

        auto b = *it++;
        if (b >= '0' && b <= '9') 
        {
            val |= (b - '0');
        } 
        else if ((b | 0x20) >= 'a' && (b | 0x20) <= 'f') 
        {
            val |= ((b | 0x20) - 'a' + 10);
        } 
        else 
        {
            JSONCONS_THROW(json_runtime_error<std::invalid_argument>("Not a hex digit. Cannot decode encoded base16 string"));
        }

        result.push_back(val);
    }
}

struct byte_traits
{
    typedef uint8_t char_type;

    static constexpr int eof() 
    {
        return std::char_traits<char>::eof();
    }

    static int compare(const char_type* s1, const char_type* s2, std::size_t count)
    {
        return std::memcmp(s1,s2,count);
    }
};

// basic_byte_string

template <class Allocator>
class basic_byte_string;

// byte_string_view
class byte_string_view
{
    const uint8_t* data_;
    size_t size_; 
public:
    typedef byte_traits traits_type;

    typedef const uint8_t* const_iterator;
    typedef const uint8_t* iterator;
    typedef std::size_t size_type;
    typedef uint8_t value_type;
    typedef uint8_t& reference;
    typedef const uint8_t& const_reference;
    typedef std::ptrdiff_t difference_type;
    typedef uint8_t* pointer;
    typedef const uint8_t* const_pointer;

    constexpr byte_string_view() noexcept
        : data_(nullptr), size_(0)
    {
    }

    constexpr byte_string_view(const uint8_t* data, size_t length) noexcept
        : data_(data), size_(length)
    {
    }
/*
    template <class Container>
    constexpr byte_string_view(const Container& cont, 
                               typename std::enable_if<std::is_same<typename Container::value_type,uint8_t>::value>::type* = 0) 
        : data_(cont.data()), size_(cont.size())
    {
    }
*/
    template <class Allocator>
    constexpr byte_string_view(const basic_byte_string<Allocator>& bytes);

    constexpr byte_string_view(const byte_string_view&) noexcept = default;

    byte_string_view(byte_string_view&& other) noexcept
        : data_(nullptr), size_(0)
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    byte_string_view& operator=(const byte_string_view&) = default;

    byte_string_view& operator=(byte_string_view&& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
    }

    constexpr const uint8_t* data() const noexcept
    {
        return data_;
    }
#if !defined(JSONCONS_NO_DEPRECATED)
    JSONCONS_DEPRECATED_MSG("Instead, use size()") 
    size_t length() const
    {
        return size_;
    }
#endif
    constexpr size_t size() const noexcept
    {
        return size_;
    }

    // iterator support 
    const_iterator begin() const noexcept
    {
        return data_;
    }
    const_iterator end() const noexcept
    {
        return data_ + size_;
    }

    uint8_t operator[](size_type pos) const 
    { 
        return data_[pos]; 
    }

    int compare(const byte_string_view& s) const 
    {
        const int rc = traits_type::compare(data_, s.data(), (std::min)(size_, s.size()));
        return rc != 0 ? rc : (size_ == s.size() ? 0 : size_ < s.size() ? -1 : 1);
    }

    template <class Allocator>
    int compare(const basic_byte_string<Allocator>& s) const 
    {
        const int rc = traits_type::compare(data_, s.data(), (std::min)(size_, s.size()));
        return rc != 0 ? rc : (size_ == s.size() ? 0 : size_ < s.size() ? -1 : 1);
    }

    template <class CharT>
    friend std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os, const byte_string_view& bstr)
    {
        std::basic_ostringstream<CharT> ss;
        ss.flags(std::ios::hex);
        ss.fill('0');

        bool first = true;
        for (auto b : bstr)
        {
            if (first)
            {
                first = false;
            }
            else 
            {
                ss << ' ';
            }
            ss << std::setw(2) << static_cast<int>(b);
        }
        os << ss.str();
        return os;
    }
};

// basic_byte_string
template <class Allocator = std::allocator<uint8_t>>
class basic_byte_string
{
    typedef typename std::allocator_traits<Allocator>:: template rebind_alloc<uint8_t> byte_allocator_type;
    std::vector<uint8_t,byte_allocator_type> data_;
public:
    typedef byte_traits traits_type;

    typedef typename std::vector<uint8_t,byte_allocator_type>::value_type value_type;
    typedef typename std::vector<uint8_t,byte_allocator_type>::size_type size_type;
    typedef typename std::vector<uint8_t,byte_allocator_type>::difference_type difference_type;
    typedef typename std::vector<uint8_t,byte_allocator_type>::reference reference;
    typedef typename std::vector<uint8_t,byte_allocator_type>::const_reference const_reference;
    typedef typename std::vector<uint8_t,byte_allocator_type>::pointer pointer;
    typedef typename std::vector<uint8_t,byte_allocator_type>::const_pointer const_pointer;
    typedef typename std::vector<uint8_t,byte_allocator_type>::iterator iterator;
    typedef typename std::vector<uint8_t,byte_allocator_type>::const_iterator const_iterator;

    basic_byte_string() = default;

    explicit basic_byte_string(const Allocator& alloc)
        : data_(alloc)
    {
    }

    basic_byte_string(std::initializer_list<uint8_t> init)
        : data_(std::move(init))
    {
    }

    basic_byte_string(std::initializer_list<uint8_t> init, const Allocator& alloc)
        : data_(std::move(init), alloc)
    {
    }

    explicit basic_byte_string(const byte_string_view& v)
        : data_(v.begin(),v.end())
    {
    }

    basic_byte_string(const basic_byte_string<Allocator>& v)
        : data_(v.data_)
    {
    }

    basic_byte_string(basic_byte_string<Allocator>&& v) noexcept
        : data_(std::move(v.data_))
    {
    }

    basic_byte_string(const byte_string_view& v, const Allocator& alloc)
        : data_(v.begin(),v.end(),alloc)
    {
    }

    basic_byte_string(const uint8_t* data, size_t length, const Allocator& alloc = Allocator())
        : data_(data, data+length,alloc)
    {
    }

    Allocator get_allocator() const
    {
        return data_.get_allocator();
    }

    basic_byte_string& operator=(const basic_byte_string& s) = default;

    basic_byte_string& operator=(basic_byte_string&& other) noexcept
    {
        data_.swap(other.data_);
        return *this;
    }

    void reserve(size_t new_cap)
    {
        data_.reserve(new_cap);
    }

    void push_back(uint8_t b)
    {
        data_.push_back(b);
    }

    void assign(const uint8_t* s, size_t count)
    {
        data_.clear();
        data_.insert(s, s+count);
    }

    void append(const uint8_t* s, size_t count)
    {
        data_.insert(s, s+count);
    }

    void clear()
    {
        data_.clear();
    }

    uint8_t operator[](size_type pos) const 
    { 
        return data_[pos]; 
    }

    // iterator support 
    iterator begin() noexcept
    {
        return data_.begin();
    }
    iterator end() noexcept
    {
        return data_.end();
    }

    const_iterator begin() const noexcept
    {
        return data_.begin();
    }
    const_iterator end() const noexcept
    {
        return data_.end();
    }

    uint8_t* data()
    {
        return data_.data();
    }

    const uint8_t* data() const
    {
        return data_.data();
    }

    size_t size() const
    {
        return data_.size();
    }

#if !defined(JSONCONS_NO_DEPRECATED)
    JSONCONS_DEPRECATED_MSG("Instead, use size()") 
    size_t length() const
    {
        return data_.size();
    }
#endif

    int compare(const byte_string_view& s) const 
    {
        const int rc = traits_type::compare(data(), s.data(), (std::min)(size(), s.size()));
        return rc != 0 ? rc : (size() == s.size() ? 0 : size() < s.size() ? -1 : 1);
    }

    int compare(const basic_byte_string& s) const 
    {
        const int rc = traits_type::compare(data(), s.data(), (std::min)(size(), s.size()));
        return rc != 0 ? rc : (size() == s.size() ? 0 : size() < s.size() ? -1 : 1);
    }

    template <class CharT>
    friend std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os, const basic_byte_string& o)
    {
        os << byte_string_view(o);
        return os;
    }
};

template <class Allocator>
constexpr byte_string_view::byte_string_view(const basic_byte_string<Allocator>& bytes) 
    : data_(bytes.data()), size_(bytes.size())
{
}

// ==
inline
bool operator==(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) == 0;
}
template<class Allocator>
bool operator==(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) == 0;
}
template<class Allocator>
bool operator==(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) == 0;
}
template<class Allocator>
bool operator==(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) == 0;
}

// !=

inline
bool operator!=(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) != 0;
}
template<class Allocator>
bool operator!=(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) != 0;
}
template<class Allocator>
bool operator!=(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) != 0;
}
template<class Allocator>
bool operator!=(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) != 0;
}

// <=

inline
bool operator<=(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) <= 0;
}
template<class Allocator>
bool operator<=(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) <= 0;
}
template<class Allocator>
bool operator<=(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) >= 0;
}
template<class Allocator>
bool operator<=(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) >= 0;
}

// <

inline
bool operator<(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) < 0;
}
template<class Allocator>
bool operator<(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) < 0;
}
template<class Allocator>
bool operator<(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) > 0;
}
template<class Allocator>
bool operator<(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) > 0;
}

// >=

inline
bool operator>=(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) >= 0;
}
template<class Allocator>
bool operator>=(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) >= 0;
}
template<class Allocator>
bool operator>=(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) <= 0;
}
template<class Allocator>
bool operator>=(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) <= 0;
}

// >

inline
bool operator>(const byte_string_view& lhs, const byte_string_view& rhs)
{
    return lhs.compare(rhs) > 0;
}
template<class Allocator>
bool operator>(const byte_string_view& lhs, const basic_byte_string<Allocator>& rhs)
{
    return lhs.compare(rhs) > 0;
}
template<class Allocator>
bool operator>(const basic_byte_string<Allocator>& lhs, const byte_string_view& rhs)
{
    return rhs.compare(lhs) < 0;
}
template<class Allocator>
bool operator>(const basic_byte_string<Allocator>& lhs, const basic_byte_string<Allocator>& rhs)
{
    return rhs.compare(lhs) < 0;
}

typedef basic_byte_string<std::allocator<uint8_t>> byte_string;


}

#endif
