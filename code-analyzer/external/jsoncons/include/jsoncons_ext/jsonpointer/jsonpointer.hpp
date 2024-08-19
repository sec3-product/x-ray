// Copyright 2017 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_JSONPOINTER_JSONPOINTER_HPP
#define JSONCONS_JSONPOINTER_JSONPOINTER_HPP

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <iostream>
#include <iterator>
#include <utility> // std::move
#include <system_error> // system_error
#include <type_traits> // std::enable_if, std::true_type
#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpointer/jsonpointer_error.hpp>
#include <jsoncons/detail/print_number.hpp>

namespace jsoncons { namespace jsonpointer {

// find_by_reference

template <class J, class Enable=void>
struct is_accessible_by_reference : std::false_type {};

template <class J>
struct is_accessible_by_reference<J, 
                            typename std::enable_if<std::is_reference<decltype(std::declval<J>().at(typename J::string_view_type()))>::value
                                                    && std::is_reference<decltype(std::declval<J>().at(0))>::value>::type> 
: std::true_type {};

namespace detail {

enum class pointer_state 
{
    start,
    escaped,
    delim
};

} // detail

// json_ptr_iterator
template <class InputIt>
class json_ptr_iterator
{
    typedef typename std::iterator_traits<InputIt>::value_type char_type;
    typedef std::basic_string<char_type> string_type;
    typedef InputIt base_iterator;

    base_iterator path_ptr_;
    base_iterator end_input_;
    base_iterator p_;
    base_iterator q_;
    jsonpointer::detail::pointer_state state_;
    size_t line_;
    size_t column_;
    std::basic_string<char_type> buffer_;
public:
    typedef string_type value_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type* pointer;
    typedef const value_type& reference;
    typedef std::input_iterator_tag iterator_category;

    json_ptr_iterator(base_iterator first, base_iterator last)
        : json_ptr_iterator(first, last, first)
    {
        std::error_code ec;
        increment(ec);
    }

    json_ptr_iterator(base_iterator first, base_iterator last, base_iterator current)
        : path_ptr_(first), end_input_(last), p_(current), q_(current), state_(jsonpointer::detail::pointer_state::start)
    {
    }

    json_ptr_iterator(const json_ptr_iterator&) = default;

    json_ptr_iterator(json_ptr_iterator&&) = default;

    json_ptr_iterator& operator=(const json_ptr_iterator&) = default;

    json_ptr_iterator& operator=(json_ptr_iterator&&) = default;

    json_ptr_iterator& operator++()
    {
        std::error_code ec;
        increment(ec);
        if (ec)
        {
            JSONCONS_THROW(jsonpointer_error(ec));
        }
        return *this;
    }

    json_ptr_iterator& increment(std::error_code& ec)
    {
        q_ = p_;
        buffer_.clear();

        bool done = false;
        while (p_ != end_input_ && !done)
        {
            switch (state_)
            {
                case jsonpointer::detail::pointer_state::start: 
                    switch (*p_)
                    {
                        case '/':
                            state_ = jsonpointer::detail::pointer_state::delim;
                            break;
                        default:
                            ec = jsonpointer_errc::expected_slash;
                            done = true;
                            break;
                    };
                    break;
                case jsonpointer::detail::pointer_state::delim: 
                    switch (*p_)
                    {
                        case '/':
                            state_ = jsonpointer::detail::pointer_state::delim;
                            done = true;
                            break;
                        case '~':
                            state_ = jsonpointer::detail::pointer_state::escaped;
                            break;
                        default:
                            buffer_.push_back(*p_);
                            break;
                    };
                    break;
                case jsonpointer::detail::pointer_state::escaped: 
                    switch (*p_)
                    {
                        case '0':
                            buffer_.push_back('~');
                            state_ = jsonpointer::detail::pointer_state::delim;
                            break;
                        case '1':
                            buffer_.push_back('/');
                            state_ = jsonpointer::detail::pointer_state::delim;
                            break;
                        default:
                            ec = jsonpointer_errc::expected_0_or_1;
                            done = true;
                            break;
                    };
                    break;
                default:
                    JSONCONS_UNREACHABLE();
                    break;
            }
            ++p_;
            ++column_;
        }
        return *this;
    }

    json_ptr_iterator operator++(int) // postfix increment
    {
        json_ptr_iterator temp(*this);
        ++(*this);
        return temp;
    }

    reference operator*() const
    {
        return buffer_;
    }

    friend bool operator==(const json_ptr_iterator& it1, const json_ptr_iterator& it2)
    {
        return it1.q_ == it2.q_;
    }
    friend bool operator!=(const json_ptr_iterator& it1, const json_ptr_iterator& it2)
    {
        return !(it1 == it2);
    }

private:
};

template <class CharT>
std::basic_string<CharT> escape_string(const std::basic_string<CharT>& s)
{
    std::basic_string<CharT> result;
    for (auto c : s)
    {
        switch (c)
        {
            case '~':
                result.push_back('~');
                result.push_back('0');
                break;
            case '/':
                result.push_back('~');
                result.push_back('1');
                break;
            default:
                result.push_back(c);
                break;
        }
    }
    return result;
}

// basic_json_ptr

template <class CharT>
class basic_json_ptr
{
public:
    std::basic_string<CharT> path_;
public:
    // Member types
    typedef CharT char_type;
    typedef std::basic_string<char_type> string_type;
    typedef basic_string_view<char_type> string_view_type;
    typedef json_ptr_iterator<typename string_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    // Constructors
    basic_json_ptr()
    {
    }
    explicit basic_json_ptr(const string_type& s)
        : path_(s)
    {
    }
    explicit basic_json_ptr(string_type&& s)
        : path_(std::move(s))
    {
    }
    explicit basic_json_ptr(const CharT* s)
        : path_(s)
    {
    }

    basic_json_ptr(const basic_json_ptr&) = default;

    basic_json_ptr(basic_json_ptr&&) = default;

    // operator=
    basic_json_ptr& operator=(const basic_json_ptr&) = default;

    basic_json_ptr& operator=(basic_json_ptr&&) = default;

    // Modifiers

    void clear()
    {
        path_.clear();
    }

    basic_json_ptr& operator/=(const string_type& s)
    {
        path_.push_back('/');
        path_.append(escape_string(s));

        return *this;
    }

    basic_json_ptr& operator+=(const basic_json_ptr& p)
    {
        path_.append(p.path_);
        return *this;
    }

    // Accessors
    bool empty() const
    {
      return path_.empty();
    }

    const string_type& string() const
    {
        return path_;
    }

    operator string_view_type() const
    {
        return path_;
    }

    // Iterators
    iterator begin() const
    {
        return iterator(path_.begin(),path_.end());
    }
    iterator end() const
    {
        return iterator(path_.begin(), path_.end(), path_.end());
    }

    // Non-member functions
    friend basic_json_ptr<CharT> operator/(const basic_json_ptr<CharT>& lhs, const string_type& rhs)
    {
        basic_json_ptr<CharT> p(lhs);
        p /= rhs;
        return p;
    }

    friend basic_json_ptr<CharT> operator+( const basic_json_ptr<CharT>& lhs, const basic_json_ptr<CharT>& rhs )
    {
        basic_json_ptr<CharT> p(lhs);
        p += rhs;
        return p;
    }

    friend bool operator==( const basic_json_ptr& lhs, const basic_json_ptr& rhs )
    {
        return lhs.path_ == rhs.path_;
    }

    friend bool operator!=( const basic_json_ptr& lhs, const basic_json_ptr& rhs )
    {
        return lhs.path_ != rhs.path_;
    }

    friend std::basic_ostream<CharT>&
    operator<<( std::basic_ostream<CharT>& os, const basic_json_ptr<CharT>& p )
    {
        os << p.path_;
        return os;
    }
};

typedef basic_json_ptr<char> json_ptr;
typedef basic_json_ptr<wchar_t> wjson_ptr;

#if !defined(JSONCONS_NO_DEPRECATED)
template<class CharT>
using basic_address = basic_json_ptr<CharT>;
JSONCONS_DEPRECATED_MSG("Instead, use json_ptr") typedef json_ptr address;
#endif

namespace detail {

template <class J,class JReference,class Enable = void>
class handle_type
{
public:
    using value_type = typename J::value_type;
    using type = value_type;

    handle_type(const value_type& val) noexcept
        : val_(val)
    {
    }

    handle_type(value_type&& val) noexcept
        : val_(std::move(val))
    {
    }

    handle_type(const handle_type& w) noexcept
        : val_(w.val_)
    {
    }

    handle_type& operator=(const handle_type&) noexcept = default;

    type get() const noexcept
    {
        return val_;
    }
private:
    value_type val_;
};

template <class J,class JReference>
class handle_type<J,JReference,
                  typename std::enable_if<is_accessible_by_reference<J>::value>::type>
{
public:
    using reference = JReference;
    using type = reference;
    using pointer = typename std::conditional<std::is_const<typename std::remove_reference<JReference>::type>::value,typename J::const_pointer,typename J::pointer>::type;

    handle_type(reference ref) noexcept
        : ptr_(std::addressof(ref))
    {
    }

    handle_type(const handle_type&) noexcept = default;

    handle_type& operator=(const handle_type&) noexcept = default;

    type get() const noexcept
    {
        return *ptr_;
    }
private:
    pointer ptr_;
};

template<class J,class JReference>
class jsonpointer_evaluator : public ser_context
{
    typedef typename handle_type<J,JReference>::type type;
    typedef typename J::char_type char_type;
    typedef typename std::basic_string<char_type> string_type;
    typedef typename J::string_view_type string_view_type;
    using reference = JReference;
    using pointer = typename std::conditional<std::is_const<typename std::remove_reference<JReference>::type>::value,typename J::const_pointer,typename J::pointer>::type;

    size_t line_;
    size_t column_;
    string_type buffer_;
    std::vector<handle_type<J,JReference>> current_;
public:
    type get_result() 
    {
        return current_.back().get();
    }

    void get(reference root, const string_view_type& path, std::error_code& ec)
    {
        evaluate(root, path, ec);
        if (ec)
        {
            return;
        }
        if (path.empty())
        {
            return;
        }
        resolve(current_, buffer_, ec);
    }

    string_type normalized_path(reference root, const string_view_type& path)
    {
        std::error_code ec;
        evaluate(root, path, ec);
        if (ec)
        {
            return string_type(path);
        }
        if (current_.back().get().is_array() && buffer_.size() == 1 && buffer_[0] == '-')
        {
            string_type p = string_type(path.substr(0,path.length()-1));
            std::string s = std::to_string(current_.back().get().size());
            for (auto c : s)
            {
                p.push_back(c);
            }
            return p;
        }
        else
        {
            return string_type(path);
        }
    }

    void insert_or_assign(reference root, const string_view_type& path, const J& value, std::error_code& ec)
    {
        evaluate(root, path, ec);
        if (ec)
        {
            return;
        }
        if (current_.back().get().is_array())
        {
            if (buffer_.size() == 1 && buffer_[0] == '-')
            {
                current_.back().get().push_back(value);
            }
            else
            {
                if (!jsoncons::detail::is_integer(buffer_.data(), buffer_.length()))
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                auto result = jsoncons::detail::to_integer<size_t>(buffer_.data(), buffer_.length());
                if (result.ec != jsoncons::detail::to_integer_errc())
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                size_t index = result.value;
                if (index > current_.back().get().size())
                {
                    ec = jsonpointer_errc::index_exceeds_array_size;
                    return;
                }
                if (index == current_.back().get().size())
                {
                    current_.back().get().push_back(value);
                }
                else
                {
                    current_.back().get().insert(current_.back().get().array_range().begin()+index,value);
                }
            }
        }
        else if (current_.back().get().is_object())
        {
            current_.back().get().insert_or_assign(buffer_,value);
        }
        else
        {
            ec = jsonpointer_errc::expected_object_or_array;
            return;
        }
    }

    void insert(reference root, const string_view_type& path, const J& value, std::error_code& ec)
    {
        evaluate(root, path, ec);
        if (ec)
        {
            return;
        }
        if (current_.back().get().is_array())
        {
            if (buffer_.size() == 1 && buffer_[0] == '-')
            {
                current_.back().get().push_back(value);
            }
            else
            {
                if (!jsoncons::detail::is_integer(buffer_.data(), buffer_.length()))
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                auto result = jsoncons::detail::to_integer<size_t>(buffer_.data(), buffer_.length());
                if (result.ec != jsoncons::detail::to_integer_errc())
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                size_t index = result.value;
                if (index > current_.back().get().size())
                {
                    ec = jsonpointer_errc::index_exceeds_array_size;
                    return;
                }
                if (index == current_.back().get().size())
                {
                    current_.back().get().push_back(value);
                }
                else
                {
                    current_.back().get().insert(current_.back().get().array_range().begin()+index,value);
                }
            }
        }
        else if (current_.back().get().is_object())
        {
            if (current_.back().get().contains(buffer_))
            {
                ec = jsonpointer_errc::key_already_exists;
                return;
            }
            else
            {
                current_.back().get().insert_or_assign(buffer_,value);
            }
        }
        else
        {
            ec = jsonpointer_errc::expected_object_or_array;
            return;
        }
    }

    void remove(reference root, const string_view_type& path, std::error_code& ec)
    {
        evaluate(root, path, ec);
        if (ec)
        {
            return;
        }
        if (current_.back().get().is_array())
        {
            if (buffer_.size() == 1 && buffer_[0] == '-')
            {
                ec = jsonpointer_errc::index_exceeds_array_size;
                return;
            }
            else
            {
                if (!jsoncons::detail::is_integer(buffer_.data(), buffer_.length()))
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                auto result = jsoncons::detail::to_integer<size_t>(buffer_.data(), buffer_.length());
                if (result.ec != jsoncons::detail::to_integer_errc())
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                size_t index = result.value;
                if (index >= current_.back().get().size())
                {
                    ec = jsonpointer_errc::index_exceeds_array_size;
                    return;
                }
                current_.back().get().erase(current_.back().get().array_range().begin()+index);
            }
        }
        else if (current_.back().get().is_object())
        {
            if (!current_.back().get().contains(buffer_))
            {
                ec = jsonpointer_errc::name_not_found;
                return;
            }
            else
            {
                current_.back().get().erase(buffer_);
            }
        }
        else
        {
            ec = jsonpointer_errc::expected_object_or_array;
            return;
        }
    }

    void replace(reference root, const string_view_type& path, const J& value, std::error_code& ec)
    {
        evaluate(root, path, ec);
        if (ec)
        {
            return;
        }
        if (current_.back().get().is_array())
        {
            if (buffer_.size() == 1 && buffer_[0] == '-')
            {
                ec = jsonpointer_errc::index_exceeds_array_size;
                return;
            }
            else
            {
                if (!jsoncons::detail::is_integer(buffer_.data(), buffer_.length()))
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                auto result = jsoncons::detail::to_integer<size_t>(buffer_.data(), buffer_.length());
                if (result.ec != jsoncons::detail::to_integer_errc())
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                size_t index = result.value;
                if (index >= current_.back().get().size())
                {
                    ec = jsonpointer_errc::index_exceeds_array_size;
                    return;
                }
                (current_.back().get())[index] = value;
            }
        }
        else if (current_.back().get().is_object())
        {
            if (!current_.back().get().contains(buffer_))
            {
                ec = jsonpointer_errc::key_already_exists;
                return;
            }
            else
            {
                current_.back().get().insert_or_assign(buffer_,value);
            }
        }
        else
        {
            ec = jsonpointer_errc::expected_object_or_array;
            return;
        }
    }

    void evaluate(reference root, const string_view_type& path, std::error_code& ec)
    {
        current_.push_back(root);

        json_ptr_iterator<typename string_view_type::iterator> it(path.begin(), path.end());
        json_ptr_iterator<typename string_view_type::iterator> end(path.begin(), path.end(), path.end());
        while (it != end)
        {
            buffer_ = *it;
            it.increment(ec);
            if (ec)
                return;
            if (it == end)
            {
                return;
            }
            resolve(current_, buffer_, ec);
            if (ec)
                return;
        }
    }

    static void resolve(std::vector<handle_type<J,JReference>>& current,
                        const string_view_type& buffer,
                        std::error_code& ec)
    {
        if (current.back().get().is_array())
        {
            if (buffer.size() == 1 && buffer[0] == '-')
            {
                ec = jsonpointer_errc::index_exceeds_array_size;
                return;
            }
            else
            {
                if (!jsoncons::detail::is_integer(buffer.data(), buffer.length()))
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                auto result = jsoncons::detail::to_integer<size_t>(buffer.data(), buffer.length());
                if (result.ec != jsoncons::detail::to_integer_errc())
                {
                    ec = jsonpointer_errc::invalid_index;
                    return;
                }
                size_t index = result.value;
                if (index >= current.back().get().size())
                {
                    ec = jsonpointer_errc::index_exceeds_array_size;
                    return;
                }
                current.push_back(current.back().get().at(index));
            }
        }
        else if (current.back().get().is_object())
        {
            if (!current.back().get().contains(buffer))
            {
                ec = jsonpointer_errc::name_not_found;
                return;
            }
            current.push_back(current.back().get().at(buffer));
        }
        else
        {
            ec = jsonpointer_errc::expected_object_or_array;
            return;
        }
    }

    // ser_context

    size_t line() const override
    {
        return line_;
    }

    size_t column() const override
    {
        return column_;
    }
};

}

template<class J>
std::basic_string<typename J::char_type> normalized_path(const J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;
    return evaluator.normalized_path(root,path);
}

template<class J>
typename std::enable_if<is_accessible_by_reference<J>::value,J&>::type
get(J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;
    std::error_code ec;
    evaluator.get(root, path, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
    return evaluator.get_result();
}

template<class J>
typename std::enable_if<is_accessible_by_reference<J>::value,const J&>::type
get(const J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;

    std::error_code ec;
    evaluator.get(root, path, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
    return evaluator.get_result();
}

template<class J>
typename std::enable_if<!is_accessible_by_reference<J>::value,J>::type
get(const J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;

    std::error_code ec;
    evaluator.get(root, path, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
    return evaluator.get_result();
}

template<class J>
typename std::enable_if<is_accessible_by_reference<J>::value,J&>::type
get(J& root, const typename J::string_view_type& path, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;
    evaluator.get(root, path, ec);
    return evaluator.get_result();
}

template<class J>
typename std::enable_if<is_accessible_by_reference<J>::value,const J&>::type
get(const J& root, const typename J::string_view_type& path, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;
    evaluator.get(root, path, ec);
    return evaluator.get_result();
}

template<class J>
typename std::enable_if<!is_accessible_by_reference<J>::value,J>::type
get(const J& root, const typename J::string_view_type& path, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;
    evaluator.get(root, path, ec);
    return evaluator.get_result();
}

template<class J>
bool contains(const J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,const J&> evaluator;
    std::error_code ec;
    evaluator.get(root, path, ec);
    return !ec ? true : false;
}

template<class J>
void insert_or_assign(J& root, const typename J::string_view_type& path, const J& value)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    std::error_code ec;
    evaluator.insert_or_assign(root, path, value, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
}

template<class J>
void insert_or_assign(J& root, const typename J::string_view_type& path, const J& value, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    evaluator.insert_or_assign(root, path, value, ec);
}

template<class J>
void insert(J& root, const typename J::string_view_type& path, const J& value)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    std::error_code ec;
    evaluator.insert(root, path, value, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
}

template<class J>
void insert(J& root, const typename J::string_view_type& path, const J& value, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    evaluator.insert(root, path, value, ec);
}

template<class J>
void remove(J& root, const typename J::string_view_type& path)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    std::error_code ec;
    evaluator.remove(root, path, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
}

template<class J>
void remove(J& root, const typename J::string_view_type& path, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    evaluator.remove(root, path, ec);
}

template<class J>
void replace(J& root, const typename J::string_view_type& path, const J& value)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    std::error_code ec;
    evaluator.replace(root, path, value, ec);
    if (ec)
    {
        JSONCONS_THROW(jsonpointer_error(ec));
    }
}

template<class J>
void replace(J& root, const typename J::string_view_type& path, const J& value, std::error_code& ec)
{
    jsoncons::jsonpointer::detail::jsonpointer_evaluator<J,J&> evaluator;

    evaluator.replace(root, path, value, ec);
}

template <class String>
void escape(const String& s, std::basic_ostringstream<typename String::value_type>& os)
{
    for (auto c : s)
    {
        if (c == '~')
        {
            os.put('~');
            os.put('0');
        }
        else if (c == '/')
        {
            os.put('~');
            os.put('1');
        }
        else
        {
            os.put(c);
        }
    }
}

}}

#endif
