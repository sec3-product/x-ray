// Copyright 2017 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_MSGPACK_MSGPACK_PARSER_HPP
#define JSONCONS_MSGPACK_MSGPACK_PARSER_HPP

#include <string>
#include <vector>
#include <memory>
#include <utility> // std::move
#include <jsoncons/json.hpp>
#include <jsoncons/source.hpp>
#include <jsoncons/json_content_handler.hpp>
#include <jsoncons/config/binary_config.hpp>
#include <jsoncons_ext/msgpack/msgpack_detail.hpp>
#include <jsoncons_ext/msgpack/msgpack_error.hpp>

namespace jsoncons { namespace msgpack {

enum class parse_mode {root,before_done,array,map_key,map_value};

struct parse_state 
{
    parse_mode mode; 
    size_t length;
    size_t index;

    parse_state(parse_mode mode, size_t length)
        : mode(mode), length(length), index(0)
    {
    }

    parse_state(const parse_state&) = default;
    parse_state(parse_state&&) = default;
};

template <class Src,class WorkAllocator=std::allocator<char>>
class basic_msgpack_parser : public ser_context
{
    typedef char char_type;
    typedef std::char_traits<char> char_traits_type;
    typedef WorkAllocator work_allocator_type;
    typedef typename std::allocator_traits<work_allocator_type>:: template rebind_alloc<char_type> char_allocator_type;
    typedef typename std::allocator_traits<work_allocator_type>:: template rebind_alloc<uint8_t> byte_allocator_type;
    typedef typename std::allocator_traits<work_allocator_type>:: template rebind_alloc<parse_state> parse_state_allocator_type;

    Src source_;
    bool more_;
    bool done_;
    std::basic_string<char,std::char_traits<char>,char_allocator_type> buffer_;
    std::vector<parse_state,parse_state_allocator_type> state_stack_;
public:
    template <class Source>
    basic_msgpack_parser(Source&& source,
                         const WorkAllocator alloc=WorkAllocator())
       : source_(std::forward<Source>(source)),
         more_(true), 
         done_(false),
         buffer_(alloc),
         state_stack_(alloc)
    {
        state_stack_.emplace_back(parse_mode::root,0);
    }

    void restart()
    {
        more_ = true;
    }

    void reset()
    {
        state_stack_.clear();
        state_stack_.emplace_back(parse_mode::root,0);
        more_ = true;
        done_ = false;
    }

    bool done() const
    {
        return done_;
    }

    bool stopped() const
    {
        return !more_;
    }

    size_t line() const override
    {
        return 0;
    }

    size_t column() const override
    {
        return source_.position();
    }

    void parse(json_content_handler& handler, std::error_code& ec)
    {
        while (!done_ && more_)
        {
            switch (state_stack_.back().mode)
            {
                case parse_mode::array:
                {
                    if (state_stack_.back().index < state_stack_.back().length)
                    {
                        ++state_stack_.back().index;
                        parse_item(handler, ec);
                        if (ec)
                        {
                            return;
                        }
                    }
                    else
                    {
                        produce_end_array(handler, ec);
                    }
                    break;
                }
                case parse_mode::map_key:
                {
                    if (state_stack_.back().index < state_stack_.back().length)
                    {
                        ++state_stack_.back().index;
                        parse_name(handler, ec);
                        if (ec)
                        {
                            return;
                        }
                        state_stack_.back().mode = parse_mode::map_value;
                    }
                    else
                    {
                        produce_end_map(handler, ec);
                    }
                    break;
                }
                case parse_mode::map_value:
                {
                    state_stack_.back().mode = parse_mode::map_key;
                    parse_item(handler, ec);
                    if (ec)
                    {
                        return;
                    }
                    break;
                }
                case parse_mode::root:
                {
                    state_stack_.back().mode = parse_mode::before_done;
                    parse_item(handler, ec);
                    if (ec)
                    {
                        return;
                    }
                    break;
                }
                case parse_mode::before_done:
                {
                    JSONCONS_ASSERT(state_stack_.size() == 1);
                    state_stack_.clear();
                    more_ = false;
                    done_ = true;
                    handler.flush();
                    break;
                }
            }
        }
    }
private:

    void parse_item(json_content_handler& handler, std::error_code& ec)
    {
        if (source_.is_error())
        {
            ec = msgpack_errc::source_error;
            return;
        }   

        uint8_t type{};
        source_.get(type);

        if (type <= 0xbf)
        {
            if (type <= 0x7f) 
            {
                // positive fixint
                more_ = handler.uint64_value(type, semantic_tag::none, *this, ec);
            }
            else if (type <= 0x8f) 
            {
                produce_begin_map(handler,type,ec); // fixmap
            }
            else if (type <= 0x9f) 
            {
                produce_begin_array(handler,type,ec); // fixarray
            }
            else 
            {
                // fixstr
                const size_t len = type & 0x1f;

                buffer_.clear();
                source_.read(std::back_inserter(buffer_), len);
                if (source_.eof())
                {
                    ec = msgpack_errc::unexpected_eof;
                    return;
                }

                auto result = unicons::validate(buffer_.begin(),buffer_.end());
                if (result.ec != unicons::conv_errc())
                {
                    ec = msgpack_errc::invalid_utf8_text_string;
                    return;
                }
                more_ = handler.string_value(basic_string_view<char>(buffer_.data(),buffer_.length()), semantic_tag::none, *this, ec);
            }
        }
        else if (type >= 0xe0) 
        {
            // negative fixint
            more_ = handler.int64_value(static_cast<int8_t>(type), semantic_tag::none, *this, ec);
        }
        else
        {
            switch (type)
            {
                case jsoncons::msgpack::detail::msgpack_format::nil_cd: 
                {
                    more_ = handler.null_value(semantic_tag::none, *this, ec);
                    break;
                }
                case jsoncons::msgpack::detail::msgpack_format::true_cd:
                {
                    more_ = handler.bool_value(true, semantic_tag::none, *this, ec);
                    break;
                }
                case jsoncons::msgpack::detail::msgpack_format::false_cd:
                {
                    more_ = handler.bool_value(false, semantic_tag::none, *this, ec);
                    break;
                }
                case jsoncons::msgpack::detail::msgpack_format::float32_cd: 
                {
                    uint8_t buf[sizeof(float)];
                    source_.read(buf, sizeof(float));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    float val = jsoncons::detail::big_to_native<float>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.double_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::float64_cd: 
                {
                    uint8_t buf[sizeof(double)];
                    source_.read(buf, sizeof(double));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    double val = jsoncons::detail::big_to_native<double>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.double_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::uint8_cd: 
                {
                    uint8_t val{};
                    source_.get(val);
                    more_ = handler.uint64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::uint16_cd: 
                {
                    uint8_t buf[sizeof(uint16_t)];
                    source_.read(buf, sizeof(uint16_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    uint16_t val = jsoncons::detail::big_to_native<uint16_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.uint64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::uint32_cd: 
                {
                    uint8_t buf[sizeof(uint32_t)];
                    source_.read(buf, sizeof(uint32_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    uint32_t val = jsoncons::detail::big_to_native<uint32_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.uint64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::uint64_cd: 
                {
                    uint8_t buf[sizeof(uint64_t)];
                    source_.read(buf, sizeof(uint64_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    uint64_t val = jsoncons::detail::big_to_native<uint64_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.uint64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::int8_cd: 
                {
                    uint8_t buf[sizeof(int8_t)];
                    source_.read(buf, sizeof(int8_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int8_t val = jsoncons::detail::big_to_native<int8_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.int64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::int16_cd: 
                {
                    uint8_t buf[sizeof(int16_t)];
                    source_.read(buf, sizeof(int16_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int16_t val = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.int64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::int32_cd: 
                {
                    uint8_t buf[sizeof(int32_t)];
                    source_.read(buf, sizeof(int32_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int32_t val = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.int64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::int64_cd: 
                {
                    uint8_t buf[sizeof(int64_t)];
                    source_.read(buf, sizeof(int64_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int64_t val = jsoncons::detail::big_to_native<int64_t>(buf,buf+sizeof(buf),&endp);
                    more_ = handler.int64_value(val, semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::str8_cd: 
                {
                    uint8_t buf[sizeof(int8_t)];
                    source_.read(buf, sizeof(int8_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int8_t len = jsoncons::detail::big_to_native<int8_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    auto result = unicons::validate(buffer_.begin(),buffer_.end());
                    if (result.ec != unicons::conv_errc())
                    {
                        ec = msgpack_errc::invalid_utf8_text_string;
                        return;
                    }
                    more_ = handler.string_value(basic_string_view<char>(buffer_.data(),buffer_.length()), semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::str16_cd: 
                {
                    uint8_t buf[sizeof(int16_t)];
                    source_.read(buf, sizeof(int16_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int16_t len = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    auto result = unicons::validate(buffer_.begin(),buffer_.end());
                    if (result.ec != unicons::conv_errc())
                    {
                        ec = msgpack_errc::invalid_utf8_text_string;
                        return;
                    }
                    more_ = handler.string_value(basic_string_view<char>(buffer_.data(),buffer_.length()), semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::str32_cd: 
                {
                    uint8_t buf[sizeof(int32_t)];
                    source_.read(buf, sizeof(int32_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int32_t len = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    auto result = unicons::validate(buffer_.begin(),buffer_.end());
                    if (result.ec != unicons::conv_errc())
                    {
                        ec = msgpack_errc::invalid_utf8_text_string;
                        return;
                    }
                    more_ = handler.string_value(basic_string_view<char>(buffer_.data(),buffer_.length()), semantic_tag::none, *this, ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::bin8_cd: 
                {
                    uint8_t buf[sizeof(int8_t)];
                    source_.read(buf, sizeof(int8_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int8_t len = jsoncons::detail::big_to_native<int8_t>(buf,buf+sizeof(buf),&endp);

                    std::vector<uint8_t> v;
                    v.reserve(len);
                    source_.read(std::back_inserter(v), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    more_ = handler.byte_string_value(byte_string_view(v.data(),v.size()), 
                                               semantic_tag::none, 
                                               *this);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::bin16_cd: 
                {
                    uint8_t buf[sizeof(int16_t)];
                    source_.read(buf, sizeof(int16_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int16_t len = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);

                    std::vector<uint8_t> v;
                    v.reserve(len);
                    source_.read(std::back_inserter(v), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    more_ = handler.byte_string_value(byte_string_view(v.data(),v.size()), 
                                               semantic_tag::none, 
                                               *this);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::bin32_cd: 
                {
                    uint8_t buf[sizeof(int32_t)];
                    source_.read(buf, sizeof(int32_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int32_t len = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);

                    std::vector<uint8_t> v;
                    v.reserve(len);
                    source_.read(std::back_inserter(v), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    more_ = handler.byte_string_value(byte_string_view(v.data(),v.size()), 
                                               semantic_tag::none, 
                                               *this);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::array16_cd: 
                case jsoncons::msgpack::detail::msgpack_format::array32_cd: 
                {
                    produce_begin_array(handler,type,ec);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::map16_cd : 
                case jsoncons::msgpack::detail::msgpack_format::map32_cd : 
                {
                    produce_begin_map(handler, type, ec);
                    break;
                }

                default:
                {
                    //error
                }
            }
        }
    }

    void parse_name(json_content_handler& handler, std::error_code& ec)
    {
        uint8_t type{};
        source_.get(type);
        if (source_.eof())
        {
            ec = msgpack_errc::unexpected_eof;
            return;
        }

        //const uint8_t* pos = input_ptr_++;
        if (type >= 0xa0 && type <= 0xbf)
        {
                // fixstr
            const size_t len = type & 0x1f;

            buffer_.clear();
            source_.read(std::back_inserter(buffer_), len);
            if (source_.eof())
            {
                ec = msgpack_errc::unexpected_eof;
                return;
            }
            auto result = unicons::validate(buffer_.begin(),buffer_.end());
            if (result.ec != unicons::conv_errc())
            {
                ec = msgpack_errc::invalid_utf8_text_string;
                return;
            }
            more_ = handler.name(basic_string_view<char>(buffer_.data(),buffer_.length()), *this);
        }
        else
        {
            switch (type)
            {
                case jsoncons::msgpack::detail::msgpack_format::str8_cd: 
                {
                    uint8_t buf[sizeof(int8_t)];
                    source_.read(buf, sizeof(int8_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int8_t len = jsoncons::detail::big_to_native<int8_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    auto result = unicons::validate(buffer_.begin(),buffer_.end());
                    if (result.ec != unicons::conv_errc())
                    {
                        ec = msgpack_errc::invalid_utf8_text_string;
                        return;
                    }
                    more_ = handler.name(basic_string_view<char>(buffer_.data(),buffer_.length()), *this);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::str16_cd: 
                {
                    uint8_t buf[sizeof(int16_t)];
                    source_.read(buf, sizeof(int16_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int16_t len = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    more_ = handler.name(basic_string_view<char>(buffer_.data(),buffer_.length()), *this);
                    break;
                }

                case jsoncons::msgpack::detail::msgpack_format::str32_cd: 
                {
                    uint8_t buf[sizeof(int32_t)];
                    source_.read(buf, sizeof(int32_t));
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }
                    const uint8_t* endp;
                    int32_t len = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);

                    buffer_.clear();
                    source_.read(std::back_inserter(buffer_), len);
                    if (source_.eof())
                    {
                        ec = msgpack_errc::unexpected_eof;
                        return;
                    }

                    more_ = handler.name(basic_string_view<char>(buffer_.data(),buffer_.length()), *this);
                    break;
                }
            }
        }
    }

    void produce_begin_array(json_content_handler& handler, uint8_t type, std::error_code& ec)
    {
        size_t len = 0;
        switch (type)
        {
            case jsoncons::msgpack::detail::msgpack_format::array16_cd: 
            {
                uint8_t buf[sizeof(int16_t)];
                source_.read(buf, sizeof(int16_t));
                if (source_.eof())
                {
                    ec = msgpack_errc::unexpected_eof;
                    return;
                }
                const uint8_t* endp;
                len = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);
                break;
            }
            case jsoncons::msgpack::detail::msgpack_format::array32_cd: 
            {
                uint8_t buf[sizeof(int32_t)];
                source_.read(buf, sizeof(int32_t));
                if (source_.eof())
                {
                    ec = msgpack_errc::unexpected_eof;
                    return;
                }
                const uint8_t* endp;
                len = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);
                break;
            }
            default:
                JSONCONS_ASSERT(type > 0x8f && type <= 0x9f) // fixarray
                len = type & 0x0f;
                break;
        }
        state_stack_.emplace_back(parse_mode::array,len);
        more_ = handler.begin_array(len, semantic_tag::none, *this, ec);
    }

    void produce_end_array(json_content_handler& handler, std::error_code&)
    {
        more_ = handler.end_array(*this);
        state_stack_.pop_back();
    }

    void produce_begin_map(json_content_handler& handler, uint8_t type, std::error_code& ec)
    {
        size_t len = 0;
        switch (type)
        {
            case jsoncons::msgpack::detail::msgpack_format::map16_cd:
            {
                uint8_t buf[sizeof(int16_t)];
                source_.read(buf, sizeof(int16_t));
                if (source_.eof())
                {
                    ec = msgpack_errc::unexpected_eof;
                    return;
                }
                const uint8_t* endp;
                len = jsoncons::detail::big_to_native<int16_t>(buf,buf+sizeof(buf),&endp);
                break; 
            }
            case jsoncons::msgpack::detail::msgpack_format::map32_cd : 
            {
                uint8_t buf[sizeof(int32_t)];
                source_.read(buf, sizeof(int32_t));
                if (source_.eof())
                {
                    ec = msgpack_errc::unexpected_eof;
                    return;
                }
                const uint8_t* endp;
                len = jsoncons::detail::big_to_native<int32_t>(buf,buf+sizeof(buf),&endp);
                break;
            }
            default:
                JSONCONS_ASSERT (type > 0x7f && type <= 0x8f) // fixmap
                len = type & 0x0f;
                break;
        }
        state_stack_.emplace_back(parse_mode::map_key,len);
        more_ = handler.begin_object(len, semantic_tag::none, *this, ec);
    }

    void produce_end_map(json_content_handler& handler, std::error_code&)
    {
        more_ = handler.end_object(*this);
        state_stack_.pop_back();
    }
};

}}

#endif
