// Copyright 2017 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_MSGPACK_MSGPACK_READER_HPP
#define JSONCONS_MSGPACK_MSGPACK_READER_HPP

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
#include <jsoncons_ext/msgpack/msgpack_parser.hpp>

namespace jsoncons { namespace msgpack {

template <class Src,class WorkAllocator=std::allocator<char>>
class basic_msgpack_reader : public ser_context
{
    basic_msgpack_parser<Src,WorkAllocator> parser_;
    json_content_handler& handler_;
public:
    template <class Source>
    basic_msgpack_reader(Source&& source, 
                         json_content_handler& handler,
                         const WorkAllocator alloc=WorkAllocator())
       : parser_(std::forward<Source>(source), alloc),
         handler_(handler)
    {
    }

    void read()
    {
        std::error_code ec;
        read(ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec,line(),column()));
        }
    }

    void read(std::error_code& ec)
    {
        parser_.reset();
        parser_.parse(handler_, ec);
        if (ec)
        {
            return;
        }
    }

    size_t line() const override
    {
        return parser_.line();
    }

    size_t column() const override
    {
        return parser_.column();
    }
};

typedef basic_msgpack_reader<jsoncons::binary_stream_source> msgpack_stream_reader;

typedef basic_msgpack_reader<jsoncons::bytes_source> msgpack_bytes_reader;

#if !defined(JSONCONS_NO_DEPRECATED)
JSONCONS_DEPRECATED_MSG("Instead, use msgpack_stream_reader") typedef msgpack_stream_reader msgpack_reader;
JSONCONS_DEPRECATED_MSG("Instead, use msgpack_bytes_reader") typedef msgpack_bytes_reader msgpack_buffer_reader;
#endif

}}

#endif
