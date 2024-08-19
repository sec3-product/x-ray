/// Copyright 2013 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_CSV_CSV_HPP
#define JSONCONS_CSV_CSV_HPP

#include <jsoncons_ext/csv/csv_options.hpp>
#include <jsoncons_ext/csv/csv_reader.hpp>
#include <jsoncons_ext/csv/csv_encoder.hpp>
#include <jsoncons_ext/csv/csv_cursor.hpp>

namespace jsoncons { namespace csv {

// decode_csv

template <class T,class CharT>
typename std::enable_if<is_basic_json_class<T>::value,T>::type 
decode_csv(const std::basic_string<CharT>& s, const basic_csv_decode_options<CharT>& options = basic_csv_decode_options<CharT>())
{
    typedef CharT char_type;

    json_decoder<T> decoder;

    basic_csv_reader<char_type,jsoncons::string_source<char_type>> reader(s,decoder,options);
    reader.read();
    return decoder.get_result();
}

template <class T,class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,T>::type 
decode_csv(const std::basic_string<CharT>& s, const basic_csv_decode_options<CharT>& options = basic_csv_decode_options<CharT>())
{
    basic_csv_cursor<CharT> cursor(s, options);
    std::error_code ec;
    T val = ser_traits<T>::decode(cursor, basic_json<CharT>(), ec);
    if (ec)
    {
        JSONCONS_THROW(ser_error(ec, cursor.context().line(), cursor.context().column()));
    }
    return val;
}

template <class T,class CharT>
typename std::enable_if<is_basic_json_class<T>::value,T>::type 
decode_csv(std::basic_istream<CharT>& is, const basic_csv_decode_options<CharT>& options = basic_csv_decode_options<CharT>())
{
    typedef CharT char_type;

    json_decoder<T> decoder;

    basic_csv_reader<char_type,jsoncons::stream_source<char_type>> reader(is,decoder,options);
    reader.read();
    return decoder.get_result();
}

template <class T,class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,T>::type 
decode_csv(std::basic_istream<CharT>& is, const basic_csv_decode_options<CharT>& options = basic_csv_decode_options<CharT>())
{
    basic_csv_cursor<CharT> cursor(is, options);
    std::error_code ec;
    T val = ser_traits<T>::decode(cursor, basic_json<CharT>(), ec);
    if (ec)
    {
        JSONCONS_THROW(ser_error(ec, cursor.context().line(), cursor.context().column()));
    }
    return val;
}
// encode_csv

template <class T,class CharT>
typename std::enable_if<is_basic_json_class<T>::value,void>::type 
encode_csv(const T& j, std::basic_string<CharT>& s, const basic_csv_encode_options<CharT>& options = basic_csv_encode_options<CharT>())
{
    typedef CharT char_type;
    basic_csv_encoder<char_type,jsoncons::string_result<std::basic_string<char_type>>> encoder(s,options);
    j.dump(encoder);
}

template <class T,class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,void>::type 
encode_csv(const T& val, std::basic_string<CharT>& s, const basic_csv_encode_options<CharT>& options = basic_csv_encode_options<CharT>())
{
    typedef CharT char_type;
    basic_csv_encoder<char_type,jsoncons::string_result<std::basic_string<char_type>>> encoder(s,options);
    std::error_code ec;
    ser_traits<T>::encode(val, encoder, json(), ec);
    if (ec)
    {
        JSONCONS_THROW(ser_error(ec));
    }
}

template <class T, class CharT>
typename std::enable_if<is_basic_json_class<T>::value,void>::type 
encode_csv(const T& j, std::basic_ostream<CharT>& os, const basic_csv_encode_options<CharT>& options = basic_csv_encode_options<CharT>())
{
    typedef CharT char_type;
    basic_csv_encoder<char_type,jsoncons::stream_result<char_type>> encoder(os,options);
    j.dump(encoder);
}

template <class T, class CharT>
typename std::enable_if<!is_basic_json_class<T>::value,void>::type 
encode_csv(const T& val, std::basic_ostream<CharT>& os, const basic_csv_encode_options<CharT>& options = basic_csv_encode_options<CharT>())
{
    typedef CharT char_type;
    basic_csv_encoder<char_type,jsoncons::stream_result<char_type>> encoder(os,options);
    std::error_code ec;
    ser_traits<T>::encode(val, encoder, json(), ec);
    if (ec)
    {
        JSONCONS_THROW(ser_error(ec));
    }
}

}} // namespace csv namespace jsoncons 

#endif
