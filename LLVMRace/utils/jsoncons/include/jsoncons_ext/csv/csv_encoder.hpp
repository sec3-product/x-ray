// Copyright 2013 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_CSV_CSV_ENCODER_HPP
#define JSONCONS_CSV_CSV_ENCODER_HPP

#include <array> // std::array
#include <string>
#include <sstream>
#include <vector>
#include <ostream>
#include <utility> // std::move
#include <unordered_map> // std::unordered_map
#include <memory> // std::allocator
#include <limits> // std::numeric_limits
#include <jsoncons/json_exception.hpp>
#include <jsoncons/json_content_handler.hpp>
#include <jsoncons/detail/print_number.hpp>
#include <jsoncons_ext/csv/csv_options.hpp>
#include <jsoncons/result.hpp>

namespace jsoncons { namespace csv {

template<class CharT,class Result=jsoncons::stream_result<CharT>,class Allocator=std::allocator<CharT>>
class basic_csv_encoder final : public basic_json_content_handler<CharT>
{
public:
    typedef CharT char_type;
    using typename basic_json_content_handler<CharT>::string_view_type;
    typedef Result result_type;

    typedef Allocator allocator_type;
    typedef typename std::allocator_traits<allocator_type>:: template rebind_alloc<CharT> char_allocator_type;
    typedef std::basic_string<CharT, std::char_traits<CharT>, char_allocator_type> string_type;
    typedef typename std::allocator_traits<allocator_type>:: template rebind_alloc<string_type> string_allocator_type;
    typedef typename std::allocator_traits<allocator_type>:: template rebind_alloc<std::pair<const string_type,string_type>> string_string_allocator_type;

private:
    static const std::array<CharT, 4>& null_k()
    {
        static constexpr std::array<CharT,4> k{'n','u','l','l'};
        return k;
    }
    static const std::array<CharT, 4>& true_k()
    {
        static constexpr std::array<CharT,4> k{'t','r','u','e'};
        return k;
    }
    static const std::array<CharT, 5>& false_k()
    {
        static constexpr std::array<CharT,5> k{'f','a','l','s','e'};
        return k;
    }

    enum class stack_item_kind
    {
        row_mapping,
        column_mapping,
        object,
        row,
        column,
        object_multi_valued_field,
        row_multi_valued_field,
        column_multi_valued_field
    };

    struct stack_item
    {
        stack_item_kind item_kind_;
        size_t count_;

        stack_item(stack_item_kind item_kind)
           : item_kind_(item_kind), count_(0)
        {
        }

        bool is_object() const
        {
            return item_kind_ == stack_item_kind::object;
        }

        stack_item_kind item_kind() const
        {
            return item_kind_;
        }
    };

    Result result_;
    const basic_csv_encode_options<CharT> options_;
    std::vector<stack_item> stack_;
    jsoncons::detail::print_double fp_;
    std::vector<string_type,string_allocator_type> strings_buffer_;

    std::unordered_map<string_type,string_type, std::hash<string_type>,std::equal_to<string_type>,string_string_allocator_type> buffered_line_;
    string_type name_;
    size_t column_index_;
    std::vector<size_t> row_counts_;

    // Noncopyable and nonmoveable
    basic_csv_encoder(const basic_csv_encoder&) = delete;
    basic_csv_encoder& operator=(const basic_csv_encoder&) = delete;
public:
    basic_csv_encoder(result_type result)
       : basic_csv_encoder(std::move(result), basic_csv_encode_options<CharT>())
    {
    }

    basic_csv_encoder(result_type result,
                      const basic_csv_encode_options<CharT>& options)
      : result_(std::move(result)),
        options_(options),
        stack_(),
        fp_(options.float_format(), options.precision()),
        column_index_(0)
    {
        jsoncons::csv::detail::parse_column_names(options.column_names(), strings_buffer_);
    }

    ~basic_csv_encoder()
    {
        JSONCONS_TRY
        {
            result_.flush();
        }
        JSONCONS_CATCH(...)
        {
        }
    }

private:

    template<class AnyWriter>
    void escape_string(const CharT* s,
                       size_t length,
                       CharT quote_char, CharT quote_escape_char,
                       AnyWriter& result)
    {
        const CharT* begin = s;
        const CharT* end = s + length;
        for (const CharT* it = begin; it != end; ++it)
        {
            CharT c = *it;
            if (c == quote_char)
            {
                result.push_back(quote_escape_char); 
                result.push_back(quote_char);
            }
            else
            {
                result.push_back(c);
            }
        }
    }

    void do_flush() override
    {
        result_.flush();
    }

    bool do_begin_object(semantic_tag, const ser_context&, std::error_code& ec) override
    {
        if (stack_.empty())
        {
            stack_.emplace_back(stack_item_kind::column_mapping);
            return true;
        }
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::row_mapping:
                stack_.emplace_back(stack_item_kind::object);
                return true;
            default: // error
                ec = csv_errc::source_error;
                return false;
        }
    }

    bool do_end_object(const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
                if (stack_[0].count_ == 0)
                {
                    for (size_t i = 0; i < strings_buffer_.size(); ++i)
                    {
                        if (i > 0)
                        {
                            result_.push_back(options_.field_delimiter());
                        }
                        result_.append(strings_buffer_[i].data(),
                                      strings_buffer_[i].length());
                    }
                    result_.append(options_.line_delimiter().data(),
                                  options_.line_delimiter().length());
                }
                for (size_t i = 0; i < strings_buffer_.size(); ++i)
                {
                    if (i > 0)
                    {
                        result_.push_back(options_.field_delimiter());
                    }
                    auto it = buffered_line_.find(strings_buffer_[i]);
                    if (it != buffered_line_.end())
                    {
                        result_.append(it->second.data(),it->second.length());
                        it->second.clear();
                    }
                }
                result_.append(options_.line_delimiter().data(), options_.line_delimiter().length());
                break;
            case stack_item_kind::column_mapping:
             {
                 for (const auto& item : strings_buffer_)
                 {
                     result_.append(item.data(), item.size());
                     result_.append(options_.line_delimiter().data(), options_.line_delimiter().length());
                 }
                 break;
             }
            default:
                break;
        }
        stack_.pop_back();
        if (!stack_.empty())
        {
            end_value();
        }
        return true;
    }

    bool do_begin_array(semantic_tag, const ser_context&, std::error_code& ec) override
    {
        if (stack_.empty())
        {
            stack_.emplace_back(stack_item_kind::row_mapping);
            return true;
        }
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::row_mapping:
                stack_.emplace_back(stack_item_kind::row);
                if (stack_[0].count_ == 0)
                {
                    for (size_t i = 0; i < strings_buffer_.size(); ++i)
                    {
                        if (i > 0)
                        {
                            result_.push_back(options_.field_delimiter());
                        }
                        result_.append(strings_buffer_[i].data(),strings_buffer_[i].length());
                    }
                    if (strings_buffer_.size() > 0)
                    {
                        result_.append(options_.line_delimiter().data(),
                                      options_.line_delimiter().length());
                    }
                }
                return true;
            case stack_item_kind::object:
                stack_.emplace_back(stack_item_kind::object_multi_valued_field);
                return true;
            case stack_item_kind::column_mapping:
                stack_.emplace_back(stack_item_kind::column);
                row_counts_.push_back(1);
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                return true;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }                
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                begin_value(bo);
                stack_.emplace_back(stack_item_kind::column_multi_valued_field);
                return true;
            }
            case stack_item_kind::row:
                begin_value(result_);
                stack_.emplace_back(stack_item_kind::row_multi_valued_field);
                return true;
            default: // error
                ec = csv_errc::source_error;
                return false;
        }
    }

    bool do_end_array(const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::row:
                result_.append(options_.line_delimiter().data(),
                              options_.line_delimiter().length());
                break;
            case stack_item_kind::column:
                ++column_index_;
                break;
            default:
                break;
        }
        stack_.pop_back();

        if (!stack_.empty())
        {
            end_value();
        }
        return true;
    }

    bool do_name(const string_view_type& name, const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            {
                name_ = string_type(name);
                buffered_line_[string_type(name)] = std::basic_string<CharT>();
                if (stack_[0].count_ == 0 && options_.column_names().size() == 0)
                {
                    strings_buffer_.push_back(string_type(name));
                }
                break;
            }
            case stack_item_kind::column_mapping:
            {
                if (strings_buffer_.empty())
                {
                    strings_buffer_.push_back(string_type(name));
                }
                else
                {
                    strings_buffer_[0].push_back(options_.field_delimiter());
                    strings_buffer_[0].append(string_type(name));
                }
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_null_value(semantic_tag, const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_null_value(bo);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_null_value(result_);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_null_value(bo);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_null_value(bo);
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_string_value(const string_view_type& sv, semantic_tag, const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_string_value(sv,bo);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_string_value(sv,result_);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_string_value(sv,bo);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_string_value(sv,bo);
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_byte_string_value(const byte_string_view& b, 
                              semantic_tag tag, 
                              const ser_context& context,
                              std::error_code& ec) override
    {
        byte_string_chars_format encoding_hint;
        switch (tag)
        {
            case semantic_tag::base16:
                encoding_hint = byte_string_chars_format::base16;
                break;
            case semantic_tag::base64:
                encoding_hint = byte_string_chars_format::base64;
                break;
            case semantic_tag::base64url:
                encoding_hint = byte_string_chars_format::base64url;
                break;
            default:
                encoding_hint = byte_string_chars_format::none;
                break;
        }
        byte_string_chars_format format = jsoncons::detail::resolve_byte_string_chars_format(encoding_hint,byte_string_chars_format::none,byte_string_chars_format::base64url);

        std::basic_string<CharT> s;
        switch (format)
        {
            case byte_string_chars_format::base16:
            {
                encode_base16(b.begin(),b.end(),s);
                do_string_value(s, semantic_tag::none, context, ec);
                break;
            }
            case byte_string_chars_format::base64:
            {
                encode_base64(b.begin(),b.end(),s);
                do_string_value(s, semantic_tag::none, context, ec);
                break;
            }
            case byte_string_chars_format::base64url:
            {
                encode_base64url(b.begin(),b.end(),s);
                do_string_value(s, semantic_tag::none, context, ec);
                break;
            }
            default:
            {
                JSONCONS_UNREACHABLE();
            }
        }

        return true;
    }

    bool do_double_value(double val, 
                         semantic_tag, 
                         const ser_context& context,
                         std::error_code& ec) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_double_value(val, context, bo, ec);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_double_value(val, context, result_, ec);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_double_value(val, context, bo, ec);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_double_value(val, context, bo, ec);
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_int64_value(int64_t val, 
                        semantic_tag, 
                        const ser_context&,
                        std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_int64_value(val,bo);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_int64_value(val,result_);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_int64_value(val, bo);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_int64_value(val, bo);
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_uint64_value(uint64_t val, 
                         semantic_tag, 
                         const ser_context&,
                         std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_uint64_value(val, bo);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_uint64_value(val,result_);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_uint64_value(val, bo);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_uint64_value(val, bo);
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool do_bool_value(bool val, semantic_tag, const ser_context&, std::error_code&) override
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::object:
            case stack_item_kind::object_multi_valued_field:
            {
                auto it = buffered_line_.find(name_);
                if (it != buffered_line_.end())
                {
                    std::basic_string<CharT> s;
                    jsoncons::string_result<std::basic_string<CharT>> bo(s);
                    write_bool_value(val,bo);
                    bo.flush();
                    if (!it->second.empty() && options_.subfield_delimiter() != char_type())
                    {
                        it->second.push_back(options_.subfield_delimiter());
                    }
                    it->second.append(s);
                }
                break;
            }
            case stack_item_kind::row:
            case stack_item_kind::row_multi_valued_field:
                write_bool_value(val,result_);
                break;
            case stack_item_kind::column:
            {
                if (strings_buffer_.size() <= row_counts_.back())
                {
                    strings_buffer_.emplace_back();
                }
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_bool_value(val, bo);
                break;
            }
            case stack_item_kind::column_multi_valued_field:
            {
                jsoncons::string_result<std::basic_string<CharT>> bo(strings_buffer_[row_counts_.back()]);
                write_bool_value(val, bo);
                break;
            }
            default:
                break;
        }
        return true;
    }

    template <class AnyWriter>
    bool string_value(const CharT* s, size_t length, AnyWriter& result)
    {
        bool quote = false;
        if (options_.quote_style() == quote_style_kind::all || options_.quote_style() == quote_style_kind::nonnumeric ||
            (options_.quote_style() == quote_style_kind::minimal &&
            (std::char_traits<CharT>::find(s, length, options_.field_delimiter()) != nullptr || std::char_traits<CharT>::find(s, length, options_.quote_char()) != nullptr)))
        {
            quote = true;
            result.push_back(options_.quote_char());
        }
        escape_string(s, length, options_.quote_char(), options_.quote_escape_char(), result);
        if (quote)
        {
            result.push_back(options_.quote_char());
        }

        return true;
    }

    template <class AnyWriter>
    void write_string_value(const string_view_type& value, AnyWriter& result)
    {
        begin_value(result);
        string_value(value.data(),value.length(),result);
        end_value();
    }

    template <class AnyWriter>
    void write_double_value(double val, const ser_context& context, AnyWriter& result, std::error_code& ec)
    {
        begin_value(result);

        if (!std::isfinite(val))
        {
            if ((std::isnan)(val))
            {
                if (options_.enable_nan_to_num())
                {
                    result.append(options_.nan_to_num().data(), options_.nan_to_num().length());
                }
                else if (options_.enable_nan_to_str())
                {
                    do_string_value(options_.nan_to_str(), semantic_tag::none, context, ec);
                }
                else
                {
                    result.append(null_k().data(), null_k().size());
                }
            }
            else if (val == std::numeric_limits<double>::infinity())
            {
                if (options_.enable_inf_to_num())
                {
                    result.append(options_.inf_to_num().data(), options_.inf_to_num().length());
                }
                else if (options_.enable_inf_to_str())
                {
                    do_string_value(options_.inf_to_str(), semantic_tag::none, context, ec);
                }
                else
                {
                    result.append(null_k().data(), null_k().size());
                }
            }
            else
            {
                if (options_.enable_neginf_to_num())
                {
                    result.append(options_.neginf_to_num().data(), options_.neginf_to_num().length());
                }
                else if (options_.enable_neginf_to_str())
                {
                    do_string_value(options_.neginf_to_str(), semantic_tag::none, context, ec);
                }
                else
                {
                    result.append(null_k().data(), null_k().size());
                }
            }
        }
        else
        {
            fp_(val, result);
        }

        end_value();

    }

    template <class AnyWriter>
    void write_int64_value(int64_t val, AnyWriter& result)
    {
        begin_value(result);

        std::basic_ostringstream<CharT> ss;
        ss << val;
        result.append(ss.str().data(),ss.str().length());

        end_value();
    }

    template <class AnyWriter>
    void write_uint64_value(uint64_t val, AnyWriter& result)
    {
        begin_value(result);

        std::basic_ostringstream<CharT> ss;
        ss << val;
        result.append(ss.str().data(),ss.str().length());

        end_value();
    }

    template <class AnyWriter>
    void write_bool_value(bool val, AnyWriter& result) 
    {
        begin_value(result);

        if (val)
        {
            result.append(true_k().data(), true_k().size());
        }
        else
        {
            result.append(false_k().data(), false_k().size());
        }

        end_value();
    }
 
    template <class AnyWriter>
    bool write_null_value(AnyWriter& result) 
    {
        begin_value(result);
        result.append(null_k().data(), null_k().size());
        end_value();
        return true;
    }

    template <class AnyWriter>
    void begin_value(AnyWriter& result)
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch (stack_.back().item_kind_)
        {
            case stack_item_kind::row:
                if (stack_.back().count_ > 0)
                {
                    result.push_back(options_.field_delimiter());
                }
                break;
            case stack_item_kind::column:
            {
                if (row_counts_.size() >= 3)
                {
                    for (size_t i = row_counts_.size()-2; i-- > 0;)
                    {
                        if (row_counts_[i] <= row_counts_.back())
                        {
                            result.push_back(options_.field_delimiter());
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                if (column_index_ > 0)
                {
                    result.push_back(options_.field_delimiter());
                }
                break;
            }
            case stack_item_kind::row_multi_valued_field:
            case stack_item_kind::column_multi_valued_field:
                if (stack_.back().count_ > 0 && options_.subfield_delimiter() != char_type())
                {
                    result.push_back(options_.subfield_delimiter());
                }
                break;
            default:
                break;
        }
    }

    void end_value()
    {
        JSONCONS_ASSERT(!stack_.empty());
        switch(stack_.back().item_kind_)
        {
            case stack_item_kind::row:
            {
                ++stack_.back().count_;
                break;
            }
            case stack_item_kind::column:
            {
                ++row_counts_.back();
                break;
            }
            default:
                ++stack_.back().count_;
                break;
        }
    }
};

typedef basic_csv_encoder<char> csv_stream_encoder;
typedef basic_csv_encoder<char,jsoncons::string_result<std::string>> csv_string_encoder;
typedef basic_csv_encoder<wchar_t> csv_wstream_encoder;
typedef basic_csv_encoder<wchar_t,jsoncons::string_result<std::wstring>> wcsv_string_encoder;

#if !defined(JSONCONS_NO_DEPRECATED)
template<class CharT, class Result = jsoncons::stream_result<CharT>, class Allocator = std::allocator<CharT>>
using basic_csv_serializer = basic_csv_encoder<CharT,Result,Allocator>;

JSONCONS_DEPRECATED_MSG("Instead, use csv_stream_encoder") typedef csv_stream_encoder csv_serializer;
JSONCONS_DEPRECATED_MSG("Instead, use csv_string_encoder") typedef csv_string_encoder csv_string_serializer;
JSONCONS_DEPRECATED_MSG("Instead, use csv_stream_encoder") typedef csv_stream_encoder csv_serializer;
JSONCONS_DEPRECATED_MSG("Instead, use csv_string_encoder") typedef csv_string_encoder csv_string_serializer;
JSONCONS_DEPRECATED_MSG("Instead, use csv_stream_encoder") typedef csv_stream_encoder csv_encoder;
JSONCONS_DEPRECATED_MSG("Instead, use wcsv_stream_encoder") typedef csv_stream_encoder wcsv_encoder;
#endif

}}

#endif
