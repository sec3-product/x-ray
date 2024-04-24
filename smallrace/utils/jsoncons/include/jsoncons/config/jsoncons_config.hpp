// Copyright 2013 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_JSONCONS_CONFIG_HPP
#define JSONCONS_JSONCONS_CONFIG_HPP

#include <stdexcept>
#include <string>
#include <cmath>
#include <exception>
#include <ostream>

// Uncomment the following line to suppress deprecated names (recommended for new code)
//#define JSONCONS_NO_DEPRECATED

// The definitions below follow the definitions in compiler_support_p.h, https://github.com/01org/tinycbor
// MIT license

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54577
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 9
#define JSONCONS_NO_ERASE_TAKING_CONST_ITERATOR 1
#endif

#if defined(__clang__) 
#  define JSONCONS_FALLTHROUGH [[clang::fallthrough]]
#elif defined(__GNUC__) && ((__GNUC__ >= 7))
#  define JSONCONS_FALLTHROUGH __attribute__((fallthrough))
#elif defined (__GNUC__)
#  define JSONCONS_FALLTHROUGH // FALLTHRU
#else
#  define JSONCONS_FALLTHROUGH
#endif

#if defined(__GNUC__) || defined(__clang__)
#define JSONCONS_LIKELY(x) __builtin_expect(!!(x), 1)
#define JSONCONS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define JSONCONS_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define JSONCONS_LIKELY(x) x
#define JSONCONS_UNLIKELY(x) x
#define JSONCONS_UNREACHABLE() __assume(0)
#else
#define JSONCONS_LIKELY(x) x
#define JSONCONS_UNLIKELY(x) x
#define JSONCONS_UNREACHABLE() do {} while (0)
#endif

// Deprecated symbols markup
#if (defined(__cplusplus) && __cplusplus >= 201402L)
#define JSONCONS_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#endif

#if !defined(JSONCONS_DEPRECATED_MSG) && defined(__GNUC__) && defined(__has_extension)
#if __has_extension(attribute_deprecated_with_message)
#define JSONCONS_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#endif
#endif

#if !defined(JSONCONS_DEPRECATED_MSG) && defined(_MSC_VER)
#if (_MSC_VER) >= 1920
#define JSONCONS_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#define JSONCONS_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
#endif
#endif

// Following boost/atomic/detail/config.hpp
#if !defined(JSONCONS_DEPRECATED_MSG) && (\
    (defined(__GNUC__) && ((__GNUC__ + 0) * 100 + (__GNUC_MINOR__ + 0)) >= 405) ||\
    (defined(__SUNPRO_CC) && (__SUNPRO_CC + 0) >= 0x5130))
    #define JSONCONS_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#endif

#if !defined(JSONCONS_DEPRECATED_MSG) && defined(__clang__) && defined(__has_extension)
    #if __has_extension(attribute_deprecated_with_message)
        #define JSONCONS_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
    #else
        #define JSONCONS_DEPRECATED_MSG(msg) __attribute__((deprecated))
    #endif
#endif

#if !defined(JSONCONS_DEPRECATED_MSG)
#define JSONCONS_DEPRECATED_MSG(msg)
#endif

#if defined(ANDROID) || defined(__ANDROID__)
#define JSONCONS_HAS_STRTOLD_L
#if __ANDROID_API__ >= 21
#else
#define JSONCONS_NO_LOCALECONV
#endif
#endif 

#if defined(_MSC_VER)
#define JSONCONS_HAS_MSC__STRTOD_L
#define JSONCONS_HAS_FOPEN_S
#endif

// Follows boost 1_68
#if !defined(JSONCONS_HAS_STRING_VIEW)
#  if defined(__clang__)
#   if (__cplusplus >= 201703)
#    if __has_include(<string_view>)
#     define JSONCONS_HAS_STRING_VIEW 1
#    endif // __has_include(<string_view>)
#   endif // (__cplusplus >= 201703)
#  endif // defined(__clang__)
#  if defined(__GNUC__)
#   if (__GNUC__ >= 7)
#    if (__cplusplus >= 201703) || (defined(_HAS_CXX17) && _HAS_CXX17 == 1)
#     define JSONCONS_HAS_STRING_VIEW 1
#    endif // (__cplusplus >= 201703)
#   endif // (__GNUC__ >= 7)
#  endif // defined(__GNUC__)
#  if defined(_MSC_VER)
#   if (_MSC_VER >= 1910 && defined(_HAS_CXX17) && (_HAS_CXX17 > 0))
#    define JSONCONS_HAS_STRING_VIEW 1
#   endif // (_MSC_VER >= 1910 && _HAS_CXX17)
#  endif // defined(_MSC_VER)
#endif // !defined(JSONCONS_HAS_STRING_VIEW)

#if !defined(JSONCONS_HAS_STRING_VIEW)
#include <jsoncons/detail/string_view.hpp>
namespace jsoncons {
using jsoncons::detail::basic_string_view;
using string_view = basic_string_view<char, std::char_traits<char>>;
using wstring_view = basic_string_view<wchar_t, std::char_traits<wchar_t>>;
}
#else 
#include <string_view>
namespace jsoncons {
using std::basic_string_view;
using std::string_view;
using std::wstring_view;
}
#endif

#if !defined(JSONCONS_HAS_SPAN)
#include <jsoncons/detail/span.hpp>
namespace jsoncons {
using nonstd::jsoncons_span_lite::span;
}
#else 
#include <span>
namespace jsoncons {
using std::span;
}
#endif
 
#define JSONCONS_STRING_LITERAL(name, ...) \
    template <class CharT> \
    const std::basic_string<CharT>& name##_literal() {\
        static constexpr CharT s[] = { __VA_ARGS__};\
        static const std::basic_string<CharT> sv(s, sizeof(s) / sizeof(CharT));\
        return sv;\
    }
 
#define JSONCONS_ARRAY_OF_CHAR(CharT, name, ...) \
    static constexpr CharT name[] = { __VA_ARGS__,0};

#define JSONCONS_EXPAND(X) X    
#define JSONCONS_QUOTE(Prefix, A) JSONCONS_EXPAND(Prefix ## #A)

#define JSONCONS_DEFINE_LITERAL( name ) \
template<class CharT> CharT const* name##_literal(); \
template<> inline char const * name##_literal<char>() { return JSONCONS_QUOTE(,name); } \
template<> inline wchar_t const* name##_literal<wchar_t>() { return JSONCONS_QUOTE(L,name); } \
template<> inline char16_t const* name##_literal<char16_t>() { return JSONCONS_QUOTE(u,name); } \
template<> inline char32_t const* name##_literal<char32_t>() { return JSONCONS_QUOTE(U,name); }

#endif

#if (!defined(JSONCONS_NO_EXCEPTIONS))
// Check if exceptions are disabled.
#if defined( __cpp_exceptions) && __cpp_exceptions == 0
# define JSONCONS_NO_EXCEPTIONS 1
#endif
#endif

#if !defined(JSONCONS_NO_EXCEPTIONS)
#if __GNUC__ && !__EXCEPTIONS
# define JSONCONS_NO_EXCEPTIONS 1
#elif _MSC_VER 
#if defined(_HAS_EXCEPTIONS) && _HAS_EXCEPTIONS == 0
# define JSONCONS_NO_EXCEPTIONS 1
#elif !defined(_CPPUNWIND)
# define JSONCONS_NO_EXCEPTIONS 1
#endif
#endif

#endif

// allow to disable exceptions
#if !defined(JSONCONS_NO_EXCEPTIONS)
    #define JSONCONS_THROW(exception) throw exception
    #define JSONCONS_RETHROW throw
    #define JSONCONS_TRY try
    #define JSONCONS_CATCH(exception) catch(exception)
#else
    #define JSONCONS_THROW(exception) std::terminate()
    #define JSONCONS_RETHROW std::terminate()
    #define JSONCONS_TRY if(true)
    #define JSONCONS_CATCH(exception) if(false)
#endif

