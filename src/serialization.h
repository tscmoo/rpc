
#pragma once

#include "allocator.h"

#include <cstddef>
#include <string_view>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>

namespace rpc {

template <typename X, typename A, typename B>
void serialize(X& x, const std::pair<A, B>& v) {
  x(v.first, v.second);
}

template <typename X, typename A, typename B>
void serialize(X& x, std::pair<A, B>& v) {
  x(v.first, v.second);
}

template <typename X, typename T>
void serialize(X& x, const std::optional<T>& v) {
  x(v.has_value());
  if (v.has_value()) {
    x(v.value());
  }
}

template <typename X, typename T> void serialize(X& x, std::optional<T>& v) {
  if (x.template read<bool>()) {
    v.emplace();
    x(v.value());
  } else {
    v.reset();
  }
}

template <typename X, typename T>
void serialize(X& x, const std::vector<T>& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}

template <typename X, typename T> void serialize(X& x, std::vector<T>& v) {
  if (std::is_trivial_v<T>) {
    std::basic_string_view<T> view;
    x(view);
    v.resize(view.size());
    std::memcpy(v.data(), view.data(), sizeof(T) * view.size());
  } else {
    size_t n = x.template read<size_t>();
    v.resize(n);
    for (size_t i = 0; i != n; ++i) {
      x(v[i]);
    }
  }
}

template <typename X, typename Key, typename Value>
void serialize(X& x, const std::unordered_map<Key, Value>& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}

template <typename X, typename Key, typename Value>
void serialize(X& x, std::unordered_map<Key, Value>& v) {
  v.clear();
  size_t n = x.template read<size_t>();
  for (; n; --n) {
    auto k = x.template read<Key>();
    v.emplace(std::move(k), x.template read<Value>());
  }
}

struct SerializationError: std::runtime_error{
  using std::runtime_error::runtime_error;
};

struct OpSize {};
struct OpWrite {};
struct OpRead {};

// This is not a cross platform serializer
struct Serializer {
  std::byte* write(OpSize, std::byte* dst, [[maybe_unused]] const void* src, size_t len) {
    return dst + len;
  }
  std::byte* write(OpWrite, std::byte* dst, const void* src, size_t len) {
    std::memcpy(dst, src, len);
    return dst + len;
  }
  template <typename Op, typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  std::byte* write(Op, std::byte* dst, T v) {
    dst = write(Op{}, dst, (void*)&v, sizeof(v));
    return dst;
  }
  template<typename Op>
  std::byte* write(Op, std::byte* dst, std::string_view str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), str.size());
    return dst;
  }
  template <typename Op, typename T>
  std::byte* write(Op, std::byte* dst, std::basic_string_view<T> str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), sizeof(T) * str.size());
    return dst;
  }
};
struct Deserializer {
  std::string_view buf;
  Deserializer() = default;
  Deserializer(std::string_view buf)
      : buf(buf) {
  }
  Deserializer(const void* data, size_t len)
      : buf((const char*)data, len) {
  }
  [[noreturn]] void eod() {
    throw SerializationError("Deserializer: reached end of data");
  }
  void consume(size_t len) {
    buf = {buf.data() + len, buf.size() - len};
  }
  template <typename T> std::basic_string_view<T> readStringView() {
    size_t len = read<size_t>();
    if (buf.size() < sizeof(T) * len) {
      len = buf.size() / sizeof(T);
    }
    T* data = (T*)buf.data();
    consume(sizeof(T) * len);
    return {data, len};
  }
  std::string_view readString() {
    size_t len = read<size_t>();
    if (buf.size() < len) {
      eod();
    }
    const char* data = buf.data();
    consume(len);
    return {data, len};
  }
  template <typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void read(T& r) {
    if (buf.size() < sizeof(T)) {
      eod();
    }
    std::memcpy(&r, buf.data(), sizeof(T));
    consume(sizeof(T));
  }
  void read(std::string_view& r) {
    r = readString();
  }
  void read(std::string& r) {
    r = readString();
  }
  template <typename T> void read(std::basic_string_view<T>& r) {
    r = readStringView<T>();
  }

  template <typename T> T read() {
    T r;
    read(r);
    return r;
  }
  std::string_view read() {
    return readString();
  }

  bool empty() {
    return buf.empty();
  }
};

template<typename Op>
struct Serialize {
  std::byte* dst = nullptr;
  template <typename T> static std::false_type has_serialize_f(...);
  template <typename T,
            typename = decltype(
                std::declval<T>().serialize(std::declval<Serialize&>()))>
  static std::true_type has_serialize_f(int);
  template <typename T>
  static const bool has_serialize =
      decltype(Serialize::has_serialize_f<T>(0))::value;
  template <typename T> static std::false_type has_builtin_write_f(...);
  template <
      typename T,
      typename = decltype(std::declval<Serializer>().write(OpWrite{}, (std::byte*)nullptr, std::declval<T>()))>
  static std::true_type has_builtin_write_f(int);
  template <typename T>
  static const bool has_builtin_write =
      decltype(Serialize::has_builtin_write_f<T>(0))::value;
  template <typename T>
  void operator()(const T& v) {
    if constexpr (has_serialize<const T>) {
      v.serialize(*this);
    } else if constexpr (has_serialize<T>) {
      const_cast<T&>(v).serialize(*this);
    } else if constexpr (has_builtin_write<const T>) {
      dst = Serializer{}.write(Op{}, dst, v);
    } else {
      serialize(*this, v);
    }
  }
  template <typename... T>
  void operator()(const T&... v) {
    ((*this)(std::forward<const T&>(v)), ...);
  }
};

struct Deserialize {
  Deserialize(Deserializer& des)
      : des(des) {
  }
  Deserializer& des;

  template <typename T> static std::false_type has_serialize_f(...);
  template <typename T,
            typename = decltype(
                std::declval<T>().serialize(std::declval<Deserialize&>()))>
  static std::true_type has_serialize_f(int);
  template <typename T>
  static const bool has_serialize =
      decltype(Deserialize::has_serialize_f<T>(0))::value;
  template <typename T> static std::false_type has_builtin_read_f(...);
  template <typename T,
            typename =
                decltype(std::declval<Deserializer>().read(std::declval<T&>()))>
  static std::true_type has_builtin_read_f(int);
  template <typename T>
  static const bool has_builtin_read =
      decltype(Deserialize::has_builtin_read_f<T>(0))::value;
  template <typename T> void operator()(T& v) {
    if constexpr (has_serialize<T>) {
      v.serialize(*this);
    } else if constexpr (has_builtin_read<T>) {
      des.read(v);
    } else {
      serialize(*this, v);
    }
  }

  template <typename... T> void operator()(T&... v) {
    ((*this)(v), ...);
  }

  template <typename T> T read() {
    if constexpr (has_serialize<T>) {
      T r;
      r.serialize(*this);
      return r;
    } else if constexpr (has_builtin_read<T>) {
      return des.read<T>();
    } else {
      T r;
      serialize(*this, r);
      return r;
    }
  }
};


}
