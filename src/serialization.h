
#pragma once

#include "buffer.h"

#include <torch/torch.h>

#include <cstddef>
#include <string_view>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>

namespace rpc {

struct SerializationError: std::runtime_error{
  using std::runtime_error::runtime_error;
};

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

template <typename X, typename T> void serialize(X& x, torch::ArrayRef<T>& v) {
  std::basic_string_view<T> v2;
  x(v2);
  v = torch::ArrayRef<T>(v2.data(), v2.size());
}
template <typename X, typename T> void serialize(X& x, const torch::ArrayRef<T>& v) {
  x(std::basic_string_view<T>(v.data(), v.size()));
}

template <typename X> void serialize(X& x, const torch::Tensor& v) {
  if (!v.device().is_cpu()) {
    serialize(x, v.cpu());
    return;
  }
  size_t size = v.numel() * v.dtype().itemsize();
  bool inlined = size <= 4096;
  bool contiguous = v.is_contiguous();
  x(inlined, contiguous);
  if (inlined) {
    void* data = v.data_ptr();
    if (contiguous) {
      x(v.scalar_type(), v.sizes(), std::string_view((const char*)data, size));
    } else {
      x(v.scalar_type(), v.sizes(), v.strides(), std::string_view((const char*)data, size));
    }
  } else {
    x.addTensor(v, x.tell());
    if (contiguous) {
      x(v.scalar_type(), v.sizes(), v.strides());
    } else {
      x(v.scalar_type(), v.sizes(), v.strides());
    }
  }
}

template<typename X> torch::Tensor serializeHelperAllocateTensor(X& x, std::byte* data, size_t datalen, BufferHandle buffer) {
  torch::Tensor t;
  decltype(t.scalar_type()) dtype;
  decltype(t.sizes()) sizes;
  decltype(t.strides()) strides;
  x(dtype, sizes, strides);

  torch::DataPtr dataPtr(data, buffer.release(), [](void* ptr) {
    Buffer* buf = (Buffer*)ptr;
    BufferHandle{buf};
  }, torch::DeviceType::CPU);
  torch::Storage storage(torch::Storage::use_byte_size_t(), datalen, std::move(dataPtr), nullptr, false);

  return torch::empty({0}, dtype).set_(std::move(storage), 0, sizes, strides);
}

template <typename X> void serialize(X& x, torch::Tensor& v) {
  bool inlined, contiguous;
  x(inlined, contiguous);

  decltype(v.scalar_type()) dtype;
  decltype(v.sizes()) sizes;
  decltype(v.strides()) strides;
  if (contiguous && inlined) {
    x(dtype, sizes);
  } else {
    x(dtype, sizes, strides);
  }

  if (inlined) {
    std::string_view data;
    x(data);
    if (v.defined() && v.scalar_type() == dtype) {
      v.resize_(sizes);
    } else {
      v = torch::empty(sizes, dtype);
    }
    if (!contiguous) {
      v.as_strided_(sizes, strides);
    }
    if ((size_t)v.numel() != data.size() / v.dtype().itemsize()) {
      throw SerializationError("numel mismatch in tensor deserialize");
    }
    std::memcpy(v.data_ptr(), data.data(), data.size());
  } else {
    v = std::move(x.getTensor().tensor);
  }
}

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
  std::byte* begin = nullptr;
  std::byte* dst = nullptr;
  TensorRef* tensors = nullptr;
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

  void addTensor(OpWrite, const torch::Tensor& x, size_t offset) {
    new (tensors++) TensorRef{x, offset};
  }
  void addTensor(OpSize, [[maybe_unused]] const torch::Tensor& x, [[maybe_unused]] size_t offset) {
    ++tensors;
  }
  void addTensor(const torch::Tensor& t, size_t offset) {
    addTensor(Op{}, t, offset);
  }

  size_t tell() const {
    return dst - begin;
  }
};

struct Deserialize {
  TensorRef* tensors = nullptr;
  TensorRef* tensorsEnd = nullptr;
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

  TensorRef& getTensor() {
    if (tensors == tensorsEnd) {
      throw SerializationError("Deserialize: reached end of tensor data");
    }
    return *tensors++;
  }
};

template<typename... T>
void serializeToBuffer(BufferHandle& buffer, const T&... v) {
  Serialize<OpSize> x{};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  size_t nTensors = x.tensors - (TensorRef*)nullptr;
  if (!buffer || buffer->capacity < size || nTensors < buffer->nTensors) {
    buffer = makeBuffer(size, nTensors);
  } else {
    shrinkBuffer(buffer, size, nTensors);
  }
  std::byte* dst = buffer->data();
  Serialize<OpWrite> x2{dst, dst, buffer->tensors()};
  (x2(v), ...);
}

template<typename... T>
BufferHandle serializeToBuffer(const T&... v) {
  BufferHandle h;
  serializeToBuffer(h, std::forward<const T&>(v)...);
  return h;
}

template<typename... T>
void serializeToStringView(std::string_view buffer, const T&... v) {
  Serialize<OpSize> x{};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  size_t nTensors = x.tensors - (TensorRef*)nullptr;
  if (buffer.size() < size || nTensors) {
    throw SerializationError("Data does not fit in target buffer");
  }
  std::byte* dst = (std::byte*)buffer.data();
  Serialize<OpWrite> x2{dst, dst, nullptr};
  (x2(v), ...);
}


template<typename... T>
std::string_view deserializeBuffer(const void* ptr, size_t len, T&... result) {
  Deserializer des(std::string_view{(const char*)ptr, len});
  Deserialize x(des);
  x(result...);
  return des.buf;
}
template<typename... T>
auto deserializeBuffer(Buffer* buffer, T&... result) {
  Deserializer des(std::string_view{(const char*)buffer->data(), buffer->size});
  Deserialize x(des);
  x.tensors = buffer->tensors();
  x.tensorsEnd = x.tensors + buffer->nTensors;
  x(result...);
  if (des.buf.size() != 0) {
    throw SerializationError("deserializeBuffer: " + std::to_string(des.buf.size()) + " trailing bytes");
  }
  return des.buf;
}

}
