#pragma once

//#include <tensorpipe/common/function.h>
#include "tensorpipe/tensorpipe/common/function.h"

#include "serialization.h"
#include "allocator.h"
#include "synchronization.h"

#include <cstddef>
#include <string_view>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <mutex>

namespace rpc {

template<typename T>
using Function = tensorpipe::Function<T>;

struct Buffer {
  Buffer* next{nullptr};
  size_t capacity = 0;
  size_t size = 0;
  std::atomic_int refcount{0};
  std::byte* data() {
    return dataptr<std::byte>(this);
  }
};

struct BufferHandle {
  Buffer* buffer_ = nullptr;
  BufferHandle() = default;
  BufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {}
  BufferHandle(const BufferHandle&) = delete;
  BufferHandle& operator=(const BufferHandle&) = delete;
  BufferHandle(BufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  BufferHandle& operator=(BufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~BufferHandle() {
    if (buffer_) {
      //printf("non shared deallocate\n");
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer_;
    buffer_ = nullptr;
    return r;
  }
};
struct SharedBufferHandle {
  Buffer* buffer_ = nullptr;
  SharedBufferHandle() = default;
  SharedBufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {
    if (buffer_) {
      if (buffer->refcount != 0) std::abort();
      addref();
    }
  }
  SharedBufferHandle(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
  }
  SharedBufferHandle& operator=(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
    return *this;
  }
  SharedBufferHandle(SharedBufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  SharedBufferHandle& operator=(SharedBufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~SharedBufferHandle() {
    if (buffer_ && decref() == 0) {
      //printf("shared deallocate\n");
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  int addref() noexcept {
    return buffer_->refcount.fetch_add(1, std::memory_order_acquire) + 1;
  }
  int decref() noexcept {
    int r = buffer_->refcount.fetch_sub(1, std::memory_order_release) - 1;
    //printf("decref %p %d\n", this, r);
    return r;
    return buffer_->refcount.fetch_sub(1, std::memory_order_release) - 1;
  }
};

template<typename... T>
void serializeToBuffer(BufferHandle& buffer, const T&... v) {
  Serialize<OpSize> x{nullptr};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  if (!buffer || buffer->capacity < size) {
    buffer = BufferHandle(rpc::allocate<Buffer, std::byte>(size));
  }
  buffer->size = size;
  std::byte* dst = dataptr<std::byte>(&*buffer);
  Serialize<OpWrite> x2{dst};
  (x2(v), ...);
}

template<typename... T>
BufferHandle serializeToBuffer(const T&... v) {
  BufferHandle h;
  serializeToBuffer(h, std::forward<const T&>(v)...);
  return h;
}

template<typename... T>
void deserializeBuffer(const void* ptr, size_t len, T&... result) {
  Deserializer des(std::string_view{(const char*)ptr, len});
  Deserialize x(des);
  x(result...);
}
template<typename... T>
void deserializeBuffer(const Buffer* buffer, T&... result) {
  const std::byte* data = dataptr<std::byte>(buffer);
  return deserializeBuffer(data, buffer->size, result...);
}
template<typename... T>
void deserializeBuffer(const BufferHandle& buffer, T&... result) {
  return deserializeBuffer(&*buffer, result...);
}

struct Error: std::exception {
  std::string str;
  Error() {}
  Error(std::string&& str) : str(std::move(str)) {}
  virtual const char* what() const noexcept override {
    return str.c_str();
  }
};

template<typename R>
using CallbackFunction = Function<void(R*, Error* error)>;

struct RPCFunctionOptions {
  std::string_view name;
  int maxSimultaneousCalls = 0;
};

template<typename R>
struct RpcCallOptions {
  std::string_view name;
  CallbackFunction<R> callback;
  std::optional<std::chrono::milliseconds> timeout;
  Function<void()> timeoutFunction;
};

struct Rpc;

template<typename T>
struct RpcDeleter {
  void operator()(T*) const noexcept;
};

struct RpcConnection {
  std::string localAddress() const;
  std::string remoteAddress() const;
  struct Impl;
  std::unique_ptr<Impl, RpcDeleter<Impl>> impl_;
};

struct RpcListener {
  void accept(Function<void(RpcConnection*, Error*)>&&);
  struct Impl;
  std::unique_ptr<Impl, RpcDeleter<Impl>> impl_;
};

struct Rpc {

  using ResponseCallback = Function<void(const void*, size_t, Error*)>;

  Rpc();
  ~Rpc();

  RpcListener listen(std::string_view url);
  RpcConnection connect(std::string_view url);

  enum class ExceptionMode {
    None,
    DeserializationOnly,
    All
  };

  void setExceptionMode(ExceptionMode mode) {
    currentExceptionMode_ = mode;
  }

  struct FBase {
    virtual ~FBase(){};
    virtual void call(const std::byte*, size_t, BufferHandle& buffer) = 0;
  };

  template <typename F>
  struct FImpl;

  enum {
    reqError,
    reqSuccess,
    reqAck,
    reqFunctionNotFound,
    reqFindFunction,
    reqPoke,

    reqCallOffset = 1000,
  };

  template <typename R, typename... Args>
  struct FImpl<R(Args...)> : FBase {
    Rpc& rpc;
    Function<R(Args...)> f;
    FImpl(Rpc& rpc, Function<R(Args...)>&& f) : rpc(rpc), f(std::move(f)) {
    }
    virtual ~FImpl() {}
    virtual void call(const std::byte* ptr, size_t len, BufferHandle& buffer) noexcept override {
      std::tuple<std::decay_t<Args>...> args;
      auto in = [&]() {
        std::apply([ptr, len](std::decay_t<Args>&... args) {
          deserializeBuffer(ptr, len, args...);
        }, args);
      };
      auto out = [&]() {
        if constexpr (std::is_same_v<void, R>) {
          std::apply(f, std::move(args));
          serializeToBuffer(buffer, (uint32_t)0, (uint32_t)reqSuccess);
        } else {
          serializeToBuffer(buffer, (uint32_t)0, (uint32_t)reqSuccess, std::apply(f, std::move(args)));
        }
      };
      if (rpc.currentExceptionMode_ == ExceptionMode::None) {
        in();
        out();
      } else if (rpc.currentExceptionMode_ == ExceptionMode::DeserializationOnly) {
        try {
          in();
        } catch (const std::exception& e) {
          serializeToBuffer(buffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
          return;
        }
        out();
      } else {
        try {
          in();
          out();
        } catch (const std::exception& e) {
          serializeToBuffer(buffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
        }
      }
    }
  };

  struct RemoteFunction {
    uint32_t id = 0;
    std::string typeId;

    template<typename X>
    void serialize(X& x) {
      x(id, typeId);
    }
  };

  template<typename F>
  void define(std::string_view name, Function<F>&& f) {
    auto ff = std::make_unique<FImpl<F>>(*this, std::move(f));
    define(name, std::move(ff));
  }

  template<typename R>
  void asyncCallbackById(RpcConnection& conn, uint32_t fid, BufferHandle&& buffer, CallbackFunction<R> callback) {
    //printf("asyncCallbackById fid %#x  %d bytes\n", fid, buffer->size);
    sendRequest(conn, std::move(buffer), fid, [callback = std::move(callback)](const void* ptr, size_t len, Error* error) noexcept {
      if (error) {
        callback(nullptr, error);
        return;
      }
      //printf("request got a response of %d bytes\n", len);
      try {
        if constexpr (std::is_same_v<R, void>) {
          char nonnull;
          callback(&nonnull, nullptr);
        } else {
          R r;
          deserializeBuffer(ptr, len, r);
          callback(&r, nullptr);
        }
      } catch (const std::exception& e) {
        Error err{std::string("Deserialization error: ") + e.what()};
        callback(nullptr, &err);
      }
    });
  }

  template<typename R, typename... Args>
  void asyncCallback(RpcConnection& conn, std::string_view name, CallbackFunction<R> callback, const Args&... args) {
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, args...);
    //printf("original buffer size is %d\n", buffer->size);
    uint32_t id = functionId(name);
    if (id == 0) {
      name = persistentString(name);
      BufferHandle buffer2;
      serializeToBuffer(buffer2, (uint32_t)0, (uint32_t)0, name);
      asyncCallbackById<RemoteFunction>(conn, reqFindFunction, std::move(buffer2), [this, &conn, name, buffer = std::move(buffer), callback = std::move(callback)](RemoteFunction* rf, Error* error) mutable noexcept {
        if (!rf) {
          std::move(callback)(nullptr, error);
        } else {
          uint32_t id = rf->id;
          //printf("got id %#x\n", id);
          if (id == 0) {
            Error err("RPC remote function '" + std::string(name) + "' does not exist");
            std::move(callback)(nullptr, &err);
            return;
          }
          setRemoteFunc(name, rf);
          asyncCallbackById<R>(conn, id, std::move(buffer), std::move(callback));
        }
      });
    } else {
      asyncCallbackById<R>(conn, id, std::move(buffer), std::move(callback));
    }
  }

  struct Impl;

private:

  void sendRequest(RpcConnection& conn, BufferHandle&& buffer, uint32_t fid, ResponseCallback response);

  ExceptionMode currentExceptionMode_ = ExceptionMode::DeserializationOnly;
  std::unique_ptr<Impl> impl_;

  uint32_t functionId(std::string_view name);
  void define(std::string_view name, std::unique_ptr<FBase>&& f);
  std::string_view persistentString(std::string_view str);
  void setRemoteFunc(std::string_view name, RemoteFunction* rf);
};


}

