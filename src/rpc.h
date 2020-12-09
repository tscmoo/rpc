#pragma once

//#include <tensorpipe/common/function.h>


#include "serialization.h"
#include "allocator.h"
#include "synchronization.h"
#include "async.h"
#include "function.h"
#include "buffer.h"

#include <cstddef>
#include <string_view>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <mutex>
#include <future>

namespace rpc {

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

struct Rpc {

  using ResponseCallback = Function<void(BufferHandle buffer, Error*)>;

  Rpc();
  ~Rpc();

  void setName(std::string_view name);
  std::string_view getName() const;
  void setOnError(Function<void(const Error&)>&&);
  void listen(std::string_view addr);
  void connect(std::string_view addr);

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
    virtual void call(BufferHandle, Function<void(BufferHandle)>) = 0;
  };

  template <typename Signature, typename F>
  struct FImpl;

  enum ReqType : uint32_t {
    reqGreeting,
    reqError,
    reqSuccess,
    reqAck,
    reqNack,
    reqFunctionNotFound,
    reqFindFunction__deprecated,
    reqPoke,
    //reqNotFound,
    reqLookingForPeer,
    reqPeerFound,
    reqClose,

    reqCallOffset = 1000,
  };


  async::SchedulerFifo scheduler;

  template <typename R, typename... Args, typename F>
  struct FImpl<R(Args...), F> : FBase {
    Rpc& rpc;
    F f;
    template<typename F2>
    FImpl(Rpc& rpc, F2&& f) : rpc(rpc), f(std::forward<F2>(f)) {
    }
    virtual ~FImpl() {}
    virtual void call(BufferHandle inbuffer, Function<void(BufferHandle)> callback) noexcept override {
      rpc.scheduler.run([this, inbuffer = std::move(inbuffer), callback = std::move(callback)]() {
        std::tuple<std::decay_t<Args>...> args;
        auto in = [&]() {
          std::apply([&](std::decay_t<Args>&... args) {
            deserializeBuffer(std::move(inbuffer), args...);
          }, args);
        };
        BufferHandle outbuffer;
        auto out = [&]() {
          if constexpr (std::is_same_v<void, R>) {
            std::apply(f, std::move(args));
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, (uint32_t)0);
          } else {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, (uint32_t)0, std::apply(f, std::move(args)));
          }
        };
        auto exceptionMode = rpc.currentExceptionMode_.load(std::memory_order_relaxed);
        if (exceptionMode == ExceptionMode::None) {
          in();
          out();
        } else if (exceptionMode == ExceptionMode::DeserializationOnly) {
          bool success = false;
          try {
            in();
            success = true;
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, (uint32_t)0, std::string_view(e.what()));
          }
          if (success) {
            out();
          }
        } else {
          try {
            in();
            out();
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, (uint32_t)0, std::string_view(e.what()));
          }
        }
        callback(std::move(outbuffer));
      });
    }
  };

  template<typename Signature, typename F>
  void define(std::string_view name, F&& f) {
    auto ff = std::make_unique<FImpl<Signature, F>>(*this, std::forward<F>(f));
    define(name, std::move(ff));
  }

  template<typename R, typename Callback, typename... Args>
  void asyncCallback(std::string_view peerName, std::string_view functionName, Callback&& callback, const Args&... args) {
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, (uint32_t)0, args...);
    //printf("original buffer size is %d\n", buffer->size);

    sendRequest(peerName, functionName, std::move(buffer), [callback = std::forward<Callback>(callback)](BufferHandle buffer, Error* error) mutable noexcept {
      if (error) {
        std::move(callback)(nullptr, error);
        return;
      }
      //printf("request got a response of %d bytes\n", len);
      try {
        if constexpr (std::is_same_v<R, void>) {
          char nonnull;
          std::move(callback)((void*)&nonnull, nullptr);
        } else {
          R r;
          deserializeBuffer(buffer, r);
          std::move(callback)(&r, nullptr);
        }
      } catch (const std::exception& e) {
        Error err{std::string("Deserialization error: ") + e.what()};
        std::move(callback)(nullptr, &err);
      }
    });
  }

  template<typename R = void, typename... Args>
  std::future<R> async(std::string_view peerName, std::string_view functionName, const Args&... args) {
    std::promise<R> promise;
    auto future = promise.get_future();
    asyncCallback<R>(peerName, functionName, [promise = std::move(promise)]([[maybe_unused]] R* ptr, Error* err) mutable {
      if (err) {
        promise.set_exception(std::make_exception_ptr(std::move(*err)));
      } else {
        if constexpr (std::is_same_v<R, void>) {
          promise.set_value();
        } else {
          promise.set_value(std::move(*ptr));
        }
      }
    }, args...);
    return future;
  }

  template<typename R = void, typename... Args>
  R sync(std::string_view peerName, std::string_view functionName, const Args&... args) {
    auto future = async<R>(peerName, functionName, args...);
    return future.get();
  }

  struct Impl;

private:

  void sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response);

  std::atomic<ExceptionMode> currentExceptionMode_ = ExceptionMode::DeserializationOnly;
  std::unique_ptr<Impl> impl_;

  void define(std::string_view name, std::unique_ptr<FBase>&& f);
};


}

