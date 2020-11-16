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

  using ResponseCallback = Function<void(const void*, size_t, Error*)>;

  Rpc();
  ~Rpc();

  void setName(std::string_view name);
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

  template <typename F>
  struct FImpl;

  enum ReqType : uint32_t {
    reqGreeting,
    reqError,
    reqSuccess,
    reqAck,
    reqNack,
    reqFunctionNotFound,
    reqFindFunction,
    reqPoke,
    reqNotFound,
    reqLookingForPeer,
    reqPeerFound,

    reqCallOffset = 1000,
  };


  async::SchedulerFifo scheduler;

  template <typename R, typename... Args>
  struct FImpl<R(Args...)> : FBase {
    Rpc& rpc;
    Function<R(Args...)> f;
    FImpl(Rpc& rpc, Function<R(Args...)>&& f) : rpc(rpc), f(std::move(f)) {
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
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess);
          } else {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqSuccess, std::apply(f, std::move(args)));
          }
        };
        auto exceptionMode = rpc.currentExceptionMode_.load(std::memory_order_relaxed);
        if (exceptionMode == ExceptionMode::None) {
          in();
          out();
        } else if (exceptionMode == ExceptionMode::DeserializationOnly) {
          try {
            in();
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
            return;
          }
          out();
        } else {
          try {
            in();
            out();
          } catch (const std::exception& e) {
            serializeToBuffer(outbuffer, (uint32_t)0, (uint32_t)reqError, std::string_view(e.what()));
          }
        }
        callback(std::move(outbuffer));
      });
    }
  };

  template<typename F>
  void define(std::string_view name, Function<F>&& f) {
    auto ff = std::make_unique<FImpl<F>>(*this, std::move(f));
    define(name, std::move(ff));
  }

  template<typename R, typename Callback, typename... Args>
  void asyncCallback(std::string_view peerName, std::string_view functionName, Callback&& callback, const Args&... args) {
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)0, (uint32_t)0, args...);
    //printf("original buffer size is %d\n", buffer->size);

    sendRequest(peerName, functionName, std::move(buffer), [callback = std::forward<Callback>(callback)](const void* ptr, size_t len, Error* error) noexcept {
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

  struct Impl;

private:

  void sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response);

  std::atomic<ExceptionMode> currentExceptionMode_ = ExceptionMode::DeserializationOnly;
  std::unique_ptr<Impl> impl_;

  void define(std::string_view name, std::unique_ptr<FBase>&& f);
};


}

