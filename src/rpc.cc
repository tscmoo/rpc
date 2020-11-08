#include "rpc.h"

#include "network.h"

#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/transport/listener.h>
#include <tensorpipe/transport/connection.h>
#include <tensorpipe/transport/shm/listener.h>
#include <tensorpipe/transport/shm/connection.h>
#include <tensorpipe/transport/uv/listener.h>
#include <tensorpipe/transport/uv/connection.h>

#include <random>
#include <thread>
#include <new>

namespace rpc {

namespace {
std::mt19937_64 rng{std::random_device{}()};
SpinMutex rngMutex;
template<typename T>
T random(T min, T max) {
  std::lock_guard l(rngMutex);
  return std::uniform_int_distribution<T>(min, max)(rng);
}
}

namespace inproc {

struct Listener;
struct Connection;

struct Manager {
  alignas(64) SpinMutex mutex;
  std::unordered_map<std::string, Listener*> listeners;
};

Manager global;

struct Context {
  std::unique_ptr<Connection> connect(std::string_view url);
  std::unique_ptr<Listener> listen(std::string_view url);
};

struct Listener {
  std::string url;
  Function<void(Error*, std::unique_ptr<Connection>&&)> acceptFunction;
  Listener(std::string url) : url(url) {
    std::lock_guard l(global.mutex);
    global.listeners[url] = this;
  }
  ~Listener() {
    std::lock_guard l(global.mutex);
    auto i = global.listeners.find(url);
    if (i != global.listeners.end() && i->second == this) {
      i->second = nullptr;
    }
  }
  void accept(Function<void(Error*, std::unique_ptr<Connection>&&)>&& callback) {
    acceptFunction = std::move(callback);
  }
};

struct alignas(64) Pipe {
  std::atomic<bool> dead{false};
  std::atomic<Buffer*> buffer = nullptr;
};

struct Connection {
  std::shared_ptr<Pipe> inpipe;
  std::shared_ptr<Pipe> outpipe;

  Connection(std::shared_ptr<Pipe> inpipe, std::shared_ptr<Pipe> outpipe);
  ~Connection();
  void close();
  void read(Function<bool(Error*, BufferHandle&&)>&&);
  void write(BufferHandle&&, Function<void(Error*)>&&);

  std::thread thread;

  alignas(64) std::atomic<tensorpipe::FunctionPointer> readCallback;
  alignas(64) std::atomic<tensorpipe::FunctionPointer> writeCallback;
};

std::unique_ptr<Connection> Context::connect(std::string_view url) {
  std::lock_guard l(global.mutex);
  auto i = global.listeners.find(std::string(url));
  if (i == global.listeners.end() || !i->second || !i->second->acceptFunction) {
    return nullptr;
  }
  auto p1 = std::make_shared<Pipe>();
  auto p2 = std::make_shared<Pipe>();
  auto c1 = std::make_unique<Connection>(p1, p2);
  auto c2 = std::make_unique<Connection>(p2, p1);
  i->second->acceptFunction(nullptr, std::move(c1));
  return c2;
}

std::unique_ptr<Listener> Context::listen(std::string_view url) {
  return std::make_unique<Listener>(std::string(url));
}

Connection::~Connection() {
  close();
  thread.join();
  readCallback = nullptr;
  writeCallback = nullptr;
}

void Connection::close() {
  inpipe->dead = true;
  outpipe->dead = true;
}

void Connection::read(Function<bool(Error*, BufferHandle&&)>&& callback) {
  auto* x = callback.release();
  //printf("callback at %p set to %p\n", &readCallback, x);
  readCallback.store(x, std::memory_order_relaxed);
}

void Connection::write(BufferHandle&& buffer, Function<void (Error*)>&& callback) {
  auto* x = buffer.release();
  //printf("%p write %p\n", &outpipe->buffer, x);
  auto* addr = &outpipe->buffer;
  long tmp, tmp2;
  asm volatile (""
  "1:mov (%1), %2\n"
  "test %2, %2\n"
  "jnz 1b\n"
  "mov %0, (%1)\n"
  : "+r"(x), "+r"(addr), "+r"(tmp)
  :
  : "memory", "cc");
//  asm volatile (""
//  "1:xbegin 2f\n"
//  "movq (%1), %2\n"
//  "test %2, %2\n"
//  "jz 3f\n"
//  "xabort $0\n"
//  "2:movq (%1), %2\n"
//  "pause\n"
//  "jmp 1b\n"
//  "3:movq %0, (%1)\n"
//  "xend"
//  : "+r"(x), "+r"(addr), "+r"(tmp), "+a"(tmp2)
//  :
//  : "memory", "cc");
//  while (outpipe->buffer);
//  auto* x = buffer.release();
//  //printf("%p write %p\n", &outpipe->buffer, x);
//  outpipe->buffer.store(x, std::memory_order_relaxed);
  //printf("%p wrote %p\n", &outpipe->buffer, x);
  std::move(callback)(nullptr);
}

Connection::Connection(std::shared_ptr<Pipe> inpipe, std::shared_ptr<Pipe> outpipe) : inpipe(std::move(inpipe)), outpipe(std::move(outpipe)) {
  thread = std::thread([this, inpipe = this->inpipe, outpipe = this->outpipe]() {
    tensorpipe::FunctionPointer callback;
    while (!readCallback);
    callback = readCallback;
    while (true) {
//      while (!inpipe->buffer.load(std::memory_order_relaxed)) {
//        if (outpipe->dead.load(std::memory_order_relaxed)) {
//          return;
//        }
//      }
//      //BufferHandle buffer(std::exchange(inpipe->buffer, nullptr));
//      BufferHandle buffer(inpipe->buffer.exchange(nullptr));
//      while (!readCallback.load(std::memory_order_relaxed)) {
//        if (outpipe->dead.load(std::memory_order_relaxed)) {
//          return;
//        }
//      }
//      (Function<void (Error*, BufferHandle&&)>(readCallback.exchange(nullptr, std::memory_order_relaxed)))(nullptr, std::move(buffer));
      Buffer* buf;
      //tensorpipe::FunctionPointer callback;
      long dead = 0;
      long tmp;
      long one = 1;
      auto* addr = &inpipe->buffer;
      asm (""
      "1:movq (%1), %0\n"
      "testq %0, %0\n"
      "jz 1b\n"
      "xorq %2, %2\n"
      "movq %2, (%1)\n"
      : "+r"(buf), "+r"(addr), "+r"(tmp)
      :
      : "memory", "cc");
//      asm (""
//      "1:xbegin 2f\n"
//      "movq (%5), %1\n"
//      "movq (%6), %2\n"
//      "xorq %0, %0\n"
//      "xorq %3, %3\n"
//      "testq %1, %1\n"
//      "cmovnzq %4, %0\n"
//      "testq %2, %2\n"
//      "cmovnzq %4, %3\n"
//      "andq %0, %3\n"
//      "testq %0, %0\n"
//      "jnz 4f\n"
//      "xabort $0\n"
//      "2:movq (%5), %0\n"
//      "movq (%6), %0\n"
//      "jmp 1b\n"
//      "4:movq $0, (%5)\n"
//      "movq $0, (%6)\n"
//      "xend\n"
//      "xorq %3, %3\n"
//      "prefetcht0 (%2)\n"
//      "prefetcht0 (%1)\n"
//      : "+a"(tmp), "+r"(buf), "+r"(callback), "+r"(dead), "+r"(one)
//      : "r"(&inpipe->buffer), "r"(&readCallback), "r"(&inpipe->dead)
//      : "memory", "cc");
//      char dead;
//      char tmp;
//      char one;
//      asm (""
//      "1:xor %1, %1\n"
//      "xor %2, %2\n"
//      "xor %%rax, %%rax\n"
//      "mov (%5), %%rax\n"
//      "mov (%4), %%rax\n"
//      "test %%rax, %%rax\n"
//      "jz 11f\n"
//      "xacquire lock cmpxchgq %1, (%4)\n"
//      "jnz 11f\n"
//      "mov %%rax, %1\n"
//      "mov (%5), %%rax\n"
//      "test %%rax, %%rax\n"
//      "jz 6f\n"
//      "xacquire lock cmpxchgq %2, (%5)\n"
//      "jnz 6f\n"
//      "mov %%rax, %2\n"
//      "jmp 5f\n"
//      "11:mov (%5), %%rax\n"
//      "test %%rax, %%rax\n"
//      "jz 4f\n"
//      "xacquire lock cmpxchgq %2, (%5)\n"
//      "jnz 4f\n"
//      "mov %%rax, %2\n"
//      "jmp 7f\n"
//      "4:movb (%6), %0\n"
//      "test %1, %1\n"
//      "jz 2f\n"
//      "test %2, %2\n"
//      "jz 3f\n"
//      "jmp 1b\n"
//      "2:pause\n"
//      "movq (%4), %1\n"
//      "testb %0, %0\n"
//      "jz 4b\n"
//      "jmp 10f\n"
//      "3:pause\n"
//      "movq (%5), %2\n"
//      "testb %0, %0\n"
//      "jz 4b\n"
//      "jmp 10f\n"
//      "6:int3\npause\n" // check %2
//      "movb (%6), %0\n"
//      "movq (%5), %2\n"
//      "testb %0, %0\n"
//      "jnz 8f\n"
//      "test %2, %2\n"
//      "jz 6b\n"
//      "xor %2, %2\n"
//      "xacquire lock xchg %2, (%5)\n"
//      "test %2, %2\n"
//      "jz 6b\n"
//      "jmp 5f\n"
//      "7:pause\n" // check %1
//      "movb (%6), %0\n"
//      "movq (%4), %1\n"
//      "testb %0, %0\n"
//      "jnz 9f\n"
//      "test %1, %1\n"
//      "jz 7b\n"
//      "xor %1, %1\n"
//      "xacquire lock xchg %1, (%4)\n"
//      "test %1, %1\n"
//      "jz 7b\n"
//      "jmp 5f\n"
//      "8:xor %2, %2\n"
//      "jmp 5f\n"
//      "9:xor %1, %1\n"
//      "jmp 5f\n"
//      "10:xor %1, %1\n"
//      "xor %2, %2\n"
//      "5:\n"
//      : "+a"(tmp), "+r"(buf), "+r"(callback), "+r"(dead)
//      : "r"(&inpipe->buffer), "r"(&readCallback), "r"(&inpipe->dead)
//      : "memory", "cc");
//      printf("%p buf is %p\n", &inpipe->buffer, buf);
//      printf("post buffer is %p\n", inpipe->buffer.load());
//      printf("%p callback is %p\n", &readCallback, callback);
//      printf("post callback is %p\n", readCallback.load());
      if (dead) {
        BufferHandle buffer(buf);
        if (callback) {
          Error err("connection closed");
          (Function<void (Error*, BufferHandle&&)>(callback))(&err, nullptr);
        }
        break;
      }
      BufferHandle buffer(buf);
      //(Function<void (Error*, BufferHandle&&)>(callback))(nullptr, std::move(buffer));
      Function<bool(Error*, BufferHandle&&)> f(callback);
      if (f(nullptr, std::move(buffer))) {
        f.release();
      } else {
        readCallback.store(nullptr);
      }
    }
  });
}

}

namespace network {

struct Connection;
struct Listener;

struct Context {
  Connection connect(std::string_view url);
  Listener listen(std::string_view url);

  ::network::Network nw;

  std::thread thread, thread2, thread3;
  Context() {
    thread = std::thread([this]() {
      while (true) {
        nw.run_one();
      }
    });
//    thread2 = std::thread([this]() {
//      while (true) {
//        nw.run_one();
//      }
//    });
//    thread3 = std::thread([this]() {
//      while (true) {
//        nw.run_one();
//      }
//    });
  }
};

struct Listener {
  ::network::Server server;

  void accept(Function<void(Error*, Connection&&)>&& callback);
};


struct Connection {
  ::network::Peer peer;

  void close() {
    peer.post_close();
  }
  void read(Function<bool(Error*, const void*, size_t)>&& callback) {
    peer.setOnMessage([callback = std::move(callback)](const void* ptr, size_t len) {
      callback(nullptr, ptr, len);
    });
  }
  void write(const void* ptr, size_t len, Function<void(Error*)>&& callback) {
    peer.sendMessage(ptr, len);
    callback(nullptr);
  }
};

void Listener::accept(Function<void (Error*, Connection&&)>&& callback) {
  server.setOnPeer([callback = std::move(callback)](auto&& peer) {
    Connection conn;
    conn.peer = std::move(peer);
    callback(nullptr, std::move(conn));
  });
}

Connection Context::connect(std::string_view url) {
  Connection c;
  c.peer = nw.connect(url);
  return c;
}
Listener Context::listen(std::string_view url) {
  Listener l;
  l.server = nw.listen(url);
  return l;
}

}

struct API_InProcess {
  using Context = inproc::Context;
  using Connection = std::unique_ptr<inproc::Connection>;
  using Listener = std::unique_ptr<inproc::Listener>;

  static constexpr bool supportsBuffer = true;
  static constexpr bool persistentRead = true;
  static constexpr bool persistentAccept = true;

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
};

struct API_TPSHM {
  using Context = tensorpipe::transport::shm::Context;
  using Connection = std::shared_ptr<tensorpipe::transport::Connection>;
  using Listener = std::shared_ptr<tensorpipe::transport::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;

  static auto& cast(Connection& x) {
    return (tensorpipe::transport::shm::Connection&)*x;
  }
  static auto& cast(Listener& x) {
    return (tensorpipe::transport::shm::Listener&)*x;
  }
  static std::string errstr(const tensorpipe::Error& err) {
    return err.what();
  }
};

struct API_TPUV {
  using Context = tensorpipe::transport::uv::Context;
  using Connection = std::shared_ptr<tensorpipe::transport::Connection>;
  using Listener = std::shared_ptr<tensorpipe::transport::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;

  static auto& cast(Connection& x) {
    return (tensorpipe::transport::uv::Connection&)*x;
  }
  static auto& cast(Listener& x) {
    return (tensorpipe::transport::uv::Listener&)*x;
  }
  static std::string errstr(const tensorpipe::Error& err) {
    return err.what();
  }
};

struct API_Network {
  using Context = network::Context;
  using Connection = network::Connection;
  using Listener = network::Listener;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = true;
  static constexpr bool persistentAccept = true;

  template<typename T> static T& cast(T& v) {
    return v;
  }
};

template<typename API>
struct APIWrapper: API {
  template<typename T, typename X = API>
  static auto errstr(T&& err) -> decltype(X::errstr(std::forward<T>(err))) {
    return API::errstr(std::forward<T>(err));
  }
  static std::string errstr(const char* str) {
    return std::string(str);
  }
  static std::string errstr(std::string_view str) {
    return std::string(str);
  }
  static std::string errstr(std::string&& str) {
    return std::move(str);
  }
};

template<typename T> struct RpcImpl;

struct RpcConnection::Impl {
  virtual ~Impl() = default;
  virtual void sendRequest(BufferHandle&& buffer, uint32_t fid, rpc::Rpc::ResponseCallback response) = 0;
};

template<typename API>
struct RpcConnectionImpl : RpcConnection::Impl {
  RpcConnectionImpl(RpcImpl<API>& rpc, typename API::Connection&& connection) : rpc(rpc), connection(std::move(connection)) {}
  RpcImpl<API>& rpc;

  typename API::Connection connection;

  std::atomic<uint32_t> sequenceId{random<uint32_t>(0, std::numeric_limits<uint32_t>::max())};

  struct ActiveRequest {
    Rpc::ResponseCallback response;
  };

  SpinMutex reqMutex;
  std::unordered_map<uint32_t, ActiveRequest> activeRequests;

  std::atomic_int activeOps{0};
  std::atomic_bool dead{false};

  struct Me {
    RpcConnectionImpl* me = nullptr;
    Me() = default;
    Me(RpcConnectionImpl* me) : me(me) {
      if (me) {
        me->activeOps.fetch_add(1, std::memory_order_relaxed);
      }
    }
    Me(const Me&) = delete;
    Me(Me&& n) noexcept {
      me = std::exchange(n.me, nullptr);
    }
    Me&operator=(const Me&) = delete;
    Me&operator=(Me&& n) noexcept {
      std::swap(me, n.me);
      return *this;
    }
    ~Me() {
      if (me) {
        me->activeOps.fetch_sub(1, std::memory_order_release);
      }
    }
    RpcConnectionImpl* operator->() const {
      return me;
    }
  };

  ~RpcConnectionImpl() {
    dead = true;
    API::cast(connection).close();
    //while (activeOps.load(std::memory_order_acquire));
  }

  //BufferHandle readBuffer{allocate<Buffer, std::byte>(4096)};

  void onError(Error* err) {
    std::lock_guard l(reqMutex);
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    for (auto& v : activeRequests) {
      v.second.response(nullptr, 0, err);
    }
    activeRequests.clear();
  }

  template<typename E>
  void onError(E&& error) {
    Error err(API::errstr(error));
    onError(&err);
  }

  void onData(const std::byte* ptr, size_t len) {
    if (len < sizeof(uint32_t) * 2) {
      onError("Received not enough data");
      return;
    }
    uint32_t rid;
    std::memcpy(&rid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    uint32_t fid;
    std::memcpy(&fid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    //printf("onData rid %#x fid %#x\n", rid, fid);
    if (rid & 1) {
      rpc.onRequest(*this, rid, fid, ptr, len);
    } else {
      Rpc::ResponseCallback response;
      {
        std::lock_guard l(reqMutex);
        auto i = activeRequests.find(rid | 1);
        if (i != activeRequests.end()) {
          response = std::move(i->second.response);
        }
      }
      if (response) {
        if (fid == 0) {
          Error err("Remote function not found");
          std::move(response)(nullptr, 0, &err);
        } else if (fid == 2) {
          std::string_view str;
          deserializeBuffer(ptr, len, str);
          Error err{"Remote exception during RPC call: " + std::string(str)};
          std::move(response)(nullptr, 0, &err);
        } else {
          std::move(response)(ptr, len, nullptr);
        }
      }
    }
  }

  void onData(const void* ptr, size_t len) {
    return onData((std::byte*)ptr, len);
  }
  void onData(BufferHandle&& buffer) {
    return onData(buffer->data(), buffer->size);
  }

  void read(Me&& me) {
    //printf("read %p\n", this);
    API::cast(connection).read([me = std::move(me)](auto&& error, auto&&... args) mutable {
      //printf("%p got data\n");
      if (me->dead.load(std::memory_order_relaxed)) {
        if constexpr (API::persistentRead) {
          return false;
        } else {
          return;
        }
      }
      if (error) {
        me->onError(error);
        if constexpr (API::persistentRead) {
          return false;
        }
      } else {
        me->onData(std::forward<decltype(args)>(args)...);
        if constexpr (API::persistentRead) {
          return true;
        } else {
          me->read(std::move(me));
        }
      }
    });
  }

  void start() {
    read(this);
  }

  void send(BufferHandle&& buffer) {
    auto* ptr = buffer->data();
    size_t size = buffer->size;
    if constexpr (API::supportsBuffer) {
      API::cast(connection).write(std::move(buffer), [me = Me(this)](auto&& error) mutable {
        if (error) {
          me->onError(error);
        }
      });
    } else {
      API::cast(connection).write(ptr, size, [buffer = std::move(buffer), me = Me(this)](auto&& error) mutable {
        if (error) {
          me->onError(error);
        }
      });
    }
  }

  virtual void sendRequest(BufferHandle&& buffer, uint32_t fid, rpc::Rpc::ResponseCallback response) override {
    auto* ptr = dataptr<std::byte>(&*buffer);
    uint32_t rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
    std::memcpy(ptr, &rid, sizeof(rid));
    ptr += sizeof(rid);
    std::memcpy(ptr, &fid, sizeof(fid));
    {
      //printf("send request %#x %#x\n", fid, rid);
      std::lock_guard l(reqMutex);
      if (dead.load(std::memory_order_relaxed)) {
        Error err("RPC call: connection is closed");
        std::move(response)(nullptr, 0, &err);
        return;
      }
      send(std::move(buffer));
      ActiveRequest& q = activeRequests[rid];
      q.response = std::move(response);
    }
  }

};

struct RpcListener::Impl {
  virtual ~Impl() = default;
  virtual void accept(Function<void(RpcConnection*, Error*)>&& callback) = 0;
};

template<typename API>
struct RpcListenerImpl : RpcListener::Impl {
  RpcListenerImpl(RpcImpl<API>& rpc, typename API::Listener&& listener) : rpc(rpc), listener(std::move(listener)) {}
  RpcImpl<API>& rpc;
  typename API::Listener listener;

  virtual void accept(Function<void(RpcConnection*, Error*)>&& callback) override {
    API::cast(listener).accept([this, callback = std::move(callback)](auto&& error, auto&& conn) mutable {
      if (error) {
        if constexpr (std::is_same_v<std::decay_t<decltype(error)>, Error*>) {
          callback(nullptr, error);
        } else {
          Error err(API::errstr(error));
          callback(nullptr, &err);
        }
      } else {
        RpcConnection c;
        std::unique_ptr<RpcConnectionImpl<API>, RpcDeleter<RpcConnection::Impl>> ci(new RpcConnectionImpl<API>(rpc, std::move(conn)));
        ci->start();
        c.impl_ = std::move(ci);
        callback(&c, nullptr);
        if (!API::persistentAccept) {
          accept(std::move(callback));
        }
      }
    });
  }
};

template<>
void RpcDeleter<RpcConnection::Impl>::operator()(RpcConnection::Impl* ptr) const noexcept {
  delete ptr;
}
template<>
void RpcDeleter<RpcListener::Impl>::operator()(RpcListener::Impl* ptr) const noexcept {
  delete ptr;
}

struct Rpc::Impl {
  virtual ~Impl() = default;

  SpinMutex mutex_;
  std::unordered_set<std::string> funcnames_;
  std::unordered_map<std::string_view, uint32_t> funcIds_;
  std::vector<std::unique_ptr<Rpc::FBase>> funcs_;
  static constexpr size_t maxFunctions_  = 0x100000;
  uint32_t baseFuncId_ = random<uint32_t>(1, std::numeric_limits<uint32_t>::max() - maxFunctions_);
  uint32_t nextFuncIndex_ = 0;

  SpinMutex remoteFuncsMutex_;
  std::unordered_map<std::string_view, Rpc::RemoteFunction> remoteFuncs_;

  uint32_t functionId(std::string_view name) {
    std::lock_guard l(remoteFuncsMutex_);
    auto i = remoteFuncs_.find(name);
    if (i != remoteFuncs_.end()) {
      return i->second.id;
    } else {
      return 0;
    }
  }
  void define(std::string_view name, std::unique_ptr<Rpc::FBase>&& f) {
    std::lock_guard l(mutex_);
    if (nextFuncIndex_ >= maxFunctions_) {
      throw Error("Too many RPC functions defined");
    }
    uint32_t id = baseFuncId_ + nextFuncIndex_++;
    size_t index = id - baseFuncId_;
    if (funcs_.size() <= index) {
      funcs_.resize(std::max(index + 1, funcs_.size() + funcs_.size() / 2));
    }
    funcIds_[*funcnames_.emplace(name).first] = id;
    funcs_[index] = std::move(f);
  }
  std::string_view persistentString(std::string_view name) {
    std::lock_guard l(mutex_);
    return *funcnames_.emplace(name).first;
  }
  void setRemoteFunc(std::string_view name, Rpc::RemoteFunction* rf) {
    std::lock_guard l(remoteFuncsMutex_);
    remoteFuncs_[name] = *rf;
  }
  virtual RpcConnection connect(std::string_view url) = 0;
  virtual RpcListener listen(std::string_view url) = 0;
  virtual void onRequest(RpcConnection::Impl& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) = 0;
};

template<typename API>
struct RpcImpl : Rpc::Impl {
  typename API::Context context;

  virtual RpcConnection connect(std::string_view url) override {
    RpcConnection c;
    std::unique_ptr<RpcConnectionImpl<API>, RpcDeleter<RpcConnection::Impl>> ci(new RpcConnectionImpl<API>(*this, context.connect(std::string(url))));
    ci->start();
    c.impl_ = std::move(ci);
    return c;
  }

  virtual RpcListener listen(std::string_view url) override {
    RpcListener r;
    std::unique_ptr<RpcListenerImpl<API>, RpcDeleter<RpcListener::Impl>> i(new RpcListenerImpl<API>(*this, context.listen(std::string(url))));
    r.impl_ = std::move(i);
    return r;
  }

  virtual void onRequest(RpcConnection::Impl& connx, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) override {
    //printf("onRequest rid %#x fid %#x %p %d\n", rid, fid, ptr, len);
    auto& conn = (RpcConnectionImpl<API>&)connx;
    rid &= ~(uint32_t)1;
    if (fid == 0) {
      std::string_view name;
      deserializeBuffer(ptr, len, name);
      Rpc::RemoteFunction rf;
      {
        std::lock_guard l(mutex_);
        auto i = funcIds_.find(name);
        if (i != funcIds_.end()) {
          rf.id = i->second;
        }
      }
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, rf.id, rf);
      conn.send(std::move(buffer));
    } else {
      size_t index = fid - baseFuncId_;
      //printf("fid %#x has index %d\n", fid, index);
      if (index >= funcs_.size()) {
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, 0);
        conn.send(std::move(buffer));
      } else {
        Rpc::FBase* f;
        {
          std::lock_guard l(mutex_);
          f = &*funcs_[index];
        }
        BufferHandle buffer;
        //printf("call with len %d\n", len);
        f->call(ptr, len, buffer);
        auto* ptr = dataptr<std::byte>(&*buffer);
        std::memcpy(ptr, &rid, sizeof(rid));
//        ptr += sizeof(rid);
//        std::memcpy(ptr, &fid, sizeof(fid));
        conn.send(std::move(buffer));
      }
    }
  }

};

void RpcListener::accept(Function<void(RpcConnection*, Error*)>&& callback) {
  impl_->accept(std::move(callback));
}

Rpc::Rpc() {
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_Network>>>();
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_TPUV>>>();
  impl_ = std::make_unique<RpcImpl<APIWrapper<API_TPSHM>>>();
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_InProcess>>>();
}
Rpc::~Rpc() {}

RpcListener Rpc::listen(std::string_view url) {
  return impl_->listen(url);
}
RpcConnection Rpc::connect(std::string_view url) {
  return impl_->connect(url);
}
void Rpc::sendRequest(RpcConnection& conn, BufferHandle&& buffer, uint32_t fid, ResponseCallback response) {
  conn.impl_->sendRequest(std::move(buffer), fid, std::move(response));
}
uint32_t Rpc::functionId(std::string_view name) {
  return impl_->functionId(name);
}
void Rpc::define(std::string_view name, std::unique_ptr<FBase>&& f) {
  impl_->define(name, std::move(f));
}
std::string_view Rpc::persistentString(std::string_view str) {
  return impl_->persistentString(str);
}
void Rpc::setRemoteFunc(std::string_view name, RemoteFunction *rf) {
  impl_->setRemoteFunc(name, rf);
}

}
