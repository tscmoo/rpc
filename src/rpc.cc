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

  alignas(64) std::atomic<tensorpipe::FunctionPointer> readCallback = nullptr;
  alignas(64) std::atomic<tensorpipe::FunctionPointer> writeCallback = nullptr;
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
  std::atomic_int activeOps{0};
};

template<typename T>
struct Me {
  T* me = nullptr;
  Me() = default;
  Me(T* me) : me(me) {
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
  T* operator->() const {
    return me;
  }
};

template<typename T>
auto makeMe(T* v) {
  return Me<T>(v);
}

template<typename API>
struct RpcConnectionImpl : RpcConnection::Impl {
  RpcConnectionImpl(RpcImpl<API>& rpc, typename API::Connection&& connection) : rpc(rpc), connection(std::move(connection)) {}
  RpcImpl<API>& rpc;

  typename API::Connection connection;

  std::atomic_bool dead{false};

  ~RpcConnectionImpl() {
    dead = true;
    API::cast(connection).close();
    while (activeOps.load(std::memory_order_acquire));
  }

  void onError(Error* err) {
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
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
    //printf("onData rid %#x fid %#x len %d\n", rid, fid, len);
    if (rid & 1) {
      rpc.onRequest(*this, rid, fid, ptr, len);
    } else {
      rpc.onResponse(*this, rid, fid, ptr, len);
    }
  }

  void onData(const void* ptr, size_t len) {
    return onData((std::byte*)ptr, len);
  }
  void onData(BufferHandle&& buffer) {
    return onData(buffer->data(), buffer->size);
  }

  void read(Me<RpcConnectionImpl>&& me) {
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

  template<typename Buffer>
  void send(Buffer buffer) {
    auto* ptr = buffer->data();
    size_t size = buffer->size;
    //printf("%p :: send %d bytes\n", this, size);
    if (random(0, 1) == 0) {
      return;
    }
    if constexpr (API::supportsBuffer) {
      BufferHandle hx = allocate<rpc::Buffer, std::byte> (buffer->size);
      hx->size = buffer->size;
      std::memcpy(hx->data(), buffer->data(), hx->size);
      API::cast(connection).write(std::move(hx), [me = Me<RpcConnectionImpl>(this)](auto&& error) mutable {
        if (error) {
          me->onError(error);
        }
      });
    } else {
      API::cast(connection).write(ptr, size, [buffer = std::move(buffer), me = Me<RpcConnectionImpl>(this)](auto&& error) mutable {
        if (error) {
          me->onError(error);
        }
      });
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

  alignas(64) SpinMutex mutex_;
  std::unordered_set<std::string> funcnames_;
  std::unordered_map<std::string_view, uint32_t> funcIds_;
  std::vector<std::unique_ptr<Rpc::FBase>> funcs_;
  static constexpr size_t maxFunctions_  = 0x100000;
  uint32_t baseFuncId_ = random<uint32_t>(Rpc::reqCallOffset, std::numeric_limits<uint32_t>::max() - maxFunctions_);
  uint32_t nextFuncIndex_ = 0;

  alignas(64) SpinMutex remoteFuncsMutex_;
  std::unordered_map<std::string_view, Rpc::RemoteFunction> remoteFuncs_;

  struct Incoming {
    Incoming* prev = nullptr;
    Incoming* next = nullptr;
    uint32_t rid;
    std::chrono::steady_clock::time_point responseTimestamp;
    SharedBufferHandle response;
  };

  struct IncomingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Incoming> map;
  };

  alignas(64) std::array<IncomingBucket, 0x10> incoming_;
  alignas(64) SpinMutex incomingFifoMutex_;
  Incoming incomingFifo_;
  std::atomic_size_t totalResponseSize_ = 0;

  std::thread timeoutThread_;
  alignas(64) Semaphore timeoutSem_;
  std::atomic<std::chrono::steady_clock::time_point> timeout_ = std::chrono::steady_clock::now();
  std::atomic<bool> timeoutDead_ = false;

  struct Outgoing {
    Outgoing* prev = nullptr;
    Outgoing* next = nullptr;
    std::chrono::steady_clock::time_point requestTimestamp;
    std::chrono::steady_clock::time_point timeout;
    uint32_t rid = 0;
    bool acked = false;
    Rpc::ResponseCallback response;
    Me<RpcConnection::Impl> conn;
  };

  struct OutgoingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Outgoing> map;
  };

  alignas(64) std::array<OutgoingBucket, 0x10> outgoing_;
  alignas(64) SpinMutex outgoingFifoMutex_;
  Incoming outgoingFifo_;
  alignas(64) std::atomic<uint32_t> sequenceId{random<uint32_t>(0, std::numeric_limits<uint32_t>::max())};

  Impl() {
    incomingFifo_.next = &incomingFifo_;
    incomingFifo_.prev = &incomingFifo_;
    outgoingFifo_.next = &outgoingFifo_;
    outgoingFifo_.prev = &outgoingFifo_;
  }
  virtual ~Impl() {
    if (timeoutThread_.joinable()) {
      timeoutThread_.join();
    }
  }

  template<typename T>
  auto& getBucket(T& arr, uint32_t rid) {
    return arr[(rid >> 1) % arr.size()];
  }

  void processTimeout(Outgoing& o) {
    if (!o.acked) {
      BufferHandle buffer;
      serializeToBuffer(buffer, o.rid, Rpc::reqAck);
      o.conn->send(std::move(buffer));
    }
  }

  void startTimeoutThread() {
    timeoutThread_ = std::thread([this]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      printf("timeout thread running!\n");
      while (!timeoutDead_.load(std::memory_order_relaxed)) {
        auto now = std::chrono::steady_clock::now();
        auto timeout = timeout_.load(std::memory_order_relaxed);
        printf("timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
        if (now < timeout) {
          printf("%p sleeping for %d\n", this, std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
          timeoutSem_.wait_for(timeout - now);
          printf("%p woke up\n", this);
          continue;
        }
        auto newTimeout = now + std::chrono::seconds(5);
        timeout_.store(newTimeout);
        for (auto& b : outgoing_) {
          std::lock_guard l(b.mutex);
          for (auto& v : b.map) {
            newTimeout = std::min(newTimeout, v.second.timeout);
            if (now >= v.second.timeout) {
              processTimeout(v.second);
            }
          }
        }
        timeout = timeout_.load(std::memory_order_relaxed);
        while (newTimeout < timeout && !timeout_.compare_exchange_weak(timeout, newTimeout));
        printf("new timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(newTimeout - now).count());
      }
    });
  }

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
  virtual void onResponse(RpcConnection::Impl& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) = 0;
  virtual void sendRequest(RpcConnection::Impl& conn, BufferHandle buffer, uint32_t fid, rpc::Rpc::ResponseCallback response) = 0;
};

namespace {
template<typename T>
void listInsert(T* at, T* item) {
  T* next = at;
  T* prev = at->prev;
  next->prev = item;
  prev->next = item;
  item->next = next;
  item->prev = prev;
}
template<typename T>
void listErase(T* at) {
  T* next = at->next;
  T* prev = at->prev;
  next->prev = prev;
  prev->next = next;
}
}

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

  virtual void onRequest(RpcConnection::Impl& connx, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) noexcept override {
    //printf("onRequest rid %#x fid %#x %p %d\n", rid, fid, ptr, len);
    auto& conn = (RpcConnectionImpl<API>&)connx;
    rid &= ~(uint32_t)1;
    if (fid == Rpc::reqFindFunction) {
      //printf("find function\n");
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
      serializeToBuffer(buffer, rid, Rpc::reqSuccess, rf);
      conn.send(std::move(buffer));
    } else if (fid == Rpc::reqAck) {
      // Peer acknowledged that it has received the response
      // (return value of an RPC call)
      IncomingBucket& bucket = incoming_[(rid >> 1) % incoming_.size()];
      std::lock_guard l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i != bucket.map.end()) {
        auto& x = i->second;
        if (x.response) {
          std::lock_guard l2(incomingFifoMutex_);
          totalResponseSize_.fetch_sub(x.response->size, std::memory_order_relaxed);
          listErase(&x);
        }
        bucket.map.erase(i);
      }
    } else if (fid == Rpc::reqPoke) {
      // Peer is poking us to check the status of an RPC call
      printf("got poke for %#x\n", rid);
      IncomingBucket& bucket = getBucket(incoming_, rid);
      std::unique_lock l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i == bucket.map.end()) {
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, Rpc::reqNotFound);
        conn.send(std::move(buffer));
      } else {
        auto& x = i->second;
        if (x.response) {
          SharedBufferHandle r = x.response;
          l.unlock();
          printf("re-sending response of %d bytes\n", r->size);
          conn.send(r);
        } else {
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqAck);
          conn.send(std::move(buffer));
        }
        return;
      }
    } else if (fid >= (uint32_t)Rpc::reqCallOffset) {
      // RPC call
      size_t index = fid - baseFuncId_;
      Rpc::FBase* f = nullptr;
      {
        std::lock_guard l(mutex_);
        if (index < funcs_.size()) {
          f = &*funcs_[index];
        }
      }
      if (!f) {
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, Rpc::reqFunctionNotFound);
        conn.send(std::move(buffer));
      } else {
        {
          IncomingBucket& bucket = getBucket(incoming_, rid);
          std::unique_lock l(bucket.mutex);
          auto i = bucket.map.try_emplace(rid);
          auto& x = i.first->second;
          if (i.second) {
            x.rid = rid;
          } else {
            if (x.response) {
              SharedBufferHandle r = x.response;
              l.unlock();
              printf("re-sending response of %d bytes\n", r->size);
              conn.send(r);
            } else {
              l.unlock();
              BufferHandle buffer;
              serializeToBuffer(buffer, rid, Rpc::reqAck);
              conn.send(std::move(buffer));
            }
            return;
          }
        }
        //BufferHandle buffer;
        //serializeToBuffer(buffer, rid, Rpc::reqAck);
        //conn.send(std::move(buffer));
        //printf("call with len %d\n", len);
        BufferHandle inbuffer = allocate<Buffer, std::byte>(len);
        std::memcpy(inbuffer->data(), ptr, len);
        inbuffer->size = len;
        f->call(std::move(inbuffer), [this, rid, conn = makeMe(&conn)](BufferHandle outbuffer) {
          auto* ptr = dataptr<std::byte>(&*outbuffer);
          std::memcpy(ptr, &rid, sizeof(rid));
  //        ptr += sizeof(rid);
  //        std::memcpy(ptr, &fid, sizeof(fid));
          SharedBufferHandle shared(outbuffer.release());
          //printf("sending response of %d bytes (%p)\n", shared->size, &*shared);
          conn->send(shared);

          auto now = std::chrono::steady_clock::now();
          IncomingBucket& bucket = getBucket(incoming_, rid);
          std::lock_guard l(bucket.mutex);
          auto& x = bucket.map[rid];
          x.responseTimestamp = now;
          size_t totalResponseSize = totalResponseSize_ += shared->size;
          x.response = std::move(shared);
          std::lock_guard l2(incomingFifoMutex_);
          listInsert(incomingFifo_.prev, &x);

          // Erase outgoing data if it has not been acknowledged within a
          // certain time period. This prevents us from using resources
          // for peers that are permanently gone.
          auto timeout = std::chrono::seconds(300);
          if (now - incomingFifo_.next->responseTimestamp >= std::chrono::seconds(5)) {
            if (totalResponseSize < 1024 * 1024 && incoming_.size() < 1024) {
              timeout = std::chrono::seconds(1800);
            } else if (totalResponseSize >= 1024 * 1024 * 1024 || incoming_.size() >= 1024 * 1024) {
              timeout = std::chrono::seconds(5);
            }
            timeout = std::chrono::seconds(0);
            while (incomingFifo_.next != &incomingFifo_ && now - incomingFifo_.next->responseTimestamp >= timeout) {
              auto* i = incomingFifo_.next;
              if (i->response) {
                totalResponseSize_ -= i->response->size;
              }
              listErase(i);
              auto& iBucket = getBucket(incoming_, i->rid);
              if (&iBucket != &bucket) {
                std::lock_guard l3(iBucket.mutex);
                iBucket.map.erase(i->rid);
              } else {
                iBucket.map.erase(i->rid);
              }
            }
          }
        });
      }
    }
  }

  virtual void onResponse(RpcConnection::Impl &connx, uint32_t rid, uint32_t fid, const std::byte *ptr, size_t len) override {
    auto& conn = (RpcConnectionImpl<API>&)connx;
    rid |= 1;
    if (fid == Rpc::reqAck) {
      //printf("got req ack, cool\n");
      return;
    }
    BufferHandle buffer;
    serializeToBuffer(buffer, rid, Rpc::reqAck);
    //printf("ack response for rid %#x\n", rid);
    conn.send(std::move(buffer));
    Rpc::ResponseCallback response;
    {
      auto& oBucket = getBucket(outgoing_, rid);
      std::lock_guard l(oBucket.mutex);
      auto i = oBucket.map.find(rid);
      if (i != oBucket.map.end()) {
        response = std::move(i->second.response);
        oBucket.map.erase(i);
      }
    }
    if (response) {
      if (fid == Rpc::reqFunctionNotFound) {
        Error err("Remote function not found");
        std::move(response)(nullptr, 0, &err);
      } else if (fid == Rpc::reqError) {
        std::string_view str;
        deserializeBuffer(ptr, len, str);
        Error err{"Remote exception during RPC call: " + std::string(str)};
        std::move(response)(nullptr, 0, &err);
      } else if (fid == Rpc::reqSuccess) {
        std::move(response)(ptr, len, nullptr);
      }
    }
  }

  virtual void sendRequest(RpcConnection::Impl &connx, BufferHandle buffer, uint32_t fid, rpc::Rpc::ResponseCallback response) override {
    auto& conn = (RpcConnectionImpl<API>&)connx;
    auto* ptr = dataptr<std::byte>(&*buffer);
    uint32_t rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
    std::memcpy(ptr, &rid, sizeof(rid));
    ptr += sizeof(rid);
    std::memcpy(ptr, &fid, sizeof(fid));
    auto now = std::chrono::steady_clock::now();
    auto myTimeout = now + std::chrono::seconds(1);
    {
      //printf("send request %#x %#x\n", rid, fid);
      if (conn.dead.load(std::memory_order_relaxed)) {
        Error err("RPC call: connection is closed");
        std::move(response)(nullptr, 0, &err);
        return;
      }
      conn.send(std::move(buffer));
      auto& oBucket = getBucket(outgoing_, rid);
      std::lock_guard l(oBucket.mutex);
      auto& q = oBucket.map[rid];
      q.conn = makeMe(&connx);
      q.requestTimestamp = now;
      q.timeout = myTimeout;
      q.response = std::move(response);
    }
    printf("myTimeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(myTimeout - now).count());
    static_assert(std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
    auto timeout = timeout_.load(std::memory_order_acquire);
    while (myTimeout < timeout) {
      if (timeout_.compare_exchange_weak(timeout, myTimeout)) {
        printf("timeout set to %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout_.load() - now).count());
        printf("waking up %p\n", this);
        timeoutSem_.post();
      }
    }
    printf("timeout post is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout_.load() - now).count());
    if (!timeoutThread_.joinable()) {
      startTimeoutThread();
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
  impl_->sendRequest(*conn.impl_, std::move(buffer), fid, std::move(response));
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
