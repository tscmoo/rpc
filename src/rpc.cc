#include "rpc.h"

#include "network.h"
#include "shm2.h"

#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/transport/listener.h>
#include <tensorpipe/transport/connection.h>
#include <tensorpipe/transport/shm/listener.h>
#include <tensorpipe/transport/shm/connection.h>
#include <tensorpipe/transport/uv/listener.h>
#include <tensorpipe/transport/uv/connection.h>
#include <tensorpipe/transport/ibv/listener.h>
#include <tensorpipe/transport/ibv/connection.h>

#include "fmt/printf.h"

#include <random>
#include <thread>
#include <new>

namespace rpc {

std::mutex logMutex;

template<typename... Args>
void log(const char* fmt, Args&&... args) {
  //return;
  std::lock_guard l(logMutex);
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto* tm = std::localtime(&now);
  char buf[0x40];
  std::strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", tm);
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  if (!s.empty() && s.back() == '\n') {
    fmt::printf("%s: %s", buf, s);
  } else {
    fmt::printf("%s: %s\n", buf, s);
  }
  fflush(stdout);
}

namespace {
auto seedRng() {
  std::random_device dev;
  auto start = std::chrono::high_resolution_clock::now();
  std::seed_seq ss({
    (uint32_t)dev(),
    (uint32_t)dev(),
    (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
    (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count(),
    (uint32_t)std::chrono::system_clock::now().time_since_epoch().count(),
    (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
    (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
    (uint32_t)dev(),
    (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
    (uint32_t)dev(),
    (uint32_t)dev(),
    (uint32_t)std::hash<std::thread::id>()(std::this_thread::get_id())
  });
  return std::mt19937_64(ss);
};
std::mt19937_64 rng{seedRng()};
thread_local std::mt19937_64 threadRng{seedRng()};
SpinMutex rngMutex;
template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T random(T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
  std::lock_guard l(rngMutex);
  return std::uniform_int_distribution<T>(min, max)(rng);
}
template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T threadRandom(T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
  return std::uniform_int_distribution<T>(min, max)(threadRng);
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
  std::unique_ptr<Connection> connect(std::string_view addr);
  std::unique_ptr<Listener> listen(std::string_view addr);
};

struct Listener {
  std::string addr;
  Function<void(Error*, std::unique_ptr<Connection>&&)> acceptFunction;
  Listener(std::string addr) : addr(addr) {
    std::lock_guard l(global.mutex);
    global.listeners[addr] = this;
  }
  ~Listener() {
    std::lock_guard l(global.mutex);
    auto i = global.listeners.find(addr);
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

  alignas(64) std::atomic<rpc_tensorpipe::FunctionPointer> readCallback = nullptr;
  alignas(64) std::atomic<rpc_tensorpipe::FunctionPointer> writeCallback = nullptr;
};

std::unique_ptr<Connection> Context::connect(std::string_view addr) {
  std::lock_guard l(global.mutex);
  auto i = global.listeners.find(std::string(addr));
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

std::unique_ptr<Listener> Context::listen(std::string_view addr) {
  return std::make_unique<Listener>(std::string(addr));
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
  //log("callback at %p set to %p\n", &readCallback, x);
  readCallback.store(x, std::memory_order_relaxed);
}

void Connection::write(BufferHandle&& buffer, Function<void (Error*)>&& callback) {
  auto* x = buffer.release();
  //log("%p write %p\n", &outpipe->buffer, x);
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
//  //log("%p write %p\n", &outpipe->buffer, x);
//  outpipe->buffer.store(x, std::memory_order_relaxed);
  //log("%p wrote %p\n", &outpipe->buffer, x);
  std::move(callback)(nullptr);
}

Connection::Connection(std::shared_ptr<Pipe> inpipe, std::shared_ptr<Pipe> outpipe) : inpipe(std::move(inpipe)), outpipe(std::move(outpipe)) {
  thread = std::thread([this, inpipe = this->inpipe, outpipe = this->outpipe]() {
    rpc_tensorpipe::FunctionPointer callback;
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
      //rpc_tensorpipe::FunctionPointer callback;
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
//      log("%p buf is %p\n", &inpipe->buffer, buf);
//      log("post buffer is %p\n", inpipe->buffer.load());
//      log("%p callback is %p\n", &readCallback, callback);
//      log("post callback is %p\n", readCallback.load());
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
  Connection connect(std::string_view addr);
  Listener listen(std::string_view addr);

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

Connection Context::connect(std::string_view addr) {
  Connection c;
  c.peer = nw.connect(addr);
  return c;
}
Listener Context::listen(std::string_view addr) {
  Listener l;
  l.server = nw.listen(addr);
  return l;
}

}

std::string randomAddress() {
  std::string s;
  for (int i = 0; i != 2; ++i) {
    uint64_t v = random<uint64_t>();
    if (!s.empty()) {
      s += "-";
    }
    for (size_t i = 0; i != 8; ++i) {
      uint64_t sv = v >> (i * 8);
      s += "0123456789abcdef"[(sv >> 4) & 0xf];
      s += "0123456789abcdef"[sv & 0xf];
    }
  }
  return s;
}

struct API_InProcess {
  using Context = inproc::Context;
  using Connection = std::unique_ptr<inproc::Connection>;
  using Listener = std::unique_ptr<inproc::Listener>;

  static constexpr bool supportsBuffer = true;
  static constexpr bool persistentRead = true;
  static constexpr bool persistentAccept = true;
  static constexpr bool addressIsIp = false;
  static constexpr bool singularWrites = false;

  static std::vector<std::string> defaultAddr() {
    return {randomAddress()};
  }
  static std::string localAddr([[maybe_unused]] const Listener& listener, std::string addr) {
    return addr;
  }
  static std::string localAddr([[maybe_unused]] const Connection&) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection&) {
    return "";
  }

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
};

struct API_TPSHM {
  using Context = rpc_tensorpipe::transport::shm::Context;
  using Connection = std::shared_ptr<rpc_tensorpipe::transport::Connection>;
  using Listener = std::shared_ptr<rpc_tensorpipe::transport::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = false;
  static constexpr bool singularWrites = true;

  static std::vector<std::string> defaultAddr() {
    return {randomAddress()};
  }
  static std::string localAddr([[maybe_unused]] const Listener& listener, std::string addr) {
    return addr;
  }
  static std::string localAddr([[maybe_unused]] const Connection&) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection&) {
    return "";
  }

  static auto& cast(Connection& x) {
    return (rpc_tensorpipe::transport::shm::Connection&)*x;
  }
  static auto& cast(Listener& x) {
    return (rpc_tensorpipe::transport::shm::Listener&)*x;
  }
  static std::string errstr(const rpc_tensorpipe::Error& err) {
    return err.what();
  }
};

struct API_TPUV {
  using Context = rpc_tensorpipe::transport::uv::Context;
  using Connection = std::shared_ptr<rpc_tensorpipe::transport::Connection>;
  using Listener = std::shared_ptr<rpc_tensorpipe::transport::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = true;
  static constexpr bool singularWrites = true;

  static std::vector<std::string> defaultAddr() {
    return {"0.0.0.0", "::"};
  }
  static std::string localAddr(const Listener& listener, [[maybe_unused]] std::string addr) {
    return listener->addr();
  }
  static std::string localAddr(const Connection& x) {
    return ((const rpc_tensorpipe::transport::uv::Connection&)*x).localAddr();
  }
  static std::string remoteAddr(const Connection& x) {
    return ((const rpc_tensorpipe::transport::uv::Connection&)*x).remoteAddr();
  }

  static auto& cast(Connection& x) {
    return (rpc_tensorpipe::transport::uv::Connection&)*x;
  }
  static auto& cast(Listener& x) {
    return (rpc_tensorpipe::transport::uv::Listener&)*x;
  }
  static std::string errstr(const rpc_tensorpipe::Error& err) {
    return err.what();
  }
};

struct API_TPIBV {
  using Context = rpc_tensorpipe::transport::ibv::Context;
  using Connection = std::shared_ptr<rpc_tensorpipe::transport::Connection>;
  using Listener = std::shared_ptr<rpc_tensorpipe::transport::Listener>;

  static constexpr bool supportsBuffer = false;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = true;
  static constexpr bool singularWrites = true;

  static std::vector<std::string> defaultAddr() {
    return {"0.0.0.0", "::"};
  }
  static std::string localAddr(const Listener& listener, [[maybe_unused]] std::string addr) {
    return listener->addr();
  }
  static std::string localAddr([[maybe_unused]] const Connection&) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection&) {
    return "";
  }

  static auto& cast(Connection& x) {
    return (rpc_tensorpipe::transport::ibv::Connection&)*x;
  }
  static auto& cast(Listener& x) {
    return (rpc_tensorpipe::transport::ibv::Listener&)*x;
  }
  static std::string errstr(const rpc_tensorpipe::Error& err) {
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

struct API_SHM2 {
  using Context = shm2::Context;
  using Connection = std::unique_ptr<shm2::Connection>;
  using Listener = std::unique_ptr<shm2::Listener>;

  static constexpr bool supportsBuffer = true;
  static constexpr bool persistentRead = false;
  static constexpr bool persistentAccept = false;
  static constexpr bool addressIsIp = false;
  static constexpr bool singularWrites = false;

  static std::vector<std::string> defaultAddr() {
    return {randomAddress()};
  }
  static std::string localAddr([[maybe_unused]] const Listener& listener, std::string addr) {
    return addr;
  }
  static std::string localAddr([[maybe_unused]] const Connection&) {
    return "";
  }
  static std::string remoteAddr([[maybe_unused]] const Connection&) {
    return "";
  }

  static auto& cast(Connection& x) {
    return *x;
  }
  static auto& cast(Listener& x) {
    return *x;
  }
  static std::string errstr(const shm2::Error* err) {
    return err->what();
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

enum class ConnectionType {
  uv,
  shm,
  ibv,
//  inproc,
//  shm2,
  count
};
static std::array<const char*, (int)ConnectionType::count> connectionTypeName = {
  "TCP/IP",
  "Shared memory",
  "InfiniBand",
//  "Shared memory2",
//  "In-process"
};

template<typename API> struct index_t;
template<> struct index_t<API_TPUV> { static constexpr ConnectionType value = ConnectionType::uv; };
template<> struct index_t<API_TPSHM> { static constexpr ConnectionType value = ConnectionType::shm; };
template<> struct index_t<API_TPIBV> { static constexpr ConnectionType value = ConnectionType::ibv; };
//template<> struct index_t<API_SHM2> { static constexpr ConnectionType value = ConnectionType::shm2; };
//template<> struct index_t<API_InProcess> { static constexpr ConnectionType value = ConnectionType::inproc; };
template<typename API> constexpr size_t index = (size_t)index_t<API>::value;

template<typename F>
auto switchOnAPI(ConnectionType t, F&& f) {
  switch (t) {
  case ConnectionType::uv:
    return f(API_TPUV{});
    break;
  case ConnectionType::shm:
    return f(API_TPSHM{});
    break;
  case ConnectionType::ibv:
    return f(API_TPIBV{});
    break;
//  case ConnectionType::shm2:
//    return f(API_SHM2{});
//    break;
//  case ConnectionType::inproc:
//    return f(API_InProcess{});
//    break;
  default:
    std::abort();
  }
}

template<typename F>
auto switchOnScheme(std::string_view str, F&& f) {
  if (str == "uv") {
    return f(API_TPUV{});
  } else if (str == "shm") {
    return f(API_TPSHM{});
  } else if (str == "ibv") {
    return f(API_TPIBV{});
  } else {
    throw std::runtime_error("Unrecognized scheme '" + std::string(str) + "'");
  }
}

template<typename T> struct RpcImpl;

bool addressIsIp(ConnectionType t) {
  return switchOnAPI(t, [](auto api) {
    return decltype(api)::addressIsIp;
  });
}

struct ConnectionTypeInfo {
  std::string_view name;
  std::vector<std::string_view> addr;

  template<typename X>
  void serialize(X& x) {
    x(name, addr);
  }
};

struct RpcConnectionImplBase {
  virtual ~RpcConnectionImplBase() {}
  virtual void close() = 0;

  virtual const std::string& localAddr() const = 0;
  virtual const std::string& remoteAddr() const = 0;
  virtual size_t apiIndex() const = 0;

  std::atomic_bool dead{false};
  std::atomic_int activeOps{0};
  std::chrono::steady_clock::time_point lastReceivedData;
  bool isExplicit = false;
  std::string connectAddr;
  std::chrono::steady_clock::time_point timeWait;
};

struct RpcListenerImplBase {
  virtual ~RpcListenerImplBase() {}

  virtual void close() = 0;
  virtual std::string localAddr() const {
    return "";
  }
};

struct Connection {
  std::atomic<bool> valid = false;
  std::atomic<float> readBanditValue = 0.0f;
  std::atomic<std::chrono::steady_clock::time_point> lastTryConnect = std::chrono::steady_clock::time_point{};
  SpinMutex mutex;
  bool outgoing = false;
  std::string addr;
  bool isExplicit = false;
  std::atomic<bool> hasConn = false;
  std::vector<std::unique_ptr<RpcConnectionImplBase>> conns;

  std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();

  std::vector<std::string_view> remoteAddresses;

  alignas(64) SpinMutex latencyMutex;
  std::chrono::steady_clock::time_point lastUpdateLatency;
  std::atomic<float> runningLatency = 0.0f;
  float writeBanditValue = 0.0f;

  alignas(64) std::atomic<uint64_t> sendCount = 0;
};

struct Listener {
  int activeCount = 0;
  int explicitCount = 0;
  int implicitCount = 0;
  std::vector<std::unique_ptr<RpcListenerImplBase>> listeners;
};

struct PeerId {
  std::array<uint64_t, 2> id{};
  template<typename X>
  void serialize(X& x) {
    x(id);
  }

  bool operator==(const PeerId& n) const noexcept {
    return id == n.id;
  }
  bool operator!=(const PeerId& n) const noexcept {
    return id != n.id;
  }

  static PeerId generate() {
    return {random<uint64_t>(), random<uint64_t>()};
  }

  std::string toString() const {
    std::string s;
    for (auto v : id) {
      if (!s.empty()) {
        s += "-";
      }
      for (size_t i = 0; i != 8; ++i) {
        uint64_t sv = v >> (i * 8);
        s += "0123456789abcdef"[(sv >> 4) & 0xf];
        s += "0123456789abcdef"[sv & 0xf];
      }
    }
    return s;
  }
};

struct RemoteFunction {
  uint32_t id = 0;
  std::string_view typeId;

  template<typename X>
  void serialize(X& x) {
    x(id, typeId);
  }
};

template<typename T>
struct Me {
  T* me = nullptr;
  Me() = default;
  Me(std::nullptr_t) noexcept {}
  explicit Me(T* me) noexcept : me(me) {
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
  T* operator->() const noexcept {
    return me;
  }
  T& operator*() const noexcept {
    return *me;
  }
  explicit operator bool() const noexcept {
    return me;
  }
};

template<typename T>
auto makeMe(T* v) {
  return Me<T>(v);
}

template<typename API>
struct RpcConnectionImpl;

struct PeerImpl {
  Rpc::Impl& rpc;
  std::atomic_int activeOps{0};
  std::atomic_bool dead{false};

  alignas(64) SpinMutex idMutex_;
  std::atomic<bool> hasId = false;
  PeerId id;
  std::string_view name;
  std::vector<ConnectionTypeInfo> info;

  std::array<Connection, (int)ConnectionType::count> connections_;

  alignas(64) SpinMutex remoteFuncsMutex_;
  std::unordered_map<std::string_view, RemoteFunction> remoteFuncs_;

  std::chrono::steady_clock::time_point lastFindPeers;

  PeerImpl(Rpc::Impl& rpc) : rpc(rpc) {}
  ~PeerImpl() {
    dead = true;
    while (activeOps.load());
  }

  void setRemoteFunc(std::string_view name, const RemoteFunction& rf) {
    std::lock_guard l(remoteFuncsMutex_);
    remoteFuncs_[name] = rf;
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

  bool isConnected(const Connection& v) {
    return v.valid.load(std::memory_order_relaxed) && v.hasConn.load(std::memory_order_relaxed);
  }

  bool willConnectOrSend(const std::chrono::steady_clock::time_point& now, const Connection& v) {
    return v.valid.load(std::memory_order_relaxed) && (v.hasConn.load(std::memory_order_relaxed) || v.lastTryConnect.load(std::memory_order_relaxed) + std::chrono::seconds(30) <= now);
  }

  template<typename Buffer>
  bool banditSend(uint32_t mask, Buffer buffer, size_t* indexUsed = nullptr, Me<RpcConnectionImplBase>* outConnection = nullptr, bool shouldFindPeer = true) noexcept {
    log("banditSend %d bytes mask %#x\n", (int)buffer->size, mask);
    auto now = std::chrono::steady_clock::now();
    thread_local std::vector<std::pair<size_t, float>> list;
    list.clear();
    float sum = 0.0f;
    for (size_t i = 0; i != connections_.size(); ++i) {
      if (~mask & (1 << i)) {
        continue;
      }
      auto& v = connections_[i];
      if (willConnectOrSend(now, v)) {
        float score = std::exp(v.readBanditValue * 4);
        log("bandit %s has score %g\n", connectionTypeName[i], score);
        sum += score;
        list.emplace_back(i, sum);
      }
    }
    if (list.size() > 0) {
      size_t index;
      if (list.size() == 1) {
        index = list[0].first;
      } else {
        float v = std::uniform_real_distribution<float>(0.0f, sum)(threadRng);
        index = std::lower_bound(list.begin(), std::prev(list.end()), v, [&](auto& a, float b) {
          return a.second < b;
        })->first;
      }
      log("bandit chose %d (%s)\n", index, connectionTypeName.at(index));
      auto& x = connections_.at(index);
      x.sendCount.fetch_add(1, std::memory_order_relaxed);
      bool b = switchOnAPI((ConnectionType)index, [&](auto api) {
        return send<decltype(api)>(now, buffer, outConnection);
      });
      if (!b && buffer) {
        mask &= ~(1 << index);
        return banditSend(mask, std::move(buffer), indexUsed);
      }
      if (b && indexUsed) {
        *indexUsed = index;
      }
      return b;
    } else {
      log("No connectivity to %s\n", name);

      if (shouldFindPeer) {
        findPeer();
      }
      return false;
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args);

  std::string_view rpcName();

  void findPeer();

  template<typename API, bool isExplicit>
  void connect(std::string_view addr);

  template<typename API, typename Buffer>
  bool send(std::chrono::steady_clock::time_point now, Buffer& buffer, Me<RpcConnectionImplBase>* outConnection) {
    auto& x = connections_[index<API>];
    std::unique_lock l(x.mutex);
    if (x.conns.empty()) {
      x.lastTryConnect = now;
      if (x.remoteAddresses.empty()) {
        x.valid = false;
      } else {
        std::string_view addr;
        if (x.remoteAddresses.size() == 1) {
          addr = x.remoteAddresses[0];
        } else {
          addr = x.remoteAddresses[threadRandom<size_t>(0, x.remoteAddresses.size() - 1)];
        }
        l.unlock();
        if (!addr.empty()) {
          log("connecting to %s::%s!! :D\n", connectionTypeName[index<API>], addr);
          connect<API, false>(addr);
        }
      }
      if (outConnection) {
        *outConnection = nullptr;
      }
      return false;
    } else {
      size_t i = threadRandom<size_t>(0, x.conns.size() - 1);
      auto& c = x.conns[i];
      if (c->dead.load(std::memory_order_relaxed) || now - c->lastReceivedData >= std::chrono::seconds(15)) {
        log("Connection through %s to %s is dead, yo!\n", connectionTypeName[index<API>], name);
        BufferHandle buffer;
        serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqClose);
        ((RpcConnectionImpl<API>&)*c).send(std::move(buffer));
        throwAway(x, i);
        if (outConnection) {
          *outConnection = nullptr;
        }
        return false;
      } else {
        ((RpcConnectionImpl<API>&)*c).send(std::move(buffer));
        if (outConnection) {
          *outConnection = makeMe(&*c);
        }
        return true;
      }
    }
  }

  void throwAway(Connection& x, size_t i) {
    auto cv = std::move(x.conns[i]);
    std::swap(x.conns.back(), x.conns[i]);
    x.conns.pop_back();
    if (x.conns.empty()) {
      x.hasConn = false;
    }
    cv->timeWait = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    throwAway(std::move(cv));
  }

  void throwAway(std::unique_ptr<RpcConnectionImplBase> c);
};

std::string emptyString;

template<typename API>
struct RpcConnectionImpl : RpcConnectionImplBase {
  RpcConnectionImpl(RpcImpl<API>& rpc, typename API::Connection&& connection) : rpc(rpc), connection(std::move(connection)) {}
  RpcImpl<API>& rpc;

  typename API::Connection connection;

  PeerImpl* peer = nullptr;

  std::atomic_bool hasReceivedData{false};

  mutable std::once_flag localAddrOnce_;
  mutable std::once_flag remoteAddrOnce_;
  mutable std::string localAddrStr_;
  mutable std::string remoteAddrStr_;

  SpinMutex sendMutex;
  Buffer* sendQueueBegin = nullptr;
  Buffer* sendQueueEnd = nullptr;

  ~RpcConnectionImpl() {
    close();
    while (activeOps.load(std::memory_order_acquire));

    if constexpr (API::singularWrites) {
      for (Buffer* buf = sendQueueBegin; buf;) {
        SharedBufferHandle h;
        h.acquire(buf);
        buf = buf->next;
      }
    }
  }

  virtual const std::string& localAddr() const override {
    if (!hasReceivedData) {
      return emptyString;
    }
    std::call_once(localAddrOnce_, [this]() {
      try {
        localAddrStr_ = API::localAddr(connection);
      } catch (const std::exception&) {}
    });
    return localAddrStr_;
  }
  virtual const std::string& remoteAddr() const override {
    if (!hasReceivedData) {
      return emptyString;
    }
    std::call_once(remoteAddrOnce_, [this]() {
      try {
        remoteAddrStr_ = API::remoteAddr(connection);
      } catch (const std::exception&) {}
    });
    return remoteAddrStr_;
  }

  virtual size_t apiIndex() const override {
    return index<API>;
  }

  virtual void close() override {
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    rpc.log("Connection %s closed\n", connectionTypeName[index<API>]);
    dead = true;
    API::cast(connection).close();
  }

  void onError([[maybe_unused]] Error* err) {
    rpc.log("Connection %s to %s error: %s\n", connectionTypeName[index<API>], connectAddr, err->what());
    close();
  }
  void onError([[maybe_unused]] const char* err) {
    rpc.log("Connection %s error: %s\n", connectionTypeName[index<API>], err);
    close();
  }

  template<typename E>
  void onError(E&& error) {
    Error err(API::errstr(error));
    onError(&err);
  }

  static constexpr uint64_t kSignature = 0xff984b883019d443;

  void onData(const std::byte* ptr, size_t len) noexcept {
    log("%s :: got %d bytes\n", connectionTypeName[index<API>], len);
    if (len < sizeof(uint32_t) * 2) {
      onError("Received not enough data");
      return;
    }
    if (!hasReceivedData.load(std::memory_order_relaxed)) {
      hasReceivedData = true;
    }
    lastReceivedData = std::chrono::steady_clock::now();
    uint32_t rid;
    std::memcpy(&rid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    uint32_t fid;
    std::memcpy(&fid, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    len -= sizeof(uint32_t);
    //log("onData rid %#x fid %#x\n", rid, fid);
    if (peer && fid != Rpc::reqGreeting) {
      if (rid & 1) {
        rpc.onRequest(*peer, *this, rid, fid, ptr, len);
      } else {
        rpc.onResponse(*peer, *this, rid, fid, ptr, len);
      }
    } else if (fid == Rpc::reqGreeting) {
      try {
        uint64_t signature;
        std::string_view peerName;
        PeerId peerId;
        std::vector<ConnectionTypeInfo> info;
        deserializeBuffer(ptr, len, signature, peerName, peerId, info);
        if (signature == kSignature) {
          rpc.onGreeting(*this, peerName, peerId, std::move(info));
        } else {
          log("signature mismatch\n");
          std::terminate();
        }
      } catch (const std::exception&) {
        log("error in greeting\n");
      }
    }
  }

  void greet(std::string_view name, PeerId peerId, const std::vector<ConnectionTypeInfo>& info) {
    //log("%p::greet(\"%s\", %s)\n", (void*)this, std::string(name).c_str(), peerId.toString().c_str());
    BufferHandle buffer;
    serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqGreeting, kSignature, name, peerId, info);
    send(std::move(buffer));
  }

  void onData(const void* ptr, size_t len) {
    return onData((std::byte*)ptr, len);
  }
  void onData(BufferHandle&& buffer) {
    return onData(buffer->data(), buffer->size);
  }

  void read(Me<RpcConnectionImpl>&& me) {
    log("read %s :: %p\n", connectionTypeName[index<API>], (void*)this);
    API::cast(connection).read([me = std::move(me)](auto&& error, auto&&... args) mutable noexcept {
      log("%s :: %p got data\n", connectionTypeName[index<API>], (void*)&*me);
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
    read(makeMe(this));
  }

  struct WriteCallback {
    WriteCallback(RpcConnectionImpl* me) : me(me) {}
    Me<RpcConnectionImpl> me;
    template<typename Error>
    void operator()(Error&& error) {
      if (error) {
        me->onError(std::forward<Error>(error));
      } else {
        if constexpr (API::singularWrites) {
          std::unique_lock l(me->sendMutex);
          SharedBufferHandle h;
          h.acquire(me->sendQueueBegin);
          log("%s: write success for %d bytes\n", connectionTypeName[index<API>], h->size);
          me->sendQueueBegin = me->sendQueueBegin->next;
          if (!me->sendQueueBegin) {
            me->sendQueueEnd = nullptr;
          }
          Buffer* buf = me->sendQueueBegin;
          l.unlock();
          if (buf) {
            auto* ptr = buf->data();
            size_t size = buf->size;
            log("next buf is %d bytes\n", size);
            auto& conn = me->connection;
            API::cast(conn).write(ptr, size, std::move(*this));
          } else {
            log("next buf is null\n");
          }
        }
      }
    }
  };

  template<typename Buffer>
  void send(Buffer buffer) {
    log("%s :: send %d bytes\n", connectionTypeName[index<API>], buffer->size);
//    if (random(0, 1) == 0) {
//      return;
//    }
    if (API::singularWrites) {
      auto shared = [&](auto&& v) {
        if constexpr (std::is_same_v<std::decay_t<Buffer>, SharedBufferHandle>) {
          return std::move(v);
        } else if constexpr (std::is_same_v<std::decay_t<Buffer>, BufferHandle>) {
          return SharedBufferHandle(v.release());
        } else {
          std::abort();
        }
      };
      auto* buf = shared(std::move(buffer)).release();
      std::unique_lock l(sendMutex);
      if (sendQueueEnd) {
        sendQueueEnd->next = buf;
        sendQueueEnd = buf;
        buf->next = nullptr;
      } else {
        sendQueueBegin = sendQueueEnd = buf;
        buf->next = nullptr;
        l.unlock();
        auto* ptr = buf->data();
        size_t size = buf->size;
        API::cast(connection).write(ptr, size, WriteCallback(this));
      }
    } else {
      auto* ptr = buffer->data();
      size_t size = buffer->size;
      API::cast(connection).write(ptr, size, WriteCallback(this));
    }
  }

};

template<typename API>
struct RpcListenerImpl : RpcListenerImplBase {
  RpcListenerImpl(RpcImpl<API>& rpc, typename API::Listener&& listener, std::string_view addr) : rpc(rpc), listener(std::move(listener)), addr(addr) {
    accept();
  }
  ~RpcListenerImpl() {
    close();
    while (activeOps.load());
  }
  RpcImpl<API>& rpc;
  typename API::Listener listener;
  bool isExplicit = false;
  bool active = false;
  std::string addr;

  std::atomic<bool> dead = false;
  std::atomic_int activeOps{0};

  virtual void close() override {
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    rpc.log("Listener %s closed\n", connectionTypeName[index<API>]);
    dead = true;
    API::cast(listener).close();
  }

  virtual std::string localAddr() const override {
    return API::localAddr(listener, addr);
  }

  void accept() {
    API::cast(listener).accept([me = makeMe(this)](auto&& error, auto&& conn) mutable {
      if (error) {
        if constexpr (std::is_same_v<std::decay_t<decltype(error)>, Error*>) {
          me->rpc.onAccept(*me, nullptr, error);
        } else {
          Error err(API::errstr(error));
          me->rpc.onAccept(*me, nullptr, &err);
        }
      } else {
        auto c = std::make_unique<RpcConnectionImpl<API>>(me->rpc, std::move(conn));
        me->rpc.onAccept(*me, std::move(c), nullptr);
        if (!API::persistentAccept && !me->dead) {
          me->accept();
        }
      }
    });
  }
};

//template<>
//void RpcDeleter<RpcConnection::Impl>::operator()(RpcConnection::Impl* ptr) const noexcept {
//  delete ptr;
//}
//template<>
//void RpcDeleter<RpcListener::Impl>::operator()(RpcListener::Impl* ptr) const noexcept {
//  delete ptr;
//}

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

  at->prev = nullptr;
  at->next = nullptr;
}
}

struct RpcImplBase {
  Rpc::Impl& rpc;
  RpcImplBase(Rpc::Impl& rpc) : rpc(rpc) {}
  virtual ~RpcImplBase() {}
};

struct Rpc::Impl {

  alignas(64) SpinMutex mutex_;
  std::list<std::string> stringList_;
  std::unordered_set<std::string_view> stringMap_;
  std::unordered_map<std::string_view, uint32_t> funcIds_;
  std::vector<std::unique_ptr<Rpc::FBase>> funcs_;
  static constexpr size_t maxFunctions_  = 0x100000;
  //uint32_t baseFuncId_ = random<uint32_t>(Rpc::reqCallOffset, std::numeric_limits<uint32_t>::max() - maxFunctions_);
  uint32_t baseFuncId_ = Rpc::reqCallOffset;
  uint32_t nextFuncIndex_ = 0;

  struct Resend {
    SharedBufferHandle buffer;
    std::chrono::steady_clock::time_point ackTimestamp;
    std::chrono::steady_clock::time_point pokeTimestamp;
    std::chrono::steady_clock::time_point lastSendTimestamp;
    std::chrono::steady_clock::time_point lastSendFailTimestamp;
    bool hasAddedFailureLatency = false;
    size_t connectionIndex = ~0;
    Me<RpcConnectionImplBase> connection = nullptr;
    int pokeCount = 0;
    int totalPokeCount = 0;
    bool acked = false;
    int nackCount = 0;
  };

  struct TensorData {
    BufferHandle buffer;
    size_t offset;
    std::string_view data;
  };

  struct Receive {
    BufferHandle buffer;
    std::vector<TensorData> tensorData;
    uint32_t receivedTensors = 0;
    bool done = false;
  };

  struct Incoming {
    Incoming* prev = nullptr;
    Incoming* next = nullptr;
    uint32_t rid;
    std::chrono::steady_clock::time_point responseTimestamp;
    PeerImpl* peer = nullptr;

    int timeoutCount = 0;
    std::chrono::steady_clock::time_point timeout;
    Receive recv;

    std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();
    Resend resend;
    std::vector<Resend> resendTensors;
    size_t partsAcked = 0;
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
  alignas(64) std::once_flag timeoutThreadOnce_;
  Semaphore timeoutSem_;
  std::atomic<std::chrono::steady_clock::time_point> timeout_ = std::chrono::steady_clock::now();
  std::atomic<bool> terminate_ = false;

  struct Outgoing {
    Outgoing* prev = nullptr;
    Outgoing* next = nullptr;
    std::chrono::steady_clock::time_point requestTimestamp;
    std::chrono::steady_clock::time_point timeout;
    uint32_t rid = 0;
    Rpc::ResponseCallback response;
    PeerImpl* peer = nullptr;
    int timeoutCount = 0;
    std::chrono::steady_clock::time_point creationTimestamp = std::chrono::steady_clock::now();
    Resend resend;
    std::vector<Resend> resendTensors;
    size_t partsAcked = 0;

    Receive recv;

//    struct ResponseTensor {
//      BufferHandle buffer;
//      std::chrono::steady_clock::time_point lastPoke;
//      size_t pokeConnectionIndex = ~0;
//      Me<RpcConnectionImplBase> pokeConnection = nullptr;
//    };

//    BufferHandle responseBuffer;
//    std::vector<ResponseTensor> responseTensors;
  };

  struct OutgoingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Outgoing> map;
  };

  alignas(64) std::array<OutgoingBucket, 0x10> outgoing_;
  alignas(64) SpinMutex outgoingFifoMutex_;
  Incoming outgoingFifo_;
  alignas(64) std::atomic<uint32_t> sequenceId{random<uint32_t>()};

  alignas(64) std::array<std::unique_ptr<RpcImplBase>, (int)ConnectionType::count> rpcs_;
  std::array<std::once_flag, (int)ConnectionType::count> rpcsInited_{};

  alignas(64) PeerId myId = PeerId::generate();
  std::string_view myName = persistentString(myId.toString());
  Function<void(const Error&)> onError_;

  alignas(64) SpinMutex listenersMutex_;
  std::array<Listener, (int)ConnectionType::count> listeners_;
  std::list<std::unique_ptr<Connection>> floatingConnections_;
  std::unordered_map<RpcConnectionImplBase*, decltype(floatingConnections_)::iterator> floatingConnectionsMap_;

  alignas(64) SpinMutex peersMutex_;
  std::unordered_map<std::string_view, PeerImpl> peers_;

  alignas(64) SpinMutex garbageMutex_;
  std::vector<std::unique_ptr<RpcConnectionImplBase>> garbageConnections_;
  std::vector<std::unique_ptr<RpcListenerImplBase>> garbageListeners_;
  SpinMutex findPeerMutex_;
  std::vector<std::string_view> findPeerList_;
  std::vector<std::string_view> findPeerLocalNameList_;
  std::vector<PeerImpl*> findPeerLocalPeerList_;

  std::atomic<std::chrono::steady_clock::time_point> lastRanMisc = std::chrono::steady_clock::time_point{};

  std::chrono::steady_clock::time_point lastPrint;

  std::atomic<bool> setupDone_ = false;
  std::atomic<bool> doingSetup_ = false;
  std::vector<ConnectionTypeInfo> info_;


  template<typename API>
  void tryInitRpc() {
    if (index<API> == (size_t)ConnectionType::ibv) {
      return;
    }
//    if (index<API> == (size_t)ConnectionType::shm) {
//      return;
//    }
    try {
      printf("init %s\n", connectionTypeName[index<API>]);
      auto u = std::make_unique<RpcImpl<API>>(*this);
      rpcs_[index<API>] = std::move(u);
    } catch (const std::exception& e) {
      log("Error during init of '%s': %s\n", connectionTypeName.at(index<API>), e.what());
    }
  }

  template<typename API>
  void lazyInitRpc() {
    std::call_once(rpcsInited_[index<API>], [this]() {
      tryInitRpc<API>();
    });
  }

  Impl() {
    log("%p peer id is %s\n", (void*)this, myId.toString().c_str());
    incomingFifo_.next = &incomingFifo_;
    incomingFifo_.prev = &incomingFifo_;
    outgoingFifo_.next = &outgoingFifo_;
    outgoingFifo_.prev = &outgoingFifo_;
  }
  virtual ~Impl() {
    terminate_ = true;
    if (timeoutThread_.joinable()) {
      timeoutSem_.post();
      timeoutThread_.join();
    }
    {
      std::lock_guard l2(garbageMutex_);
      std::lock_guard l(listenersMutex_);
      std::lock_guard l3(peersMutex_);
      for (auto& v : listeners_) {
        for (auto& v2 : v.listeners) {
          garbageListeners_.push_back(std::move(v2));
        }
        v.listeners.clear();
      }
      for (auto& v : peers_) {
        for (auto& v2 : v.second.connections_) {
          for (auto& v3 : v2.conns) {
            garbageConnections_.push_back(std::move(v3));
          }
          v2.conns.clear();
        }
      }
      for (auto& v : floatingConnections_) {
        for (auto& v2 : v->conns) {
          garbageConnections_.push_back(std::move(v2));
        }
      }
      floatingConnections_.clear();
    }
    {
      std::unique_lock l(garbageMutex_);
      collect(l, garbageListeners_);
      collect(l, garbageConnections_);
    }
    for (size_t i = 0; i != (size_t)ConnectionType::count; ++i) {
      rpcs_[i] = nullptr;
    }
  }

  template<typename T>
  auto& getBucket(T& arr, uint32_t rid) {
    return arr[(rid >> 1) % arr.size()];
  }

  template<typename T>
  std::chrono::steady_clock::time_point processTimeout(T& o, std::chrono::steady_clock::time_point now, Resend& s, uint32_t partIndex) {
    auto newTimeout = now + std::chrono::seconds(1);
    if (partIndex && s.acked) {
      return newTimeout;
    }
    if (s.connection) {
      if (!s.hasAddedFailureLatency && now - s.lastSendTimestamp >= std::chrono::milliseconds(1000)) {
        s.hasAddedFailureLatency = true;
        log("  -- rid %#x to %s   %s failed \n", o.rid, o.peer->name, connectionTypeName.at(s.connectionIndex));
        switchOnAPI((ConnectionType)s.connectionIndex, [&](auto api) {
          addLatency<decltype(api)>(*o.peer, now, std::chrono::seconds(1));
        });
        fflush(stdout);
        std::exit(1);
        std::abort();
      }
      if (now - s.connection->lastReceivedData >= std::chrono::seconds(8)) {
        log("Closing connection %s to %s due to timeout!\n", connectionTypeName.at(s.connectionIndex), o.peer->name);
        auto& x = o.peer->connections_.at(s.connectionIndex);
        std::lock_guard l(x.mutex);
        for (size_t i = 0; i != x.conns.size(); ++i) {
          if (&*x.conns[i] == &*s.connection) {
            BufferHandle buffer;
            serializeToBuffer(buffer, (uint32_t)0, (uint32_t)Rpc::reqClose);
            switchOnAPI((ConnectionType)s.connectionIndex, [&](auto api) {
              ((RpcConnectionImpl<decltype(api)>&)*x.conns[i]).send(std::move(buffer));
            });
            o.peer->throwAway(x, i);
            break;
          }
        }
        s.connection = nullptr;
      }
    }
    if (!s.connection) {
      s.pokeCount = 0;
      s.acked = false;
    }
    if (s.pokeCount < 2) {
      log("timeout sending poke for rid %#x part %d (destined for %s)\n", o.rid, partIndex, o.peer->name);
      BufferHandle buffer;
      serializeToBuffer(buffer, o.rid, Rpc::reqPoke, partIndex);
      size_t index;
      bool b = o.peer->banditSend(~0, std::move(buffer), &index);
      log("timeout bandit send result: %d\n", b);
      if (b) {
        if (s.pokeCount == 0) {
          s.pokeTimestamp = now;
        }
        ++s.pokeCount;
        ++s.totalPokeCount;

        newTimeout = now + std::chrono::milliseconds((int)std::ceil(o.peer->connections_.at(index).runningLatency.load(std::memory_order_relaxed) * (4 * o.timeoutCount)));
        newTimeout = std::max(newTimeout, now + std::chrono::milliseconds(s.acked ? 1000 : 100));
        newTimeout = std::min(newTimeout, now + std::chrono::seconds(2));
      } else {
        newTimeout = now + std::chrono::milliseconds(250);
      }
      if (s.totalPokeCount >= 4) {
        newTimeout = now + std::chrono::seconds(2);
      }
    }
    return newTimeout;
  }

  template<typename T>
  void processTimeout(std::chrono::steady_clock::time_point now, T& o) {
    ++o.timeoutCount;
    auto newTimeout = now + std::chrono::seconds(1);
    //log("process timeout!\n");
    if (o.peer) {
      newTimeout = std::min(newTimeout, processTimeout(o, now, o.resend, 0));
      for (size_t i = 0; i != o.resendTensors.size(); ++i) {
        newTimeout = std::min(newTimeout, processTimeout(o, now, o.resendTensors[i], 1 + i));
      }
    }
    o.timeout = newTimeout;
  }

  template<typename L, typename T>
  void collect(L& lock, T& ref, bool respectTimeWait = false) noexcept {
    if (!ref.empty()) {
      auto now = std::chrono::steady_clock::now();
      thread_local T tmp;
      std::swap(ref, tmp);
      lock.unlock();
      for (auto& v : tmp) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(v)>, std::unique_ptr<RpcConnectionImplBase>>) {

          auto checkBucket = [&](auto& bucket) {
            std::lock_guard l(bucket.mutex);
            for (auto& [id, x] : bucket.map) {
              if (x.resend.connection && &*x.resend.connection == &*v) {
                x.resend.connection = nullptr;
              }
            }
          };
          for (auto& bucket : incoming_) {
            checkBucket(bucket);
          }
          for (auto& bucket : outgoing_) {
            checkBucket(bucket);
          }

          if (respectTimeWait && now < v->timeWait) {
            log("time wait %s connection %s <-> %s\n", connectionTypeName.at(v->apiIndex()), v->localAddr(), v->remoteAddr());
            lock.lock();
            ref.push_back(std::move(v));
            lock.unlock();
          } else {
            log("collecting %s connection %s <-> %s\n", connectionTypeName.at(v->apiIndex()), v->localAddr(), v->remoteAddr());

            if (v->isExplicit) {
              switchOnAPI((ConnectionType)v->apiIndex(), [&](auto api) {
                log("Reconnecting to %s...\n", v->connectAddr);
                connect<decltype(api)>(v->connectAddr);
              });
            }
          }
        }
      }
      tmp.clear();
      //log("GARBAGE COLLECTED YEY!\n");
      lock.lock();
    }
  }

  void collectFloatingConnections(std::chrono::steady_clock::time_point now) {
    for (auto i = floatingConnections_.begin(); i != floatingConnections_.end();) {
      auto& c = *i;
      if (now - c->creationTimestamp >= std::chrono::seconds(10)) {
        log("Collecting floating connection\n");
        for (auto& v2 : c->conns) {
          garbageConnections_.push_back(std::move(v2));
        }
        i = floatingConnections_.erase(i);
      } else {
        ++i;
      }
    }
  }

  void collectGarbage() {
    std::unique_lock l(garbageMutex_);
    collect(l, garbageConnections_, true);
    collect(l, garbageListeners_);
  }

  void timeoutThreadEntry() {
    async::setCurrentThreadName("timeout");
    //std::this_thread::sleep_for(std::chrono::milliseconds(250));
    //log("timeout thread running (%s)!\n", std::string(myName).c_str());
    auto lastP = std::chrono::steady_clock::now();
    while (!terminate_.load(std::memory_order_relaxed)) {
      auto now = std::chrono::steady_clock::now();
      if (lastRanMisc.load() + std::chrono::milliseconds(750) <= now) {
        lastRanMisc.store(now);
        collectFloatingConnections(now);
        collectGarbage();
        findPeersImpl();
        now = std::chrono::steady_clock::now();
      }

      auto timeout = timeout_.load(std::memory_order_relaxed);
      log("timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
      while (now < timeout) {
        log("%p sleeping for %d\n", (void*)this, std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
        if (timeout - now > std::chrono::milliseconds(250)) {
          timeout = now + std::chrono::milliseconds(250);
        }
        timeoutSem_.wait_for(timeout - now);
        log("%p woke up\n", (void*)this);
        now = std::chrono::steady_clock::now();
        if (now - lastP >= std::chrono::milliseconds(200)) {
          lastP = now;
          if (true) {
            std::lock_guard l(peersMutex_);
            for (auto& v : peers_) {
              auto& p = v.second;
              std::lock_guard l(p.idMutex_);
              log("Peer %s (%s)\n", std::string(p.name).c_str(), p.id.toString().c_str());
              for (size_t i = 0; i != p.connections_.size(); ++i) {
                auto& x = p.connections_[i];
                log(" %s x%d  latency %g bandit %g\n", connectionTypeName[i], x.sendCount.load(std::memory_order_relaxed), x.runningLatency.load(), x.readBanditValue.load());
                log("    %d conns:\n", x.conns.size());
                for (auto& v : x.conns) {
                  float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - v->lastReceivedData).count();
                  log("      %s (%s, %s) [%s] age %g\n", v->dead ? "dead" : "alive", v->localAddr(), v->remoteAddr(), v->connectAddr, t);
                }
              }
            }
          }
        }
        continue;
      }
      auto newTimeout = now + std::chrono::seconds(5);
      timeout_.store(newTimeout);
      auto process = [&](auto& container) {
        auto absTimeoutDuration = std::chrono::seconds(60);
        for (auto& b : container) {
          std::unique_lock l(b.mutex);
          bool anyToRemove = false;
          for (auto& v : b.map) {
            if (now - v.second.creationTimestamp >= absTimeoutDuration) {
              anyToRemove = true;
            }
            if (v.second.resend.buffer) {
              if (now >= v.second.timeout) {
                processTimeout(now, v.second);
              }
              newTimeout = std::min(newTimeout, v.second.timeout);
            }

            constexpr bool isIncoming = std::is_same_v<std::decay_t<decltype(v.second)>, Incoming>;
            float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - v.second.creationTimestamp).count();
            if constexpr (isIncoming) {
              log("Response %#x age: %g\n", v.second.rid, t);
            } else {
              log("Request %#x age: %g\n", v.second.rid, t);
            }
          }
          if (anyToRemove) {
            std::unique_lock l2(incomingFifoMutex_, std::defer_lock);
            constexpr bool isIncoming = std::is_same_v<std::decay_t<decltype(b.map.begin()->second)>, Incoming>;
            if (isIncoming) {
              if (!l2.try_lock()) {
                l.unlock();
                l2.lock();
                l.lock();
              }
            }
            for (auto i = b.map.begin(); i != b.map.end();) {
              auto& v = i->second;
              if (now - v.creationTimestamp >= absTimeoutDuration) {
                if constexpr (isIncoming) {
                  if (v.resend.buffer) {
                    listErase(&v);
                  }
                  log("Response %#x timed out for real\n", v.rid);
                } else {
                  log("Request %#x timed out for real\n", v.rid);
                }
                i = b.map.erase(i);
              } else {
                ++i;
              }
            }
          }
        }
      };
      process(outgoing_);
      process(incoming_);
      timeout = timeout_.load(std::memory_order_relaxed);
      while (newTimeout < timeout && !timeout_.compare_exchange_weak(timeout, newTimeout));
      log("new timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(newTimeout - now).count());


      if (now - lastPrint >= std::chrono::seconds(30)) {
        lastPrint = now;
        std::lock_guard l(peersMutex_);
        for (auto& v : peers_) {
          auto& p = v.second;
          std::lock_guard l(p.idMutex_);
          log("Peer %s (%s)\n", std::string(p.name).c_str(), p.id.toString().c_str());
          for (size_t i = 0; i != p.connections_.size(); ++i) {
            auto& x = p.connections_[i];
            log(" %s x%d  latency %g bandit %g\n", connectionTypeName[i], x.sendCount.load(std::memory_order_relaxed), x.runningLatency.load(), x.readBanditValue.load());
          }
        }
      }
    }
  }

  void startTimeoutThread() {
    timeoutThread_ = std::thread([this]() noexcept {
      timeoutThreadEntry();
    });
  }

  void define(std::string_view name, std::unique_ptr<Rpc::FBase>&& f) {
    name = persistentString(name);
    std::lock_guard l(mutex_);
    if (nextFuncIndex_ >= maxFunctions_) {
      throw Error("Too many RPC functions defined");
    }
    //uint32_t id = baseFuncId_ + nextFuncIndex_++;
    uint32_t id = baseFuncId_ + std::hash<std::string_view>()(name) % maxFunctions_;
    size_t index = id - baseFuncId_;
    if (funcs_.size() <= index) {
      funcs_.resize(std::max(index + 1, funcs_.size() + funcs_.size() / 2));
    }
    if (funcIds_[name]) {
      throw Error("Function " + std::string(name) + " already defined (or hash collision)");
    }
    funcIds_[name] = id;
    funcs_[index] = std::move(f);
  }
  std::string_view persistentString(std::string_view name) {
    std::lock_guard l(mutex_);
    auto i = stringMap_.find(name);
    if (i != stringMap_.end()) {
      return *i;;
    }
    stringList_.emplace_back(name);
    return *stringMap_.emplace(stringList_.back()).first;
  }

  template<typename API>
  void setup() noexcept {
    lazyInitRpc<API>();
    auto& x = listeners_.at(index<API>);
    if (x.implicitCount > 0) {
      return;
    }
    for (auto& addr : API::defaultAddr()) {
      listen<API, false>(addr);
    }
  }

  auto& getInfo() noexcept {
    if (!setupDone_) {
      if (doingSetup_.exchange(true)) {
        while (!setupDone_);
      } else {
        for (size_t i = 0; i != (size_t)ConnectionType::count; ++i) {
          switchOnAPI(ConnectionType(i), [&](auto api) {
            setup<decltype(api)>();
          });
        }

        std::lock_guard l(listenersMutex_);
        info_.clear();
        for (size_t i = 0; i != listeners_.size(); ++i) {
          ConnectionTypeInfo ci;
          ci.name = connectionTypeName.at(i);
          for (auto& v : listeners_[i].listeners) {
            try {
              const auto& str = v->localAddr();
              if (!str.empty()) {
                ci.addr.push_back(persistentString(str));
              }
            } catch (const std::exception& e) {
            }
          }
          info_.push_back(std::move(ci));
        }

        setupDone_ = true;
      }
    }
    return info_;
  }

  template<typename API, bool explicit_ = true>
  void connect(std::string_view addr) {
    log("Connecting with %s to %s\n", connectionTypeName[index<API>], addr);
    lazyInitRpc<API>();
    auto* u = getImpl<API>();
    if (!u) {
      throw std::runtime_error("Backend " + std::string(connectionTypeName.at(index<API>)) + " is not available");
    }

    getInfo();

    auto c = std::make_unique<Connection>();
    std::unique_lock l(listenersMutex_);
    if (terminate_.load(std::memory_order_relaxed)) {
      return;
    }
    c->outgoing = true;
    c->isExplicit = explicit_;
    c->addr = persistentString(addr);
    RpcConnectionImpl<API> xx(*u, u->context.connect(std::string(addr)));
    auto cu = std::make_unique<RpcConnectionImpl<API>>(*u, u->context.connect(std::string(addr)));
    cu->isExplicit = explicit_;
    cu->connectAddr = addr;
    cu->greet(myName, myId, info_);
    cu->start();
    c->conns.push_back(std::move(cu));
    floatingConnections_.push_back(std::move(c));
    floatingConnectionsMap_[&*floatingConnections_.back()->conns.back()] = std::prev(floatingConnections_.end());
  }

  template<typename API, bool explicit_ = true>
  auto listen(std::string_view addr) {
    lazyInitRpc<API>();
    auto* u = getImpl<API>();
    if (!u) {
      if constexpr (!explicit_) {
        return false;
      } else {
        throw std::runtime_error("Backend " + std::string(connectionTypeName.at(index<API>)) + " is not available");
      }
    }
    auto ul = u->context.listen(std::string(addr));
    std::lock_guard l(listenersMutex_);
    auto& x = listeners_.at(index<API>);
    std::unique_ptr<RpcListenerImpl<API>> i;
    try {
      i = std::make_unique<RpcListenerImpl<API>>(*u, std::move(ul), addr);
      i->active = true;
      i->isExplicit = explicit_;
      ++x.activeCount;
      ++(explicit_ ? x.explicitCount : x.implicitCount);
      x.listeners.push_back(std::move(i));
      log("%s::listen<%s, %d>(%s) success\n", myName, connectionTypeName[index<API>], explicit_, std::string(addr).c_str());
    } catch (const std::exception& e) {
      std::lock_guard l(garbageMutex_);
      garbageListeners_.push_back(std::move(i));
      log("error in listen<%s, %d>(%s): %s\n", myName, connectionTypeName[index<API>], explicit_, e.what());
      if constexpr (!explicit_) {
        return false;
      } else {
        throw;
      }
    }
    if constexpr (!explicit_) {
      return true;
    }
  }

  void setOnError(Function<void(const Error&)>&& callback) {
    if (onError_) {
      throw std::runtime_error("onError callback already set");
    }
    onError_ = std::move(callback);
  }

  template<typename API>
  auto* getImpl() {
    return (RpcImpl<API>*)&*rpcs_.at(index<API>);
  }

  static size_t computeStorageNbytes(
      torch::IntArrayRef sizes,
      torch::IntArrayRef strides,
      size_t itemsize_bytes) {
    // size of the underlying storage is 1 bigger than the offset
    // of the last element according to stride
    size_t size = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
      if (sizes[i] == 0) {
        return 0;
      }
      size += strides[i]*(sizes[i]-1);
    }
    return size * itemsize_bytes;
  }

  bool resend(PeerImpl& peer, Resend& s) {
    size_t index;
    Me<RpcConnectionImplBase> connection;
    if (peer.banditSend(~0, s.buffer, &index, &connection)) {
      s.lastSendTimestamp = std::chrono::steady_clock::now();
      s.connection = std::move(connection);
      s.connectionIndex = index;
      return true;
    } else {
      s.lastSendFailTimestamp = std::chrono::steady_clock::now();
      s.connection = nullptr;
      return false;
    }
  }

  void sendRequest(PeerImpl& peer, uint32_t fid, BufferHandle buffer, rpc::Rpc::ResponseCallback response) noexcept {
    if (buffer->size != (uint32_t)buffer->size) {
      throw std::runtime_error("RPC request is too large!");
    }
    auto* ptr = dataptr<std::byte>(&*buffer);
    uint32_t rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
    if (rid % 0x100 == 0xff) {
      sequenceId.store(random<uint32_t>());
    }
    auto* ridPtr = ptr;
    std::memcpy(ptr, &rid, sizeof(rid));
    ptr += sizeof(rid);
    std::memcpy(ptr, &fid, sizeof(fid));
    ptr += sizeof(fid);
    uint32_t nTensors = buffer->nTensors;
    std::vector<SharedBufferHandle> tensorbuffers;
    if (fid >= Rpc::reqCallOffset) {
      std::memcpy(ptr, &nTensors, sizeof(nTensors));
      ptr += sizeof(nTensors);

      for (uint32_t i = 0; i != nTensors; ++i) {
        auto& tensorRef = buffer->tensors()[i];
        auto& tensor = tensorRef.tensor;
        size_t nBytes = computeStorageNbytes(tensor.sizes(), tensor.strides(), tensor.itemsize());
        BufferHandle tmp;
        serializeToBuffer(tmp, rid, fid, (uint32_t)~0, i, (uint32_t)tensorRef.offset, std::string_view((const char*)tensor.data_ptr(), nBytes));
        tensorbuffers.push_back(SharedBufferHandle(tmp.release()));
      }
    } else {
      if (nTensors) {
        std::abort();
      }
    }
    auto now = std::chrono::steady_clock::now();
    {
      log("send request %#x %s::%#x\n", rid, peer.name, fid);
//      if (conn.dead.load(std::memory_order_relaxed)) {
//        Error err("RPC call: connection is closed");
//        std::move(response)(nullptr, 0, &err);
//        return;
//      }
      SharedBufferHandle shared(buffer.release());
      auto& oBucket = getBucket(outgoing_, rid);
      std::unique_lock l(oBucket.mutex);
      auto in = oBucket.map.try_emplace(rid);
      while (!in.second) {
        rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
        if (rid % 0x100 == 0xff) {
          sequenceId.store(random<uint32_t>());
        }
        std::memcpy(ridPtr, &rid, sizeof(rid));
        in = oBucket.map.try_emplace(rid);
      }
      //log("sending request with rid %#x\n", rid);
      auto& q = in.first->second;
      q.rid = rid;
      q.peer = &peer;
      q.requestTimestamp = now;
      q.timeout = now + std::chrono::milliseconds(100);
      q.response = std::move(response);
      q.resend.buffer = shared;
      resend(peer, q.resend);

//      size_t index = -1;
//      if (peer.banditSend(~0, std::move(shared), &index)) {
//        myTimeout = now + std::chrono::milliseconds((int)std::ceil(peer.connections_.at(index).runningLatency.load(std::memory_order_relaxed) * 4));
//        myTimeout = std::min(myTimeout, now + std::chrono::seconds(2));
//      } else {
//        myTimeout = now + std::chrono::milliseconds(100);
//      }

      if (nTensors) {
        for (uint32_t i = 0; i != nTensors; ++i) {
          q.resendTensors.emplace_back();
          q.resendTensors.back().buffer = std::move(tensorbuffers[i]);
          resend(peer, q.resendTensors.back());
        }
      }
    }
    updateTimeout(now + std::chrono::seconds(1));
  }

  void updateTimeout(std::chrono::steady_clock::time_point myTimeout) {
    static_assert(std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
    auto timeout = timeout_.load(std::memory_order_acquire);
    while (myTimeout < timeout) {
      if (timeout_.compare_exchange_weak(timeout, myTimeout)) {
        timeoutSem_.post();
        break;
      }
    }
    std::call_once(timeoutThreadOnce_, [&]() {
      startTimeoutThread();
    });
  }

  PeerImpl& getPeer(std::string_view name) {
    std::lock_guard l(peersMutex_);
    auto i = peers_.try_emplace(name, *this);
    auto& p = i.first->second;
    if (i.second) {
      p.name = persistentString(name);
      const_cast<std::string_view&>(i.first->first) = p.name;
    }
    return p;
  }

  void sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response) noexcept {
    log("sendRequest %s::%s %d bytes\n", peerName, functionName, buffer->size);
    auto& peer = getPeer(peerName);
    uint32_t fid = peer.functionId(functionName);
    if (fid == 0) {
      functionName = persistentString(functionName);
      BufferHandle buffer2;
      serializeToBuffer(buffer2, (uint32_t)0, (uint32_t)0, functionName);
      sendRequest(peer, reqFindFunction, std::move(buffer2), [this, peer = &peer, functionName = functionName, buffer = std::move(buffer), response = std::move(response)](BufferHandle recvbuffer, Error* error) mutable noexcept {
        if (error) {
          std::move(response)(nullptr, error);
        } else {
          //log("got %d bytes\n", recvbuffer->size);
          RemoteFunction rf;
          deserializeBuffer(recvbuffer, rf);
          uint32_t fid = rf.id;
          //log("got id %#x\n", id);
          if (fid == 0) {
            Error err("RPC remote function " + std::string(peer->name) + "::'" + std::string(functionName) + "' does not exist");
            std::move(response)(nullptr, &err);
            return;
          }
          rf.typeId = persistentString(rf.typeId);
          peer->setRemoteFunc(functionName, rf);
          sendRequest(*peer, fid, std::move(buffer), std::move(response));
        }
      });
    } else {
      sendRequest(peer, fid, std::move(buffer), std::move(response));
    }
  }

  template<typename API>
  void onAccept(RpcListenerImpl<API>& listener, std::unique_ptr<RpcConnectionImpl<API>> conn, Error* err) {
    //log("onAccept!()\n");
    getInfo();
    std::unique_lock l(listenersMutex_);
    if (terminate_.load(std::memory_order_relaxed)) {
      return;
    }
    if (err) {
      log("accept error: %s\n", err->what());
      listener.active = false;
      bool isExplicit = listener.isExplicit;
      auto& x = listeners_.at(index<API>);
      --x.activeCount;
      --(isExplicit ? x.explicitCount : x.implicitCount);
      if (isExplicit) {
        if (onError_) {
          onError_(*err);
        }
//        int nExplicit = 0;
//        for (auto& v : listeners_) {
//          if (v.explicitCount) {
//            ++nExplicit;
//          }
//        }
//        if (nExplicit == 0) {
//          l.unlock();
//          if (onError_) {
//            onError_(*err);
//          }
//        }
      }
    } else {
      //log("accept got connection!\n");
      auto c = std::make_unique<Connection>();
      conn->greet(myName, myId, info_);
      conn->start();
      c->conns.push_back(std::move(conn));
      floatingConnections_.push_back(std::move(c));
      floatingConnectionsMap_[&*floatingConnections_.back()->conns.back()] = std::prev(floatingConnections_.end());
    }
  }

  void setName(std::string_view name) {
    myName = persistentString(name);
  }

  std::pair<std::string_view, int> decodeIpAddress(std::string_view address) {
    std::string_view hostname = address;
    int port = 0;
    auto bpos = address.find('[');
    if (bpos != std::string_view::npos) {
      auto bepos = address.find(']', bpos);
      if (bepos != std::string_view::npos) {
        hostname = address.substr(bpos + 1, bepos - (bpos + 1));
        address = address.substr(bepos + 1);
      }
    }
    auto cpos = address.find(':');
    if (cpos != std::string_view::npos) {
      if (hostname == address) {
        hostname = address.substr(0, cpos);
      }
      ++cpos;
      while (cpos != address.size()) {
        char c = address[cpos];
        if (c < '0' || c > '9') {
          break;
        }
        port *= 10;
        port += c - '0';
        ++cpos;
      }
    }
    return {hostname, port};
  }

  template<typename API>
  void onGreeting(RpcConnectionImpl<API>& conn, std::string_view peerName, PeerId peerId, std::vector<ConnectionTypeInfo>&& info) {
    //log("%s::%s::onGreeting!(\"%s\", %s)\n", std::string(myName).c_str(), connectionTypeName[index<API>], std::string(peerName).c_str(), peerId.toString().c_str());
//    for (auto& v : info) {
//      log(" %s\n", std::string(v.name).c_str());
//      for (auto& v2 : v.addr) {
//        log("  @ %s\n", std::string(v2).c_str());
//      }
//    }
    std::unique_lock l(listenersMutex_);
    auto i = floatingConnectionsMap_.find(&conn);
    if (i != floatingConnectionsMap_.end()) {
      auto i2 = i->second;
      floatingConnectionsMap_.erase(i);
      auto cptr = std::move(*i2);
      floatingConnections_.erase(i2);

      l.unlock();

      if (peerId == myId) {
        std::lock_guard l(garbageMutex_);
        for (auto& c : cptr->conns) {
          garbageConnections_.push_back(std::move(c));
        }
        log("I connected to myself! oops!\n");
        return;
      }
      if (peerName == myName) {
        std::lock_guard l(garbageMutex_);
        for (auto& c : cptr->conns) {
          garbageConnections_.push_back(std::move(c));
        }
        log("Peer with same name as me! Refusing connection!\n");
        return;
      }

      PeerImpl& peer = getPeer(peerName);
      {
        std::lock_guard l(peer.idMutex_);
        peer.id = peerId;
        peer.info = std::move(info);
        peer.hasId = true;
      }
      if (&conn != &*cptr->conns.back()) {
        std::abort();
      }
      conn.peer = &peer;
      std::unique_ptr<RpcConnectionImplBase> oldconn;
      {
        auto& x = peer.connections_[index<API>];
        std::lock_guard l(x.mutex);
        x.isExplicit = cptr->isExplicit;
        x.outgoing = cptr->outgoing;
        x.addr = std::move(cptr->addr);
        for (auto& c : cptr->conns) {
          x.conns.push_back(std::move(c));
        }
        x.hasConn = true;
        x.valid = true;
      }
      if (oldconn) {
        std::lock_guard l(garbageMutex_);
        garbageConnections_.push_back(std::move(oldconn));
      }

      {
        std::lock_guard l(peer.idMutex_);
        for (auto& v : peer.info) {
          for (size_t i = 0; i != connectionTypeName.size(); ++i) {
            if (v.name == connectionTypeName[i]) {
              auto& x = peer.connections_.at(i);
              std::lock_guard l(x.mutex);
              x.valid = true;
              std::string addr;
              if (API::addressIsIp && addressIsIp((ConnectionType)i)) {
                addr = conn.remoteAddr();
              }
              if (!addr.empty()) {
                auto remote = decodeIpAddress(addr);
                bool remoteIpv6 = remote.first.find(':') != std::string_view::npos;
                for (auto& v2 : v.addr) {
                  auto v3 = decodeIpAddress(v2);
                  bool ipv6 = v3.first.find(':') != std::string_view::npos;
                  if (ipv6 != remoteIpv6) {
                    continue;
                  }
                  std::string newAddr = std::string(remote.first) + ":" + std::to_string(v3.second);
                  if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), newAddr) == x.remoteAddresses.end()) {
                    x.remoteAddresses.push_back(persistentString(newAddr));
                    if (x.remoteAddresses.size() > 48) {
                      x.remoteAddresses.erase(x.remoteAddresses.begin(), x.remoteAddresses.begin() + 24 - x.remoteAddresses.size());
                    }
                  }
                }
              } else if (!addressIsIp((ConnectionType)i)) {
                for (auto& v2 : v.addr) {
                  if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), v2) == x.remoteAddresses.end()) {
                    x.remoteAddresses.push_back(persistentString(v2));
                    if (x.remoteAddresses.size() > 48) {
                      x.remoteAddresses.erase(x.remoteAddresses.begin(), x.remoteAddresses.begin() + 24 - x.remoteAddresses.size());
                    }
                  }
                }
              }
//              for (auto& v : x.remoteAddresses) {
//                log(" -- %s -- has a remote address %s\n", std::string(peer.name).c_str(), std::string(v).c_str());
//              }
              if (x.remoteAddresses.size() > 24) {
                x.remoteAddresses.erase(x.remoteAddresses.begin(), x.remoteAddresses.begin() + 24 - x.remoteAddresses.size());
              }
            }
          }
        }
      }

      for (auto& b : outgoing_) {
        std::lock_guard l(b.mutex);
        for (auto& v : b.map) {
          auto& o = v.second;
          if (o.peer == &peer && !o.resend.connection) {
            log("poking on newly established connection\n");
            BufferHandle buffer;
            serializeToBuffer(buffer, o.rid, Rpc::reqPoke, o.resend.acked, (uint32_t)0);
            conn.send(std::move(buffer));
            for (size_t i = 0; i != o.resendTensors.size(); ++i) {
              auto& t = o.resendTensors[i];
              if (!t.connection) {
                serializeToBuffer(buffer, o.rid, Rpc::reqPoke, o.resend.acked, uint32_t(1 + i));
                conn.send(std::move(buffer));
              }
            }
          }
        }
      }
    }
  }

  void findPeersImpl() {
    std::vector<std::string_view> nameList;
    std::vector<PeerImpl*> peerList;
    {
      std::lock_guard l(findPeerMutex_);
      if (findPeerList_.empty()) {
        return;
      }
      std::swap(nameList, findPeerLocalNameList_);
      std::swap(nameList, findPeerList_);
      std::swap(peerList, findPeerLocalPeerList_);
      findPeerList_.clear();
      peerList.clear();
    }
    size_t n = 0;
    {
      std::lock_guard l(peersMutex_);
      n = peers_.size();
    }
    peerList.reserve(n + n / 4);
    {
      std::lock_guard l(peersMutex_);
      for (auto& v : peers_) {
        peerList.push_back(&v.second);
      }
    }
    auto now = std::chrono::steady_clock::now();
    peerList.erase(std::remove_if(peerList.begin(), peerList.end(), [&](PeerImpl* p) {
      if (!p->hasId.load(std::memory_order_relaxed)) {
        return true;
      }
      for (auto& v : p->connections_) {
        if (p->isConnected(v)) {
          return false;
        }
      }
      if (now - p->lastFindPeers <= std::chrono::seconds(5)) {
        return false;
      }
      return true;
    }), peerList.end());

    //log("findPeers has %d/%d peers with live connection\n", peerList.size(), n);

    bool anySuccess = false;

    if (!peerList.empty()) {
      size_t nToKeep = std::min((size_t)std::ceil(std::log2(n)), peerList.size());
      nToKeep = std::max(nToKeep, std::min(n, (size_t)2));
      while (peerList.size() > nToKeep) {
        std::swap(peerList.back(), peerList.at(threadRandom<size_t>(0, peerList.size() - 1)));
        peerList.pop_back();
      }
      log("looking among %d peers\n", peerList.size());
      BufferHandle buffer;
      serializeToBuffer(buffer, (uint32_t)1, (uint32_t)Rpc::reqLookingForPeer, nameList);
      SharedBufferHandle shared{buffer.release()};
      for (auto* p : peerList) {
        p->lastFindPeers = now;
        anySuccess |= p->banditSend(~0, shared, nullptr, nullptr, false);
      }
    }

    if (anySuccess) {
      std::lock_guard l(findPeerMutex_);
      std::swap(nameList, findPeerLocalNameList_);
      std::swap(peerList, findPeerLocalPeerList_);
    } else {
      //log("No connectivity to any peers for search; keeping find list\n");
      std::lock_guard l(findPeerMutex_);
      std::swap(nameList, findPeerList_);
      std::swap(peerList, findPeerLocalPeerList_);
    }
  }

  void findPeer(std::string_view name) {
    if (name == myName) {
      return;
    }
    std::call_once(timeoutThreadOnce_, [&]() {
      startTimeoutThread();
    });
    std::lock_guard l(findPeerMutex_);
    if (std::find(findPeerList_.begin(), findPeerList_.end(), name) == findPeerList_.end()) {
      //log("%s looking for %s\n", std::string(myName).c_str(), std::string(name).c_str());
      findPeerList_.push_back(name);
    }
    timeoutSem_.post();
  }

  template<typename API>
  void addLatency(PeerImpl& peer, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration duration, int partIndex = -1) {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    //float latency = std::min(us, (uint64_t)1000) / 100.0f + std::min(us / 1000.0f, 1000.0f);
    float latency = std::min(us / 1000.0f, 10000.0f);

    log("add latency: connection %s part %d, latency %g\n", connectionTypeName[index<API>], partIndex, latency);

    if (latency >= 500) {
      //log("WARNING: connection %s part %d us is %lld, latency is %g\n", connectionTypeName[index<API>], partIndex, us, latency);
      //std::abort();
    }

    Connection& cx = peer.connections_[index<API>];
    std::lock_guard l(cx.latencyMutex);

    float t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(now - cx.lastUpdateLatency).count();
    cx.lastUpdateLatency = now;
    float a = std::pow(0.25f, std::min(t, 0.25f));
    if (!std::isfinite(a)) {
      a = 0.0f;
    }

    float runningLatency = cx.runningLatency.load(std::memory_order_relaxed);
    runningLatency = runningLatency * a + latency * (1.0f - a);
    cx.runningLatency.store(runningLatency, std::memory_order_relaxed);
    float min = runningLatency;
    for (auto& v : peer.connections_) {
      if (v.valid.load(std::memory_order_relaxed) && v.hasConn.load(std::memory_order_relaxed)) {
        float l = v.runningLatency.load(std::memory_order_relaxed);
        //log("running latency for %d is %g\n", &v - peer.connections_.data(), l);
        min = std::min(min, l);
      }
    }
    //log("runningLatency is %g\n", runningLatency);
    float r = runningLatency <= min ? 1.0f : -1.0f;
    float banditValue = cx.writeBanditValue * a + r * (1.0f - a);
    cx.writeBanditValue = banditValue;
    if (std::abs(banditValue - cx.readBanditValue.load(std::memory_order_relaxed)) >= 0.001f) {
      //log("update bandit value %g -> %g\n", cx.readBanditValue.load(), banditValue);
      cx.readBanditValue.store(banditValue, std::memory_order_relaxed);
    } else {
      //log("bandit value of %g does not need update\n", cx.readBanditValue.load(std::memory_order_relaxed));
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args) {
    auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
    rpc::log("%s: %s", myName, s);
  }

};

template<typename API>
struct RpcImpl : RpcImplBase {
  typename API::Context context;

  RpcImpl(Rpc::Impl& rpc) : RpcImplBase(rpc) {}

  void onAccept(RpcListenerImpl<API>& listener, std::unique_ptr<RpcConnectionImpl<API>>&& conn, Error* err) {
    rpc.onAccept(listener, std::move(conn), err);
  }

  void onGreeting(RpcConnectionImpl<API>& conn, std::string_view peerName, PeerId peerId, std::vector<ConnectionTypeInfo>&& info) {
    rpc.onGreeting(conn, peerName, peerId, std::move(info));
  }

  template<typename T>
  void handlePoke(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, const std::byte* ptr, size_t len) {
    uint32_t partIndex;
    deserializeBuffer(ptr, len, partIndex);
    auto& bucket = rpc.getBucket(container, rid);
    std::unique_lock l(bucket.mutex);
    auto i = bucket.map.find(rid);
    if (i == bucket.map.end()) {
      log("got poke for unknown rid, nack %d\n", partIndex);
      BufferHandle buffer;
      //serializeToBuffer(buffer, rid, Rpc::reqNotFound);
      serializeToBuffer(buffer, rid, Rpc::reqNack, partIndex);
      conn.send(std::move(buffer));
    } else {
      auto& x = i->second;
      if (x.rid != rid) {
        log("rid %#x is not set!\n", rid);
        std::abort();
      }
      if (x.peer != &peer) {
        log("peer %p vs %p\n", (void*)x.peer, (void*)&peer);
        log("rid collision on poke! (not fatal!)\n");
        std::terminate();
      }
      bool ack = false;
      if (x.recv.done) {
        ack = true;
      } else {
        if (partIndex == 0) {
          ack = x.recv.buffer != nullptr;
        } else {
          size_t tensorIndex = partIndex - 1;
          if (tensorIndex < x.recv.tensorData.size()) {
            ack = x.recv.tensorData[tensorIndex].buffer != nullptr;
          }
        }
        l.unlock();
      }
      log("got poke %s %d\n", ack ? "ack" : "nack", partIndex);
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, ack ? Rpc::reqAck : Rpc::reqNack, partIndex);
      conn.send(std::move(buffer));
    }
  }

  template<bool allowNew, typename T>
  BufferHandle handleRecv(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid,  const std::byte* ptr, size_t len) {
    uint32_t nTensors;
    auto view = deserializeBuffer(ptr, len, nTensors);
    ptr = (const std::byte*)view.data();
    len = view.size();
    bool isTensor = nTensors == (uint32_t)~0;
    if (isTensor) {
      nTensors = 0;
    }
    auto find = [&](auto& bucket, auto& l) {
      auto check = [&](auto i) -> decltype(&i->second) {
        if (i == bucket.map.end()) {
          return nullptr;
        }
        auto& x = i->second;
        if (x.rid != rid) {
          log("rid %#x is not set!\n", rid);
          std::abort();
        }
        if (x.peer != &peer) {
          log("peer %p vs %p\n", (void*)x.peer, (void*)&peer);
          log("rid collision on recv! (not fatal!)\n");
          // but we probably need a rid collision message so that the client can change rid
          //   -- what? the client can't change rid ...
          //std::terminate();
          return nullptr;
        }
        if (x.recv.done) {
          l.unlock();
          //log("recv for rid %#x already done\n", rid);
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqAck, (uint32_t)0);
          conn.send(std::move(buffer));
          return nullptr;
        }
        return &x;
      };
      if constexpr (allowNew) {
        auto i = bucket.map.try_emplace(rid);
        auto& x = i.first->second;
        if (i.second) {
          x.peer = &peer;
          x.rid = rid;
        }
        return check(i.first);
      } else {
        return check(bucket.map.find(rid));
      }
    };
    //log("call with len %d\n", len);
    auto createTensors = [&](auto& x, auto& l) {
      auto data = std::move(x.recv.tensorData);
      BufferHandle buffer = std::move(x.recv.buffer);
      l.unlock();
      for (size_t i = 0; i != data.size(); ++i) {
        auto& d = data[i];
        Deserializer des(std::string_view((const char*)buffer->data(), buffer->size));
        Deserialize xd(des);
        des.consume(d.offset - 12);
        buffer->tensors()[i].tensor = serializeHelperAllocateTensor(xd, (std::byte*)d.data.data(), d.data.size(), std::move(d.buffer));
      }
      return buffer;
    };
    BufferHandle inbuffer = makeBuffer(len, nTensors);
    std::memcpy(inbuffer->data(), ptr, len);
    inbuffer->size = len;
    if (!isTensor) {
      log("recv part 0 of rid %#x\n", rid);
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqAck, (uint32_t)0);
      conn.send(std::move(buffer));
      auto& bucket = rpc.getBucket(container, rid);
      std::unique_lock l(bucket.mutex);
      auto xptr = find(bucket, l);
      if (xptr) {
        //log("recv %d tensors\n", nTensors);
        auto& x = *xptr;
        if (nTensors) {
          if (x.recv.done) {
            //log("already called !? main\n");
            return nullptr;
          }
          x.recv.buffer = std::move(inbuffer);
          x.recv.tensorData.resize(nTensors);
          if (x.recv.receivedTensors >= nTensors) {
            //log("got all %d tensors, let's go bitches!\n", nTensors);
            x.recv.done = true;
            return createTensors(x, l);
          }
        } else {
          x.recv.done = true;
          return inbuffer;
        }
      }
    } else {
      uint32_t tensorIndex;
      uint32_t offset;
      std::string_view data;
      deserializeBuffer(inbuffer, tensorIndex, offset, data);

      log("recv part %d of rid %#x\n", 1 + tensorIndex, rid);

      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqAck, 1 + tensorIndex);
      conn.send(std::move(buffer));

      auto& bucket = rpc.getBucket(container, rid);
      std::unique_lock l(bucket.mutex);
      auto xptr = find(bucket, l);
      if (xptr) {
        auto& x = *xptr;
        if (x.recv.done) {
          //log("already called !? tensor %d\n", tensorIndex);
          return nullptr;
        }
        if (x.recv.tensorData.size() <= tensorIndex) {
          if (x.recv.buffer) {
            // If we received the main part of the request, then tensorData is resized
            // to the exact number of tensors in the request.
            log("received bad tensor\n");
            std::abort();
            return nullptr;
          }
          x.recv.tensorData.resize(std::max((size_t)tensorIndex + 16, x.recv.tensorData.size() + x.recv.tensorData.size() / 2));
        } else {
          if (x.recv.tensorData[tensorIndex].buffer) {
            log("already has tensor data for this index!\n");
            //std::abort();
            return nullptr;
          }
        }
        x.recv.tensorData[tensorIndex] = {std::move(inbuffer), offset, data};
        ++x.recv.receivedTensors;
        if (x.recv.buffer && (size_t)x.recv.receivedTensors >= x.recv.tensorData.size()) {
          //log("do call yo got the last tensor\n");
          x.recv.done = true;
          return createTensors(x, l);
        }
      }
    }
    return nullptr;
  }

  template<bool isIncoming, typename T>
  void handleAck(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, const std::byte* ptr, size_t len) {
    uint32_t partIndex;
    deserializeBuffer(ptr, len, partIndex);
    auto& bucket = rpc.getBucket(container, rid);
    std::optional<std::chrono::steady_clock::duration> duration;
    {
      std::unique_lock l2(rpc.incomingFifoMutex_, std::defer_lock);
      if constexpr (isIncoming) {
        l2.lock();
      }
      std::lock_guard l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i != bucket.map.end()) {
        auto& x = i->second;

        auto now = std::chrono::steady_clock::now();
        if (partIndex && partIndex - 1 >= x.resendTensors.size()) {
          return;
        }
        auto& s = partIndex == 0 ? x.resend : x.resendTensors[partIndex - 1];
        if (!s.acked) {
          log("handleAck got ack for part %d\n", partIndex);
          s.nackCount = 0;
          s.acked = true;
          s.ackTimestamp = now;
          duration = now - s.lastSendTimestamp;
          ++x.partsAcked;
        }

        if (x.partsAcked == 1 + x.resendTensors.size()) {
          if constexpr (isIncoming) {
            if (x.resend.buffer) {
              rpc.totalResponseSize_.fetch_sub(x.resend.buffer->size, std::memory_order_relaxed);
            }
            for (auto& v : x.resendTensors) {
              if (v.buffer) {
                rpc.totalResponseSize_.fetch_sub(v.buffer->size, std::memory_order_relaxed);
              }
            }
            listErase(&x);
            //log("%#x acked and freed\n", rid);
            bucket.map.erase(i);
          }
        }
      }
    }
    if (duration) {
      rpc.addLatency<API>(peer, std::chrono::steady_clock::now(), *duration, partIndex);
    }
  }

  template<typename T>
  void handleNack(T& container, PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, const std::byte* ptr, size_t len) {
    uint32_t partIndex = 0;
    deserializeBuffer(ptr, len, partIndex);
    log("got nack %d\n", partIndex);
    auto& bucket = rpc.getBucket(container, rid);
    std::unique_lock l(bucket.mutex);
    auto i = bucket.map.find(rid);
    if (i != bucket.map.end() && i->second.peer != &peer) {
      log("rid collision on nack! (not fatal error)\n");
      //std::terminate();
      return;
    }
    if (i != bucket.map.end() && i->second.peer == &peer) {
      auto& x = i->second;
      if (partIndex && partIndex - 1 >= x.resendTensors.size()) {
        return;
      }
      auto& s = partIndex == 0 ? x.resend : x.resendTensors[partIndex - 1];
      if (s.buffer) {
        ++s.nackCount;
        log("nackCount is now %d\n", s.nackCount);
        if (!s.connection) {
          log("nack resend\n");
          rpc.resend(peer, s);
        } else {
          log("nack but resend already in progress\n");
        }
      }
    }
  }

  void onRequest(PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) noexcept {
    log("onRequest rid %#x fid %#x  from %s\n", rid, fid, peer.name);
    rid &= ~(uint32_t)1;
    switch (fid) {
    case Rpc::reqFindFunction: {
      // Look up function id by name
      std::string_view name;
      deserializeBuffer(ptr, len, name);
      //log("find function '%s'\n", name);
      RemoteFunction rf;
      {
        std::lock_guard l(rpc.mutex_);
        auto i = rpc.funcIds_.find(name);
        if (i != rpc.funcIds_.end()) {
          rf.id = i->second;
        }
      }
      //log("returning fid %#x\n", rf.id);
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqSuccess, (uint32_t)0, rf);
      conn.send(std::move(buffer));
      break;
    }
    case Rpc::reqAck: {
      // Peer acknowledged that it has received the response
      // (return value of an RPC call)
      handleAck<true>(rpc.incoming_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqPoke: {
      // Peer is poking us to check the status of an RPC call
      //log("got poke for %#x\n", rid);
      handlePoke(rpc.incoming_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqNack: {
      // Peer nacked a poke for us; this means we may need to
      // resend a rpc response
      handleNack(rpc.incoming_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqLookingForPeer: {
      // Peer is looking for some other peer(s)
      std::vector<std::string_view> names;
      deserializeBuffer(ptr, len, names);
      std::vector<PeerImpl*> foundPeers;
      std::unordered_map<std::string_view, std::vector<ConnectionTypeInfo>> info;
      {
        std::lock_guard l(rpc.peersMutex_);
        for (auto name : names) {
          auto i = rpc.peers_.find(name);
          if (i != rpc.peers_.end() && i->second.hasId.load(std::memory_order_relaxed)) {
            //log("Peer is looking for '%s', and we know them!\n", std::string(name).c_str());
            foundPeers.push_back(&i->second);
          } else {
            //log("Peer is looking for '%s', but we don't know them :/\n", std::string(name).c_str());
          }
        }
      }
      if (!foundPeers.empty()) {
        for (auto* ptr : foundPeers) {
          PeerImpl& p = *ptr;
          std::lock_guard l(p.idMutex_);
          if (!p.connections_.empty()) {
            std::vector<ConnectionTypeInfo> vec;
            for (auto& x : p.connections_) {
              if (!x.remoteAddresses.empty()) {
                vec.emplace_back();
                vec.back().name = connectionTypeName.at(&x - p.connections_.data());
                vec.back().addr = x.remoteAddresses;
              }
            }
            if (!vec.empty()) {
              info[p.name] = std::move(vec);
            }
          }
        }
        if (!info.empty()) {
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqPeerFound, info);
          conn.send(std::move(buffer));
        }
      }
      break;
    }
    default:
      if (fid < (uint32_t)Rpc::reqCallOffset) {
        return;
      }
      // RPC call
      size_t index = fid - rpc.baseFuncId_;
      Rpc::FBase* f = nullptr;
      {
        std::lock_guard l(rpc.mutex_);
        if (index < rpc.funcs_.size()) {
          f = &*rpc.funcs_[index];
        }
      }
      if (!f) {
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, Rpc::reqFunctionNotFound, (uint32_t)0);
        conn.send(std::move(buffer));
      } else {
        auto recvbuffer = handleRecv<true>(rpc.incoming_, peer, conn, rid, ptr, len);
        if (recvbuffer) {
          f->call(std::move(recvbuffer), [this, peer = &peer, rid](BufferHandle outbuffer) {
            auto* ptr = dataptr<std::byte>(&*outbuffer);
            std::memcpy(ptr, &rid, sizeof(rid));
            ptr += sizeof(rid);
            uint32_t outFid;
            std::memcpy(&outFid, ptr, sizeof(outFid));
            ptr += sizeof(outFid);

            std::vector<SharedBufferHandle> tensorbuffers;

            if (outFid == Rpc::reqSuccess) {
              uint32_t nTensors = outbuffer->nTensors;
              std::memcpy(ptr, &nTensors, sizeof(nTensors));
              ptr += sizeof(nTensors);
              for (uint32_t i = 0; i != nTensors; ++i) {
                auto& tensorRef = outbuffer->tensors()[i];
                auto& tensor = tensorRef.tensor;
                size_t nBytes = Rpc::Impl::computeStorageNbytes(tensor.sizes(), tensor.strides(), tensor.itemsize());
                BufferHandle tmp;
                serializeToBuffer(tmp, rid, outFid, (uint32_t)~0, i, (uint32_t)tensorRef.offset, std::string_view((const char*)tensor.data_ptr(), nBytes));
                tensorbuffers.push_back(SharedBufferHandle(tmp.release()));
              }
            }

            //std::memcpy(ptr, &fid, sizeof(fid));
            SharedBufferHandle shared(outbuffer.release());
            //log("sending response for rid %#x of %d bytes to %s\n", rid, shared->size, peer->name);
            //conn->send(shared);

            auto now = std::chrono::steady_clock::now();
            Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
            std::lock_guard l2(rpc.incomingFifoMutex_);
            std::unique_lock l(bucket.mutex);
            //peer->banditSend(~0, shared);
            size_t totalResponseSize;
            auto i = bucket.map.find(rid);
            if (i != bucket.map.end()) {
              auto& x = bucket.map[rid];
              x.responseTimestamp = now;
              totalResponseSize = rpc.totalResponseSize_ += shared->size;
              x.timeout = now + std::chrono::milliseconds(250);
              x.resend.buffer = std::move(shared);
              //log("x is %p, resend.buffer is %p\n", (void*)&x, (void*)&*x.resend.buffer);
              rpc.resend(*peer, x.resend);
              listInsert(rpc.incomingFifo_.prev, &x);

              for (auto& buffer : tensorbuffers) {
                totalResponseSize = rpc.totalResponseSize_ += buffer->size;
                x.resendTensors.emplace_back();
                x.resendTensors.back().buffer = std::move(buffer);
                rpc.resend(*peer, x.resendTensors.back());
              }
              rpc.updateTimeout(now + std::chrono::seconds(1));
            } else {
              totalResponseSize = rpc.totalResponseSize_;
            }
            l.unlock();

            // Erase outgoing data if it has not been acknowledged within a
            // certain time period. This prevents us from using resources
            // for peers that are permanently gone.
            auto timeout = std::chrono::seconds(300);
            if (now - rpc.incomingFifo_.next->responseTimestamp >= std::chrono::seconds(5)) {
              if (totalResponseSize < 1024 * 1024 && rpc.incoming_.size() < 1024) {
                timeout = std::chrono::seconds(1800);
              } else if (totalResponseSize >= 1024 * 1024 * 1024 || rpc.incoming_.size() >= 1024 * 1024) {
                timeout = std::chrono::seconds(5);
              }
              timeout = std::chrono::seconds(0);
              while (rpc.incomingFifo_.next != &rpc.incomingFifo_ && now - rpc.incomingFifo_.next->responseTimestamp >= timeout) {
                Rpc::Impl::Incoming* i = rpc.incomingFifo_.next;
                listErase(i);
                auto& iBucket = rpc.getBucket(rpc.incoming_, i->rid);
                std::lock_guard l3(iBucket.mutex);
                if (i->resend.buffer) {
                  rpc.totalResponseSize_ -= i->resend.buffer->size;
                }
                for (auto& v : i->resendTensors) {
                  if (v.buffer) {
                    rpc.totalResponseSize_ -= v.buffer->size;
                  }
                }
                iBucket.map.erase(i->rid);
                //log("permanent timeout of response !?\n");
                //std::abort();
              }
            }
          });
        }
      }
    }
  }

  void onResponse(PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte *ptr, size_t len) noexcept {
    log("onResponse rid %#x fid %#x  from %s\n", rid, fid, peer.name);
    rid |= 1;
    switch (fid) {
    case Rpc::reqClose: {
      log("got reqClose\n");
      auto& x = peer.connections_.at(index<API>);
      std::lock_guard l(x.mutex);
      for (size_t i = 0; i != x.conns.size(); ++i) {
        if (&*x.conns[i] == &conn) {
          peer.throwAway(x, i);
          break;
        }
      }
      break;
    }
    case Rpc::reqPoke: {
      handlePoke(rpc.outgoing_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqAck: {
      handleAck<false>(rpc.outgoing_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqNack: {
      handleNack(rpc.outgoing_, peer, conn, rid, ptr, len);
      break;
    }
    case Rpc::reqPeerFound: {
      std::unordered_map<std::string_view, std::vector<ConnectionTypeInfo>> info;
      deserializeBuffer(ptr, len, info);
      for (auto& [name, vec] : info) {
        //log("%s:: Received some connection info about peer %s\n", std::string(rpc.myName).c_str(), std::string(name).c_str());
        PeerImpl& peer = rpc.getPeer(name);
        std::lock_guard l(peer.idMutex_);
        for (auto& n : vec) {
          for (size_t i = 0; i != peer.connections_.size(); ++i) {
            if (connectionTypeName[i] == n.name) {
              auto& x = peer.connections_[i];
              std::lock_guard l(x.mutex);
              x.valid = true;
              for (auto& v2 : n.addr) {
                if (std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), v2) == x.remoteAddresses.end()) {
                  //log("Adding address %s\n", std::string(v2).c_str());
                  x.remoteAddresses.push_back(rpc.persistentString(v2));
                  if (x.remoteAddresses.size() > 48) {
                    x.remoteAddresses.erase(x.remoteAddresses.begin(), x.remoteAddresses.begin() + 24 - x.remoteAddresses.size());
                  }
                }
              }
              if (x.remoteAddresses.size() > 24) {
                x.remoteAddresses.erase(x.remoteAddresses.begin(), x.remoteAddresses.begin() + 24 - x.remoteAddresses.size());
              }
            }
          }
        }
      }
      break;
    }
    case Rpc::reqFunctionNotFound:
    case Rpc::reqError:
    case Rpc::reqSuccess: {
      auto buffer = handleRecv<false>(rpc.outgoing_, peer, conn, rid, ptr, len);
      if (buffer) {
        Rpc::ResponseCallback response;
        {
          auto& oBucket = rpc.getBucket(rpc.outgoing_, rid);
          std::lock_guard l(oBucket.mutex);
          auto i = oBucket.map.find(rid);
          if (i != oBucket.map.end() && i->second.peer != &peer) {
            log("rid collision on response! (not fatal error)\n");
            log("so this actually can only mean that the request timed out "
                "in between handleRecv and here, and *then* there was another "
                "request with the same rid. It's so unlikely that I'm going to "
                "leave this message here and damned me if it ever gets logged");
            std::terminate();
          }
          if (i != oBucket.map.end() && i->second.peer == &peer) {
            log("got response for rid %#x from %s\n", rid, peer.name);
            response = std::move(i->second.response);
            oBucket.map.erase(i);
          } else {
            log("got response for unknown rid %#x from %s\n", rid, peer.name);
          }
        }
        if (response) {
          if (fid == Rpc::reqFunctionNotFound) {
            Error err("Remote function not found");
            std::move(response)(std::move(buffer), &err);
          } else if (fid == Rpc::reqError) {
            std::string_view str;
            deserializeBuffer(std::move(buffer), str);
            Error err{"Remote exception during RPC call: " + std::string(str)};
            std::move(response)(nullptr, &err);
          } else if (fid == Rpc::reqSuccess) {
            std::move(response)(std::move(buffer), nullptr);
          }
        }
      }
      break;
    }
    default:
      log("onResponse: unknown fid %#x\n", fid);
    }
  }

  template<typename... Args>
  void log(const char* fmt, Args&&... args) {
    rpc.log(fmt, std::forward<Args>(args)...);
  }

};


template<typename... Args>
void PeerImpl::log(const char* fmt, Args&&... args) {
  rpc.log(fmt, std::forward<Args>(args)...);
}

std::string_view PeerImpl::rpcName() {
  return rpc.myName;
}

void PeerImpl::findPeer() {
  rpc.findPeer(name);
}

void PeerImpl::throwAway(std::unique_ptr<RpcConnectionImplBase> c) {
  std::lock_guard l(rpc.garbageMutex_);
  rpc.garbageConnections_.push_back(std::move(c));
}

template<typename API, bool explicit_>
void PeerImpl::connect(std::string_view addr) {
  rpc.connect<API, explicit_>(addr);
}

//void RpcListener::accept(Function<void(RpcConnection*, Error*)>&& callback) {
//  impl_->accept(std::move(callback));
//}

Rpc::Rpc() {
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_Network>>>();
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_TPUV>>>();
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_TPSHM>>>();
  //impl_ = std::make_unique<RpcImpl<APIWrapper<API_InProcess>>>();

  impl_ = std::make_unique<Rpc::Impl>();
}
Rpc::~Rpc() {}

std::pair<std::string_view, std::string_view> splitUri(std::string_view uri) {
  const char* s = uri.begin();
  const char* e = uri.end();
  const char* i = s;
  while (i != e) {
    if (*i == ':') {
      ++i;
      if (i != e && i + 1 != e && *i == '/' && i[1] == '/') {
        return {std::string_view(uri.begin(), i - 1 - s), std::string_view(i + 2, e - (i + 2))};
      } else {
        return {std::string_view(uri.begin(), i - 1 - s), std::string_view(i, e - i)};
      }
    }
    if (!std::islower(*i)) {
      break;
    }
    ++i;
  }
  return {std::string_view(), std::string_view(s, e - s)};
}

void Rpc::listen(std::string_view addr) {
  auto [scheme, path] = splitUri(addr);
  if (!scheme.empty()) {
    switchOnScheme(scheme, [&, path = path](auto api) {
      impl_->setupDone_ = true;
      impl_->listen<decltype(api)>(path);
    });
  } else {
    impl_->listen<API_TPUV>(addr);
  }
}
void Rpc::connect(std::string_view addr) {
  auto [scheme, path] = splitUri(addr);
  if (!scheme.empty()) {
    switchOnScheme(scheme, [&, path = path](auto api) {
      impl_->setupDone_ = true;
      impl_->connect<decltype(api)>(path);
    });
  } else {
    impl_->connect<API_TPUV>(addr);
  }
}
void Rpc::setName(std::string_view name) {
  impl_->setName(name);
}
std::string_view Rpc::getName() const {
  return impl_->myName;
}
void Rpc::sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response) {
  impl_->sendRequest(peerName, functionName, std::move(buffer), std::move(response));
}
void Rpc::define(std::string_view name, std::unique_ptr<FBase>&& f) {
  impl_->define(name, std::move(f));
}


}
