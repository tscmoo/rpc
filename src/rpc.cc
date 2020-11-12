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

  alignas(64) std::atomic<tensorpipe::FunctionPointer> readCallback = nullptr;
  alignas(64) std::atomic<tensorpipe::FunctionPointer> writeCallback = nullptr;
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
  static constexpr bool addressIsIp = false;

  static std::vector<std::string> defaultAddr() {
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
    return {s};
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
  static constexpr bool addressIsIp = true;

  static std::vector<std::string> defaultAddr() {
    return {"0.0.0.0", "::"};
  }
  static std::string localAddr(const Listener& listener, [[maybe_unused]] std::string addr) {
    return listener->addr();
  }
  static std::string localAddr(const Connection& x) {
    return ((const tensorpipe::transport::uv::Connection&)*x).localAddr();
  }
  static std::string remoteAddr(const Connection& x) {
    return ((const tensorpipe::transport::uv::Connection&)*x).remoteAddr();
  }

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

enum class ConnectionType {
  uv,
  shm,
  count
};
static std::array<const char*, (int)ConnectionType::count> connectionTypeName = {
  "TCP/IP",
  "Shared memory"
};

template<typename API> struct index_t;
template<> struct index_t<API_TPUV> { static constexpr ConnectionType value = ConnectionType::uv; };
template<> struct index_t<API_TPSHM> { static constexpr ConnectionType value = ConnectionType::shm; };
template<typename API> constexpr size_t index = (size_t)index_t<API>::value;

template<typename T> struct RpcImpl;

bool addressIsIp(ConnectionType t) {
  switch (t) {
  case ConnectionType::uv:
    return true;
  case ConnectionType::shm:
    return false;
  default:
    std::abort();
  }
}

struct ConnectionTypeInfo {
  std::string name;
  std::vector<std::string> addr;

  template<typename X>
  void serialize(X& x) {
    x(name, addr);
  }
};

struct RpcConnectionImplBase {
  virtual ~RpcConnectionImplBase() {}
  virtual void close() = 0;

  virtual std::string localAddr() const {
    return "";
  }
  virtual std::string remoteAddr() const {
    return "";
  }
};

struct RpcListenerImplBase {
  virtual ~RpcListenerImplBase() {}

  virtual std::string localAddr() const {
    return "";
  }
};

struct Connection {
  std::atomic<bool> valid = false;
  std::atomic<float> banditScore = 0.0f;
  std::atomic<std::chrono::steady_clock::time_point> lastTryConnect = std::chrono::steady_clock::time_point::min();
  SpinMutex mutex;
  bool outgoing = false;
  std::string addr;
  std::atomic<bool> hasConn = false;
  std::unique_ptr<RpcConnectionImplBase> conn;

  std::vector<std::string> remoteAddresses;
};

struct Listener {
  int activeCount = 0;
  int explicitCount = 0;
  int implicitCount = 0;
  std::vector<std::unique_ptr<RpcListenerImplBase>> listeners;
};

struct PeerId {
  std::array<uint64_t, 2> id;
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

template<typename API>
struct RpcConnectionImpl;

struct PeerImpl {
  Rpc::Impl& rpc;
  std::atomic_int activeOps{0};
  std::atomic_bool dead{false};

  alignas(64) SpinMutex idMutex_;
  PeerId id;
  std::string_view name;
  std::vector<ConnectionTypeInfo> info;

  std::array<Connection, (int)ConnectionType::count> connections_;

  alignas(64) SpinMutex remoteFuncsMutex_;
  std::unordered_map<std::string_view, RemoteFunction> remoteFuncs_;

  alignas(64) SpinMutex banditMutex_;

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

  template<typename Buffer>
  void banditSend(uint32_t mask, Buffer buffer) noexcept {
    //printf("banditSend %d bytes mask %#x\n", (int)buffer->size, mask);
    auto now = std::chrono::steady_clock::now();
    thread_local std::vector<std::pair<size_t, float>> list;
    list.clear();
    float sum = 0.0f;
    for (size_t i = 0; i != connections_.size(); ++i) {
      if (~mask & (1 << i)) {
        continue;
      }
      auto& v = connections_[i];
      if (v.valid.load(std::memory_order_acquire) && (v.hasConn.load() || v.lastTryConnect.load(std::memory_order_relaxed) + std::chrono::seconds(30) <= now)) {
        float score = v.banditScore.load(std::memory_order_relaxed);
        score = std::exp(score * 4);
        sum += score;
        list.emplace_back(i, score);
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
      //printf("bandit chose %d (%s)\n", index, connectionTypeName.at(index));
      //std::this_thread::sleep_for(std::chrono::milliseconds(250));
      auto& x = connections_.at(index);
      if (x.valid) {
        bool b;
        switch (index) {
        case (size_t)ConnectionType::uv:
          b = send<API_TPUV>(now, buffer);
          break;
        case (size_t)ConnectionType::shm:
          b = send<API_TPSHM>(now, buffer);
          break;
        default:
          printf("Unknown connection type %d\n", index);
          std::abort();
        }
        if (!b && buffer) {
          mask &= ~(1 << index);
          return banditSend(mask, std::move(buffer));
        }
      }
    } else {
      printf("no valid connections to bandit among :(\n");
    }
  }

  template<typename API>
  void connect(std::string_view addr);

  template<typename API, typename Buffer>
  bool send(std::chrono::steady_clock::time_point now, Buffer& buffer) {
    auto& x = connections_[index<API>];
    std::unique_lock l(x.mutex);
    if (!x.conn) {
      x.lastTryConnect = now;
      if (x.remoteAddresses.empty()) {
        x.valid = false;
      } else {
        std::string addr;
        if (x.remoteAddresses.size() == 1) {
          addr = x.remoteAddresses[0];
        } else {
          addr = x.remoteAddresses[threadRandom<size_t>(0, x.remoteAddresses.size() - 1)];
        }
        l.unlock();
        if (!addr.empty()) {
          printf("connecting to %s!! :D\n", addr.c_str());
          connect<API>(addr);
        }
      }
      return false;
    } else {
      ((RpcConnectionImpl<API>&)*x.conn).send(std::move(buffer));
      return true;
    }
  }
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
  T* operator->() const noexcept {
    return me;
  }
  T& operator*() const noexcept {
    return *me;
  }
};

template<typename T>
auto makeMe(T* v) {
  return Me<T>(v);
}

template<typename API>
struct RpcConnectionImpl : RpcConnectionImplBase {
  RpcConnectionImpl(RpcImpl<API>& rpc, typename API::Connection&& connection) : rpc(rpc), connection(std::move(connection)) {}
  RpcImpl<API>& rpc;

  typename API::Connection connection;

  PeerImpl* peer = nullptr;

  std::atomic_int activeOps{0};
  std::atomic_bool dead{false};

  ~RpcConnectionImpl() {
    close();
    while (activeOps.load(std::memory_order_acquire));
  }

  virtual std::string localAddr() const override {
    return API::localAddr(connection);
  }
  virtual std::string remoteAddr() const override {
    return API::remoteAddr(connection);
  }

  virtual void close() override {
    if (dead.exchange(true, std::memory_order_relaxed)) {
      return;
    }
    dead = true;
    API::cast(connection).close();
  }

  void onError(Error* err) {
    close();
  }
  void onError(const char* err) {
    close();
  }

  template<typename E>
  void onError(E&& error) {
    Error err(API::errstr(error));
    onError(&err);
  }

  static constexpr uint64_t kSignature = 0xff984b883019d443;

  void onData(const std::byte* ptr, size_t len) noexcept {
    //printf("got %d bytes\n", len);
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
          printf("signature mismatch\n");
          std::terminate();
        }
      } catch (const std::exception&) {
        printf("error in greeting\n");
      }
    }
  }

  void greet(std::string_view name, PeerId peerId, const std::vector<ConnectionTypeInfo>& info) {
    printf("%p::greet(\"%s\", %s)\n", this, std::string(name).c_str(), peerId.toString().c_str());
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
    //printf("read %p\n", this);
    API::cast(connection).read([me = std::move(me)](auto&& error, auto&&... args) mutable noexcept {
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
//    if (random(0, 1) == 0) {
//      return;
//    }
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

template<typename API>
struct RpcListenerImpl : RpcListenerImplBase {
  RpcListenerImpl(RpcImpl<API>& rpc, typename API::Listener&& listener, std::string_view addr) : rpc(rpc), listener(std::move(listener)), addr(addr) {
    accept();
  }
  ~RpcListenerImpl() {
    dead = true;
    while (activeOps.load());
  }
  RpcImpl<API>& rpc;
  typename API::Listener listener;
  bool isExplicit = false;
  bool active = false;
  std::string addr;

  std::atomic<bool> dead = false;
  std::atomic_int activeOps{0};

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

struct RpcImplBase {
  Rpc::Impl& rpc;
  RpcImplBase(Rpc::Impl& rpc) : rpc(rpc) {}
};

struct Rpc::Impl {

  alignas(64) SpinMutex mutex_;
  std::list<std::string> stringList_;
  std::unordered_set<std::string_view> stringMap_;
  std::unordered_map<std::string_view, uint32_t> funcIds_;
  std::vector<std::unique_ptr<Rpc::FBase>> funcs_;
  static constexpr size_t maxFunctions_  = 0x100000;
  uint32_t baseFuncId_ = random<uint32_t>(Rpc::reqCallOffset, std::numeric_limits<uint32_t>::max() - maxFunctions_);
  uint32_t nextFuncIndex_ = 0;

  struct Incoming {
    Incoming* prev = nullptr;
    Incoming* next = nullptr;
    uint32_t rid;
    std::chrono::steady_clock::time_point responseTimestamp;
    SharedBufferHandle responseBuffer;
    PeerImpl* peer = nullptr;
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
    PeerImpl* peer = nullptr;
    SharedBufferHandle requestBuffer;
  };

  struct OutgoingBucket {
    alignas(64) SpinMutex mutex;
    std::unordered_map<uint32_t, Outgoing> map;
  };

  alignas(64) std::array<OutgoingBucket, 0x10> outgoing_;
  alignas(64) SpinMutex outgoingFifoMutex_;
  Incoming outgoingFifo_;
  alignas(64) std::atomic<uint32_t> sequenceId{random<uint32_t>(0, std::numeric_limits<uint32_t>::max())};

  alignas(64) std::array<std::unique_ptr<RpcImplBase>, (int)ConnectionType::count> rpcs_;

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

  std::atomic<bool> setupDone_ = false;
  std::atomic<bool> doingSetup_ = false;
  std::vector<ConnectionTypeInfo> info_;

  template<typename API>
  void tryInitRpc(size_t index) {
    try {
      auto u = std::make_unique<RpcImpl<API>>(*this);
      rpcs_[index] = std::move(u);
    } catch (const std::exception& e) {
      printf("Error during init of '%s': %s\n", connectionTypeName.at(index), e.what());
    }
  }

  Impl() {
    printf("%p peer id is %s\n", this, myId.toString().c_str());
    incomingFifo_.next = &incomingFifo_;
    incomingFifo_.prev = &incomingFifo_;
    outgoingFifo_.next = &outgoingFifo_;
    outgoingFifo_.prev = &outgoingFifo_;

    tryInitRpc<API_TPUV>((size_t)ConnectionType::uv);
    tryInitRpc<API_TPSHM>((size_t)ConnectionType::shm);
  }
  virtual ~Impl() {
    printf("~Impl deadlock, fixme\n");
    if (timeoutThread_.joinable()) {
      timeoutThread_.join();
    }
  }

  template<typename T>
  auto& getBucket(T& arr, uint32_t rid) {
    return arr[(rid >> 1) % arr.size()];
  }

  void processTimeout(std::chrono::steady_clock::time_point now, Outgoing& o) {
    auto newTimeout = now + std::chrono::seconds(1);
    printf("process timeout!\n");
    if (o.peer) {
      if (!o.acked) {
        printf("timeout sending poke for rid %#x\n", o.rid);
        BufferHandle buffer;
        serializeToBuffer(buffer, o.rid, Rpc::reqPoke);
        o.peer->banditSend(~0, std::move(buffer));
      }
    }
    o.timeout = newTimeout;
  }

  template<typename L, typename T>
  void collect(L& lock, T& ref) {
    if (!garbageConnections_.empty()) {
      thread_local T tmp;
      std::swap(ref, tmp);
      lock.unlock();
      tmp.clear();
      printf("GARBAGE COLLECTED YEY!\n");
      lock.lock();
    }
  }

  void collectGarbage() {
    std::unique_lock l(garbageMutex_);
    collect(l, garbageConnections_);
    collect(l, garbageListeners_);
  }

  void startTimeoutThread() {
    timeoutThread_ = std::thread([this]() {
      async::setCurrentThreadName("timeout");
      //std::this_thread::sleep_for(std::chrono::milliseconds(250));
      //printf("timeout thread running!\n");
      while (!timeoutDead_.load(std::memory_order_relaxed)) {
        auto now = std::chrono::steady_clock::now();
        auto timeout = timeout_.load(std::memory_order_relaxed);
        printf("timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
        if (now < timeout) {
          printf("%p sleeping for %d\n", this, std::chrono::duration_cast<std::chrono::milliseconds>(timeout - now).count());
          timeoutSem_.wait_for(timeout - now);
          printf("%p woke up\n", this);
          collectGarbage();
          continue;
        }
        auto newTimeout = now + std::chrono::seconds(5);
        timeout_.store(newTimeout);
        for (auto& b : outgoing_) {
          std::lock_guard l(b.mutex);
          for (auto& v : b.map) {
            if (now >= v.second.timeout) {
              processTimeout(now, v.second);
            }
            newTimeout = std::min(newTimeout, v.second.timeout);
          }
        }
        timeout = timeout_.load(std::memory_order_relaxed);
        while (newTimeout < timeout && !timeout_.compare_exchange_weak(timeout, newTimeout));
        printf("new timeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(newTimeout - now).count());
      }
    });
  }

  void define(std::string_view name, std::unique_ptr<Rpc::FBase>&& f) {
    name = persistentString(name);
    std::lock_guard l(mutex_);
    if (nextFuncIndex_ >= maxFunctions_) {
      throw Error("Too many RPC functions defined");
    }
    uint32_t id = baseFuncId_ + nextFuncIndex_++;
    size_t index = id - baseFuncId_;
    if (funcs_.size() <= index) {
      funcs_.resize(std::max(index + 1, funcs_.size() + funcs_.size() / 2));
    }
    funcIds_[name] = id;
    funcs_[index] = std::move(f);
  }
  std::string_view persistentString(std::string_view name) {
    std::lock_guard l(mutex_);
    auto i = stringMap_.find(name);
    if (i != stringMap_.end()) {
      return name;
    }
    stringList_.emplace_back(name);
    return *stringMap_.emplace(stringList_.back()).first;
  }

  template<typename API>
  void setup() noexcept {
    auto& x = listeners_.at(index<API>);
    if (x.implicitCount > 0) {
      return;
    }
    for (auto& addr : API::defaultAddr()) {
      listen<API, false>(addr);
    }
  }

  auto& getInfo() noexcept {
    printf("%p::getInfo()\n", this);
    if (!setupDone_) {
      if (doingSetup_.exchange(true)) {
        while (!setupDone_);
      } else {
        setup<API_TPUV>();
        setup<API_TPSHM>();

        std::lock_guard l(listenersMutex_);
        info_.clear();
        for (size_t i = 0; i != listeners_.size(); ++i) {
          ConnectionTypeInfo ci;
          ci.name = connectionTypeName.at(i);
          for (auto& v : listeners_[i].listeners) {
            try {
              ci.addr.push_back(v->localAddr());
            } catch (const std::exception& e) {
              printf(":: %s\n", e.what());
            }
          }
          info_.push_back(std::move(ci));
        }

        setupDone_ = true;
      }
    }
    return info_;
  }

  template<typename API>
  void connect(std::string_view addr) {
    auto* u = getImpl<API>();
    if (!u) {
      throw std::runtime_error("Backend " + std::string(connectionTypeName.at(index<API>)) + " is not available");
    }

    getInfo();

    auto c = std::make_unique<Connection>();
    std::unique_lock l(listenersMutex_);
    c->outgoing = true;
    c->addr = persistentString(addr);
    auto cu = std::make_unique<RpcConnectionImpl<API>>(*u, u->context.connect(std::string(addr)));
    cu->greet(myName, myId, info_);
    cu->start();
    c->conn = std::move(cu);
    floatingConnections_.push_back(std::move(c));
    floatingConnectionsMap_[&*floatingConnections_.back()->conn] = std::prev(floatingConnections_.end());
  }

  template<typename API, bool explicit_ = true>
  auto listen(std::string_view addr) {
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
//    if (x.listener) {
//      if constexpr (!explicit_) {
//        return false;
//      } else {
//        throw std::runtime_error("Already listening on backend " + std::string(connectionTypeName.at(index<API>)));
//      }
//    }
    std::unique_ptr<RpcListenerImpl<API>> i;
    try {
      i = std::make_unique<RpcListenerImpl<API>>(*u, std::move(ul), addr);
      i->active = true;
      i->isExplicit = explicit_;
      ++x.activeCount;
      ++(explicit_ ? x.explicitCount : x.implicitCount);
      i->localAddr();
      x.listeners.push_back(std::move(i));
      printf("%s::listen(%s) success\n", std::string(myName).c_str(), std::string(addr).c_str());
    } catch (const std::exception& e) {
      std::lock_guard l(garbageMutex_);
      garbageListeners_.push_back(std::move(i));
      printf("errror in listen(%s): %s\n", std::string(addr).c_str(), e.what());
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

  void sendRequest(PeerImpl& peer, uint32_t fid, BufferHandle buffer, rpc::Rpc::ResponseCallback response) noexcept {
    auto* ptr = dataptr<std::byte>(&*buffer);
    uint32_t rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
    auto* ridPtr = ptr;
    std::memcpy(ptr, &rid, sizeof(rid));
    ptr += sizeof(rid);
    std::memcpy(ptr, &fid, sizeof(fid));
    auto now = std::chrono::steady_clock::now();
    auto myTimeout = now + std::chrono::seconds(1);
    {
      //printf("send request %#x %#x\n", rid, fid);
//      if (conn.dead.load(std::memory_order_relaxed)) {
//        Error err("RPC call: connection is closed");
//        std::move(response)(nullptr, 0, &err);
//        return;
//      }
      SharedBufferHandle shared(buffer.release());
      peer.banditSend(~0, shared);
      auto& oBucket = getBucket(outgoing_, rid);
      std::lock_guard l(oBucket.mutex);
      auto in = oBucket.map.try_emplace(rid);
      while (!in.second) {
        rid = sequenceId.fetch_add(1, std::memory_order_relaxed) << 1 | 1;
        std::memcpy(ridPtr, &rid, sizeof(rid));
        in = oBucket.map.try_emplace(rid);
      }
      auto& q = in.first->second;
      q.rid = rid;
      q.peer = &peer;
      q.requestTimestamp = now;
      q.timeout = myTimeout;
      q.response = std::move(response);
      q.requestBuffer = std::move(shared);
    }
    //printf("myTimeout is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(myTimeout - now).count());
    static_assert(std::atomic<std::chrono::steady_clock::time_point>::is_always_lock_free);
    auto timeout = timeout_.load(std::memory_order_acquire);
    while (myTimeout < timeout) {
      if (timeout_.compare_exchange_weak(timeout, myTimeout)) {
        //printf("timeout set to %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout_.load() - now).count());
        //printf("waking up %p\n", this);
        timeoutSem_.post();
      }
    }
    //printf("timeout post is in %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(timeout_.load() - now).count());
    if (!timeoutThread_.joinable()) {
      startTimeoutThread();
    }
  }

  PeerImpl& getPeer(std::string_view name) {
    std::lock_guard l(peersMutex_);
    auto i = peers_.try_emplace(name, *this);
    auto& p = i.first->second;
    if (i.second) {
      p.name = name;
    }
    return p;
  }

  void sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response) noexcept {
    auto& peer = getPeer(peerName);
    uint32_t fid = peer.functionId(functionName);
    if (fid == 0) {
      functionName = persistentString(functionName);
      BufferHandle buffer2;
      serializeToBuffer(buffer2, (uint32_t)0, (uint32_t)0, functionName);
      sendRequest(peer, reqFindFunction, std::move(buffer2), [this, peer = &peer, functionName = functionName, buffer = std::move(buffer), response = std::move(response)](const void* ptr, size_t len, Error* error) mutable noexcept {
        if (error) {
          std::move(response)(nullptr, 0, error);
        } else {
          RemoteFunction rf;
          deserializeBuffer(ptr, len, rf);
          uint32_t fid = rf.id;
          //printf("got id %#x\n", id);
          if (fid == 0) {
            Error err("RPC remote function " + std::string(peer->name) + "::" + std::string(functionName) + "' does not exist");
            std::move(response)(nullptr, 0, &err);
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
    printf("onAccept!()\n");
    getInfo();
    std::unique_lock l(listenersMutex_);
    if (err) {
      printf("accept error: %s\n", err->what());
      listener.active = false;
      bool isExplicit = listener.isExplicit;
      auto& x = listeners_.at(index<API>);
      --x.activeCount;
      --(isExplicit ? x.explicitCount : x.implicitCount);
      if (isExplicit) {
        onError_(*err);
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
      printf("accept got connection!\n");
      auto c = std::make_unique<Connection>();
      conn->greet(myName, myId, info_);
      conn->start();
      c->conn = std::move(conn);
      floatingConnections_.push_back(std::move(c));
      floatingConnectionsMap_[&*floatingConnections_.back()->conn] = std::prev(floatingConnections_.end());
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
    printf("onGreeting!(\"%s\", %s)\n", std::string(peerName).c_str(), peerId.toString().c_str());
    for (auto& v : info) {
      printf(" %s\n", v.name.c_str());
      for (auto& v2 : v.addr) {
        printf("  @ %s\n", v2.c_str());
      }
    }
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
        garbageConnections_.push_back(std::move(cptr->conn));
        printf("I connected to myself! oops!\n");
        return;
      }
      if (peerName == myName) {
        std::lock_guard l(garbageMutex_);
        garbageConnections_.push_back(std::move(cptr->conn));
        printf("Peer with same name as me! Refusing connection!\n");
        return;
      }

      peerName = persistentString(peerName);

      PeerImpl& peer = getPeer(peerName);
      {
        std::lock_guard l(peer.idMutex_);
        peer.name = peerName;
        peer.id = peerId;
        peer.info = std::move(info);
      }
      if (&conn != &*cptr->conn) {
        std::abort();
      }
      conn.peer = &peer;
      std::unique_ptr<RpcConnectionImplBase> oldconn;
      {
        auto& x = peer.connections_[index<API>];
        std::lock_guard l(x.mutex);
        x.outgoing = cptr->outgoing;
        x.addr = std::move(cptr->addr);
        if (x.conn) {
          oldconn = std::move(x.conn);
        }
        x.hasConn = true;
        x.conn = std::move(cptr->conn);
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
              if (x.remoteAddresses.size() >= 24) {
                continue;
              }
              x.valid = true;
              std::string addr;
              if (API::addressIsIp && addressIsIp((ConnectionType)i)) {
                try {
                  addr = conn.remoteAddr();
                } catch (const std::exception& e) {}
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
                  if (x.remoteAddresses.size() < 24 && std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), newAddr) == x.remoteAddresses.end()) {
                    x.remoteAddresses.push_back(newAddr);
                  }
                }
              } else {
                for (auto& v2 : v.addr) {
                  if (x.remoteAddresses.size() < 24 && std::find(x.remoteAddresses.begin(), x.remoteAddresses.end(), v2) == x.remoteAddresses.end()) {
                    x.remoteAddresses.push_back(v2);
                  }
                }
              }
            }
          }
        }
      }

      for (auto& b : outgoing_) {
        std::lock_guard l(b.mutex);
        for (auto& v : b.map) {
          auto& o = v.second;
          if (o.peer == &peer && !o.acked && o.requestBuffer) {
            printf("poking on newly established connection\n");
            BufferHandle buffer;
            serializeToBuffer(buffer, o.rid, Rpc::reqPoke);
            peer.banditSend(1 << index<API>, std::move(buffer));
          }
        }
      }
    }
  }

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
struct RpcImpl : RpcImplBase {
  typename API::Context context;

  RpcImpl(Rpc::Impl& rpc) : RpcImplBase(rpc) {}

  void onAccept(RpcListenerImpl<API>& listener, std::unique_ptr<RpcConnectionImpl<API>>&& conn, Error* err) {
    rpc.onAccept(listener, std::move(conn), err);
  }

  void onGreeting(RpcConnectionImpl<API>& conn, std::string_view peerName, PeerId peerId, std::vector<ConnectionTypeInfo>&& info) {
    rpc.onGreeting(conn, peerName, peerId, std::move(info));
  }

  void onRequest(PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte* ptr, size_t len) noexcept {
    //printf("onRequest rid %#x fid %#x %p %d\n", rid, fid, ptr, len);
    rid &= ~(uint32_t)1;
    if (fid == Rpc::reqFindFunction) {
      //printf("find function\n");
      std::string_view name;
      deserializeBuffer(ptr, len, name);
      RemoteFunction rf;
      {
        std::lock_guard l(rpc.mutex_);
        auto i = rpc.funcIds_.find(name);
        if (i != rpc.funcIds_.end()) {
          rf.id = i->second;
        }
      }
      BufferHandle buffer;
      serializeToBuffer(buffer, rid, Rpc::reqSuccess, rf);
      conn.send(std::move(buffer));
    } else if (fid == Rpc::reqAck) {
      // Peer acknowledged that it has received the response
      // (return value of an RPC call)
      Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
      std::lock_guard l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i != bucket.map.end()) {
        auto& x = i->second;
        if (x.responseBuffer) {
          //printf("responsed acked, yey, freeing response buffer\n");
          std::lock_guard l2(rpc.incomingFifoMutex_);
          rpc.totalResponseSize_.fetch_sub(x.responseBuffer->size, std::memory_order_relaxed);
          listErase(&x);
        }
        bucket.map.erase(i);
      }
    } else if (fid == Rpc::reqPoke) {
      // Peer is poking us to check the status of an RPC call
      printf("got poke for %#x\n", rid);
      Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
      std::unique_lock l(bucket.mutex);
      auto i = bucket.map.find(rid);
      if (i == bucket.map.end()) {
        printf("got poke for unknown rid\n");
        BufferHandle buffer;
        serializeToBuffer(buffer, rid, Rpc::reqNotFound);
        conn.send(std::move(buffer));
      } else {
        auto& x = i->second;
        if (x.responseBuffer) {
          SharedBufferHandle r = x.responseBuffer;
          l.unlock();
          printf("re-sending response of %d bytes\n", r->size);
          conn.send(r);
        } else {
          printf("poke ack\n");
          BufferHandle buffer;
          serializeToBuffer(buffer, rid, Rpc::reqAck);
          conn.send(std::move(buffer));
        }
        return;
      }
    } else if (fid >= (uint32_t)Rpc::reqCallOffset) {
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
        serializeToBuffer(buffer, rid, Rpc::reqFunctionNotFound);
        conn.send(std::move(buffer));
      } else {
        {
          Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
          std::unique_lock l(bucket.mutex);
          auto i = bucket.map.try_emplace(rid);
          auto& x = i.first->second;
          if (i.second) {
            x.rid = rid;
          } else {
            if (x.peer != &peer) {
              printf("rid collision! (not fatal!)\n");
              std::terminate();
            }
            if (x.responseBuffer) {
              SharedBufferHandle r = x.responseBuffer;
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
        f->call(std::move(inbuffer), [this, peer = &peer, rid, conn = makeMe(&conn)](BufferHandle outbuffer) {
          auto* ptr = dataptr<std::byte>(&*outbuffer);
          std::memcpy(ptr, &rid, sizeof(rid));
  //        ptr += sizeof(rid);
  //        std::memcpy(ptr, &fid, sizeof(fid));
          SharedBufferHandle shared(outbuffer.release());
          //printf("sending response of %d bytes (%p)\n", shared->size, &*shared);
          //conn->send(shared);
          peer->banditSend(~0, shared);

          auto now = std::chrono::steady_clock::now();
          Rpc::Impl::IncomingBucket& bucket = rpc.getBucket(rpc.incoming_, rid);
          std::unique_lock l(bucket.mutex);
          auto& x = bucket.map[rid];
          x.responseTimestamp = now;
          size_t totalResponseSize = rpc.totalResponseSize_ += shared->size;
          x.responseBuffer = std::move(shared);
          l.unlock();
          std::lock_guard l2(rpc.incomingFifoMutex_);
          listInsert(rpc.incomingFifo_.prev, &x);

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
              auto* i = rpc.incomingFifo_.next;
              listErase(i);
              auto& iBucket = rpc.getBucket(rpc.incoming_, i->rid);
              std::lock_guard l3(iBucket.mutex);
              if (i->responseBuffer) {
                rpc.totalResponseSize_ -= i->responseBuffer->size;
              }
              iBucket.map.erase(i->rid);
            }
          }
        });
      }
    }
  }

  void onResponse(PeerImpl& peer, RpcConnectionImpl<API>& conn, uint32_t rid, uint32_t fid, const std::byte *ptr, size_t len) noexcept {
    rid |= 1;
    if (fid == Rpc::reqAck) {
      printf("got req ack, cool\n");
      return;
    } else if (fid == Rpc::reqNotFound) {
      printf("req not found, re-sending\n");
      auto& oBucket = rpc.getBucket(rpc.outgoing_, rid);
      std::unique_lock l(oBucket.mutex);
      auto i = oBucket.map.find(rid);
      if (i != oBucket.map.end() && i->second.peer != &peer) {
        printf("rid collision! (not fatal error)\n");
        std::terminate();
      }
      if (i != oBucket.map.end() && i->second.peer == &peer) {
        Rpc::Impl::Outgoing& x = i->second;
        if (x.requestBuffer) {
          SharedBufferHandle r = x.requestBuffer;
          l.unlock();
          printf("re-sending request of %d bytes\n", r->size);
          conn.send(r);
        }
      }
      return;
    }
    BufferHandle buffer;
    serializeToBuffer(buffer, rid, Rpc::reqAck);
    //printf("ack response for rid %#x\n", rid);
    conn.send(std::move(buffer));
    Rpc::ResponseCallback response;
    {
      auto& oBucket = rpc.getBucket(rpc.outgoing_, rid);
      std::lock_guard l(oBucket.mutex);
      auto i = oBucket.map.find(rid);
      if (i != oBucket.map.end() && i->second.peer != &peer) {
        printf("rid collision! (not fatal error)\n");
        std::terminate();
      }
      if (i != oBucket.map.end() && i->second.peer == &peer) {
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

};


template<typename API>
void PeerImpl::connect(std::string_view addr) {
  rpc.connect<API>(addr);
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

void Rpc::listen(std::string_view addr) {
  impl_->listen<API_TPUV>(addr);
}
void Rpc::connect(std::string_view addr) {
  impl_->connect<API_TPUV>(addr);
}
void Rpc::setName(std::string_view name) {
  impl_->setName(name);
}
void Rpc::sendRequest(std::string_view peerName, std::string_view functionName, BufferHandle buffer, ResponseCallback response) {
  impl_->sendRequest(peerName, functionName, std::move(buffer), std::move(response));
}
void Rpc::define(std::string_view name, std::unique_ptr<FBase>&& f) {
  impl_->define(name, std::move(f));
}


}
