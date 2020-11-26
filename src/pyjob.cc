
#include "pybind11/pybind11.h"

#include <pybind11/numpy.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include "job.h"

#include "rpc.h"

#include "fmt/printf.h"

#include <cmath>
#include <random>
#include <fstream>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <x86intrin.h>

extern "C" {
//#include "../nle/include/hack.h"
#include "../nle/include/nledl.h"
}


namespace py = pybind11;

py::object pyLogging;

std::mutex logMutex;

template<typename... Args>
void log(const char* fmt, Args&&... args) {
  std::lock_guard l(logMutex);
  if (pyLogging.is_none()) {
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
  } else {
    auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
    if (s.size() && s.back() == '\n') {
      s.pop_back();
    }
    py::gil_scoped_acquire gil;
    pyLogging.attr("info")(s);
  }
}

std::pair<std::string_view, py::object> pyStrView(const py::handle& v) {
  char *buffer;
  ssize_t length;
  if (PyUnicode_Check(v.ptr())) {
      py::object o = py::reinterpret_steal<py::object>(PyUnicode_AsUTF8String(v.ptr()));
      if (!o) {
          py::pybind11_fail("Unable to extract string contents! (encoding issue)");
      }
      if (PYBIND11_BYTES_AS_STRING_AND_SIZE(o.ptr(), &buffer, &length)) {
          py::pybind11_fail("Unable to extract string contents! (invalid type)");
      }
      return {std::string_view(buffer, (size_t) length), std::move(o)};
  }
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(v.ptr(), &buffer, &length)) {
      py::pybind11_fail("Unable to extract string contents! (invalid type)");
  }
  return {std::string_view(buffer, (size_t) length), {}};
}

namespace rpc {

enum pyTypes : uint8_t {
  bool_,
  float_,
  dict,
  str,
  array,
  int_,
};

template <typename X>
void serialize(X& x, const py::bool_& v) {
  if (v.ptr() == Py_True) {
    x(true);
  } else if (v.ptr() == Py_False) {
    x(false);
  } else {
    log("bad bool\n");
    std::abort();
  }
}

template <typename X>
void serialize(X& x, const py::float_& v) {
  x((float)v);
}

template <typename X>
void serialize(X& x, const py::dict& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}
template <typename X>
void serialize(X& x, const py::str& v) {
  x(pyStrView(v).first);
}
template <typename X>
void serialize(X& x, const py::array& v) {
  ssize_t ndim = v.ndim();
  if (ndim < 0) {
    throw std::runtime_error("Cant serialize this array");
  }
  x(ndim);
  auto* shape = v.shape();
  auto* strides = v.strides();
  for (ssize_t i = 0; i != v.ndim(); ++i) {
    x((ssize_t)shape[i]);
    x((ssize_t)strides[i]);
  }
  size_t bytes = 1;
  for (ssize_t i = 0; i != v.ndim(); ++i) {
    if (shape[i] == 0) {
      bytes = 0;
    }
    bytes += strides[i]*(shape[i]-1);
  }
  bytes *= v.itemsize();
  x(std::string_view((const char*)v.data(), bytes));
}

template <typename X>
void serialize(X& x, const py::handle& v) {
  if (v.ptr() == Py_True) {
    x(pyTypes::bool_, true);
  } else if (v.ptr() == Py_False) {
    x(pyTypes::bool_, false);
  } else if (py::isinstance<py::float_>(v)) {
    x(pyTypes::float_, (float)py::reinterpret_borrow<py::float_>(v));
  } else if (py::isinstance<py::dict>(v)) {
    x(pyTypes::dict, py::reinterpret_borrow<py::dict>(v));
  } else if (py::isinstance<py::str>(v)) {
    x(pyTypes::str, py::reinterpret_borrow<py::str>(v));
  } else if (py::isinstance<py::array>(v)) {
    x(pyTypes::array, py::reinterpret_borrow<py::array>(v));
  } else if (py::isinstance<py::int_>(v)) {
    x(pyTypes::int_, (int64_t)py::reinterpret_borrow<py::int_>(v));
  } else {
    throw std::runtime_error("Can't serialize python type " + std::string(py::str(v.get_type())));
  }
}

template <typename X>
void serialize(X& x, const py::tuple& v) {
  size_t n = v.size();
  x(n);
  for (auto& v2 : v) {
    x(v2);
  }
}

}

struct SharedMemory {
  int fd = -1;
  size_t size = 1024 * 1024 * 100;
  std::byte* data = nullptr;
  std::string name;
  bool unlinked = false;
  SharedMemory(std::string_view name) : name(name) {
    log("creating shm %s\n", name);
    fd = shm_open(std::string(name).c_str(), O_RDWR | O_CREAT, ACCESSPERMS);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category(), "shm_open");
    }
    if (ftruncate(fd, size)) {
      throw std::system_error(errno, std::system_category(), "ftruncate");
    }
    data = (std::byte*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!data) {
      throw std::system_error(errno, std::system_category(), "mmap");
    }
  }
  ~SharedMemory() {
    munmap(data, size);
    close(fd);
    if (!unlinked) {
      shm_unlink(name.c_str());
    }
  }
  void unlink() {
    if (!unlinked) {
      shm_unlink(name.c_str());
      unlinked = true;
    }
  }
  template<typename T>
  T& as() {
    if (sizeof(T) > size) {
      log("%s is too big for shm :(\n", typeid(T).name());
      std::abort();
    }
    ((T*)data)->init(size);
    return *(T*)data;
  }
};

class Semaphore {
  sem_t sem;

 public:
  Semaphore() noexcept {
    sem_init(&sem, 1, 0);
  }
  ~Semaphore() {
    sem_destroy(&sem);
  }
  void post() noexcept {
    sem_post(&sem);
  }
  void wait() noexcept {
    while (sem_wait(&sem)) {
      if (errno != EINTR) {
        std::abort();
      }
    }
  }
  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    struct timespec ts;
    auto absduration = std::chrono::system_clock::now().time_since_epoch() + duration;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(absduration);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(nanoseconds);
    ts.tv_sec = seconds.count();
    ts.tv_nsec = (nanoseconds - seconds).count();
    while (sem_timedwait(&sem, &ts)) {
      if (errno == ETIMEDOUT) {
        break;
      }
      if (errno != EINTR) {
        std::abort();
      }
    }
  }
  template<typename Clock, typename Duration>
  void wait_until(const std::chrono::time_point<Clock, Duration>& timePoint) noexcept {
    wait_for(timePoint - Clock::now());
  }

  Semaphore(const Semaphore&) = delete;
  Semaphore(const Semaphore&&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&&) = delete;
};



thread_local std::minstd_rand threadRng(std::random_device{}());

std::atomic_int counter = 0;
std::atomic_int resetCount = 0;
std::atomic_int namecounter = 0;

std::atomic_int lastcount = 0;

template<typename Duration>
float seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(duration).count();
}

struct Timer {
  std::chrono::steady_clock::time_point start;
  Timer() {
    reset();
  }
  void reset() {
    start = std::chrono::steady_clock::now();
  }
  float elapsedAt(std::chrono::steady_clock::time_point now) {
    return std::chrono::duration_cast<
               std::chrono::duration<float, std::ratio<1, 1>>>(now - start)
        .count();
  }
  float elapsed() {
    return elapsedAt(std::chrono::steady_clock::now());
  }
  float elapsedReset() {
    auto now = std::chrono::steady_clock::now();
    float r = elapsedAt(now);
    start = now;
    return r;
  }
};

std::atomic<std::chrono::steady_clock::duration::rep> steptime{};
std::atomic<std::chrono::steady_clock::duration::rep> resettime{};

std::chrono::steady_clock::time_point astart;

struct TestTask: moo::Task {

  TestTask() {
  }


};

std::string randomName() {
  std::string r;
  for (int i = 0; i != 16; ++i) {
    r += "0123456789abcdef"[std::uniform_int_distribution<int>(0, 15)(threadRng)];
  }
  return r;
}

template<typename T>
struct SharedPointer {
  size_t offset = 0;
  template<typename Shared>
  T* operator()(Shared* shared) {
    return (T*)((std::byte*)shared + offset);
  }
};

template<typename T>
struct SharedArray {
  size_t size;
  SharedPointer<T> data;

  template<typename Shared>
  std::basic_string_view<T> view(Shared* shared) {
    return {data(shared), size};
  }

  template<typename Shared>
  T& operator()(Shared* shared, size_t index) {
    return data(shared)[index];
  }
  template<typename Shared>
  T* operator()(Shared* shared) {
    return data(shared);
  }
};

struct SharedMapEntry {
  SharedArray<char> key;
  SharedArray<int64_t> shape;
  SharedArray<std::byte> data;
  size_t elements;
  size_t itemsize;
  char dtype;
};

struct BatchData {
  SharedArray<SharedMapEntry> data;
};

constexpr size_t maxClients = 0x100;
constexpr size_t maxEnvs = 0x400;

constexpr int kInvalidAction = 0xffff;

struct Shared {
  std::atomic_bool initialized;
  std::atomic_bool initializing;
  size_t size;
  std::atomic_size_t allocated = sizeof(*this);
  void init(size_t size) {
    if (initialized) {
      return;
    }
    if (initializing.exchange(true)) {
      while (!initialized);
      return;
    }
    new (this) Shared();
    this->size = size;
    initialized = true;
  }

  struct Buffer {

    struct ClientInput {
      std::atomic_size_t nStepsIn = 0;
      std::atomic_size_t resultOffset = 0;
    };
    struct alignas(64) ClientOutput {
      std::atomic_size_t nStepsOut = 0;
    };

    struct EnvInput {
      std::atomic<int> action = kInvalidAction;
    };

    size_t size = 0;
    size_t stride = 0;

    std::array<ClientInput, maxClients> clientInputs;
    std::array<EnvInput, maxEnvs> envInputs{};
    std::array<ClientOutput, maxClients> clientOutputs;

    std::atomic_bool batchAllocated;
    std::atomic_bool batchAllocating;
    BatchData batchData;
  };

  alignas(64) std::atomic_int clients = 0;
  std::array<Buffer, 2> buffers;


  std::byte* allocateNonAligned(size_t n) {
    size_t offset = allocated.fetch_add(n, std::memory_order_relaxed);
    if (offset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return (std::byte*)this + offset;
  }

  template<typename T>
  SharedArray<T> allocate(size_t n) {
    size_t offset = allocated.fetch_add(sizeof(T) * n, std::memory_order_relaxed);
    if (offset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return {n, {offset}};
  }

  template<typename T>
  SharedArray<T> allocateAligned(size_t n) {
    size_t offset = allocated.load(std::memory_order_relaxed);
    size_t newOffset;
    do {
      newOffset = (offset + 63) / 64 * 64;
    } while (!allocated.compare_exchange_weak(offset, newOffset + sizeof(T) * n, std::memory_order_relaxed));
    if (newOffset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return {n, {newOffset}};
  }

  template<typename T>
  SharedArray<T> allocateString(std::basic_string_view<T> str) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto r = allocate<T>(str.size());
    std::memcpy(r(this), str.data(), sizeof(T) * str.size());
    return r;
  }
};

struct BatchBuilderHelper {
  std::unordered_map<std::string_view, bool> added;
  struct I {
    std::string key;
    std::vector<int64_t> shape;
    size_t elements;
    size_t itemsize;
    char dtype;
  };
  std::vector<I> fields;
  void add(std::string_view key, size_t dims, const ssize_t* shape, size_t itemsize, char dtype) {
    if (std::exchange(added[key], true)) {
      throw std::runtime_error("key " + std::string(key) + " already exists in batch!");
    }
    size_t elements = 1;
    for (size_t i = 0; i != dims; ++i) {
      elements *= shape[i];
    }
    fields.emplace_back();
    auto& f = fields.back();
    f.key = key;
    f.shape.assign(shape, shape + dims);
    f.elements = elements;
    f.itemsize = itemsize;
    f.dtype = dtype;
  }
};

torch::ScalarType getTensorDType(char dtype, int itemsize) {
  switch (dtype) {
  case 'f':
    if (itemsize == 2) {
      return torch::kFloat16;
    } else if (itemsize == 4) {
      return torch::kFloat32;
    } else if (itemsize == 8) {
      return torch::kFloat64;
    } else {
      throw std::runtime_error("Unexpected itemsize for float");
    }
    break;
  case 'i':
    if (itemsize == 1) {
      return torch::kInt8;
    } else if (itemsize == 2) {
      return torch::kInt16;
    } else if (itemsize == 4) {
      return torch::kInt32;
    } else if (itemsize == 8) {
      return torch::kInt64;
    } else throw std::runtime_error("Unexpected itemsize for int");
    break;
  case 'u':
    if (itemsize == 1) {
      return torch::kUInt8;
    } else throw std::runtime_error("Unexpected itemsize for unsigned int");
    break;
  case 'b':
    if (itemsize == 1) {
      return torch::kBool;
    } else throw std::runtime_error("Unexpected itemsize for boolean");
    break;
  default:
    throw std::runtime_error("Unsupported dtype '" + std::string(1, dtype) + "'");
  }
}

struct Env {
  py::object env_;
  py::object reset_;
  py::object step_;

  uint64_t steps = 0;
  //std::string_view outbuffer;

  uint32_t episodeStep = 0;
  float episodeReturn = 0;

  Env(py::object env) {
    py::gil_scoped_acquire gil;
    env_ = std::move(env);
    reset_ = env_.attr("reset");
    step_ = env_.attr("step");
  }

  ~Env() {
    py::gil_scoped_acquire gil;
    env_ = {};
    reset_ = {};
    step_ = {};
  }

  void allocateBatch(Shared* shared, BatchData& batch, const py::dict& obs) {

    BatchBuilderHelper bb;

    for (auto& [key, value] : obs) {
      auto [str, stro] = pyStrView(key);
      py::array arr = py::reinterpret_borrow<py::object>(value);
      bb.add(str, arr.ndim(), arr.shape(), arr.itemsize(), arr.dtype().kind());
    }

    std::array<ssize_t, 1> s{1};
    bb.add("done", 1, s.data(), 1, 'b');
    bb.add("rewards", 1, s.data(), 4, 'f');
    bb.add("episode_step", 1, s.data(), 4, 'i');
    bb.add("episode_return", 1, s.data(), 4, 'f');

    batch.data = shared->allocate<SharedMapEntry>(bb.fields.size());
    for (size_t i = 0; i != bb.fields.size(); ++i) {
      auto& f = bb.fields[i];
      auto& d = batch.data(shared, i);
      d.key = shared->allocateString(std::string_view(f.key));
    }
    for (size_t i = 0; i != bb.fields.size(); ++i) {
      auto& f = bb.fields[i];
      auto& d = batch.data(shared, i);
      d.data = shared->allocateAligned<std::byte>(f.itemsize * f.elements * maxEnvs);
      d.shape = shared->allocate<int64_t>(f.shape.size());
      for (size_t i2 = 0; i2 != f.shape.size(); ++i2) {
        d.shape(shared, i2) = f.shape[i2];
      }
      d.elements = f.elements;
      d.itemsize = f.itemsize;
      d.dtype = f.dtype;
    }
    log("allocated %d bytes\n", (int)shared->allocated);
  }

  void fillBatch(Shared* shared, BatchData& batch, size_t batchIndex, std::string_view key, void* src, size_t len) {
    auto* map = batch.data(shared);
    for (size_t i = 0; i != batch.data.size; ++i) {
      auto& v = map[i];
      if (v.key.view(shared) == key) {
        std::byte* dst = v.data(shared);
        dst += v.itemsize * v.elements * batchIndex;
        if (len != v.itemsize * v.elements) {
          throw std::runtime_error("fill batch size mismatch");
        }
        std::memcpy(dst, src, v.itemsize * v.elements);
        return;
      }
    }
    throw std::runtime_error(std::string(key) + ": batch key not found");
  }

  void step(Shared* shared, size_t bufferIndex, size_t batchIndex) {
    //log("step buffer %d index %d\n", bufferIndex, batchIndex);
    auto start = std::chrono::steady_clock::now();
    ++steps;
    int action = kInvalidAction;
    if (steps != 1) {
      auto& sa = shared->buffers[bufferIndex].envInputs[batchIndex].action;
      do {
        action = sa.load(std::memory_order_acquire);
      } while (action == kInvalidAction);
      sa.store(kInvalidAction, std::memory_order_release);
      //log("buffer %d index %d got action %d\n", bufferIndex, batchIndex, action);
    }
    try {
      py::gil_scoped_acquire gil;
      bool done;
      float reward;
      py::dict obs;
      float localEpisodeReturn = episodeReturn;
      uint32_t localEpisodeStep = episodeStep;
      //log("buffer %d index %d step\n", bufferIndex, batchIndex);
      if (steps == 1) {
        done = false;
        reward = 0.0f;
        obs = reset_();
      } else {
        py::tuple tup = step_(action);
        obs = tup[0];
        reward = (py::float_)tup[1];
        done = (py::bool_)tup[2];
        localEpisodeReturn = episodeReturn += reward;
        localEpisodeStep = ++episodeStep;
        if (done) {
          obs = reset_();
          episodeStep = 0;
          episodeReturn = 0;
        }
      }
      auto& buffer = shared->buffers[bufferIndex];
      auto& batch = buffer.batchData;
      if (!buffer.batchAllocated.load(std::memory_order_acquire)) {
        if (buffer.batchAllocating.exchange(true)) {
          while (!buffer.batchAllocated);
        } else {
          allocateBatch(shared, batch, obs);
          buffer.batchAllocated = true;
        }
      }
      fillBatch(shared, batch, batchIndex, "done", &done, sizeof(bool));
      fillBatch(shared, batch, batchIndex, "rewards", &reward, sizeof(float));
      fillBatch(shared, batch, batchIndex, "episode_step", &localEpisodeStep, sizeof(uint32_t));
      fillBatch(shared, batch, batchIndex, "episode_return", &localEpisodeReturn, sizeof(float));
      for (auto& [key, value] : obs) {
        auto [str, stro] = pyStrView(key);
        py::array arr(py::reinterpret_borrow<py::array>(value));
        fillBatch(shared, batch, batchIndex, str, (float*)arr.data(), arr.nbytes());
      }
//      if (outbuffer.size() == 0) {
//        rpc::BufferHandle buffer = rpc::serializeToBuffer(tup);
//        log("Result serialized to %d bytes\n", buffer->size);
//        outbuffer = shared->allocate(buffer->size + buffer->size / 2);
//        log("allocated to %p, size %d\n", outbuffer.data(), outbuffer.size());
//      }
//      rpc::serializeToStringView(outbuffer);
    } catch (const pybind11::error_already_set &e) {
      log("step error %s\n", e.what());
      throw;
    }
    counter.fetch_add(1, std::memory_order_relaxed);
    steptime += (std::chrono::steady_clock::now() - start).count();
    //if (steps >= 50000) std::abort();
  }

  void reset() {
    auto start = std::chrono::steady_clock::now();
    if (++resetCount == 0) {
      astart = start;
    }
    steps = 0;
    try {
      py::gil_scoped_acquire gil;
      reset_();
      if (start - astart >= std::chrono::seconds(5)) {
        float s = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(start - astart).count();

        int cn = counter;
        int n = cn - lastcount;
        float nps = n / s;
        log("%d,  %g/s\n", n, nps);

        lastcount = cn;

        astart = start;
      }
    } catch (const pybind11::error_already_set &e) {
      log("reset error %s\n", e.what());
      throw;
    }
    resettime += (std::chrono::steady_clock::now() - start).count();
  }
};

struct EnvBatch {
  py::object envInit_;
  std::vector<Env> envs;
  EnvBatch() = default;
  EnvBatch(py::object envInit) : envInit_(std::move(envInit)) {}
  ~EnvBatch() {
    py::gil_scoped_acquire gil;
    envInit_ = {};
  }
  void step(size_t size, Shared* shared, size_t bufferIndex, size_t batchIndex) {
    while (envs.size() < size) {
      py::gil_scoped_acquire gil;
      envs.emplace_back(envInit_());
    }
    for (auto& v : envs) {
      v.step(shared, bufferIndex, batchIndex);
      ++batchIndex;
    }
  }
};

struct LocalBuffer {
  std::vector<torch::Tensor> inputShared;
  std::vector<torch::Tensor> inputPinned;
  std::vector<torch::Tensor> modelStates;
  torch::Tensor actionPinned;
  std::optional<c10::cuda::CUDAStream> cudaStream;

  std::map<std::string, torch::Tensor> inputMap;
  std::map<std::string, torch::Tensor> outputMap;

  std::vector<torch::Tensor> inputDeviceSeq;
  std::vector<torch::Tensor> outputDeviceSeq;

  size_t currentSequenceIndex = 0;

  alignas(64) std::atomic_int stepCount = 0;
  std::atomic_bool busy = false;
};

std::mutex profileMutex;
std::map<std::string_view, float> profileTimes;

struct Profile {
  std::string_view name;
  Timer t;
  bool running = true;
  Profile(std::string_view name) : name(name) {}
  ~Profile() {
    stop();
  }
  void stop() {
    if (running) {
      running = false;
      std::lock_guard l(profileMutex);
      profileTimes[name] += t.elapsed();
    }
  }
};

template<typename T>
struct Future {
private:
  using IT = std::conditional_t<std::is_same_v<T, void>, std::nullptr_t, T>;
  struct S {
    std::optional<IT> value;
    std::atomic_bool hasValue = false;
  };
  std::shared_ptr<S> s;
public:
  Future() {
    s = std::make_shared<S>();
  }
  void set() {
    s->value.emplace();
    s->hasValue = true;
  }
  template<typename T2>
  void set(T2&& val) {
    s->value = std::move(val);
    s->hasValue = true;
  }
  operator bool() const noexcept {
    return s->hasValue;
  }
  IT& operator*() {
    return *s->value;
  }
  IT* operator->() {
    return &*s->value;
  }
};

template<typename T, typename... Args>
Future<T> callImpl(rpc::Rpc& rpc, std::string_view peerName, std::string_view funcName, Args&&... args) {
  Future<T> retval;
  rpc.asyncCallback<T>(peerName, funcName, [retval](T* value, rpc::Error* err) mutable {
    if (value) {
      if constexpr (!std::is_same_v<T, void>) {
        retval.set(*value);
      } else {
        retval.set();
      }
    } else {
      log("RPC error: %s\n", err->what());
    }
  }, std::forward<Args>(args)...);
  return retval;
}

struct TestJob {

  std::thread runThread;

  std::atomic_bool terminate_ = false;

  std::string shmname = "nle-" + randomName();
  SharedMemory shm{shmname};
  Shared* shared = &shm.as<Shared>();

  int numClients_;
  py::object model_;
  std::string deviceStr_;
  py::object learner_;

  std::optional<rpc::Rpc> rpc;

  Future<void> pingFuture;
  bool hasPinged = false;
  std::string groupName_;

  uint32_t numUpdates_ = 0;
  std::string syncMaster;

  std::mutex syncMutex;
  uint32_t syncId_ = -1;
  std::vector<std::string> members_;

  bool haveNewParameters = false;
  uint32_t newNumUpdates = 0;
  std::vector<torch::Tensor> newParameters;
  std::vector<torch::Tensor> newBuffers;

  size_t numSyncedGradients = 0;
  std::vector<torch::Tensor> syncedGrads;

  std::string myName;
  std::vector<torch::Tensor> modelParameters;
  std::vector<torch::Tensor> modelBuffers;

  bool gradMode = false;

  template<typename T, typename... Args>
  Future<T> call(std::string_view peerName, std::string_view funcName, Args&&... args) {
    return callImpl<T>(*rpc, peerName, funcName, std::forward<Args>(args)...);
  }

  TestJob(std::string brokerAddress, std::string groupName) : groupName_(groupName) {
    if (!brokerAddress.empty() && !groupName.empty()) {
      rpc.emplace();

      myName = rpc->getName();

      rpc->define<uint32_t(std::string_view, uint32_t)>("sync", [this](std::string_view group, uint32_t syncId) {
        std::lock_guard l(syncMutex);
        if (group == groupName_) {
          syncId_ = syncId;
          log("synced with id %#x\n", syncId);
          return syncId;
        } else {
          return (uint32_t)-1;
        }
      });

      rpc->define<std::string_view(std::string_view, std::vector<std::string>)>("update", [this](std::string_view group, std::vector<std::string> members) {
        std::lock_guard l(syncMutex);
        if (group == groupName_ && !members.empty()) {
          std::string s = "{";
          for (auto& v : members) {
            if (s.size() != 1) {
              s += ", ";
            }
            s += v;
          }
          s += "}";
          log("group %s update members %s\n", group, s);
          members_ = std::move(members);
          return members_.front();
        } else {
          return std::string();
        }
      });

      rpc->define<bool(std::string_view, uint32_t, uint32_t, std::vector<torch::Tensor>)>("grads", [this](std::string_view group, uint32_t syncId, uint32_t numUpdates, std::vector<torch::Tensor> grads) {
        std::lock_guard l(syncMutex);
        if (syncMaster != myName) {
          log("Got grads but I am not the master!\n");
          return false;
        }
        if (group != groupName_) {
          log("Got grads for wrong group %s\n", group);
          return false;
        }
        if (syncId != syncId_) {
          log("Got grads for wrong syncId (%#x, should be %#x)\n", syncId, syncId_);
          return false;
        }
        if (grads.size() != syncedGrads.size()) {
          log("Got grads for wrong number of grads (%d, should be %d)\n", grads.size(), syncedGrads.size());
          return false;
        }
        log("Recv grads %#x %#x\n", syncId, numUpdates);
        if (numUpdates_ - numUpdates <= 1) {
          ++numSyncedGradients;
          for (size_t i = 0; i != grads.size(); ++i) {
            syncedGrads[i].add_(grads[i]);
          }
          return true;
        } else {
          log("Ignoring grads as they are too old (age %d)\n", numUpdates_ - numUpdates);
          return false;
        }
      });

      rpc->define<bool(std::string_view, uint32_t, uint32_t, std::vector<torch::Tensor>, std::vector<torch::Tensor>)>("modelUpdate", [this](std::string_view group, uint32_t syncId, uint32_t numUpdates, std::vector<torch::Tensor> parameters, std::vector<torch::Tensor> buffers) {
        std::lock_guard l(syncMutex);
        if (group != groupName_) {
          log("Got model update for wrong group %s\n", group);
          return false;
        }
        if (syncId != syncId_) {
          log("Got model update for wrong syncId (%#x, should be %#x)\n", syncId, syncId_);
          return false;
        }
        if (parameters.size() != modelParameters.size()) {
          log("Got model update for wrong number of parameters (%d, should be %d)\n", parameters.size(), modelParameters.size());
          return false;
        }
        if (buffers.size() != modelBuffers.size()) {
          log("Got model update for wrong number of parameters (%d, should be %d)\n", buffers.size(), modelBuffers.size());
          return false;
        }
        log("got modelUpdate %d\n", numUpdates);
        haveNewParameters = true;
        newNumUpdates = numUpdates;
        newParameters = std::move(parameters);
        newBuffers = std::move(buffers);
        return true;
      });

      log("My name is %s, connecting to broker at %s\n", myName, brokerAddress);
      rpc->connect(brokerAddress);
      auto fut = rpc->async<size_t>("broker", "groupSize", groupName);
      if (fut.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        throw std::runtime_error("Timed out connecting to broker at " + brokerAddress);
      }
      size_t n = fut.get();
      log("Group %s has %d existing members\n", groupName, n);
    }
  }
  ~TestJob() {
    terminate_ = true;
    //shared->doneSem.post();
    //log("Note: Hackily quick-exiting\n");
    std::quick_exit(1);
    py::gil_scoped_release gil;
    runThread.join();
  }

  void start(int numClients, py::object model, std::string deviceStr, py::object learner) {
    numClients_ = numClients;
    model_ = model;
    deviceStr_ = deviceStr;
    learner_ = learner;
    runThread = std::thread([this]() {
      run();
    });
  }

  struct EnvState {
    std::vector<torch::Tensor> modelState;
  };

  struct ClientState {
    std::vector<EnvState> states;
  };

  template<typename T>
  std::string sizesStr(T&& sizes) {
    std::string s = "{";
    for (auto& v : sizes) {
      if (s.size() > 1) {
        s += ", ";
      }
      s += std::to_string(v);
    }
    s += "}";
    return s;
  }

  struct ModelUpdateTracker {
    std::optional<Future<bool>> future;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::unordered_map<std::string, ModelUpdateTracker> modelUpdateTracker;

  void run() {

    torch::Device device(deviceStr_);
    bool isCuda = device.is_cuda();
    torch::NoGradGuard ng;

    log("isCuda: %d\n", isCuda);

    Timer t;
    int count = 0;

    size_t bufferCounter = 0;

    std::deque<LocalBuffer> local;

    local.resize(shared->buffers.size());

    async::SchedulerFifo asyncforward;
    asyncforward.pool.maxThreads = 1;

    async::SchedulerFifo asyncAction;
    asyncAction.pool.maxThreads = 2;

    async::SchedulerFifo asyncSync;

    std::mutex modelMutex;

    size_t nClients = 0;

    std::optional<Future<bool>> gradFuture;
    std::chrono::steady_clock::time_point gradTimestamp;
    int masterLagCount = 0;

    while (shared->clients != numClients_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (terminate_) {
        return;
      }
    }

    shm.unlink();

    nClients = numClients_;

    int batchSize;
    int unrollLength;
    {
      py::gil_scoped_acquire gil;
      batchSize = learner_.attr("batch_size").cast<int>();
      unrollLength = learner_.attr("unroll_length").cast<int>();
    }

    log("Using an unroll length of %d and a batch size of %d\n", unrollLength, batchSize);

    {
      py::gil_scoped_acquire gil;
      size_t numelp = 0;
      size_t numelb = 0;
      for (py::handle h : model_.attr("parameters")()) {
        auto t = h.cast<torch::Tensor>();
        numelp += t.numel();
        modelParameters.push_back(std::move(t));
      }
      for (py::handle h : model_.attr("buffers")()) {
        auto t = h.cast<torch::Tensor>();
        numelb += t.numel();
        modelBuffers.push_back(std::move(t));
      }
      log("Model has %d parameters across %d tensors\n", numelp, modelParameters.size());
      log("Model has %d buffers across %d tensors\n", numelb, modelBuffers.size());
    }

    Timer mainTimer;

    while (!terminate_) {

      if (nClients > maxClients) {
        throw std::runtime_error("Too many clients connected!");
      }

      auto doForwardBuffer = [&](size_t bufferIndex) {
        auto& buffer = shared->buffers[bufferIndex];
        auto& lbuf = local[bufferIndex];

        while (lbuf.busy);

        std::optional<c10::cuda::CUDAStreamGuard> streamGuard;
        if (isCuda) {
          lbuf.cudaStream = c10::cuda::getStreamFromPool(false, device.index());
          streamGuard.emplace(*lbuf.cudaStream);
        }

        size_t size = buffer.size;
        size_t stride = buffer.stride;
        size_t clientIndex = 0;
        if (size == 0) {

          size_t size = batchSize;

          size_t strideDivisor = nClients;
          size_t stride = (size + strideDivisor - 1) / strideDivisor;

          buffer.size = size;
          buffer.stride = stride;

          size_t clientIndex = 0;
          for (size_t i = 0; i < size; i += stride, ++clientIndex) {
            int nSteps = std::min(size - i, stride);
            auto& input = buffer.clientInputs[clientIndex];
  //          if (hasAction) {
  //            for (size_t s = i; s != i + nSteps; ++s) {
  //              //log("setting action %d for buffer %d index %d\n", (*acc)[0][s], bufferIndex, s);
  //              buffer.envInputs[s].action.store((*acc)[0][s], std::memory_order_relaxed);
  //            }
  //          }
            input.resultOffset.store(i, std::memory_order_release);
            input.nStepsIn.fetch_add(nSteps, std::memory_order_acq_rel);
          }

          auto& ms = lbuf.modelStates;
          if (ms.empty()) {
            py::gil_scoped_acquire gil;
            py::tuple is = model_.attr("initial_state")(size);
            for (auto& v : is) {
              auto t = v.cast<torch::Tensor>();
              ms.push_back(t.to(device));
            }
          } else {
            throw std::runtime_error("fixme: resize ms");
          }

          return;
        }
        if (lbuf.currentSequenceIndex == (size_t)unrollLength - 1) {
          for (auto& v : local) {
            while (v.busy);
          }
        }
        auto start = std::chrono::steady_clock::now();
        lbuf.busy = true;
        for (size_t i = 0; i < size; i += stride, ++clientIndex) {
//          int nSteps = std::min(size - i, stride);
//          if (clientStates.size() <= clientIndex || clientStates[clientIndex].states.size() < nSteps) {
//            throw std::runtime_error("clientStates mismatch");
//          }
          Profile pf("wait");
          auto& input = buffer.clientInputs[clientIndex];
          auto& output = buffer.clientOutputs[clientIndex];
          size_t prevSteps = input.nStepsIn.load(std::memory_order_acquire);
          uint32_t timeCheckCounter = 0x10000;
          while (output.nStepsOut.load(std::memory_order_relaxed) != prevSteps && !terminate_) {
            _mm_pause();

            if (--timeCheckCounter == 0) {
              timeCheckCounter = 0x10000;
              auto now = std::chrono::steady_clock::now();
              if (now - start >= std::chrono::seconds(600)) {
                log("Timed out waiting for clients\n");
                std::exit(1);
                break;
              }
            }
          }
          if (terminate_) {
            return;
          }
        }
        if (lbuf.inputMap.empty()) {
          std::map<std::string, torch::Tensor> map;
          auto* src = buffer.batchData.data(shared);
          for (size_t i = 0; i != buffer.batchData.data.size; ++i) {
            auto key = src[i].key.view(shared);
            auto& v = src[i];
            torch::TensorOptions opts = getTensorDType(v.dtype, v.itemsize);
            std::vector<int64_t> sizes(v.shape(shared), v.shape(shared) + v.shape.size);
            sizes.insert(sizes.begin(), (int64_t)maxEnvs);
            map[std::string(key)] = torch::from_blob(v.data(shared), sizes, opts);
          }
          for (auto& [key, value] : map) {
            log("input [%s] sizes = %s\n", key, sizesStr(value.sizes()));
            lbuf.inputMap[key] = torch::Tensor();
            lbuf.inputShared.push_back(value);
            if (isCuda) {
              lbuf.inputPinned.push_back(value.pin_memory());
            }

            auto opts = torch::TensorOptions(value.dtype()).device(device);
            std::vector<int64_t> sizes = value.sizes().vec();
            sizes.at(0) = batchSize;
            sizes.insert(sizes.begin(), (int64_t)unrollLength);
            lbuf.inputDeviceSeq.push_back(torch::empty(sizes,  opts));
          }
        }
//        auto& input = inputTmp;
//        for (auto& [key, value] : mapCpuShared) {
//          auto n = value.narrow(0, 0, size);
//          if (key != "done") {
//            n = n.unsqueeze(0); // Create T dimension
//          }
//          input[key] = n.to(device);
//        }

//          for (auto& [key, value] : input) {
//            log("input [%s] sizes = %s\n", key, sizesStr(value.sizes()));
//          }
        asyncforward.run([&, size, stride, bufferIndex]() {
          Profile p("aforward");
          std::optional<c10::cuda::CUDAStreamGuard> streamGuard;
          if (isCuda) {
            streamGuard.emplace(*lbuf.cudaStream);
          }

          std::lock_guard lm(modelMutex);

          torch::AutoGradMode ng(false);

          //std::lock_guard l(modelMutex);

          Profile pprepare("prepare");
          auto& input = lbuf.inputMap;
          auto& ms = lbuf.modelStates;
          size_t index = 0;
          for (auto& [key, value] : input) {
            auto device = lbuf.inputDeviceSeq.at(index).select(0, lbuf.currentSequenceIndex).narrow(0, 0, size);
            value = device;
            auto shared = lbuf.inputShared.at(index).narrow(0, 0, size);
            if (isCuda) {
              auto pinned = lbuf.inputPinned.at(index).narrow(0, 0, size);
              pinned.copy_(shared, true);
              if (!pinned.is_pinned()) {
                throw std::runtime_error("pinned is not pinned!");
              }
              value.copy_(pinned, true);
            } else {
              value.copy_(shared);
            }
            value = value.unsqueeze(0);
            ++index;
          }

          size_t clientIndex = 0;
          for (size_t i = 0; i < size; i += stride, ++clientIndex) {
            int nSteps = std::min(size - i, stride);
            auto& input = buffer.clientInputs[clientIndex];
  //          if (hasAction) {
  //            for (size_t s = i; s != i + nSteps; ++s) {
  //              //log("setting action %d for buffer %d index %d\n", (*acc)[0][s], bufferIndex, s);
  //              buffer.envInputs[s].action.store((*acc)[0][s], std::memory_order_relaxed);
  //            }
  //          }
            input.resultOffset.store(i, std::memory_order_release);
            input.nStepsIn.fetch_add(nSteps, std::memory_order_acq_rel);
          }

//          for (auto& [key, value] : input) {
//            log("input [%s] sizes = %s\n", key, sizesStr(value.sizes()));
//            log("sum %g\n", value.sum().item<float>());
//          }

          pprepare.stop();

          //std::lock_guard lm(modelMutex);

          torch::Tensor action;
          std::map<std::string, torch::Tensor> outputMap;
          {
            torch::AutoGradMode grad(gradMode);
            Profile p("python model forward");
            py::gil_scoped_acquire gil;
            //log("model forward %d\n", bufferIndex);
            py::tuple tup = model_(input, ms, true);
//            while (true) {
//              tup = model_(input, ms, true);

//              count += batchSize;

//              if (t.elapsed() >= 1.0f) {
//                float tx = t.elapsedReset();
//                int n = count;
//                count = 0;
//                log("rate %g/s\n", n / tx);
//              }
//            }
            py::dict output = tup[0];
            py::tuple outstate = tup[1];
            for (size_t i = 0; i != ms.size(); ++i) {
              ms[i] = outstate[i].cast<torch::Tensor>();
            }
            outputMap = output.cast<std::map<std::string, torch::Tensor>>();

            //log("model forward %d done\n", bufferIndex);
          }
          Profile paction("action");
          action = outputMap["action"];
          if (action.dim() != 2 || action.size(0) != 1 || (size_t)action.size(1) != size) {
            throw std::runtime_error("Expected action output of size {1, " + std::to_string(size) + "}, got " + sizesStr(action.sizes()));
          }
          auto& actionPinned = lbuf.actionPinned;
          if (isCuda) {
            if (!actionPinned.defined()) {
              auto vec = action.sizes().vec();
              vec.at(1) = maxEnvs;
              actionPinned = torch::empty(vec, torch::TensorOptions(action.dtype()).pinned_memory(true));
            }
            if (!actionPinned.is_pinned()) {
              throw std::runtime_error("actionPinned is not pinned!");
            }
            actionPinned.narrow(1, 0, size).copy_(action, true);
          } else {
            actionPinned = action;
          }
          paction.stop();
          if (lbuf.outputMap.empty()) {
            lbuf.outputMap = outputMap;
            for (auto& [key, value] : outputMap) {
              auto sizes = value.sizes().vec();
              if (sizes.size() < 2) {
                throw std::runtime_error("Model output has not enough dimensions");
              }
              sizes.at(0) = unrollLength;
              sizes.at(1) = batchSize;
              auto opts = torch::TensorOptions(value.dtype()).device(device);
              lbuf.outputDeviceSeq.push_back(torch::empty(sizes, opts));

              log("new output device seq [%s] = %s\n", key, sizesStr(sizes));
            }
          } else if (lbuf.outputMap.size() != outputMap.size()) {
            throw std::runtime_error("Inconsistent model output dict");
          }

          Profile poutput("output");
          torch::GradMode::set_enabled(gradMode);
          index = 0;
          for (auto& [key, value] : outputMap) {
            if (value.size(0) != 1) {
              throw std::runtime_error("Expected size 1 in first dimension, got " + sizesStr(value.sizes()));
            }
            lbuf.outputDeviceSeq[index].select(0, lbuf.currentSequenceIndex).copy_(value.select(0, 0), true);
            ++index;
          }
          poutput.stop();
          torch::GradMode::set_enabled(false);
          asyncAction.run([this, &buffer, &lbuf, size, isCuda, stream = isCuda ? std::make_optional<c10::cuda::CUDAStream>(streamGuard->current_stream()) : std::optional<c10::cuda::CUDAStream>{}]() {
            if (isCuda) {
              Profile p("synchronize");
              stream->synchronize();
            }
            auto& action = lbuf.actionPinned;
            if (action.scalar_type() != torch::kLong) {
              std::cout << "action is " << action.dtype() << "\n";
              throw std::runtime_error("model output action type mismatch");
            }
            auto acc = action.accessor<long, 2>();
            if (acc.size(0) != 1 || (size_t)acc.size(1) < size) {
              log("size is %d, acc size is %s\n", size, sizesStr(acc.sizes()));
              throw std::runtime_error("model output action size mismatch");
            }
            for (size_t i = 0; i != size; ++i) {
              buffer.envInputs[i].action.store(acc[0][i], std::memory_order_relaxed);
              //buffer.envInputs[i].action.store(10, std::memory_order_relaxed);
            }
          });
          if (gradMode == false) {
            if (lbuf.currentSequenceIndex == (size_t)unrollLength - 1) {
              lbuf.currentSequenceIndex = 0;
            } else {
              ++lbuf.currentSequenceIndex;
            }
          } else {
            if (lbuf.currentSequenceIndex == (size_t)unrollLength - 1) {
              Profile p("step");
              //std::lock_guard lm(modelMutex);
              torch::GradMode::set_enabled(true);
              //log("Stepping buffer %d\n", bufferIndex);
              index = 0;
              for (auto& [key, value] : input) {
                //log("input [%s] = %s\n", key, sizesStr(value.sizes()));
                value = lbuf.inputDeviceSeq.at(index);
                ++index;
              }
              index = 0;
              //log("lbuf.outputDeviceSeq.size() is %d\n", lbuf.outputDeviceSeq.size());
              for (auto& [key, value] : outputMap) {
                //log("output [%s] = %s\n", key, sizesStr(value.sizes()));
                value = lbuf.outputDeviceSeq.at(index);
                ++index;
              }
              //auto msStack = torch::stack(ms);
              {
  //              float sum = 0.0f;
  //              for (auto& v : modelParameters) {
  //                auto& grad = v.grad();
  //                if (grad.defined()) {
  //                  sum += grad.sum().item<float>();
  //                }
  //              }
  //              log("pre step sum of grads: %g\n", sum);
                Profile p("python step");
                py::gil_scoped_acquire gil;
                //log("step %d\n", bufferIndex);
                learner_.attr("step")(input, outputMap);
                //log("step %d done\n", bufferIndex);
              }
              lbuf.currentSequenceIndex = 0;

              for (auto& v : ms) {
                v.detach_();
              }
              for (auto& v : lbuf.inputDeviceSeq) {
                v.detach_();
              }
              for (auto& v : lbuf.outputDeviceSeq) {
                v.detach_();
              }

              torch::GradMode::set_enabled(false);

              //log("Buffer %d stepped\n", bufferIndex);

              int stepCount = ++lbuf.stepCount;

              if (bufferIndex == shared->buffers.size() - 1) {
                //log("Stepping optimizer\n");
                Profile p("optimizer stuff");
                for (auto& v : local) {
                  while (v.stepCount != stepCount);
                }

                if (rpc) {
                  if (hasPinged && !pingFuture) {
                    log("Broker is lagging behind!");
                  } else {
                    hasPinged = true;
                    pingFuture = call<void>("broker", "ping", groupName_, myName, numUpdates_);
                  }
  //                float sum = 0.0f;
  //                for (auto& v : modelParameters) {
  //                  auto& grad = v.grad();
  //                  if (grad.defined()) {
  //                    sum += grad.sum().item<float>();
  //                  }
  //                }
  //                log("sum of grads: %g\n", sum);

                  auto zeroGrad = [&]() {
                    for (auto& v : modelParameters) {
                      auto& grad = v.grad();
                      if (grad.defined()) {
                        grad.zero_();
                      }
                    }
                  };

                  std::unique_lock l(syncMutex);

                  if (!members_.empty()) {
                    std::string_view master = members_.front();
                    if (syncMaster != master) {
                      gradFuture.reset();
                      if (syncMaster.empty()) {
                        log("Master is %s\n", master);
                      } else {
                        log("Master changed from %s to %s\n", syncMaster, master);
                      }
                      if (master == myName) {
                        log("I am now the master!\n");
                      }
                      syncMaster = master;
                    }
                    if (master == myName) {
                      if (syncedGrads.empty()) {
                        for (auto& v : modelParameters) {
                          auto& grad = v.grad();
                          if (grad.defined()) {
                            syncedGrads.push_back(torch::zeros_like(grad, torch::TensorOptions(torch::kCPU).pinned_memory(true)));
                          }
                        }
                      } else {
                        log("Adding in %d gradients\n", numSyncedGradients);
                        if (numSyncedGradients) {
                          size_t i = 0;
                          for (auto& v : modelParameters) {
                            auto& grad = v.grad();
                            if (grad.defined()) {
                              if (i == syncedGrads.size()) {
                                throw std::runtime_error("grads grew?");
                              }
                              grad.add_(syncedGrads[i].to(device, true));
                              grad.mul_(1.0f / (1 + numSyncedGradients));
                              syncedGrads[i].zero_();
                              ++i;
                            }
                          }
                          numSyncedGradients = 0;
                        }
                      }
                      ++numUpdates_;
                      log("numUpdates is now %d\n", numUpdates_);

                      l.unlock();
                      log("Stepping optimizer\n");
                      Profile p("python step_optimizer");
                      {
                        py::gil_scoped_acquire gil;
                        learner_.attr("step_optimizer")();
                      }
                      l.lock();

                      std::vector<torch::Tensor> sendParameters;
                      std::vector<torch::Tensor> sendBuffers;

                      for (auto& v : modelParameters) {
                        sendParameters.push_back(v.to(torch::kCPU, true, true));
                      }
                      for (auto& v : modelBuffers) {
                        sendBuffers.push_back(v.to(torch::kCPU, true, true));
                      }

                      for (auto& n : members_) {
                        if (n != myName) {
                          call<bool>(n, "modelUpdate", groupName_, syncId_, numUpdates_, sendParameters, sendBuffers);
                        }
                      }

                    } else {
                      if (std::find(members_.begin(), members_.end(), myName) != members_.end()) {
                        std::vector<torch::Tensor> grads;
                        for (auto& v : modelParameters) {
                          auto& grad = v.grad();
                          if (grad.defined()) {
                            grads.push_back(grad.to(torch::kCPU, false, true));
                          }
                        }
                        log("Master %s, sync id {%#x, %#x}. Sending gradients\n", syncMaster, syncId_, numUpdates_);
                        auto now = std::chrono::steady_clock::now();
                        if (gradFuture && !*gradFuture && now - gradTimestamp < std::chrono::seconds(30)) {
                          log("Warning: master is lagging behind for grad sync, skipping this one\n");
                          ++masterLagCount;
                          if (masterLagCount == 4) {
                            log("Requesting resync\n");
                            call<void>("broker", "resync", groupName_);
                          }
                        } else {
                          //gradFuture = call<bool>(master, "grads", groupName_, syncId_, numUpdates_, grads);
                          gradTimestamp = now;
                          Future<bool> retval;
                          rpc->asyncCallback<bool>(master, "grads", [retval](bool* value, rpc::Error* err) mutable {
                            if (value) {
                              log("grads returned %d!\n", *value);
                              retval.set(*value);
                            } else {
                              log("RPC error: %s\n", err->what());
                            }
                          }, groupName_, syncId_, numUpdates_, grads);
                          gradFuture = std::move(retval);
                        }
                      } else {
                        log("I am not a member, not participating in sync!\n");
                      }
                    }
                    zeroGrad();
                  } else {
                    log("No members, zeroing grad\n");
                    zeroGrad();
                  }

                  if (haveNewParameters) {
                    log("Got new parameters, numUpdates %d -> %d\n", numUpdates_, newNumUpdates);
                    haveNewParameters = false;
                    numUpdates_ = newNumUpdates;

                    if (modelParameters.size() != newParameters.size()) {
                      throw std::runtime_error("Model parameters size mismatch in update!");
                    }
                    if (modelBuffers.size() != newBuffers.size()) {
                      throw std::runtime_error("Model parameters size mismatch in update!");
                    }

                    for (size_t i = 0; i != modelParameters.size(); ++i) {
                      modelParameters[i].copy_(newParameters[i], true);
                    }
                    for (size_t i = 0; i != modelBuffers.size(); ++i) {
                      modelBuffers[i].copy_(newBuffers[i], true);
                    }
                  }

                  if (isCuda) {
                    streamGuard->current_stream().synchronize();
                  }

                } else {
                  Profile p("python step_optimizer");
                  {
                    py::gil_scoped_acquire gil;
                    //log("optimizer %d\n", bufferIndex);
                    learner_.attr("step_optimizer")();
                    //log("optimizer %d done \n", bufferIndex);
                  }
                }
              }

              p.stop();

  //            float tt = mainTimer.elapsed();
  //            auto ts = [&](float v) {
  //              return fmt::sprintf("%g (%g%%)", v, v * 100 / tt);
  //            };
  //            std::string s;
  //            std::lock_guard l(profileMutex);
  //            for (auto& [key, value] : profileTimes) {
  //              if (!s.empty()) {
  //                s += " ";
  //              }
  //              s += key;
  //              s += ": ";
  //              s += ts(value);
  //            }
  //            log("total time: %g, %s\n", tt, s);
            } else {
              ++lbuf.currentSequenceIndex;
            }
          }
          lbuf.busy = false;
        });
        if (lbuf.currentSequenceIndex == (size_t)unrollLength - 1) {
          while (lbuf.busy);
        }
      };

      size_t forwardBufferIndex = (bufferCounter) % shared->buffers.size();
      doForwardBuffer(forwardBufferIndex);
      ++bufferCounter;

      if (terminate_) {
        break;
      }

      count += batchSize;

      if (t.elapsed() >= 1.0f) {
        float tx = t.elapsedReset();
        int n = count;
        count = 0;
        log("rate %g/s\n", n / tx);
      }
    }



  }

  std::string_view getName() {
    return shmname;
  }

};

struct TestClient {

  std::atomic_bool terminate_ = false;

  std::string serverAddress_;
  std::thread runThread;

  std::array<EnvBatch, 10> batch;

  TestClient(py::object envInit) {
    for (auto& b : batch) {
      b.envInit_ = envInit;
    }
  }

  ~TestClient() {
    py::gil_scoped_release gil;
    terminate_ = true;
    runThread.join();
  }

  void start(std::string serverAddress) {
    serverAddress_ = serverAddress;
    runThread = std::thread([this]() {
      run();
    });
  }

  void run() {
    SharedMemory shm(serverAddress_);
    Shared& shared = shm.as<Shared>();

    int myIndex = shared.clients++;

    log("my index is %d\n", myIndex);

    if ((size_t)myIndex > maxClients) {
      throw std::runtime_error("Client index is too high");
    }

    size_t bufferCounter = 0;

    auto lastUpdate = std::chrono::steady_clock::now();

    while (true) {
      size_t bufferIndex = bufferCounter % shared.buffers.size();
      auto& buffer = shared.buffers[bufferIndex];
      ++bufferCounter;
      auto& input = buffer.clientInputs[myIndex];
      auto& output = buffer.clientOutputs[myIndex];
      size_t stepsDone = output.nStepsOut.load(std::memory_order_acquire);
      size_t nSteps = input.nStepsIn.load(std::memory_order_acquire);
      //log("client %d waiting for work\n", myIndex);
      uint32_t timeCheckCounter = 0x10000;
      while (nSteps == stepsDone) {
        _mm_pause();
        nSteps = input.nStepsIn.load(std::memory_order_acquire);
        if (terminate_.load(std::memory_order_relaxed)) {
          break;
        }
        if (--timeCheckCounter == 0) {
          timeCheckCounter = 0x10000;
          auto now = std::chrono::steady_clock::now();
          if (now - lastUpdate >= std::chrono::seconds(600)) {
            log("Client timed out\n");
            terminate_ = true;
            break;
          }
        }
      }
      if (terminate_.load(std::memory_order_relaxed)) {
        break;
      }
      lastUpdate = std::chrono::steady_clock::now();
      size_t offset = input.resultOffset;
      //log("oh boy i got a task for %d steps\n", nSteps - stepsDone);
      //log("client %d got work for buffer %d offset [%d, %d)\n", myIndex, bufferIndex, offset, offset + (nSteps - stepsDone));
      batch.at(bufferIndex).step(nSteps - stepsDone, &shared, bufferIndex, offset);
      output.nStepsOut.store(nSteps, std::memory_order_release);

//        if (buffer.remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
//          //shared->doneSem.post();
//        }
    }

  }
};

struct Broker {

  struct Peer {
    std::string name;
    std::chrono::steady_clock::time_point lastPing;
    std::optional<Future<uint32_t>> syncFuture;
    std::optional<Future<std::string>> updateFuture;
    uint32_t numUpdates = 0;
    bool active = false;
    size_t order = 0;
  };

  struct Group {
    std::mutex mutex;
    std::string name;
    std::unordered_map<std::string, Peer> peers;
    bool needsUpdate = false;
    std::chrono::steady_clock::time_point lastUpdate;
    uint32_t syncId = 0x42;
    size_t updateCount = 0;
    size_t orderCounter = 0;

    std::vector<std::string> active;

    Peer& getPeer(std::string name) {
      auto i = peers.try_emplace(name);
      if (i.second) {
        auto& p = i.first->second;
        p.name = name;
        p.order = orderCounter++;
      }
      return i.first->second;
    }
  };

  std::mutex groupsMutex;
  std::unordered_map<std::string, Group> groups;

  Group& getGroup(const std::string& name) {
    std::lock_guard l(groupsMutex);
    auto i = groups.try_emplace(name);
    if (i.second) {
      auto& g = i.first->second;
      g.name = name;
    }
    return i.first->second;
  }

  std::atomic_bool terminate_ = false;

  std::string address;
  rpc::Rpc server;
  std::thread runThread;

  Broker(std::string address) : address(address) {}
  ~Broker() {
    terminate_ = true;
    //log("Note: Hackily quick-exiting\n");
    std::quick_exit(1);
    runThread.join();
  }

  template<typename T, typename... Args>
  Future<T> call(std::string_view peerName, std::string_view funcName, Args&&... args) {
    return callImpl<T>(server, peerName, funcName, std::forward<Args>(args)...);
  }

  void start() {
    runThread = std::thread([this]() {
      run();
    });
  }

  void run() {

    log("This is a broker process!\n");

    server.setName("broker");

    server.define<size_t(std::string)>("groupSize", [this](std::string group) {
      log("groupSize called!\n");
      auto& g = getGroup(group);
      std::lock_guard l(g.mutex);
      return g.peers.size();
    });

    server.define<void(std::string, std::string, uint32_t)>("ping", [this](std::string group, std::string name, uint32_t numUpdates) {
      auto& g = getGroup(group);
      std::lock_guard l(g.mutex);
      auto& p = g.getPeer(name);
      p.lastPing = std::chrono::steady_clock::now();
      p.numUpdates = numUpdates;
      if (!p.active) {
        g.needsUpdate = true;
      }
      //log("got ping for %s::%s\n", group, name);
    });

    server.define<void(std::string)>("resync", [this](std::string group) {
      auto& g = getGroup(group);
      std::lock_guard l(g.mutex);
      if (!g.needsUpdate) {
        log("Got resync request for %s\n", group);
        g.needsUpdate = true;
      }
    });

    log("Broker listening on %s\n", address);
    server.listen(address);

    std::unordered_set<Group*> syncSet;
    std::chrono::steady_clock::time_point lastCheckTimeouts;

    std::vector<Group*> tmpGroups;
    std::vector<Peer*> tmpPeers;

    while (!terminate_) {
      if (syncSet.empty()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto now = std::chrono::steady_clock::now();

        for (auto i = syncSet.begin(); i != syncSet.end();) {
          auto& g = **i;
          std::lock_guard l(g.mutex);
          size_t total = 0;
          size_t ready = 0;
          for ([[maybe_unused]] auto& [pname, p] : g.peers) {
            if (p.syncFuture) {
              ++total;
              if (*p.syncFuture) {
                if (**p.syncFuture == g.syncId) {
                  ++ready;
                } else {
                  --total;
                }
              }
            }
          }
          log("Sync midway %s %d/%d in %gs\n", g.name, ready, total, seconds(now - g.lastUpdate));
          if (ready >= total || now - g.lastUpdate >= std::chrono::seconds(1)) {
            log("Sync %s %d/%d in %gs\n", g.name, ready, total, seconds(now - g.lastUpdate));

            tmpPeers.clear();
            for ([[maybe_unused]] auto& [pname, p] : g.peers) {
              if (p.syncFuture && *p.syncFuture && **p.syncFuture == g.syncId) {
                tmpPeers.push_back(&p);
                p.active = true;
              } else {
                p.active = false;
              }
            }
            std::sort(tmpPeers.begin(), tmpPeers.end(), [](Peer* a, Peer* b) {
              if (a->numUpdates == b->numUpdates) {
                return a->order < b->order;
              }
              return a->numUpdates >= b->numUpdates;
            });
            g.active.clear();
            for (auto* p : tmpPeers) {
              log("%s has %d updates\n", p->name, p->numUpdates);
              g.active.push_back(p->name);
            }
            if (!g.active.empty()) {
              log("%s is the master\n", g.active.front());
            }
            for ([[maybe_unused]] auto& [pname, p] : g.peers) {
              if (p.syncFuture && *p.syncFuture && **p.syncFuture == g.syncId) {
                p.updateFuture = call<std::string>(pname, "update", g.name, g.active);
              }
            }

            i = syncSet.erase(i);
          } else {
            ++i;
          }
        }

        if (now - lastCheckTimeouts < std::chrono::milliseconds(500)) {
          continue;
        }
      }
      auto now = std::chrono::steady_clock::now();
      lastCheckTimeouts = now;
      tmpGroups.clear();
      {
        std::lock_guard l(groupsMutex);
        for (auto& [gname, g] : groups) {
          tmpGroups.push_back(&g);
        }
      }
      for (auto* pg : tmpGroups) {
        auto& g = *pg;
        std::lock_guard l2(g.mutex);
        for (auto i = g.peers.begin(); i != g.peers.end();) {
          auto& p = i->second;
          if (now - p.lastPing >= std::chrono::seconds(60)) {
            log("Peer %s::%s timed out\n", g.name, p.name);
            if (p.active) {
              g.needsUpdate = true;
            }
            i = g.peers.erase(i);
          } else {
            ++i;
          }
        }
        auto mintime = std::chrono::seconds(30);
        if (g.updateCount < 10) {
          mintime = std::chrono::seconds(2);
        }
        if (g.needsUpdate && (now - g.lastUpdate >= mintime)) {
          log("Initiating update of group %s\n", g.name);
          ++g.updateCount;
          g.lastUpdate = now;
          g.needsUpdate = false;
          uint32_t syncId = ++g.syncId;
          for ([[maybe_unused]] auto& [pname, p] : g.peers) {
            p.syncFuture = call<uint32_t>(pname, "sync", g.name, syncId);
          }
          syncSet.insert(&g);
        }
      }
    }
  }
};

PYBIND11_MODULE(pyjob, m) {
  m.def("set_logging", [](py::object logging) {
    pyLogging = logging;
  });
  py::class_<Broker>(m, "Broker")
      .def(py::init<std::string>())
      .def("start", &Broker::start);
  py::class_<TestJob>(m, "TestJob")
      .def(py::init<std::string, std::string>())
      .def("start", &TestJob::start)
      .def("get_name", &TestJob::getName);
  py::class_<TestClient>(m, "TestClient")
      .def(py::init<py::object>())
      .def("start", &TestClient::start);
}
