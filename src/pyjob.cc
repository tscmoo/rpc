
#include "pybind11/pybind11.h"

#include <pybind11/numpy.h>

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
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto* tm = std::localtime(&now);
  char buf[0x40];
  std::strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", tm);
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  if (pyLogging.is_none()) {
    if (!s.empty() && s.back() == '\n') {
      fmt::printf("%s: %s", buf, s);
    } else {
      fmt::printf("%s: %s\n", buf, s);
    }
  } else {
    if (s.size() && s.back() == '\n') {
      s.pop_back();
    }
    py::gil_scoped_acquire gil;
    pyLogging.attr("info")(fmt::sprintf("%s: %s", buf, s));
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
  SharedMemory(std::string_view name) : name(name) {
    log("creating shm %s\n", name);
    fd = shm_open(std::string(name).c_str(), O_RDWR | O_CREAT, ACCESSPERMS);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category());
    }
    ftruncate(fd, size);
    data = (std::byte*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!data) {
      throw std::system_error(errno, std::system_category(), "mmap");
    }
  }
  ~SharedMemory() {
    munmap(data, size);
    close(fd);
    shm_unlink(name.c_str());
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
constexpr size_t maxEnvs = 0x1000;

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
      int action = 0;
    };

    size_t size = 0;
    size_t stride = 0;

    std::array<ClientInput, maxClients> clientInputs;
    std::array<EnvInput, maxEnvs> envInputs;
    std::array<ClientOutput, maxClients> clientOutputs;

    std::once_flag batchAllocated;
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

struct Env {
  py::object env_;
  py::object reset_;
  py::object step_;

  int steps = 0;
  //std::string_view outbuffer;

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
    std::unordered_map<std::string_view, bool> added;
    struct I {
      std::string key;
      std::vector<int64_t> shape;
      size_t elements;
      size_t itemsize;
      char dtype;
    };
    std::vector<I> fields;
    auto add = [&](std::string_view key, size_t dims, const ssize_t* shape, size_t itemsize, char dtype) {
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
    };
    for (auto& [key, value] : obs) {
      auto [str, stro] = pyStrView(key);
      py::array arr = py::reinterpret_borrow<py::object>(value);
      add(str, arr.ndim(), arr.shape(), arr.itemsize(), arr.dtype().kind());
    }
    std::array<ssize_t, 1> s{1};
    add("done", 1, s.data(), 1, 'b');
    add("reward", 1, s.data(), 4, 'f');

    batch.data = shared->allocate<SharedMapEntry>(fields.size());
    for (size_t i = 0; i != fields.size(); ++i) {
      auto& f = fields[i];
      auto& d = batch.data(shared, i);
      d.key = shared->allocateString(std::string_view(f.key));
    }
    for (size_t i = 0; i != fields.size(); ++i) {
      auto& f = fields[i];
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
    auto start = std::chrono::steady_clock::now();
    ++steps;
    int action = std::uniform_int_distribution<size_t>(0, 10)(threadRng);
    try {
      py::gil_scoped_acquire gil;
      bool done;
      float reward;
      py::dict obs;
      if (steps == 1) {
        done = false;
        reward = 0.0f;
        obs = reset_();
        //pyLogging.attr("info")(py::str(obs["blstats"]));
      } else {
        py::tuple tup = step_(action);
        obs = tup[0];
        reward = (py::float_)tup[1];
        done = (py::bool_)tup[2];
        if (done) {
          obs = reset_();
        }
      }
      auto& batch = shared->buffers[bufferIndex].batchData;
      std::call_once(shared->buffers[bufferIndex].batchAllocated, [&]() {
        allocateBatch(shared, batch, obs);
      });
      fillBatch(shared, batch, batchIndex, "done", &done, sizeof(bool));
      fillBatch(shared, batch, batchIndex, "reward", &reward, sizeof(float));
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
  std::vector<torch::Tensor> inputDevice;
  std::map<std::string, torch::Tensor> inputMap;
  std::vector<torch::Tensor> modelStates;
};

struct TestJob {

  std::thread runThread;

  std::atomic_bool terminate_ = false;

  std::string shmname = "nle-" + randomName();
  SharedMemory shm{shmname};
  Shared* shared = &shm.as<Shared>();

  py::object model_;
  std::string deviceStr_;

  TestJob(py::object model, std::string deviceStr) : model_(std::move(model)), deviceStr_(deviceStr) {
  }
  ~TestJob() {
    terminate_ = true;
    //shared->doneSem.post();
    py::gil_scoped_release gil;
    runThread.join();
  }

  void start() {
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

  void run() {

    torch::NoGradGuard ng;

    torch::Device device(deviceStr_);

    Timer t;
    int count = 0;

    size_t bufferCounter = 0;

    std::vector<LocalBuffer> local;

    local.resize(shared->buffers.size());

    async::SchedulerFifo asyncforward;
    asyncforward.pool.maxThreads = shared->buffers.size();

    while (!terminate_) {

      if (shared->clients == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      size_t nClients = shared->clients;

      if (nClients > maxClients) {
        throw std::runtime_error("Too many clients connected!");
      }

      auto forwardBuffer = [&](size_t bufferIndex) {
        auto& buffer = shared->buffers[bufferIndex];
        auto& lbuf = local[bufferIndex];

        size_t size = buffer.size;
        size_t stride = buffer.stride;
        size_t clientIndex = 0;
        if (size == 0) {
          return;
        }
        for (size_t i = 0; i < size; i += stride, ++clientIndex) {
//          int nSteps = std::min(size - i, stride);
//          if (clientStates.size() <= clientIndex || clientStates[clientIndex].states.size() < nSteps) {
//            throw std::runtime_error("clientStates mismatch");
//          }
          auto& input = buffer.clientInputs[clientIndex];
          auto& output = buffer.clientOutputs[clientIndex];
          size_t prevSteps = input.nStepsIn.load(std::memory_order_acquire);
          while (output.nStepsOut.load(std::memory_order_relaxed) != prevSteps && !terminate_) {
            _mm_pause();
          }
        }
        if (lbuf.inputMap.empty()) {
          std::map<std::string, torch::Tensor> map;
          auto* src = buffer.batchData.data(shared);
          for (size_t i = 0; i != buffer.batchData.data.size; ++i) {
            auto key = src[i].key.view(shared);
            auto& v = src[i];
            torch::TensorOptions opts;
            switch (v.dtype) {
            case 'f':
              if (v.itemsize == 2) {
                opts = opts.dtype(torch::kFloat16);
              } else if (v.itemsize == 4) {
                opts = opts.dtype(torch::kFloat32);
              } else if (v.itemsize == 8) {
                opts = opts.dtype(torch::kFloat64);
              } else {
                throw std::runtime_error("Unexpected itemsize for float");
              }
              break;
            case 'i':
              if (v.itemsize == 1) {
                opts = opts.dtype(torch::kInt8);
              } else if (v.itemsize == 2) {
                opts = opts.dtype(torch::kInt16);
              } else if (v.itemsize == 4) {
                opts = opts.dtype(torch::kInt32);
              } else if (v.itemsize == 8) {
                opts = opts.dtype(torch::kInt64);
              } else throw std::runtime_error("Unexpected itemsize for int");
              break;
            case 'u':
              if (v.itemsize == 1) {
                opts = opts.dtype(torch::kUInt8);
              } else throw std::runtime_error("Unexpected itemsize for unsigned int");
              break;
            case 'b':
              if (v.itemsize == 1) {
                opts = opts.dtype(torch::kBool);
              } else throw std::runtime_error("Unexpected itemsize for boolean");
              break;
            default:
              throw std::runtime_error("Unsupported dtype '" + std::string(1, v.dtype) + "'");
            }

            std::vector<int64_t> sizes(v.shape(shared), v.shape(shared) + v.shape.size);
            sizes.insert(sizes.begin(), (int64_t)maxEnvs);
            map[std::string(key)] = torch::from_blob(v.data(shared), sizes, opts);
          }
          for (auto& [key, value] : map) {
            //log("map [%s] sizes = %s\n", key, sizesStr(value.sizes()));
            lbuf.inputMap[key] = torch::Tensor();
            lbuf.inputShared.push_back(value);
            lbuf.inputPinned.push_back(value.pin_memory());
            lbuf.inputDevice.push_back(value.to(device));
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
        auto& input = lbuf.inputMap;
        size_t index = 0;
        for (auto& [key, value] : input) {
          auto shared = lbuf.inputShared.at(index).narrow(0, 0, size);
          auto pinned = lbuf.inputPinned.at(index).narrow(0, 0, size);
          auto device = lbuf.inputDevice.at(index).narrow(0, 0, size);
          pinned.copy_(shared);
          device.copy_(pinned, true);
          value = device;
          if (key != "done") {
            value = value.unsqueeze(0);
          }
          ++index;
        }
//        for (auto& [key, value] : input) {
//          log("input [%s] sizes = %s\n", key, sizesStr(value.sizes()));
//        }
//        auto& state = stateTmp;
//        state.resize(ms.size());
//        for (size_t i = 0; i != state.size(); ++i) {
//          if ((size_t)ms[i].size(1) != size) {
//            throw std::runtime_error("model state size mismatch");
//          }
//          state[i] = ms[i].narrow(1, 0, size);
//          //log("state[%d].sum() is %g\n", i, state[i].sum().item<float>());
//        }
//        for (size_t i = 0; i != state.size(); ++i) {
//          log("state [%d] sizes = {%s}\n", i, sizesStr(state[i].sizes()));
//        }
//        log("doing forward!\n");
        py::gil_scoped_acquire gil;
        py::tuple tup = model_(input, lbuf.modelStates);
//        py::dict output = tup[0];
//        py::tuple outstate = tup[1];
//        for (size_t i = 0; i != ms.size(); ++i) {
//          ms[i] = outstate[i].cast<torch::Tensor>();
//        }
      };

      auto fillBuffer = [&](size_t bufferIndex) {
        //buffer.remaining.store((size + stride - 1) / stride, std::memory_order_relaxed);
        auto& buffer = shared->buffers[bufferIndex];
        auto& ms = local[bufferIndex].modelStates;

        size_t size = 256;

        size_t strideDivisor = nClients;
        size_t stride = (size + strideDivisor - 1) / strideDivisor;

        if (buffer.size != size) {
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
        }

        buffer.size = size;
        buffer.stride = stride;

        size_t clientIndex = 0;
        for (size_t i = 0; i < size; i += stride, ++clientIndex) {
          int nSteps = std::min(size - i, stride);
//          while (clientStates.size() <= clientIndex) {
//            clientStates.emplace_back();
//          }
//          auto& cs = clientStates[clientIndex];
//          while (cs.states.size() < (size_t)nSteps) {
//            py::gil_scoped_acquire gil;
//            cs.states.emplace_back();
//            auto is = model_.attr("initial_state")();
//            for (auto& v : is.cast<py::tuple>()) {
//              log("its a tuple okay\n");
//              log("its a %s\n", std::string(py::str(v.get_type())).c_str());
//              cs.states.back().modelState.push_back(v.cast<torch::Tensor>());
//            }
//            //cs.states.back().modelState = .cast<std::vector<torch::Tensor>>();;
//            log("got new initial state yey\n");
//          }
          auto& input = buffer.clientInputs[clientIndex];
//          auto& output = buffer.clientOutputs[clientIndex];
//          //input.nSteps = nSteps;
          size_t prevSteps = input.nStepsIn.load(std::memory_order_acquire);
//          while (input.nStepsOut.load(std::memory_order_relaxed) != prevSteps && !terminate_) {
//            _mm_pause();
//          }
          input.resultOffset.store(i, std::memory_order_relaxed);
          input.nStepsIn.store(prevSteps + nSteps, std::memory_order_release);
        }
      };

//      localBatch.step(stride);
//      count += stride;

      size_t bufferIndex = bufferCounter % shared->buffers.size();
      auto& buffer = shared->buffers[bufferIndex];
      forwardBuffer(bufferIndex);
      fillBuffer(bufferIndex);
      ++bufferCounter;
      //size_t currentIndex = bufferCounter % shared->buffers.size();

      //while (!terminate_.load(std::memory_order_relaxed) && shared->buffers[currentIndex].remaining.load(std::memory_order_relaxed));

      //shared->doneSem.wait();
      if (terminate_) {
        break;
      }

      count += buffer.size;

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
  std::atomic_bool running_ = false;

  std::array<EnvBatch, 10> batch;

  TestClient(py::object envInit) {
    for (auto& b : batch) {
      b.envInit_ = envInit;
    }
  }

  ~TestClient() {
    terminate_ = true;
    while (running_);
  }

  void run(std::string serverAddress) {
    py::gil_scoped_release gil;
    running_ = true;
    try {

  //    rpc.define<void()>("quit", [this]() {
  //      terminate_ = true;
  //    });

  //    rpc.define<bool(int)>("step", [this](int nParallel) {
  //      //step(nParallel);
  //      bool done;
  //      float xx = 0.1;
  //      for (int i = 0; i != 1000 * nParallel; ++i) {
  //        xx = std::log(std::exp(xx));
  //      }
  //      done = xx < 0;
  //      return !done;
  //      //std::this_thread::sleep_for(std::chrono::milliseconds(nParallel));
  //      return true;
  //    });

  //    rpc.connect("shm://" + serverAddress);

  //    rpc.sync(serverAddress, "hello", rpc.getName());

      SharedMemory shm(serverAddress);
      Shared& shared = shm.as<Shared>();

      int myIndex = shared.clients++;

      log("my index is %d\n", myIndex);

      if ((size_t)myIndex > maxClients) {
        throw std::runtime_error("Client index is too high");
      }

      size_t bufferCounter = 0;

      while (true) {
        size_t bufferIndex = bufferCounter % shared.buffers.size();
        auto& buffer = shared.buffers[bufferIndex];
        ++bufferCounter;
        auto& input = buffer.clientInputs[myIndex];
        auto& output = buffer.clientOutputs[myIndex];
        size_t stepsDone = output.nStepsOut.load(std::memory_order_relaxed);
        size_t nSteps = input.nStepsIn.load(std::memory_order_acquire);
        while (nSteps == stepsDone) {
          _mm_pause();
          nSteps = input.nStepsIn.load(std::memory_order_acquire);
          if (terminate_.load(std::memory_order_relaxed)) {
            break;
          }
        }
        if (terminate_.load(std::memory_order_relaxed)) {
          break;
        }
        size_t offset = input.resultOffset;
        //log("oh boy i got a task for %d steps\n", nSteps - stepsDone);
        batch.at(bufferIndex).step(nSteps - stepsDone, &shared, bufferIndex, offset);
        output.nStepsOut.store(nSteps, std::memory_order_relaxed);

//        if (buffer.remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
//          //shared->doneSem.post();
//        }
      }
    } catch (...) {
      running_ = false;
      throw;
    }

  }
};


PYBIND11_MODULE(pyjob, m) {
  m.def("set_logging", [](py::object logging) {
    pyLogging = logging;
  });
  py::class_<TestJob>(m, "TestJob")
      .def(py::init<py::object, std::string>())
      .def("start", &TestJob::start)
      .def("get_name", &TestJob::getName);
  py::class_<TestClient>(m, "TestClient")
      .def(py::init<py::object>())
      .def("run", &TestClient::run);
}
