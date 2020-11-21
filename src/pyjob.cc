
#include "pybind11/pybind11.h"

#include <pybind11/numpy.h>

#include "job.h"

#include "rpc.h"

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
    printf("bad bool\n");
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
  py::object temp = v;
  if (PyUnicode_Check(v.ptr())) {
      temp = py::reinterpret_steal<py::object>(PyUnicode_AsUTF8String(v.ptr()));
      if (!temp) {
          py::pybind11_fail("Unable to extract string contents! (encoding issue)");
      }
  }
  char *buffer;
  ssize_t length;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length)) {
      py::pybind11_fail("Unable to extract string contents! (invalid type)");
  }
  x(std::string_view(buffer, (size_t) length));
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
  size_t size = 1024 * 1024 * 10;
  std::byte* data = nullptr;
  std::string name;
  SharedMemory(std::string_view name) : name(name) {
    printf("creating shm %s\n", std::string(name).c_str());
    fd = shm_open(std::string(name).c_str(), O_RDWR | O_CREAT, ACCESSPERMS);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category());
    }
    ftruncate(fd, size);
    data = (std::byte*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!data) {
      throw std::system_error(errno, std::system_category());
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
      printf("%s is too big for shm :(\n", typeid(T).name());
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

    struct Input {
      //Semaphore sem;
      //std::atomic_int nSteps = 0;
      alignas(64) std::atomic_size_t nStepsIn = 0;
      alignas(64) std::atomic_size_t nStepsOut = 0;

    };

    //alignas(64) std::atomic_int remaining = 0;
    //Semaphore doneSem;

    std::array<Input, 0x100> inputs;

  };

  alignas(64) std::atomic_int clients = 0;
  std::array<Buffer, 2> buffers;


  std::string_view allocate(size_t n) {
    n = (n + 63) / 64 * 64;
    size_t offset = allocated.fetch_add(n, std::memory_order_relaxed);
    if (offset + n > size) {
      throw std::runtime_error("Out of space in shared memory buffer");
    }
    return {(const char*)this + offset, n};
  }
};

struct Env {
  py::object env_;
  py::object reset_;
  py::object step_;

  int steps = 0;
  std::string_view outbuffer;

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

  bool step(Shared* shared) {
    auto start = std::chrono::steady_clock::now();
    ++steps;
    int action = std::uniform_int_distribution<size_t>(0, 10)(threadRng);
    bool done;
    try {
      py::gil_scoped_acquire gil;
      py::tuple tup = step_(action);
      //printf("tuple has %zu\n", tup.size());
      done = (py::bool_)tup[2];
      if (outbuffer.size() == 0) {
        rpc::BufferHandle buffer = rpc::serializeToBuffer(tup);
        printf("Result serialized to %d bytes\n", buffer->size);
        outbuffer = shared->allocate(buffer->size + buffer->size / 2);
        printf("allocated to %p, size %d\n", outbuffer.data(), outbuffer.size());
      }
      rpc::serializeToStringView(outbuffer);
    } catch (const pybind11::error_already_set &e) {
      printf("step error %s\n", e.what());
      throw;
    }
    counter.fetch_add(1, std::memory_order_relaxed);
    steptime += (std::chrono::steady_clock::now() - start).count();
    //if (steps >= 50000) std::abort();
    return !done;
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
        printf("%d,  %g/s\n", n, nps);

        lastcount = cn;

        astart = start;
      }
    } catch (const pybind11::error_already_set &e) {
      printf("reset error %s\n", e.what());
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
  void step(size_t size, Shared* shared) {
    while (envs.size() < size) {
      py::gil_scoped_acquire gil;
      envs.emplace_back(envInit_());
      envs.back().reset();
    }
    for (auto& v : envs) {
      while (!v.step(shared)) {
        v.reset();
      }
    }
  }
};

struct TestJob {

  std::thread runThread;

  std::atomic_bool terminate_ = false;

  std::string shmname = "nle-" + randomName();
  SharedMemory shm{shmname};
  Shared* shared = &shm.as<Shared>();

  EnvBatch localBatch;

  TestJob(py::object envInit) : localBatch(std::move(envInit)) {
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

  void run() {

    Timer t;
    int count = 0;

    size_t bufferCounter = 0;

    while (!terminate_) {

      if (shared->clients == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      size_t nClients = shared->clients;

      if (nClients > shared->buffers[0].inputs.size()) {
        throw std::runtime_error("Too many clients connected!");
      }

      size_t size = 100;

      size_t strideDivisor = nClients;
      size_t stride = (size + strideDivisor - 1) / strideDivisor;

      auto fillBuffer = [&](Shared::Buffer& buffer) {
        //buffer.remaining.store((size + stride - 1) / stride, std::memory_order_relaxed);
        size_t clientIndex = 0;
        for (size_t i = 0; i < size; i += stride, ++clientIndex) {
          int nSteps = std::min(size - i, stride);
          auto& input = buffer.inputs[clientIndex];
          //input.nSteps = nSteps;
          size_t prevSteps = input.nStepsIn.load(std::memory_order_relaxed);
          while (input.nStepsOut.load(std::memory_order_relaxed) != prevSteps && !terminate_) {
            _mm_pause();
          }
          input.nStepsIn.store(prevSteps + nSteps, std::memory_order_relaxed);
        }
      };

//      localBatch.step(stride);
//      count += stride;

      fillBuffer(shared->buffers[bufferCounter % shared->buffers.size()]);
      ++bufferCounter;
      //size_t currentIndex = bufferCounter % shared->buffers.size();

      //while (!terminate_.load(std::memory_order_relaxed) && shared->buffers[currentIndex].remaining.load(std::memory_order_relaxed));

      //shared->doneSem.wait();
      if (terminate_) {
        break;
      }

      count += size;

      if (t.elapsed() >= 1.0f) {
        float tx = t.elapsedReset();
        int n = count;
        count = 0;
        printf("rate %g/s\n", n / tx);
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

      printf("my index is %d\n", myIndex);

      if ((size_t)myIndex > shared.buffers[0].inputs.size()) {
        throw std::runtime_error("Client index is too high");
      }

      size_t bufferCounter = 0;

      while (true) {
        size_t bufferIndex = bufferCounter % shared.buffers.size();
        auto& buffer = shared.buffers[bufferIndex];
        ++bufferCounter;
        auto& input = buffer.inputs[myIndex];
        size_t stepsDone = input.nStepsOut.load(std::memory_order_relaxed);
        size_t nSteps = input.nStepsIn.load(std::memory_order_relaxed);
        while (nSteps == stepsDone) {
          _mm_pause();
          nSteps = input.nStepsIn.load(std::memory_order_relaxed);
          if (terminate_.load(std::memory_order_relaxed)) {
            break;
          }
        }
        if (terminate_.load(std::memory_order_relaxed)) {
          break;
        }
        //printf("oh boy i got a task for %d steps\n", nSteps - stepsDone);
        batch.at(bufferIndex).step(nSteps - stepsDone, &shared);
        input.nStepsOut.store(nSteps, std::memory_order_relaxed);

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
  py::class_<TestJob>(m, "TestJob")
      .def(py::init<py::object>())
      .def("start", &TestJob::start)
      .def("get_name", &TestJob::getName);
  py::class_<TestClient>(m, "TestClient")
      .def(py::init<py::object>())
      .def("run", &TestClient::run);
}
