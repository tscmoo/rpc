
#include "pybind11/pybind11.h"

#include <pybind11/numpy.h>

#include "job.h"

#include "rpc.h"

#include <cmath>
#include <random>
#include <fstream>
#include <unistd.h>

extern "C" {
//#include "../nle/include/hack.h"
#include "../nle/include/nledl.h"
}


namespace py = pybind11;

constexpr int actions[] = {
    13, 32, 107, 108, 106, 104, 117, 110, 98, 121,
    75, 76,  74,  72,  85,  78,  66,  89,
};
constexpr size_t actionsSize = sizeof(actions) / sizeof(actions[0]);

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

struct TestJob {

  std::thread runThread;

  rpc::Rpc rpc;

  std::atomic_bool terminate_ = false;

  std::mutex mutex;
  std::vector<std::string> newClients;

  std::vector<std::string> clients;

  TestJob() {

    rpc.define<void(std::string)>("hello", [this](std::string name) {
      std::lock_guard l(mutex);
      newClients.push_back(std::move(name));
    });

    rpc.listen("shm://" + std::string(rpc.getName()));
  }
  ~TestJob() {
    terminate_ = true;
    py::gil_scoped_release gil;
    runThread.join();
  }

  void start() {
    runThread = std::thread([this]() {
      run();
    });
  }

  void run() {

    std::atomic_int remaining = 0;
    rpc::Semaphore sem;

    Timer t;
    int count = 0;

    while (!terminate_) {

      {
        std::lock_guard l(mutex);
        for (auto& v : newClients) {
          clients.push_back(std::move(v));
        }
        newClients.clear();
      }

      if (clients.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      size_t size = 2000;

      size_t strideDivisor = clients.size();
      size_t stride = (size + strideDivisor - 1) / strideDivisor;
      remaining.store((size + stride - 1) / stride, std::memory_order_relaxed);
      size_t clientIndex = 0;
      for (size_t i = 0; i < size; i += stride, ++clientIndex) {
        int nSteps = std::min(size - i, stride);
        auto& client = clients.at(clientIndex);
        rpc.asyncCallback<bool>(client, "step", [&remaining, &sem](bool* r, rpc::Error* error) {
          if (error) {
            throw *error;
          }
          //printf("got result %d\n", *r);
          if (remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
            sem.post();
          }
        }, nSteps);
      }

      sem.wait();

      count += size;

      if (t.elapsed() >= 1.0f) {
        float tx = t.elapsedReset();
        int n = count;
        count = 0;
        printf("rate %g/s\n", n / tx);
      }
    }

    for (auto& name : clients) {
      rpc.async(name, "quit");
    }

  }

  std::string_view getName() {
    return rpc.getName();
  }

};

struct TestClient {

  std::atomic_bool terminate_ = false;
  py::object envInit_;

  struct Env {
    py::object env_;
    py::object reset_;
    py::object step_;

    int steps = 0;

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

    bool step() {
      {
        bool done;
        float xx = 0.1;
        for (int i = 0; i != 1000; ++i) {
          xx = std::log(std::exp(xx));
        }
        done = xx < 0;
        if (done) {
          printf("err done?\n");
          std::abort();
        }
        return !done;
      }
      auto start = std::chrono::steady_clock::now();
      ++steps;
      int action = std::uniform_int_distribution<size_t>(0, 10)(threadRng);
      bool done;
      try {
        py::gil_scoped_acquire gil;
//        py::tuple tup = step_(action);
//        //printf("tuple has %zu\n", tup.size());
//        done = (py::bool_)tup[2];
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
      return;
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

  std::mutex envMutex;
  std::vector<Env> envs;

  rpc::Rpc rpc;

  TestClient(py::object envInit) : envInit_(envInit) {
  }

  ~TestClient() {
    {
      py::gil_scoped_acquire gil;
      envInit_ = {};
    }
  }

  void step(int nParallel) {
    std::lock_guard l(envMutex);
    //printf("step %d\n", nParallel);
    while (envs.size() < (size_t)nParallel) {
      py::gil_scoped_acquire gil;
      envs.emplace_back(envInit_());
      envs.back().reset();
    }
    for (auto& v : envs) {
      while (!v.step()) {
        v.reset();
      }
    }
  }

  void run(std::string serverAddress) {
    py::gil_scoped_release gil;

    rpc.define<void()>("quit", [this]() {
      terminate_ = true;
    });

    rpc.define<bool(int)>("step", [this](int nParallel) {
      //step(nParallel);
      bool done;
      float xx = 0.1;
      for (int i = 0; i != 1000 * nParallel; ++i) {
        xx = std::log(std::exp(xx));
      }
      done = xx < 0;
      return !done;
      //std::this_thread::sleep_for(std::chrono::milliseconds(nParallel));
      return true;
    });

    rpc.connect("shm://" + serverAddress);

    rpc.sync(serverAddress, "hello", rpc.getName());

    while (!terminate_) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

  }
};


PYBIND11_MODULE(pyjob, m) {
  py::class_<TestJob>(m, "TestJob")
      .def(py::init<>())
      .def("start", &TestJob::start)
      .def("get_name", &TestJob::getName);
  py::class_<TestClient>(m, "TestClient")
      .def(py::init<py::object>())
      .def("run", &TestClient::run);
}
