#pragma once

#include "function.h"
#include "synchronization.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <list>
#include <deque>
#include <vector>


namespace async {

inline void setCurrentThreadName(const std::string& name) {
#ifdef __APPLE__
  pthread_setname_np(name.c_str());
#elif __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

template<typename T>
using Function = rpc::Function<T>;
using FunctionPointer = rpc::FunctionPointer;

struct Thread {
  alignas(64) rpc::Semaphore sem;
  FunctionPointer f;
  std::thread thread;
  int n = 0;
  template<typename WaitFunction>
  void entry(WaitFunction&& waitFunction) noexcept {
    while (true) {
      Function<void()>{f}();
      f = nullptr;
      waitFunction(this);
    }
  }
};

struct ThreadPool {
  alignas(64) rpc::SpinMutex mutex;
  std::list<Thread> threads;
  std::atomic_size_t numThreads = 0;
  size_t maxThreads = 0;
  ThreadPool() noexcept {
    maxThreads = std::thread::hardware_concurrency();
    if (maxThreads < 1) {
      maxThreads = 1;
    }
  }
  ~ThreadPool() {
    for (auto& v : threads) {
      v.thread.join();
    }
  }
  template<typename WaitFunction>
  Thread* addThread(Function<void()>& f, WaitFunction&& waitFunction) noexcept {
    std::unique_lock l(mutex);
    if (numThreads >= maxThreads) {
      printf("throw away function!?\n");
      std::abort();
      return nullptr;
    }
    int n = ++numThreads;
    threads.emplace_back();
    auto* t = &threads.back();
    t->n = threads.size() - 1;
    t->f = f.release();
    rpc::Semaphore sem;
    std::atomic<bool> started = false;
    t->thread = std::thread([&sem, &started, n, t, waitFunction = std::forward<WaitFunction>(waitFunction)]() mutable {
      setCurrentThreadName("async " + std::to_string(n));
      started = true;
      sem.post();
      t->entry(std::move(waitFunction));
    });
    while (!started) {
      sem.wait();
    }
    return t;
  }
};

struct SchedulerFifo {
  rpc::SpinMutex mutex;
  std::vector<Thread*> idle;
  std::deque<FunctionPointer> queue;

  ThreadPool pool;

  void wait(Thread* t) noexcept {
    std::unique_lock l(mutex);
    if (!queue.empty()) {
      t->f = queue.front();
      queue.pop_front();
    } else {
      idle.push_back(t);
      l.unlock();
      t->sem.wait();
    }
  }

  void run(Function<void()> f) noexcept {
    std::unique_lock l(mutex);
    if (idle.empty()) {
      if (pool.numThreads.load(std::memory_order_relaxed) < pool.maxThreads) {
        if (pool.addThread(f, [this](Thread* t) { wait(t); })) {
          return;
        }
      }
      queue.push_back(f.release());
    } else {
      Thread* t = idle.back();
      idle.pop_back();
      l.unlock();
      t->f = f.release();
      t->sem.post();
    }
  }

};

struct SchedulerPrio {

  void run(float priority, rpc::Function<void>&& f);

};

struct SchedulerFair {

  void run(uint32_t id, rpc::Function<void>&& f);

};

}
