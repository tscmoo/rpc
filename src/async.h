#pragma once

#include "function.h"
#include "synchronization.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <list>
#include <deque>

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
  alignas(64) std::atomic<Thread*> next = nullptr;
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
  std::atomic<Thread*> idle = nullptr;
  std::mutex mutex;
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
  Thread* addThread(FunctionPointer f, WaitFunction&& waitFunction) noexcept {
    std::unique_lock l(mutex);
    if (numThreads >= maxThreads) {
      return nullptr;
    }
    int n = ++numThreads;
    threads.emplace_back();
    auto* t = &threads.back();
    t->n = threads.size() - 1;
    t->f = f;
    t->thread = std::thread([l = std::move(l), n, t, waitFunction = std::forward<WaitFunction>(waitFunction)]() mutable {
      setCurrentThreadName("async " + std::to_string(n));
      l.unlock();
      t->entry(std::move(waitFunction));
    });
    return t;
  }
};


struct SchedulerFifo {
  std::mutex mutex_;
  std::deque<FunctionPointer> queue_;

  ThreadPool pool;

  void wait(Thread* t) noexcept {
    Thread* idle = pool.idle.load(std::memory_order_acquire);
    while (idle) {
      t->next = idle;
      if (pool.idle.compare_exchange_weak(idle, t)) {
        t->sem.wait();
        return;
      }
    }
    std::unique_lock l(mutex_);
    if (!queue_.empty()) {
      t->f = queue_.front();
      queue_.pop_front();
      return;
    }
    idle = pool.idle.load(std::memory_order_acquire);
    while (true) {
      t->next = idle;
      if (pool.idle.compare_exchange_weak(idle, t)) {
        l.unlock();
        t->sem.wait();
        return;
      }
    }
  }

  void run(Function<void()> f) noexcept {
    Thread* idle = pool.idle.load(std::memory_order_acquire);
    while (idle && !pool.idle.compare_exchange_weak(idle, idle->next.load(std::memory_order_acquire)));
    if (idle) {
      idle->f = f.release();
      idle->sem.post();
    } else {
      if (pool.numThreads.load(std::memory_order_relaxed) < pool.maxThreads) {
        idle = pool.addThread(f.release(), [this](Thread* t) {
          wait(t);
        });
        if (idle) {
          return;
        }
      }
      std::lock_guard l(mutex_);
      Thread* idle = pool.idle.load(std::memory_order_acquire);
      while (idle && !pool.idle.compare_exchange_weak(idle, idle->next.load(std::memory_order_acquire)));
      if (idle) {
        idle->f = f.release();
        idle->sem.post();
      } else {
        queue_.push_back(f.release());
      }
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
