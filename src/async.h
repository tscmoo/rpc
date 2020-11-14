#pragma once

#include "function.h"
#include "synchronization.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <list>
#include <deque>
#include <unordered_map>
#include <vector>

#include <fmt/printf.h>

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

struct Guard {
  volatile uint64_t guard = 0xb00bb00b;
  void test() {
    if (guard != 0xb00bb00b) {
      printf("guard mismatch\n");
      std::abort();
    }
  }
};

struct Thread {
  Guard g1;
  alignas(64) std::atomic<Thread*> next = nullptr;
  Guard g2;
  alignas(64) rpc::Semaphore sem;
  Guard g3;
  std::atomic<FunctionPointer> f;
  Guard g4;
  std::thread thread;
  Guard g5;
  int n = 0;
  Guard g6;
  std::atomic<bool> waiting = false;
  Guard g7;
  std::atomic<bool> bad = false;
  Guard g8;
  template<typename WaitFunction>
  void entry(WaitFunction&& waitFunction) noexcept {
    while (true) {
      if (!f) {
        printf("%p: null async function\n", this);
        std::abort();
      }
      if (next) {
        printf("%p: next is not null!\n", this);
        std::abort();
      }
      if (bad) {
        printf("%p: thread is bad!\n", this);
        std::abort();
      }
      Function<void()>{f.load()}();
      f = nullptr;
      waiting = true;
      waitFunction(this);
      waiting = false;
    }
  }
};

struct ThreadPool {
  std::atomic<Thread*> idle = nullptr;
  rpc::SpinMutex mutex;
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
      printf("throw away function!?\n");
      std::abort();
      return nullptr;
    }
    int n = ++numThreads;
    threads.emplace_back();
    auto* t = &threads.back();
    t->n = threads.size() - 1;
    t->f = f;
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

inline std::atomic<int> threacountler = 1;
inline thread_local int threadn = ++threacountler;

inline rpc::SpinMutex testm;
inline std::unordered_map<Thread*, int> testx;

struct SchedulerFifo {
  rpc::SpinMutex mutex_;
  std::deque<FunctionPointer> queue_;

  ThreadPool pool;

  rpc::SpinMutex seqMutex;
  struct S {
    int threadid;
    const char* descr;
    Thread* t;
    Thread* t2;
    std::vector<Thread*> q;
  };

  std::vector<S> sequence;

  std::mutex supermutex;

  void addSeq(const char* descr, Thread* t, Thread* t2 = nullptr) {
    S s;
    for (auto* p = pool.idle.load(); p; p = p->next) {
      s.q.push_back(p);
    }
    s.threadid = threadn;
    s.descr = descr;
    s.t = t;
    s.t2 = t2;
    std::lock_guard l(seqMutex);
    sequence.push_back(std::move(s));
  }

  void fail() {
    std::lock_guard l(seqMutex);
    size_t x = 0;
    if (sequence.size() > 50) x = sequence.size() - 50;
    for (size_t i = sequence.size(); i > x;) {
      --i;
      std::string qs;
      for (auto* v : sequence[i].q) {
        if (!qs.empty()) qs += " ";
        qs += fmt::sprintf("%p", (void*)v);
      }
      printf("thread %02x -> %s %p %p - %s\n", sequence[i].threadid, sequence[i].descr, sequence[i].t, sequence[i].t2, qs.c_str());
    }
    std::abort();
  }

  int count(Thread* t) {
    int r = 0;
    for (auto* p = pool.idle.load(); p; p = p->next) {
      if (p == t) {
        ++r;
      }
    }
    return r;
  }

  void wait(Thread* t) noexcept {
    {
      std::lock_guard l(testm);
      int& n = testx[t];
      if (n == 0) n = threadn;
      if (n != threadn) {
        printf("mismatch yo....\n");
        fail();
      }
    }
    t->g1.test();
    t->g2.test();
    t->g3.test();
    t->g4.test();
    t->g5.test();
    t->g6.test();
    t->g7.test();
    t->g8.test();
    if (!t->waiting) {
      printf("wtffff\n");
      fail();
    }
    Thread* idle = pool.idle.load();
    if (idle == t) {
      printf("%p is idle 1!!!\n", t);
      fail();
    }
    if (t->next) {
      printf("%p already has next 1!\n", t);
      fail();
    }
    if (t->f) {
      printf("%p already has f 1!\n", t);
      fail();
    }
    if (count(t) != 0) {
      printf("wait %p but count %d 0\n", t, count(t));
      fail();
    }
    while (idle) {
      if (idle == t) {
        printf("%p is idle 1.5!!!\n", t);
        fail();
      }
      t->next = idle;
      addSeq("next1", t, idle);
      if (pool.idle.compare_exchange_weak(idle, t)) {
        addSeq("queue 1", t);
//        if (count(t) != 1 && !t->f) {
//          printf("%p found with count %d 0\n", t, count(t));
//          fail();
//        }
        while (!t->f) {
          t->sem.wait();
          addSeq("wake", t);
          //if (count(t) != 1 && !t->f) {
            //printf("%p found with count %d 0.5\n", t, count(t));
            //fail();
          //}
        }
        addSeq("wake with f", t);
        if (t->next) {
          printf("%p is still in idle after wait 1\n", t);
          fail();
        }
        if (pool.idle.load() == t) {
          printf("%p is still THE idle after wait 1\n", t);
          fail();
        }
        if (count(t) != 0) {
          printf("%p found with count %d 1\n", t, count(t));
          fail();
        }
        return; // wtf is wrong here man...
      }
    }
    t->next = nullptr;
    if (t->next) {
      printf("%p already has next 2!\n", t);
      fail();
    }
    if (t->f) {
      printf("%p already has f 2!\n", t);
      fail();
    }
    printf("%p is proceeding to take mutex\n", t);
    if (pool.idle.load() == t) {
      printf("%p is already idle at mutex!\n", t);
      fail();
    }
    addSeq("wait for mutex", t);
    std::unique_lock l(mutex_);
    addSeq("got mutex", t);
    if (!queue_.empty()) {
      printf("popping queue, yo (%p)\n", this);
      fail();
      t->f = queue_.front();
      queue_.pop_front();
      return;
    }
    idle = pool.idle.load();
    if (idle == t) {
      printf("%p is idle 2.5!!!\n", t);
      fail();
    }
    printf("wait:: mutex is %p, idle is %p,\n", &mutex_, idle);
    t->g1.test();
    t->g2.test();
    t->g3.test();
    t->g4.test();
    t->g5.test();
    t->g6.test();
    t->g7.test();
    t->g8.test();
    if (t->next) {
      printf("%p already has next 3!\n", t);
      fail();
    }
    if (t->f) {
      printf("%p already has f 2.5!\n", t);
      fail();
    }
    while (true) {
      if (idle == t) {
        printf("%p is idle 3!!!\n", t);
        fail();
      }
      t->next = idle;
      addSeq("next2", t, idle);
      if (t->f) {
        printf("%p already has f 3!\n", t);
        fail();
      }
      if (pool.idle.compare_exchange_weak(idle, t)) {
        addSeq("queue in mutex", t);
        printf("inserted into idle yey (%p)\n", t);
        l.unlock();
        while (!t->f) {
          t->sem.wait();
        }
        addSeq("wake 2", t);
        printf("sem wait returned, yey (%p)\n", t);
        if (t->next) {
          printf("%p is still in idle after wait 2\n", t);
          fail();
        }
        if (pool.idle.load() == t) {
          printf("%p is still THE idle after wait 2\n", t);
          fail();
        }
        if (count(t) != 0) {
          printf("%p found with count %d 2\n", t, count(t));
          fail();
        }
        return;
      }
    }
  }

  void run(Function<void()> f) noexcept {
    addSeq("run", nullptr);
    Thread* idle = pool.idle.load();
    while (idle && !pool.idle.compare_exchange_weak(idle, idle->next.load())) {
      if (idle && idle->next.load() == idle) {
        printf("next is myself yo!!\n");
        fail();
      }
    }
    if (idle) { // sigh
      addSeq("extract", idle);
      //printf("schedule %p, pool.idle.load() is %p\n", idle, pool.idle.load());
      if (pool.idle.load() == nullptr) {
        printf("exhausted idle list while scheduling %p\n", idle);
      }
      if (count(idle)) {
        printf("schedule %p count is %d\n", idle, count(idle));
        fail();
      }
      if (idle->bad) {
        printf("bad thread 1 %p\n", idle);
        fail();
      }
//      if (!idle->f) {
//        printf("queue null function 1 %p\n", idle);
//        fail();
//      }
      if (!idle->waiting) {
        idle->bad = true;
        printf("thread is not waiting 1 %p\n", idle);
        while (!idle->waiting);
        printf("oh there now it's waiting... %p\n", idle);
        fail();
      }
      if (pool.idle.load() == idle) {
        printf("double schedule 1\n");
        fail();
      }
      addSeq("setting f", idle);
      idle->next = nullptr;
      idle->f = f.release();
      addSeq("post", idle);
      idle->sem.post();
    } else {
      printf("add thread yo!?\n");
      if (pool.numThreads.load(std::memory_order_relaxed) < pool.maxThreads) {
        auto ptr = f.release();
        idle = pool.addThread(ptr, [this](Thread* t) {
          wait(t);
        });
        if (idle) {
          return;
        }
        f = ptr;
      }
      printf("shouldn't really get here\n");
      std::abort();
      std::lock_guard l(mutex_);
      printf("&pool.idle is %p\n", &pool.idle);
      idle = pool.idle.load(std::memory_order_acquire);
      printf("run:: mutex is %p, idle is %p,\n", &mutex_, idle);
      while (idle && !pool.idle.compare_exchange_weak(idle, idle->next.load()));
      if (idle) {
        addSeq("extract 2", idle);
//        if (!idle->f) {
//          printf("queue null function 2\n");
//          fail();
//        }
        if (!idle->waiting) {
          printf("thread is not waiting 2\n");
          fail();
        }
        if (pool.idle.load() == idle) {
          printf("double schedule 2\n");
          fail();
        }
        printf("inserted into idle, post()\n");
        addSeq("setting f 2", idle);
        idle->next = nullptr;
        idle->f = f.release();
        addSeq("post 2", idle);
        idle->sem.post();
      } else {
        printf("pushing to queue\n");
        fail();
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
