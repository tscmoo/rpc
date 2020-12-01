#pragma once

#include <atomic>
#include <mutex>

#ifdef _POSIX_C_SOURCE
#include <semaphore.h>
#endif

namespace rpc {

#if 1
using SpinMutex = std::mutex;
#elif __GNUC__ && __x86_64__

class SpinMutex {
  int lock_ = 0;
public:
  void lock() {
    long tmp, tmp2;
    asm volatile (""
    "3:movl $1, %%edx\n"
    "xor %%rax, %%rax\n"
    "xacquire lock cmpxchgl %%edx, (%2)\n"
    "jz 2f\n"
    "1:\n"
    "pause\n"
    "movl (%2), %%eax\n"
    "testl %%eax, %%eax\n"
    "jnz 1b\n"
    "jmp 3b\n"
    "2:\n"
    : "+a"(tmp), "+d"(tmp2)
    : "r"(&lock_)
    : "memory", "cc");
  }
  void unlock() {
    asm volatile ("xrelease movl $0, (%0)"::"r"(&lock_):"memory");
  }
};

#else
class alignas(64) SpinMutex {
  std::atomic<bool> locked_{false};
public:
  void lock() {
    do {
      while (locked_.load(std::memory_order_acquire));
    } while (locked_.exchange(true, std::memory_order_acquire));
  }
  void unlock() {
    locked_.store(false, std::memory_order_release);
  }
  bool try_lock() {
    if (locked_.load(std::memory_order_acquire)) {
      return false;
    }
    return !locked_.exchange(true, std::memory_order_acquire);
  }
};
#endif


#ifdef _POSIX_C_SOURCE
class Semaphore {
  sem_t sem;

 public:
  Semaphore() noexcept {
    sem_init(&sem, 0, 0);
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
#else
class Semaphore {
  int count_ = 0;
  std::mutex mut_;
  std::condition_variable cv_;

 public:
  void post() {
    std::unique_lock l(mut_);
    if (++count_ >= 1) {
      cv_.notify_one();
    }
  }
  void wait() {
    std::unique_lock l(mut_);
    while (count_ == 0) {
      cv_.wait(l);
    }
    --count_;
  }
};
#endif

}
