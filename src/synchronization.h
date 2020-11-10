#pragma once

#include <atomic>
#include <mutex>

namespace rpc {

#if 0
using SpinMutex = std::mutex;
#elif __GNUC__ && __x86_64__

class SpinMutex {
  int lock_;
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

}
