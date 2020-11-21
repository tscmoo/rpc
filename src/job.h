#pragma once

#include "async.h"

#include <vector>
#include <memory>

namespace moo {

struct alignas(64) Task {
  virtual ~Task() {}
  virtual bool step() = 0;
  virtual void reset() = 0;
};

struct Job {

  std::vector<std::unique_ptr<Task>> tasks;

  virtual std::unique_ptr<Task> startTask() = 0;

  async::SchedulerFifo scheduler;

  std::atomic_bool terminate_ = false;

  void stop() {
    terminate_ = true;
  }

  void run() {

    for (int i = 0; i != 1; ++i) {
      tasks.push_back(startTask());
    }

    scheduler.pool.maxThreads = 1;
    size_t maxThreads = scheduler.pool.maxThreads;

    std::atomic_int remaining = 0;
    rpc::Semaphore sem;

    while (!terminate_) {

      if (tasks.size() == 1) {
        auto& task = tasks[0];
        while (!task->step()) {
          task->reset();
        }
      } else if (scheduler.pool.maxThreads == 1) {
        for (auto& task : tasks) {
          while (!task->step()) {
            task->reset();
          }
        }
      } else {

        size_t strideDivisor = std::max(maxThreads, (size_t)1);
        size_t stride = (tasks.size() + strideDivisor - 1) / strideDivisor;
        remaining.store((tasks.size() + stride - 1) / stride, std::memory_order_relaxed);
        for (size_t i = 0; i < tasks.size(); i += stride) {
          size_t end = std::min(tasks.size(), i + stride);
          scheduler.run([&, this, tasks = tasks.data(), offset = i, end]() {
            for (size_t i = offset; i != end; ++i) {
              auto& task = tasks[i];
              while (!task->step()) {
                task->reset();
              }
            }
            if (remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
              sem.post();
            }
          });
        }

        //while (remaining.load());
        sem.wait();
      }

    }

    tasks.clear();

  }

};


}



