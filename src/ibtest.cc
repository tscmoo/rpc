 

#include "tensorpipe/tensorpipe.h"

#include "tensorpipe/transport/uv/listener.h"
#include "tensorpipe/transport/uv/connection.h"

#include <thread>
#include <atomic>
#include <chrono>
#include <random>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

namespace tensorpipe = rpc_tensorpipe;

std::atomic<int> nallocs{0};

std::atomic<unsigned long> alloctime{0};

std::unordered_map<std::string, int> addrcounts;

void addaddrImpl() {
  std::array<void*, 16> addrs;
  size_t size = 0;
  size = backtrace(addrs.data(), addrs.size());
  char** str = backtrace_symbols(addrs.data(), size);

  std::string s;
  for (size_t i = 0; i != size; ++i) {
    //printf("%s\n", str[i]);
    if (i != 0) {
      s += ' ';
    }
    s += str[i];

    //++addrcounts[s];
  }
  ++addrcounts[s];
}

thread_local bool innew = false;

//void* operator new(std::size_t n) {
//  if (!innew && false) {
//    innew = true;
//    std::array<void*, 16> addrs;
//    size_t size = 0;
//    size = backtrace(addrs.data(), addrs.size());
//    char** str = backtrace_symbols(addrs.data(), size);

//    std::string s;
//    for (size_t i = 0; i != size; ++i) {
//      //printf("%s\n", str[i]);
//      if (i != 0) {
//        s += ' ';
//      }
//      s += str[i];

//      ++addrcounts[s];
//    }

//    free(str);
//    innew = false;
//  }

//  auto start = std::chrono::steady_clock::now();

//  ++nallocs;
//  void* ptr = malloc(n);
//  if (!ptr) {
//    throw std::bad_alloc();
//  }
//  alloctime += (std::chrono::steady_clock::now() - start).count();
//  return ptr;
//}
//void operator delete(void* ptr) noexcept {
//  auto start = std::chrono::steady_clock::now();
//  std::free(ptr);
//  alloctime += (std::chrono::steady_clock::now() - start).count();
//}


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

std::atomic<int> supersum;

std::atomic<int> totalwrites{0};

void test(std::shared_ptr<tensorpipe::Pipe>& client, std::shared_ptr<tensorpipe::Pipe>& pipe) {

  tensorpipe::Message topmessage;
  tensorpipe::Message::Tensor t;
  t.buffer.type = tensorpipe::DeviceType::kCpu;
  t.buffer.cpu.ptr = (void*)"hello world";
  t.buffer.cpu.length = 11;
  topmessage.tensors.push_back(t);

  std::vector<char> data;

  std::atomic<bool> done{false};
  std::atomic<int> count{0};
  std::atomic<int> idle{0};

  std::function<void()> one = [&]() {
    idle.store(0, std::memory_order_relaxed);

    ++totalwrites;

    client->write(std::move(topmessage), [&](const tensorpipe::Error& error, tensorpipe::Message&& message) {
      if (error) {
        printf("write error: %s\n", error.what().c_str());
      } else {
        //printf("write success\n");
        //printf("message has %d tensors\n", message.tensors.capacity());
        topmessage = std::move(message);
        if (idle.fetch_add(1, std::memory_order_relaxed) + 1 == 2) {
          if (!done) {
            one();
          }
        }
      }
    });
  };

  int npre = nallocs;
  std::chrono::steady_clock::duration tpre(alloctime.load());

  Timer timer;
  one();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  done = true;

  int n = count;
  float tt = timer.elapsed();

  int npost = nallocs;
  std::chrono::steady_clock::duration tpost(alloctime.load());

  printf("%d in %gs -- %g/s\n", n, tt, n / tt);

  while (idle != 2);

  printf("allocs pre %d  dur %d\n", npre, npost - npre);
  printf("alloctime pre %g  dur %g\n", std::chrono::duration_cast<
         std::chrono::duration<float, std::ratio<1, 1>>>(tpre)
  .count(), std::chrono::duration_cast<
         std::chrono::duration<float, std::ratio<1, 1>>>(tpost - tpre)
  .count());

  std::vector<std::pair<int, std::string_view>> sorted;
  for (auto& v : addrcounts) {
    sorted.emplace_back(v.second, v.first);
  }
  std::stable_sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i != std::min(size_t(20), sorted.size()); ++i) {
    auto&v = sorted[sorted.size() - i];
    printf("%dx  %s\n", v.first, v.second.data());
  }

  supersum += n;

}

std::vector<std::thread> threads;

void test(std::shared_ptr<tensorpipe::transport::Connection> conn) {

  auto lastRecv = std::chrono::steady_clock::now();

  std::minstd_rand rng(std::random_device{}());

  std::atomic_int nLive = 0;

  std::function<void()> readloop = [&]() {
    conn->read([&, conn](const tensorpipe::Error& error, const void* ptr, size_t len) {
      if (error) {
        printf("nLive is %d\n", (int)nLive);
        printf("read error: %s\n", error.what().c_str());
        throw std::runtime_error(error.what());
      } else {

        printf("recv %d bytes\n", len);

        lastRecv = std::chrono::steady_clock::now();

        if (false && rng() % 2 == 0) {
          std::vector<char> data;
          data.resize(42);

          auto* ptr = data.data();
          size_t len = data.size();

          conn->write(ptr, len, [&, data = std::move(data), conn](const tensorpipe::Error& error) {
            if (error) {
              printf("nLive is %d\n", (int)nLive);
              printf("write error: %s\n", error.what().c_str());
              throw std::runtime_error(error.what());
            } else {
              //printf("write success\n");
//              printf("+");
//              fflush(stdout);
            }
          });
        }

        readloop();
      }
    });
  };

  readloop();

  std::vector<char> data;
  data.resize(std::uniform_int_distribution<size_t>(0, 1024 * 1024)(rng));

  while (true) {

    //std::this_thread::sleep_for(std::chrono::milliseconds(std::uniform_int_distribution<int>(0, 1000)(rng)));

    while (nLive > 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(std::uniform_int_distribution<int>(0, 1000)(rng)));
    }

    ++nLive;
    conn->write("hello world", 10, [&, conn](const tensorpipe::Error& error) {
      if (error) {
        printf("nLive is %d\n", (int)nLive);
        printf("write error: %s\n", error.what().c_str());
        throw std::runtime_error(error.what());
      } else {
        --nLive;
        //printf("write success\n");
      }
    });

    auto now = std::chrono::steady_clock::now();
    if (now - lastRecv >= std::chrono::seconds(2)) {
      throw std::runtime_error("timeout");
    }

    //printf(".%d", (int)nLive);
    //fflush(stdout);

  }

}

void startTest(std::shared_ptr<tensorpipe::transport::Connection> conn) {
  threads.push_back(std::thread([conn]() {
    test(conn);
  }));
}

std::string connectTo;

void testit(int n) {
  tensorpipe::transport::ibv::Context ctx;

//  ctx.registerTransport(0, "uv", std::make_shared<tensorpipe::transport::uv::Context>());
//#if TENSORPIPE_HAS_IBV_TRANSPORT
//  ctx.registerTransport(1, "ibv", std::make_shared<tensorpipe::transport::ibv::Context>());
//#endif
//#if TENSORPIPE_HAS_SHM_TRANSPORT
//  //ctx.registerTransport(2, "shm", std::make_shared<tensorpipe::transport::shm::Context>());
//#endif // TENSORPIPE_HAS_SHM_TRANSPORT
//  ctx.registerChannel(0, "basic", std::make_shared<tensorpipe::channel::basic::Context>());
//#if TENSORPIPE_HAS_CMA_CHANNEL
//  //ctx.registerChannel(1, "cma", std::make_shared<tensorpipe::channel::cma::Context>());
//  //ctx.registerChannel(1, "xth", std::make_shared<tensorpipe::channel::xth::Context>());
//#endif // TENSORPIPE_HAS_CMA_CHANNEL

  if (connectTo.empty()) {
    auto server = ctx.listen("0.0.0.0:" + std::to_string(8999 + n));

    std::vector<char> data;

    data.resize(1024 * 1024 * 400);

    std::function<void()> one = [&]() {

      server->accept([&](const tensorpipe::Error& error, std::shared_ptr<tensorpipe::transport::Connection> pipe) {
        if (error) {
          printf("accept error: %s\n", error.what().c_str());
        } else {
          printf("accept success yey\n");

          one();

          startTest(pipe);
        }
      });
    };

    one();

    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

  }

  //auto client = ctx.connect(connectTo);
  std::vector<std::shared_ptr<tensorpipe::transport::Connection>> conns;
  for (int i = 0; i != 10; ++i) {
    startTest(ctx.connect(connectTo));
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  ctx.join();
}

int main(int argc, char** argv) {

  if (argc > 1) {
    connectTo = argv[1];
  }

  std::vector<std::thread> threads;
  for (int i = 0; i != 1; ++i) {
    threads.emplace_back(testit, i);
  }

  for (auto& v : threads) {
    v.join();
  }

  printf("total writes: %d\n", (int)totalwrites);
  //printf("message moves: %d\n", (int)messageMoveCount);

  printf("sum: %d\n", (int)supersum);

  return 0;
}
