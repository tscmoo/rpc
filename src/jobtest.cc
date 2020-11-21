
#include "job.h"

#include <cmath>
#include <random>
#include <fstream>
#include <unistd.h>

extern "C" {
//#include "../nle/include/hack.h"
#include "../nle/include/nledl.h"
}

constexpr int actions[] = {
    13, 32, 107, 108, 106, 104, 117, 110, 98, 121,
    75, 76,  74,  72,  85,  78,  66,  89,
};
constexpr size_t actionsSize = sizeof(actions) / sizeof(actions[0]);

thread_local std::minstd_rand threadRng(std::random_device{}());

std::atomic_int counter = 0;
std::atomic_int resetCount = 0;
std::atomic_int namecounter = 0;

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

struct TestTask: moo::Task {

  nle_ctx_t *nle = nullptr;

  nle_obs obs{};
  static constexpr int dungeon_size = 21 * (80 - 1);
  short glyphs[dungeon_size];
  unsigned char chars[dungeon_size];
  unsigned char colors[dungeon_size];
  unsigned char specials[dungeon_size];
  unsigned char message[256];
  long blstats[NLE_BLSTATS_SIZE];
  int program_state[NLE_PROGRAM_STATE_SIZE];
  int internal[NLE_INTERNAL_SIZE];

  int steps = 0;

  TestTask() {
    obs.glyphs = &glyphs[0];
    obs.chars = &chars[0];
    obs.colors = &colors[0];
    obs.specials = &specials[0];
    obs.message = &message[0];
    obs.blstats = &blstats[0];
    obs.internal = &internal[0];
    obs.program_state = &program_state[0];

    std::string localname = "/home/a/rpc/build/libnethack-" + std::to_string(++namecounter) + ".so";

    {
      std::ifstream  src("/home/a/rpc/nle/nle/libnethack.so", std::ios::binary);
      std::ofstream  dst(localname.c_str(), std::ios::binary);

      dst << src.rdbuf();
    }

    printf("using localname %s\n", localname.c_str());

//    std::string localname = "/home/a/rpc/nle/nle/libnethack.so";
    nle = nle_start(localname.c_str(), &obs, nullptr, nullptr);

    //unlink(localname.c_str());

    reset();
  }

  virtual bool step() override {
    auto start = std::chrono::steady_clock::now();
    ++steps;
    obs.action = actions[std::uniform_int_distribution<size_t>(0, actionsSize - 1)(threadRng)];
    nle_step(nle, &obs);
    counter.fetch_add(1, std::memory_order_relaxed);
    steptime += (std::chrono::steady_clock::now() - start).count();
    if (steps >= 50000) std::abort();
    return !obs.done;
  }

  virtual void reset() override {
    auto start = std::chrono::steady_clock::now();
    ++resetCount;
    steps = 0;
    nle_reset(nle, &obs, nullptr, nullptr);
    resettime += (std::chrono::steady_clock::now() - start).count();
  }

};

struct TestJob: moo::Job {

  virtual std::unique_ptr<moo::Task> startTask() override {
    return std::make_unique<TestTask>();
  }

};

struct Batcher {

};

int main() {

  TestJob job;

  std::thread t([&] {
    job.run();
  });

  std::this_thread::sleep_for(std::chrono::seconds(5));
  printf("start!\n");

  int startcounter = counter;
  int s = 20;
  std::this_thread::sleep_for(std::chrono::seconds(s));

  int n = counter - startcounter;
  printf("%d\n", (int)n);
  printf("%d\n", (int)n / s);

  printf("%d reset\n", (int)resetCount);

  printf("step time: %dms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::duration(steptime)).count());
  printf("reset time: %dms\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::duration(resettime)).count());

  std::quick_exit(0);

  t.join();

  return 0;
}

