
#include <stdexcept>

#include "function.h"
#include "buffer.h"

namespace shm2 {

using rpc::Function;
using rpc::Buffer;
using rpc::BufferHandle;
using rpc::SharedBufferHandle;

struct Error: std::runtime_error {
};

struct Connection;
struct Listener;

struct Context {

  std::unique_ptr<Connection> connect(std::string_view addr);
  std::unique_ptr<Listener> listen(std::string_view addr);

};

struct Connection {

  void close();
  void read(Function<void(Error*, BufferHandle&&)>&&);
  void write(const void* data, size_t len, Function<void(Error*)>&&);

};

struct Listener {

  void close();
  void accept(Function<void(Error*, std::unique_ptr<Connection>&&)>&& callback);

};

}

