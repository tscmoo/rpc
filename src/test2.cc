#include "rpc.h"

#include <string>
#include <thread>
#include <list>

void serverManyClientsTest() {

  rpc::Rpc server;
  struct Peer {
    rpc::RpcConnection connection;
  };
  std::mutex peersMutex;
  std::list<Peer> peers;

  struct Client {
    rpc::Rpc rpc;
    rpc::RpcConnection connection;
    std::string name;
  };
  std::list<Client> clients;

  std::atomic_int count = 0;

  server.define<int(std::string_view)>("test", [&](std::string_view src) {
    return count.fetch_add(1, std::memory_order_relaxed) + 1;
  });
  auto listener = server.listen("127.0.0.1:7411");
  listener.accept([&](rpc::RpcConnection* conn, rpc::Error* err) {
    if (err) {
      throw std::runtime_error(err->what());
    } else {
      std::lock_guard l(peersMutex);
      peers.emplace_back();
      peers.back().connection = std::move(*conn);
    }
  });

  rpc::Function<void(Client&)> one = [&](Client& client) {
    client.rpc.asyncCallback<int>(client.connection, "test", [&](int* r, rpc::Error* err) {
      one(client);
    }, client.name);
  };

  for (int i = 0; i != 1; ++i) {
    clients.emplace_back();
    Client& client = clients.back();
    client.name = "client " + std::to_string(i);
    client.connection = client.rpc.connect("127.0.0.1:7411");
    one(client);
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));

  printf("count is %d\n", (int)count);

  std::quick_exit(0);

}

int main() {
  serverManyClientsTest();
  return 0;

  std::vector<std::unordered_map<std::string, float>> vec;
  vec.push_back({{"moo" , 42.5f}, {"baa", 0.0f}});
  vec.push_back({{"mooz" , 4.5f}, {"baaz", 0.5f}});
//  auto buffer = rpc::serializeToBuffer(42, 43, 44, vec);

//  std::vector<std::unordered_map<std::string_view, float>> vec2;

//  int a, b, c;
//  rpc::deserializeBuffer(buffer, a, b, c, vec2);

//  printf(" %d %d %d\n", a, b, c);

  //printf("eq %d\n", vec == vec2);

  rpc::Rpc server, client;

  int counter = 0;
  server.define<int(std::string_view, std::vector<std::unordered_map<std::string, float>>&&)>("test", [&](std::string_view str, std::vector<std::unordered_map<std::string, float>>&&) {
    //printf("str is %s\n", std::string(str).c_str());
    return ++counter;
  });

  int lastcount = 0;
  {
    rpc::RpcConnection serverconn;

    auto listener = server.listen("shm://testwee");
    listener.accept([&](rpc::RpcConnection* connection, rpc::Error* e) {
      if (connection) {
        printf("accepted connection yey\n");
        serverconn = std::move(*connection);
      } else {
        printf("accept error\n");
        throw std::runtime_error(e->what());
      }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    rpc::Function<void()> one;

    rpc::RpcConnection clientconn = client.connect("shm://testwee");

    std::string str = "hello world";
    one = [&]() {
      client.asyncCallback<int>(clientconn, "test", [&](int* n, rpc::Error* e) {
        if (e) {
          printf("oh no there was an error: %s\n", e->what());
        } else {
          int x = *n;
          if (x != lastcount + 1) {
            printf("miscount!\n");
          } else {
            lastcount = x;
          }
          one();
        }
      }, std::string_view("hello world"), vec);
    };

    one();

    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  printf("counter is at %d\n", counter);
  printf("lastcount is %d\n", lastcount);

  return 0;
}
