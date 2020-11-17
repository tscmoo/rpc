#pragma once

#include "synchronization.h"

#include <cstddef>
#include <cstdlib>
#include <array>

namespace rpc {

namespace allocimpl {

template<typename Header, typename Data, size_t Size>
struct Storage {
  static constexpr size_t size = Size;
  Header* freelist = nullptr;
  size_t freelistSize = 0;

  ~Storage() {
    for (Header* ptr = freelist; ptr;) {
      Header* next = ptr->next;
      std::free(ptr);
      ptr = next;
    }
  }

  Header* allocate() {
    static_assert(alignof(Header) <= 64 && alignof(Data) <= 64 && alignof(Data) <= sizeof(Header));
    Header* r = freelist;
    if (!r) {
      r = (Header*)std::aligned_alloc(64, size);
      new (r) Header();
      r->capacity = size - sizeof(Header);
    } else {
      --freelistSize;
      freelist = r->next;
      if (r->capacity != size - sizeof(Header)) {
        std::abort();
      }
      if (r->refcount != 0) {
        std::abort();
      }
    }
    if (r->refcount != 0) {
      std::abort();
    }
    return r;
  }
  void deallocate(Header* ptr) {
    if (ptr->refcount != 0) {
      std::abort();
    }
    if (freelistSize >= 1024 * 1024 / Size) {
      std::free(ptr);
      return;
    }
    ++freelistSize;
    ptr->next = freelist;
    freelist = ptr;
  }

};

template<typename Header, typename Data>
inline thread_local Storage<Header, Data, 64> storage_64;
template<typename Header, typename Data>
inline thread_local Storage<Header, Data, 256> storage_256;
template<typename Header, typename Data>
inline thread_local Storage<Header, Data, 1024> storage_1024;
template<typename Header, typename Data>
inline thread_local Storage<Header, Data, 4096> storage_4096;

}

template<typename Header, typename Data>
Header* allocate(size_t n) {
  constexpr size_t overhead = sizeof(Header);
  if (n + overhead <= 64) {
    return allocimpl::storage_64<Header, Data>.allocate();
  } else if (n + overhead <= 256) {
    return allocimpl::storage_256<Header, Data>.allocate();
  } else if (n + overhead <= 1024) {
    return allocimpl::storage_1024<Header, Data>.allocate();
  } else if (n + overhead <= 4096) {
    return allocimpl::storage_4096<Header, Data>.allocate();
  } else {
    Header* h = (Header*)std::aligned_alloc(64, sizeof(Header) + sizeof(Data) * n);
    new (h) Header();
    h->capacity = n;
    return h;
  }
}
template<typename Header, typename Data>
void deallocate(Header* ptr) {
  const size_t n = ptr->capacity + sizeof(Header);
  switch (n) {
  case 64:
    allocimpl::storage_64<Header, Data>.deallocate(ptr);
    break;
  case 256:
    allocimpl::storage_256<Header, Data>.deallocate(ptr);
    break;
  case 1024:
    allocimpl::storage_1024<Header, Data>.deallocate(ptr);
    break;
  case 4096:
    allocimpl::storage_4096<Header, Data>.deallocate(ptr);
    break;
  default:
    if (n <= 4096 || ptr->refcount != 0) {
      std::abort();
    }
    std::free(ptr);
  }
}
template<typename Data, typename Header>
Data* dataptr(Header* ptr) {
  return (Data*)(ptr + 1);
}

}
