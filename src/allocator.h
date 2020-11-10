#pragma once

#include "synchronization.h"

#include <cstddef>
#include <cstdlib>
#include <mutex>

namespace rpc {

namespace allocimpl {

template<typename Header, typename Data>
struct Storage {
  Header* freelist = nullptr;

  Header* allocate(size_t n) {
    //printf("allocate %p %s %d\n", this, typeid(*this).name(), n);
    static_assert(alignof(Header) >= alignof(Data));
//    Header* h = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
//    h->capacity = n;
//    return h;
    Header* r = freelist;
    if (!r) {
      r = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
      new (r) Header();
    } else {
      freelist = r->next;
      if (r->capacity < n) {
        if (r->refcount != 0) {
          std::abort();
        }
        r->~Header();
        std::free(r);
        r = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
        new (r) Header();
      }
    }
    //printf("allocate %p refcount %d\n", r, (int)r->refcount);
    if (r->refcount != 0) {
      std::abort();
    }
    return r;
  }
  void deallocate(Header* ptr) {
    //printf("deallocate %p refcount %d\n", ptr, (int)ptr->refcount);
    if (ptr->refcount != 0) {
      std::abort();
    }
//    std::free(ptr);
//    return;
    ptr->next = freelist;
    freelist = ptr;
  }
//  std::atomic<Header*> freelist = nullptr;

//  Header* allocate(size_t n) {
//    //printf("allocate %p %s\n", this, typeid(*this).name());
//    static_assert(alignof(Header) >= alignof(Data));
////    Header* h = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
////    h->capacity = n;
////    return h;
//    Header* r = freelist.load(std::memory_order_relaxed);
//    while (r && !freelist.compare_exchange_weak(r, r->next, std::memory_order_relaxed));
//    if (!r) {
//      r = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
//      r->capacity = n;
//    } else if (r->capacity < n) {
//      std::free(r);
//      r = (Header*)std::malloc(sizeof(Header) + sizeof(Data) * n);
//      r->capacity = n;
//    }
//    return r;
//  }
//  void deallocate(Header* ptr) {
////    std::free(ptr);
////    return;
//    Header* next = freelist.load(std::memory_order_relaxed);
//    do {
//      ptr->next = next;
//    } while (!freelist.compare_exchange_weak(next, ptr, std::memory_order_relaxed));
//  }
};

template<typename Header, typename Data>
inline thread_local Storage<Header, Data> storage;

}

template<typename Header, typename Data>
Header* allocate(size_t n) {
  return allocimpl::storage<Header, Data>.allocate(n);
}
template<typename Header, typename Data>
void deallocate(Header* ptr) {
  return allocimpl::storage<Header, Data>.deallocate(ptr);
}
template<typename Data, typename Header>
Data* dataptr(Header* ptr) {
  return (Data*)(ptr + 1);
}

}
