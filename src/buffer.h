#pragma once

#include "allocator.h"

#include <torch/torch.h>

#include <cstddef>
#include <cstdlib>
#include <new>

namespace rpc {

struct TensorRef {
  torch::Tensor tensor;
  size_t offset;
};

struct Buffer {
  Buffer* next{nullptr};
  size_t capacity = 0;
  size_t size = 0;
  size_t nTensors = 0;
  std::atomic_int refcount{0};
  std::byte* data() {
    return dataptr<std::byte>(this);
  }
  static constexpr size_t roundUpSizeForTensors(size_t size) {
    constexpr auto alignment = alignof(TensorRef);
    return (size + alignment - 1) / alignment * alignment;
  }
  TensorRef* tensors() {
    return (TensorRef*)(data() + roundUpSizeForTensors(size));
  }
};

inline void destroyBuffer(Buffer* buffer) noexcept {
  if (buffer->nTensors) {
    for (size_t i = buffer->nTensors; i;) {
      --i;
      buffer->tensors()[i].~TensorRef();
    }
    buffer->nTensors = 0;
  }
}

inline void shrinkBuffer(Buffer* buffer, size_t size, size_t nTensors) {
  for (size_t i = buffer->nTensors; i != nTensors;) {
    --i;
    buffer->tensors()[i].~TensorRef();
  }
  buffer->nTensors = nTensors;
  buffer->size = size;
}

struct BufferHandle {
  Buffer* buffer_ = nullptr;
  BufferHandle() = default;
  BufferHandle(std::nullptr_t) noexcept {}
  explicit BufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {}
  BufferHandle(const BufferHandle&) = delete;
  BufferHandle& operator=(const BufferHandle&) = delete;
  BufferHandle(BufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  BufferHandle& operator=(BufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~BufferHandle() {
    if (buffer_) {
      destroyBuffer(buffer_);
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer_;
    buffer_ = nullptr;
    return r;
  }
};
struct SharedBufferHandle {
  Buffer* buffer_ = nullptr;
  SharedBufferHandle() = default;
  SharedBufferHandle(std::nullptr_t) noexcept {}
  explicit SharedBufferHandle(Buffer* buffer) noexcept : buffer_(buffer) {
    if (buffer_) {
      if (buffer->refcount != 0) {
        std::abort();
      }
      addref();
    }
  }
  SharedBufferHandle(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
  }
  SharedBufferHandle& operator=(const SharedBufferHandle& n) noexcept {
    buffer_ = n.buffer_;
    if (buffer_) {
      addref();
    }
    return *this;
  }
  SharedBufferHandle(SharedBufferHandle&& n) noexcept {
    buffer_ = n.buffer_;
    n.buffer_ = nullptr;
  }
  SharedBufferHandle& operator=(SharedBufferHandle&& n) noexcept {
    std::swap(buffer_, n.buffer_);
    return *this;
  }
  ~SharedBufferHandle() {
    if (buffer_ && decref() == 0) {
      destroyBuffer(buffer_);
      deallocate<Buffer, std::byte>(buffer_);
    }
  }
  explicit operator bool() const noexcept {
    return buffer_;
  }
  Buffer* operator->() const noexcept {
    return buffer_;
  }
  operator Buffer*() const noexcept {
    return buffer_;
  }
  int addref() noexcept {
    return buffer_->refcount.fetch_add(1, std::memory_order_acquire) + 1;
  }
  int decref() noexcept {
    return buffer_->refcount.fetch_sub(1, std::memory_order_release) - 1;
  }
};

inline BufferHandle makeBuffer(size_t size, size_t nTensors) noexcept {
  size_t allocsize = size;
  if (nTensors) {
    allocsize = Buffer::roundUpSizeForTensors(allocsize) + sizeof(TensorRef) * nTensors;
  }
  BufferHandle buffer{allocate<Buffer, std::byte>(allocsize)};
  buffer->size = size;
  buffer->nTensors = nTensors;
  if (nTensors) {
    for (size_t i = 0; i != nTensors; ++i) {
      new (buffer->tensors() + i) TensorRef{};
    }
  }
  return buffer;
}

}
