#pragma once

#include "tensorpipe/tensorpipe/common/function.h"

namespace rpc {

template<typename T>
using Function = rpc_tensorpipe::Function<T>;
using FunctionPointer = rpc_tensorpipe::FunctionPointer;

}
