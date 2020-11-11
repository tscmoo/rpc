#pragma once

#include "tensorpipe/tensorpipe/common/function.h"

namespace rpc {

template<typename T>
using Function = tensorpipe::Function<T>;
using FunctionPointer = tensorpipe::FunctionPointer;

}
