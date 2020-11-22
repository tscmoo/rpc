cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -march=native")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

message(STATUS "Adding PyTorch compilation flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TP_BUILD_BENCHMARK ON)

find_package(Threads REQUIRED)

add_subdirectory(src/fmt)
add_subdirectory(src/tensorpipe)
add_subdirectory(src/pybind11)

add_executable(rpctest src/test.cc)
target_include_directories(rpctest PUBLIC src/tensorpipe)
target_link_libraries(rpctest fmt tensorpipe)

add_executable(rpctest2 src/test2.cc src/rpc.cc src/network.cc)
target_include_directories(rpctest2 PUBLIC src/tensorpipe src)
target_link_libraries(rpctest2 fmt tensorpipe ${TORCH_LIBRARIES})
target_include_directories(rpctest2 SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

add_library(rpc src/rpc.cc src/network.cc)
target_include_directories(rpc PUBLIC src/tensorpipe src)
target_link_libraries(rpc fmt tensorpipe ${TORCH_LIBRARIES})
target_include_directories(rpc SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})


add_executable(jobtest src/jobtest.cc nle/sys/unix/nledl.c)
target_link_libraries(jobtest Threads::Threads dl)
target_include_directories(jobtest PUBLIC nle/include/)

pybind11_add_module(pyjob src/pyjob.cc)
target_link_libraries(pyjob PUBLIC rpc ${TorchPath}/lib/libtorch_python.so)
