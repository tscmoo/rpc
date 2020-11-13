cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rpc)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS
#    "${CMAKE_CXX_FLAGS} -pg")

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -fsanitize=thread")


set(TP_BUILD_BENCHMARK ON)

add_subdirectory(src/fmt)
add_subdirectory(src/tensorpipe)

add_executable(rpctest src/test.cc)
target_include_directories(rpctest PUBLIC src/tensorpipe)
target_link_libraries(rpctest fmt tensorpipe)

add_executable(rpctest2 src/test2.cc src/rpc.cc src/network.cc)
target_include_directories(rpctest2 PUBLIC src/tensorpipe src)
target_link_libraries(rpctest2 fmt tensorpipe)


