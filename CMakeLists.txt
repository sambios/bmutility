cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)
set(UTILITY_TOP ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${UTILITY_TOP}/bmutility)
include_directories(${FFMPEG_INCLUDE_DIRS})

add_library(bmutility stream_sei.cpp
        stream_demuxer.cpp
        stream_decode.cpp
        bmutility_timer.cpp
        bmutility_string.cpp
        )

