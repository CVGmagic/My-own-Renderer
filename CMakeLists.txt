cmake_minimum_required(VERSION 3.10)
project(Renderer)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
include_directories(libs/glfw/include)

add_executable(app
    src/main.cpp
    src/glad.c
)

target_link_directories(app PRIVATE libs/glfw/lib-mingw-w64)

target_link_libraries(app glfw3 opengl32 gdi32)