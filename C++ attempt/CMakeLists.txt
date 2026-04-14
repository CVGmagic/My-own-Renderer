cmake_minimum_required(VERSION 3.20)
project(MyRenderer)

set(CMAKE_CXX_STANDARD 17)

add_executable(app
    src/main.cpp
    src/glad.c
)

target_include_directories(app PRIVATE
    include
    libs/glfw/include
)

# Link GLFW explicitly (NO link_directories!)
set(GLFW_LIB "${CMAKE_SOURCE_DIR}/libs/glfw/lib-vc2022/glfw3.lib")

target_link_libraries(app PRIVATE
    ${GLFW_LIB}
    opengl32
    gdi32
)