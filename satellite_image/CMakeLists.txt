cmake_minimum_required(VERSION 3.12)
project(SatelliteAligner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(satellite_viewer
    src/main.cpp
)

target_include_directories(satellite_viewer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
)

target_link_libraries(satellite_viewer PRIVATE
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
)
