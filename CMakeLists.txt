cmake_minimum_required(VERSION 3.10)

project(TRABFINAL)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/include/opencv/build)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OpenCV_DIR})
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Collect all .cpp files in the src directory
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set the output directory for the executable
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})