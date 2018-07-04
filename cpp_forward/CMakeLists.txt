cmake_minimum_required(VERSION 2.8)

project(cifar_forward)

set(EXECUTABLE classifier)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
file(GLOB SRC_FILES main.cpp)

include(FindPackageHandleStandardArgs)
unset(CAFFE2_FOUND)
find_library(Caffe2_LIBRARY_GPU caffe2_gpu)
find_library(Caffe2_LIBRARY_CPU caffe2)
find_package_handle_standard_args(Caffe2 DEFAULT_MSG Caffe2_LIBRARY_CPU Caffe2_LIBRARY_GPU)
if(CAFFE2_FOUND)
    set(Caffe2_LIBRARY ${Caffe2_LIBRARY_CPU} ${Caffe2_LIBRARY_GPU})
    set(Caffe2_INCLUDE_DIR "/usr/local/include")
endif()


# find packages
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


set(PROJECT_LIBRARIES ${Caffe2_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${OpenCV_LIBS} )
set(PROJECT_INCLUDES ${Caffe2_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
set(LIBS -ldl make)

message(Project_include_path: ${PROJECT_INCLUDES})

include_directories(${PROJECT_INCLUDES})

CUDA_ADD_EXECUTABLE(${EXECUTABLE}  ${SRC_FILES})

target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES} -L/usr/local/cuda/lib64 -lcuda -lcudart -lprotobuf -lglog -lgflags -ldl -lrt -L/usr/local/lib/)



