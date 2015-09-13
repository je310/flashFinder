cmake_minimum_required(VERSION 2.8)
project( flashFinder CXX)

set (CMAKE_CXX_FLAGS "-g -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")

find_package( OpenCV REQUIRED )
find_package( FFTW3 REQUIRED )

add_executable( flashFinder flashFinder.cpp )
add_executable( spreadSpec spreadSpec.cpp )

target_link_libraries( flashFinder ${OpenCV_LIBS} )
target_link_libraries( spreadSpec ${OpenCV_LIBS} ${FFTW3_LIBRARIES} )
find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


