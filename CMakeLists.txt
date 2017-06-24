cmake_minimum_required (VERSION 2.6)


project (ransac)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable (ransac ransac3.cpp)
target_link_libraries( ransac ${OpenCV_LIBS})