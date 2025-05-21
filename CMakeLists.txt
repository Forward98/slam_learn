cmake_minimum_required(VERSION 3.10)
project(SLAM_LEARN)

# 设置 C++ 编译器
set(CMAKE_CXX_COMPILER "D:\\mingw64\\mingw64\\bin\\g++.exe")

# 添加 Eigen 库的头文件路径
include_directories(D:/mingw64/mingw64/include/eigen3)

# 添加头文件的搜索路径
include_directories(${PROJECT_SOURCE_DIR}/header)

# 指定 OpenCV 的头文件和库文件路径
set(OpenCV_DIR "D:/OpenCV/opencv/build/x64/MinGW/install")
include_directories(${OpenCV_DIR}/include)
link_directories(${OpenCV_DIR}/x64/mingw/bin)


# 将源文件列表写在变量 SrcFiles 中
aux_source_directory(./src SrcFiles)

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 设置可执行文件的名称
add_executable(myprint ${SrcFiles})

# 链接 OpenCV 库
target_link_libraries(myprint opencv_world4100)

# 添加调试标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

