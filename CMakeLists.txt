cmake_minimum_required(VERSION 3.10)
project(IrisTrading)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES 
    main.cpp 
    src/utils.cpp
    src/coin.cpp
    src/candle.cpp
    src/indicators.cpp
    src/smt_analysis.cpp
)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(iris ${SOURCES})

target_include_directories(iris PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(iris PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)