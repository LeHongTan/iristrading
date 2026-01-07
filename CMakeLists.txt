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
    simulator/src/bybit_simulator.cpp
    simulator/src/trading_engine.cpp
    simulator/src/backtest_engine.cpp
)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(iris ${SOURCES})

# Backtest example executable (không include main.cpp)
set(BACKTEST_SOURCES 
    examples/backtest_example.cpp
    src/utils.cpp
    src/coin.cpp
    src/candle.cpp
    src/indicators.cpp
    src/smt_analysis.cpp
    simulator/src/bybit_simulator.cpp
    simulator/src/trading_engine.cpp
    simulator/src/backtest_engine.cpp
)

add_executable(backtest_example ${BACKTEST_SOURCES})
target_include_directories(backtest_example PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simulator/include
)
target_link_libraries(backtest_example PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

target_include_directories(iris PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simulator/include
)
target_link_libraries(iris PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)