#include "../include/backtest_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

BacktestEngine::BacktestEngine(const SimulatorConfig& config) : tradingEngine(config) {
    peakEquity = 0.0;
    maxDrawdown = 0.0;
}

// void BacktestEngine::loadData(const std::string& symbol, const std::vector<Candle>& candles) {
//     candleData[symbol] = candles;
//     currentIndex[symbol] = 0;
//     tradingEngine.loadCandleData(symbol, candles);
// }

// Thay thế hàm loadData cũ bằng hàm này:
void BacktestEngine::loadData(const std::string& symbol, const std::vector<Candle>& candles) {
    if (candles.empty()) {
        std::cout << "[ERROR] Loaded candles vector is empty!" << std::endl;
        return;
    }

    // DEBUG: In ra dữ liệu TRƯỚC khi sort
    std::cout << "--- DEBUG DATA INFO ---" << std::endl;
    std::cout << "Total Candles: " << candles.size() << std::endl;
    std::cout << "Before Sort - First Candle Time: " << candles.front().getTimeOpen() << std::endl;
    std::cout << "Before Sort - Last Candle Time:  " << candles.back().getTimeOpen() << std::endl;

    // Sort nến theo thời gian tăng dần
    std::vector<Candle> sortedCandles = candles;
    std::sort(sortedCandles.begin(), sortedCandles.end(), 
        [](const Candle& a, const Candle& b) {
            return a.getTimeOpen() < b.getTimeOpen();
        });

    // DEBUG: In ra dữ liệu SAU khi sort
    std::cout << "After Sort  - First Candle Time: " << sortedCandles.front().getTimeOpen() << std::endl;
    std::cout << "After Sort  - Last Candle Time:  " << sortedCandles.back().getTimeOpen() << std::endl;
    std::cout << "-----------------------" << std::endl;

    // Lưu dữ liệu đã sort
    candleData[symbol] = sortedCandles;
    currentIndex[symbol] = 0;
    tradingEngine.loadCandleData(symbol, sortedCandles);
}

// void BacktestEngine::loadDataFromCSV(const std::string& filepath) {
//     std::ifstream file(filepath);
//     if (!file.is_open()) {
//         return;
//     }
    
//     std::string line;
//     std::getline(file, line);  // Skip header
    
//     std::vector<Candle> candles;
    
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string token;
//         std::vector<std::string> tokens;
        
//         while (std::getline(ss, token, ',')) {
//             tokens.push_back(token);
//         }
        
//         if (tokens.size() >= 7) {
//             // Parse: Time,Open,High,Low,Close,Volume,Turnover
//             // Time format: "YYYY-MM-DD HH:MM:SS" - need to convert to timestamp
//             long long timestamp = 0;
//             try {
//                 // Try parsing as timestamp first
//                 timestamp = std::stoll(tokens[0]);
//             } catch (...) {
//                 // If not timestamp, parse as date string
//                 // Format: "YYYY-MM-DD HH:MM:SS"
//                 std::string timeStr = tokens[0];
//                 std::tm tm = {};
//                 std::istringstream ss(timeStr);
//                 ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
//                 timestamp = std::mktime(&tm) * 1000;  // Convert to milliseconds
//             }
//             double open = std::stod(tokens[1]);
//             double high = std::stod(tokens[2]);
//             double low = std::stod(tokens[3]);
//             double close = std::stod(tokens[4]);
//             double volume = std::stod(tokens[5]);
//             double turnover = std::stod(tokens[6]);
            
//             Candle candle(timestamp, timestamp, open, high, low, close, volume, turnover);
//             candles.push_back(candle);
//         }
//     }
    
//     // Extract symbol from filename
//     std::string symbol = filepath;
//     size_t lastSlash = symbol.find_last_of("/\\");
//     if (lastSlash != std::string::npos) {
//         symbol = symbol.substr(lastSlash + 1);
//     }
//     size_t lastDot = symbol.find_last_of(".");
//     if (lastDot != std::string::npos) {
//         symbol = symbol.substr(0, lastDot);
//     }
    
//     loadData(symbol, candles);
// }
void BacktestEngine::loadDataFromCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << filepath << std::endl;
        return;
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    std::vector<Candle> candles;
    int lineCount = 0;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 7) {
            long long timestamp = 0;
            std::string timeStr = tokens[0];

            // [FIX] Kiểm tra nếu là chuỗi ngày tháng (chứa dấu "-") thì parse theo format
            if (timeStr.find("-") != std::string::npos) {
                std::tm tm = {};
                std::istringstream ssTime(timeStr);
                // Parse format: YYYY-MM-DD HH:MM:SS
                ssTime >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                
                // Mẹo: Cần set tm_isdst = -1 để mktime tự xác định Daylight Saving Time
                tm.tm_isdst = -1; 
                
                std::time_t timeT = std::mktime(&tm);
                if (timeT == -1) {
                    // Fallback nếu parse lỗi
                    continue; 
                }
                timestamp = static_cast<long long>(timeT) * 1000; // Đổi sang milliseconds
            } else {
                // Nếu không có dấu "-", thử parse trực tiếp thành số (trường hợp timestamp raw)
                try {
                    timestamp = std::stoll(timeStr);
                } catch (...) {
                    continue; // Skip dòng lỗi
                }
            }

            double open = std::stod(tokens[1]);
            double high = std::stod(tokens[2]);
            double low = std::stod(tokens[3]);
            double close = std::stod(tokens[4]);
            double volume = std::stod(tokens[5]);
            double turnover = std::stod(tokens[6]);
            
            Candle candle(timestamp, timestamp, open, high, low, close, volume, turnover);
            candles.push_back(candle);
            lineCount++;
        }
    }
    
    std::cout << "[INFO] Loaded " << lineCount << " lines from CSV." << std::endl;

    // Extract symbol logic (giữ nguyên)
    std::string symbol = filepath;
    size_t lastSlash = symbol.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        symbol = symbol.substr(lastSlash + 1);
    }
    size_t lastDot = symbol.find_last_of(".");
    if (lastDot != std::string::npos) {
        symbol = symbol.substr(0, lastDot);
    }
    
    loadData(symbol, candles);
}

void BacktestEngine::setStrategy(StrategyCallback strategyFunc) {
    strategy = strategyFunc;
}

void BacktestEngine::setInitialBalance(double balance) {
    tradingEngine.setInitialBalance(balance);
    peakEquity = balance;
    maxDrawdown = 0.0;
    equityCurve.clear();
    equityTimestamps.clear();
}

void BacktestEngine::updateEquityCurve(long long timestamp) {
    double currentEquity = tradingEngine.getAccount().balance + tradingEngine.getUnrealizedPnL();
    equityCurve.push_back(currentEquity);
    equityTimestamps.push_back(timestamp);
    
    if (currentEquity > peakEquity) {
        peakEquity = currentEquity;
    }
    
    double drawdown = peakEquity - currentEquity;
    if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
    }
}

void BacktestEngine::calculateStatistics(BacktestResult& result) {
    Account account = tradingEngine.getAccount();
    
    result.initialBalance = account.balance;
    result.finalBalance = account.balance + account.unrealizedPnl;
    result.totalReturn = result.finalBalance - result.initialBalance;
    result.totalReturnPercent = (result.totalReturn / result.initialBalance) * 100.0;
    result.totalPnL = tradingEngine.getTotalPnL();
    result.realizedPnL = tradingEngine.getRealizedPnL();
    result.unrealizedPnL = tradingEngine.getUnrealizedPnL();
    result.totalFees = tradingEngine.getTotalFees();
    result.totalTrades = tradingEngine.getTotalTrades();
    result.maxDrawdown = maxDrawdown;
    result.maxDrawdownPercent = (maxDrawdown / peakEquity) * 100.0;
    result.equityCurve = equityCurve;
    result.equityTimestamps = equityTimestamps;
    
    // Calculate win rate
    int wins = 0;
    int losses = 0;
    double totalWin = 0.0;
    double totalLoss = 0.0;
    
    // Simplified calculation - in real implementation, need to track entry/exit pairs
    for (const auto& trade : account.tradeHistory) {
        // This is simplified - need proper entry/exit tracking
    }
    
    result.winningTrades = wins;
    result.losingTrades = losses;
    result.winRate = result.totalTrades > 0 ? (static_cast<double>(wins) / result.totalTrades) * 100.0 : 0.0;
    result.averageWin = wins > 0 ? totalWin / wins : 0.0;
    result.averageLoss = losses > 0 ? totalLoss / losses : 0.0;
    result.profitFactor = totalLoss != 0.0 ? totalWin / std::abs(totalLoss) : 0.0;
    
    // Calculate Sharpe Ratio (simplified)
    if (equityCurve.size() > 1) {
        std::vector<double> returns;
        for (size_t i = 1; i < equityCurve.size(); ++i) {
            double ret = (equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1];
            returns.push_back(ret);
        }
        
        if (!returns.empty()) {
            double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
            double variance = 0.0;
            for (double ret : returns) {
                variance += (ret - mean) * (ret - mean);
            }
            variance /= returns.size();
            double stdDev = std::sqrt(variance);
            
            result.sharpeRatio = stdDev != 0.0 ? mean / stdDev : 0.0;
        }
    }
    
    // Count both sides hit
    result.bothSidesHitCount = 0;
    for (const auto& order : account.openOrders) {
        if (order.isBothSidesHit) {
            result.bothSidesHitCount++;
        }
    }
}

// BacktestResult BacktestEngine::run(const std::string& symbol, int startIndex, int endIndex) {
//     if (candleData.find(symbol) == candleData.end()) {
//         BacktestResult empty;
//         return empty;
//     }
    
//     const std::vector<Candle>& candles = candleData[symbol];
//     if (startIndex < 0) startIndex = 0;
//     if (endIndex < 0 || endIndex >= static_cast<int>(candles.size())) {
//         endIndex = candles.size() - 1;
//     }
    
//     // Reset
//     tradingEngine.reset();
//     setInitialBalance(tradingEngine.getAccount().balance);
//     equityCurve.clear();
//     equityTimestamps.clear();
//     peakEquity = tradingEngine.getAccount().balance;
//     maxDrawdown = 0.0;
    
//     // Run backtest với delay execution để tránh look-ahead bias
//     // Strategy ở nến i chỉ nhìn dữ liệu đến nến i-1, và lệnh sẽ được xử lý ở nến i+1
//     for (int i = startIndex; i <= endIndex; ++i) {
//         const Candle& candle = candles[i];
//         long long timestamp = candle.getTimeOpen();
        
//         // Bước 1: Xử lý lệnh đã đặt ở nến trước (nếu có)
//         // Lệnh đặt ở nến i-1 sẽ được xử lý ở nến i
//         if (i > startIndex) {
//             tradingEngine.processNextCandle(symbol, candle, timestamp);
//         }
        
//         // Bước 2: Strategy nhìn dữ liệu đến nến i-1, đặt lệnh
//         // Lệnh này sẽ được xử lý ở nến i+1
//         if (strategy && i < endIndex) {
//             // Get history (chỉ đến nến i-1 để tránh look-ahead)
//             // QUAN TRỌNG: History chỉ chứa data từ startIndex đến i-1
//             // Điều này đảm bảo khi test (startIndex = testStartIndex), 
//             // history không bao gồm future data từ test phase
//             std::vector<Candle> history;
//             if (i > startIndex) {
//                 int historySize = std::min(100, i - startIndex);
//                 int historyStart = std::max(startIndex, i - historySize);
//                 for (int j = historyStart; j < i; ++j) {
//                     history.push_back(candles[j]);
//                 }
//             }
            
//             // Strategy nhìn dữ liệu đến i-1, hành động lên nến i
//             // Nhưng lệnh sẽ được delay đến nến i+1
//             strategy(tradingEngine, symbol, candle, timestamp, history);
//         }
        
//         // Update equity curve
//         updateEquityCurve(timestamp);
//     }
    
//     // Xử lý lệnh cuối cùng nếu có (lệnh đặt ở nến cuối)
//     if (endIndex >= startIndex && endIndex < static_cast<int>(candles.size())) {
//         const Candle& lastCandle = candles[endIndex];
//         tradingEngine.processNextCandle(symbol, lastCandle, lastCandle.getTimeOpen());
//         updateEquityCurve(lastCandle.getTimeOpen());
//     }
    
//     // Calculate results
//     BacktestResult result;
//     calculateStatistics(result);
    
//     return result;
// }
// [QUAN TRỌNG] Thay thế hàm BacktestEngine::run cũ bằng hàm này
BacktestResult BacktestEngine::run(const std::string& symbol, int startIndex, int endIndex) {
    if (candleData.find(symbol) == candleData.end()) {
        BacktestResult empty;
        return empty;
    }
    
    const std::vector<Candle>& candles = candleData[symbol];
    if (startIndex < 0) startIndex = 0;
    if (endIndex < 0 || endIndex >= static_cast<int>(candles.size())) {
        endIndex = candles.size() - 1;
    }
    
    // Reset
    tradingEngine.reset();
    setInitialBalance(tradingEngine.getAccount().balance);
    equityCurve.clear();
    equityTimestamps.clear();
    peakEquity = tradingEngine.getAccount().balance;
    maxDrawdown = 0.0;

    // [MỚI] Mở file log trong thư mục report (Ghi đè mới mỗi lần chạy)
    std::string logPath = "report/backtest_run.log";
    std::ofstream logFile(logPath); // Mở chế độ write để clear file cũ
    if (logFile.is_open()) {
        logFile << "=== START BACKTEST: " << symbol << " ===" << std::endl;
        logFile << "Range: " << startIndex << " to " << endIndex << std::endl;
        logFile.close();
    }
    
    int totalSteps = endIndex - startIndex + 1;
    int logInterval = totalSteps / 100; // Log mỗi 1% tiến độ
    if (logInterval == 0) logInterval = 1;

    // Run backtest
    for (int i = startIndex; i <= endIndex; ++i) {
        const Candle& candle = candles[i];
        long long timestamp = candle.getTimeOpen();
        
        // --- [MỚI] PHẦN IN TIẾN ĐỘ ---
        if ((i - startIndex) % logInterval == 0 || i == endIndex) {
            double progress = (double)(i - startIndex) / totalSteps * 100.0;
            
            // 1. In ra Console (dùng \r để ghi đè dòng cũ cho gọn)
            std::cout << "\r[Running] Progress: " << std::fixed << std::setprecision(1) 
                      << progress << "% (" << (i - startIndex) << "/" << totalSteps << ")" 
                      << " | Balance: " << tradingEngine.getAccount().balance << std::flush;

            // 2. Ghi vào file Report (Append mode)
            std::ofstream log(logPath, std::ios::app);
            if (log.is_open()) {
                log << "Index: " << i << " | Time: " << timestamp 
                    << " | Balance: " << tradingEngine.getAccount().balance 
                    << " | PnL: " << tradingEngine.getUnrealizedPnL() << std::endl;
            }
        }
        // -----------------------------
        
        // Bước 1: Xử lý lệnh đã đặt
        if (i > startIndex) {
            tradingEngine.processNextCandle(symbol, candle, timestamp);
        }
        
        // Bước 2: Strategy đặt lệnh
        if (strategy && i < endIndex) {
            std::vector<Candle> history;
            if (i > startIndex) {
                int historySize = std::min(100, i - startIndex);
                int historyStart = std::max(startIndex, i - historySize);
                for (int j = historyStart; j < i; ++j) {
                    history.push_back(candles[j]);
                }
            }
            strategy(tradingEngine, symbol, candle, timestamp, history);
        }
        
        // Update equity curve
        updateEquityCurve(timestamp);
    }

    // Xuống dòng sau khi chạy xong progress bar
    std::cout << std::endl; 
    
    // Xử lý nến cuối cùng
    if (endIndex >= startIndex && endIndex < static_cast<int>(candles.size())) {
        const Candle& lastCandle = candles[endIndex];
        tradingEngine.processNextCandle(symbol, lastCandle, lastCandle.getTimeOpen());
        updateEquityCurve(lastCandle.getTimeOpen());
    }
    
    // Calculate results
    BacktestResult result;
    calculateStatistics(result);
    
    return result;
}

BacktestResult BacktestEngine::runAll(const std::string& symbol) {
    return run(symbol, 0, -1);
}

TradingEngine& BacktestEngine::getTradingEngine() {
    return tradingEngine;
}

const TradingEngine& BacktestEngine::getTradingEngine() const {
    return tradingEngine;
}

std::vector<double> BacktestEngine::getEquityCurve() const {
    return equityCurve;
}

std::vector<long long> BacktestEngine::getEquityTimestamps() const {
    return equityTimestamps;
}

void BacktestEngine::reset() {
    tradingEngine.reset();
    setInitialBalance(tradingEngine.getAccount().balance);
    equityCurve.clear();
    equityTimestamps.clear();
    peakEquity = tradingEngine.getAccount().balance;
    maxDrawdown = 0.0;
}

// Get train/test split indices
std::pair<int, int> BacktestEngine::getTrainTestSplit(const std::string& symbol, double trainRatio) {
    if (candleData.find(symbol) == candleData.end()) {
        return {0, 0};
    }
    
    const std::vector<Candle>& candles = candleData[symbol];
    int totalSize = candles.size();
    
    if (totalSize == 0) {
        return {0, 0};
    }
    
    // Train: 0 to trainEndIndex (2/3)
    // Test: testStartIndex to end (1/3)
    int trainEndIndex = static_cast<int>(totalSize * trainRatio) - 1;
    int testStartIndex = trainEndIndex + 1;
    
    // Đảm bảo có ít nhất 1 candle cho test
    if (testStartIndex >= totalSize) {
        testStartIndex = totalSize - 1;
        trainEndIndex = testStartIndex - 1;
    }
    
    return {trainEndIndex, testStartIndex};
}

// Validate no data leakage
bool BacktestEngine::validateNoDataLeakage(const std::string& symbol, int trainEndIndex, int testStartIndex) {
    if (candleData.find(symbol) == candleData.end()) {
        return false;
    }
    
    const std::vector<Candle>& candles = candleData[symbol];
    
    // Kiểm tra trainEndIndex < testStartIndex
    if (trainEndIndex >= testStartIndex) {
        return false;
    }
    
    // Kiểm tra timestamp: tất cả train candles phải có timestamp < test candles
    if (trainEndIndex >= 0 && testStartIndex < static_cast<int>(candles.size())) {
        long long lastTrainTime = candles[trainEndIndex].getTimeOpen();
        long long firstTestTime = candles[testStartIndex].getTimeOpen();
        
        if (lastTrainTime >= firstTestTime) {
            return false;  // Data leakage: train có timestamp >= test
        }
    }
    
    return true;
}

// Run train/test split backtest
BacktestResult BacktestEngine::runTrainTest(const std::string& symbol, double trainRatio) {
    if (candleData.find(symbol) == candleData.end()) {
        BacktestResult empty;
        return empty;
    }
    
    // Get split indices
    auto [trainEndIndex, testStartIndex] = getTrainTestSplit(symbol, trainRatio);
    
    // Validate no data leakage
    if (!validateNoDataLeakage(symbol, trainEndIndex, testStartIndex)) {
        std::cerr << "ERROR: Data leakage detected! Train and test data overlap." << std::endl;
        BacktestResult empty;
        return empty;
    }
    
    const std::vector<Candle>& candles = candleData[symbol];
    int totalSize = candles.size();
    
    std::cout << "=== TRAIN/TEST SPLIT ===" << std::endl;
    std::cout << "Total candles: " << totalSize << std::endl;
    std::cout << "Train: 0 to " << trainEndIndex << " (" << (trainEndIndex + 1) << " candles, " 
              << ((trainEndIndex + 1) * 100.0 / totalSize) << "%)" << std::endl;
    std::cout << "Test: " << testStartIndex << " to " << (totalSize - 1) << " (" 
              << (totalSize - testStartIndex) << " candles, " 
              << ((totalSize - testStartIndex) * 100.0 / totalSize) << "%)" << std::endl;
    std::cout << "Validation: No data leakage ✓" << std::endl;
    std::cout << std::endl;
    
    // Bước 1: Train phase (2/3 data)
    std::cout << "=== TRAINING PHASE ===" << std::endl;
    reset();
    BacktestResult trainResult = run(symbol, 0, trainEndIndex);
    
    std::cout << "Train Results:" << std::endl;
    std::cout << "  Initial Balance: $" << trainResult.initialBalance << std::endl;
    std::cout << "  Final Balance: $" << trainResult.finalBalance << std::endl;
    std::cout << "  Return: " << trainResult.totalReturnPercent << "%" << std::endl;
    std::cout << "  Total Trades: " << trainResult.totalTrades << std::endl;
    std::cout << "  Max Drawdown: " << trainResult.maxDrawdownPercent << "%" << std::endl;
    std::cout << std::endl;
    
    // Bước 2: Test phase (1/3 data) - Model không được train trên test data
    std::cout << "=== TESTING PHASE (Out-of-Sample) ===" << std::endl;
    std::cout << "IMPORTANT: Model will only see test data, not train data." << std::endl;
    std::cout << "History in strategy callback will only contain candles up to current test candle." << std::endl;
    std::cout << std::endl;
    
    // Lưu lại state sau training (model đã học)
    // Không reset để model giữ lại kiến thức từ training
    // Tuy nhiên, khi chạy test, history chỉ chứa data từ testStartIndex trở về trước
    // Điều này đảm bảo model không nhìn thấy future data
    
    // Chạy test trên data mới (model chưa thấy)
    // Lưu ý: run() sẽ chỉ truyền history từ startIndex trở về trước, không bao gồm future
    BacktestResult testResult = run(symbol, testStartIndex, totalSize - 1);
    
    std::cout << "Test Results (Out-of-Sample):" << std::endl;
    std::cout << "  Initial Balance: $" << testResult.initialBalance << std::endl;
    std::cout << "  Final Balance: $" << testResult.finalBalance << std::endl;
    std::cout << "  Return: " << testResult.totalReturnPercent << "%" << std::endl;
    std::cout << "  Total Trades: " << testResult.totalTrades << std::endl;
    std::cout << "  Max Drawdown: " << testResult.maxDrawdownPercent << "%" << std::endl;
    std::cout << "  Both Sides Hit: " << testResult.bothSidesHitCount << std::endl;
    std::cout << std::endl;
    
    // Return test result (out-of-sample performance)
    return testResult;
}

