#include "../include/backtest_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

BacktestEngine::BacktestEngine(const SimulatorConfig& config) : tradingEngine(config) {
    peakEquity = 0.0;
    maxDrawdown = 0.0;
}

void BacktestEngine::loadData(const std::string& symbol, const std::vector<Candle>& candles) {
    candleData[symbol] = candles;
    currentIndex[symbol] = 0;
    tradingEngine.loadCandleData(symbol, candles);
}

void BacktestEngine::loadDataFromCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return;
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    std::vector<Candle> candles;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 7) {
            // Parse: Time,Open,High,Low,Close,Volume,Turnover
            // Time format: "YYYY-MM-DD HH:MM:SS" - need to convert to timestamp
            long long timestamp = 0;
            try {
                // Try parsing as timestamp first
                timestamp = std::stoll(tokens[0]);
            } catch (...) {
                // If not timestamp, parse as date string
                // Format: "YYYY-MM-DD HH:MM:SS"
                std::string timeStr = tokens[0];
                std::tm tm = {};
                std::istringstream ss(timeStr);
                ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                timestamp = std::mktime(&tm) * 1000;  // Convert to milliseconds
            }
            double open = std::stod(tokens[1]);
            double high = std::stod(tokens[2]);
            double low = std::stod(tokens[3]);
            double close = std::stod(tokens[4]);
            double volume = std::stod(tokens[5]);
            double turnover = std::stod(tokens[6]);
            
            Candle candle(timestamp, timestamp, open, high, low, close, volume, turnover);
            candles.push_back(candle);
        }
    }
    
    // Extract symbol from filename
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
    
    // Run backtest với delay execution để tránh look-ahead bias
    // Strategy ở nến i chỉ nhìn dữ liệu đến nến i-1, và lệnh sẽ được xử lý ở nến i+1
    for (int i = startIndex; i <= endIndex; ++i) {
        const Candle& candle = candles[i];
        long long timestamp = candle.getTimeOpen();
        
        // Bước 1: Xử lý lệnh đã đặt ở nến trước (nếu có)
        // Lệnh đặt ở nến i-1 sẽ được xử lý ở nến i
        if (i > startIndex) {
            tradingEngine.processNextCandle(symbol, candle, timestamp);
        }
        
        // Bước 2: Strategy nhìn dữ liệu đến nến i-1, đặt lệnh
        // Lệnh này sẽ được xử lý ở nến i+1
        if (strategy && i < endIndex) {
            // Get history (chỉ đến nến i-1 để tránh look-ahead)
            std::vector<Candle> history;
            if (i > 0) {
                int historySize = std::min(100, i);
                for (int j = std::max(0, i - historySize); j < i; ++j) {
                    history.push_back(candles[j]);
                }
            }
            
            // Strategy nhìn dữ liệu đến i-1, hành động lên nến i
            // Nhưng lệnh sẽ được delay đến nến i+1
            strategy(tradingEngine, symbol, candle, timestamp, history);
        }
        
        // Update equity curve
        updateEquityCurve(timestamp);
    }
    
    // Xử lý lệnh cuối cùng nếu có (lệnh đặt ở nến cuối)
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

