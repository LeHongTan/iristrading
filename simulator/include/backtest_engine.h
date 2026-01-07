#ifndef BACKTEST_ENGINE_H
#define BACKTEST_ENGINE_H

#include <vector>
#include <string>
#include <functional>
#include "../../include/candle.h"
#include "trading_engine.h"
#include "order.h"

struct BacktestResult {
    double initialBalance = 0.0;
    double finalBalance = 0.0;
    double totalReturn = 0.0;
    double totalReturnPercent = 0.0;
    double totalPnL = 0.0;
    double realizedPnL = 0.0;
    double unrealizedPnL = 0.0;
    double totalFees = 0.0;
    int totalTrades = 0;
    int winningTrades = 0;
    int losingTrades = 0;
    double winRate = 0.0;
    double averageWin = 0.0;
    double averageLoss = 0.0;
    double profitFactor = 0.0;
    double maxDrawdown = 0.0;
    double maxDrawdownPercent = 0.0;
    double sharpeRatio = 0.0;
    int bothSidesHitCount = 0;  // Số lần quét 2 đầu
    std::vector<double> equityCurve;  // Đường cong vốn
    std::vector<long long> equityTimestamps;
};

// Callback function type cho strategy
using StrategyCallback = std::function<void(
    TradingEngine& engine,
    const std::string& symbol,
    const Candle& candle,
    long long timestamp,
    const std::vector<Candle>& history
)>;

class BacktestEngine {
    private:
        TradingEngine tradingEngine;
        std::map<std::string, std::vector<Candle>> candleData;
        std::map<std::string, int> currentIndex;
        StrategyCallback strategy;
        std::vector<double> equityCurve;
        std::vector<long long> equityTimestamps;
        double peakEquity;
        double maxDrawdown;
        
        // Helper functions
        void updateEquityCurve(long long timestamp);
        void calculateStatistics(BacktestResult& result);
        
    public:
        BacktestEngine(const SimulatorConfig& config = SimulatorConfig());
        
        // Load dữ liệu
        void loadData(const std::string& symbol, const std::vector<Candle>& candles);
        void loadDataFromCSV(const std::string& filepath);
        
        // Set strategy
        void setStrategy(StrategyCallback strategyFunc);
        
        // Set initial balance
        void setInitialBalance(double balance);
        
        // Run backtest
        BacktestResult run(const std::string& symbol, int startIndex = 0, int endIndex = -1);
        BacktestResult runAll(const std::string& symbol);
        
        // Train/Test split (2/3 train, 1/3 test)
        BacktestResult runTrainTest(const std::string& symbol, double trainRatio = 0.67);
        
        // Get data split indices
        std::pair<int, int> getTrainTestSplit(const std::string& symbol, double trainRatio = 0.67);
        
        // Validate no data leakage (kiểm tra model không nhìn thấy future data)
        bool validateNoDataLeakage(const std::string& symbol, int trainEndIndex, int testStartIndex);
        
        // Get current state
        TradingEngine& getTradingEngine();
        const TradingEngine& getTradingEngine() const;
        
        // Get equity curve
        std::vector<double> getEquityCurve() const;
        std::vector<long long> getEquityTimestamps() const;
        
        // Reset
        void reset();
};

#endif

