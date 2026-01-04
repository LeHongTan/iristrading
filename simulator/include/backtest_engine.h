#ifndef BACKTEST_ENGINE_H
#define BACKTEST_ENGINE_H

#include <vector>
#include <string>
#include <functional>
#include "../../include/candle.h"
#include "trading_engine.h"
#include "order.h"

struct BacktestResult {
    double initialBalance;
    double finalBalance;
    double totalReturn;
    double totalReturnPercent;
    double totalPnL;
    double realizedPnL;
    double unrealizedPnL;
    double totalFees;
    int totalTrades;
    int winningTrades;
    int losingTrades;
    double winRate;
    double averageWin;
    double averageLoss;
    double profitFactor;
    double maxDrawdown;
    double maxDrawdownPercent;
    double sharpeRatio;
    int bothSidesHitCount;  // Số lần quét 2 đầu
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

