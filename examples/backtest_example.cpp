#include <iostream>
#include <vector>
#include <iomanip>
#include "backtest_engine.h"
#include "indicators.h"

// Simple strategy example - Moving Average Crossover
void simpleMAStrategy(TradingEngine& engine, const std::string& symbol,
                     const Candle& candle, long long timestamp,
                     const std::vector<Candle>& history) {
    
    // Cần ít nhất 20 nến để tính MA
    if (history.size() < 20) return;
    
    // Tính Simple Moving Average (SMA) 10 và 20
    double sma10 = 0.0;
    double sma20 = 0.0;
    
    for (size_t i = history.size() - 10; i < history.size(); ++i) {
        sma10 += history[i].getClose();
    }
    sma10 /= 10.0;
    
    for (size_t i = history.size() - 20; i < history.size(); ++i) {
        sma20 += history[i].getClose();
    }
    sma20 /= 20.0;
    
    // Strategy logic: Golden Cross / Death Cross
    bool hasPosition = engine.hasPosition(symbol);
    
    if (!hasPosition) {
        // Golden Cross: SMA10 > SMA20 → Buy
        if (sma10 > sma20) {
            double entryPrice = candle.getClose();
            double stopLoss = entryPrice * 0.98;  // 2% stop loss
            double takeProfit = entryPrice * 1.04;  // 4% take profit
            
            engine.placeOrder(symbol, MARKET, BUY, LONG, 0.01, 
                            entryPrice, stopLoss, takeProfit);
        }
    } else {
        // Death Cross: SMA10 < SMA20 → Close long position
        Position pos = engine.getPosition(symbol);
        if (pos.side == LONG && sma10 < sma20) {
            // Close position
            engine.placeOrder(symbol, MARKET, SELL, LONG, pos.size, 
                            candle.getClose(), 0, 0);
        }
    }
}

// Strategy sử dụng FVG và Order Blocks
void smartMoneyStrategy(TradingEngine& engine, const std::string& symbol,
                       const Candle& candle, long long timestamp,
                       const std::vector<Candle>& history) {
    
    if (history.size() < 50) return;
    
    // Detect FVG và Order Blocks
    std::vector<FVG> fvgs = Indicators::detectFVG(history);
    std::vector<OrderBlock> obs = Indicators::detectOrderBlocks(history);
    
    bool hasPosition = engine.hasPosition(symbol);
    
    if (!hasPosition) {
        // Tìm Bullish Order Block chưa bị break
        for (const auto& ob : obs) {
            if (ob.isBullish && !ob.isBroken && ob.candleIndex >= static_cast<int>(history.size()) - 10) {
                // Entry tại OB
                double entryPrice = candle.getClose();
                double stopLoss = ob.bottom * 0.995;  // Slightly below OB
                double takeProfit = entryPrice + (entryPrice - stopLoss) * 2.0;  // 2:1 R:R
                
                engine.placeOrder(symbol, MARKET, BUY, LONG, 0.01,
                                entryPrice, stopLoss, takeProfit);
                break;
            }
        }
    } else {
        // Check if FVG is broken
        Position pos = engine.getPosition(symbol);
        if (pos.side == LONG) {
            for (const auto& fvg : fvgs) {
                if (fvg.isBullish && fvg.isBroken && fvg.candleIndex >= static_cast<int>(history.size()) - 5) {
                    // Close position if bullish FVG is broken
                    engine.placeOrder(symbol, MARKET, SELL, LONG, pos.size,
                                    candle.getClose(), 0, 0);
                    break;
                }
            }
        }
    }
}

int main() {
    std::cout << "=== IRIS TRADING BACKTEST EXAMPLE ===" << std::endl;
    std::cout << std::endl;
    
    // Tạo config với fixed seed để có kết quả nhất quán
    SimulatorConfig config;
    config.randomSeed = 12345;  // Fixed seed cho deterministic results
    config.makerFeeRate = 0.0002;
    config.takerFeeRate = 0.0008;
    config.fundingRateBase = 0.0003;
    config.slippageBase = 0.0005;
    config.penalizeBothSidesHit = true;
    
    // Tạo backtest engine
    BacktestEngine engine(config);
    engine.setInitialBalance(10000.0);
    
    // Load data từ CSV
    std::string dataFile = "data/BTCUSDT_15.csv";
    std::cout << "Loading data from: " << dataFile << std::endl;
    engine.loadDataFromCSV(dataFile);
    
    if (engine.getTradingEngine().getAccount().balance == 0) {
        std::cerr << "ERROR: Failed to load data or data file not found!" << std::endl;
        std::cerr << "Please make sure the CSV file exists in the data/ directory." << std::endl;
        return 1;
    }
    
    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << std::endl;
    
    // Set strategy
    engine.setStrategy(simpleMAStrategy);
    // Hoặc dùng: engine.setStrategy(smartMoneyStrategy);
    
    // Chạy train/test split (2/3 train, 1/3 test)
    std::cout << "Running Train/Test Split Backtest..." << std::endl;
    std::cout << std::endl;
    
    BacktestResult testResult = engine.runTrainTest("BTCUSDT_15", 0.67);
    
    // In kết quả chi tiết
    std::cout << "=== FINAL TEST RESULTS (Out-of-Sample) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Initial Balance: $" << testResult.initialBalance << std::endl;
    std::cout << "Final Balance: $" << testResult.finalBalance << std::endl;
    std::cout << "Total Return: $" << testResult.totalReturn << " (" 
              << testResult.totalReturnPercent << "%)" << std::endl;
    std::cout << "Realized PnL: $" << testResult.realizedPnL << std::endl;
    std::cout << "Unrealized PnL: $" << testResult.unrealizedPnL << std::endl;
    std::cout << "Total Fees: $" << testResult.totalFees << std::endl;
    std::cout << "Total Trades: " << testResult.totalTrades << std::endl;
    std::cout << "Winning Trades: " << testResult.winningTrades << std::endl;
    std::cout << "Losing Trades: " << testResult.losingTrades << std::endl;
    std::cout << "Win Rate: " << testResult.winRate << "%" << std::endl;
    std::cout << "Average Win: $" << testResult.averageWin << std::endl;
    std::cout << "Average Loss: $" << testResult.averageLoss << std::endl;
    std::cout << "Profit Factor: " << testResult.profitFactor << std::endl;
    std::cout << "Max Drawdown: $" << testResult.maxDrawdown << " (" 
              << testResult.maxDrawdownPercent << "%)" << std::endl;
    std::cout << "Sharpe Ratio: " << testResult.sharpeRatio << std::endl;
    std::cout << "Both Sides Hit Count: " << testResult.bothSidesHitCount << std::endl;
    
    return 0;
}

