#ifndef BYBIT_SIMULATOR_H
#define BYBIT_SIMULATOR_H

#include <vector>
#include <string>
#include <map>
#include "../../include/candle.h"
#include "order.h"

// Cấu hình môi trường giả lập (khắc nghiệt hơn thực tế)
struct SimulatorConfig {
    // Trading Fees (Bybit thực tế: Maker 0.01%, Taker 0.06%)
    double makerFeeRate = 0.0002;  // 0.02% - cao hơn thực tế
    double takerFeeRate = 0.0008;  // 0.08% - cao hơn thực tế
    
    // Funding Rate (Bybit thực tế: -0.01% đến 0.01% mỗi 8h)
    double fundingRateBase = 0.0003;  // 0.03% - cao hơn thực tế
    double fundingRateVolatility = 0.0002;  // Biến động funding rate
    int fundingIntervalHours = 8;  // Funding mỗi 8 giờ
    
    // Slippage (Trượt giá)
    double slippageBase = 0.0005;  // 0.05% - cao hơn thực tế
    double slippageVolatility = 0.0003;  // Biến động slippage
    double lowVolumeSlippageMultiplier = 3.0;  // Nhân thêm khi volume thấp
    
    // Volume threshold để xác định volume thấp
    double lowVolumeThreshold = 0.5;  // 50% của volume trung bình
    
    // Minimum volume để execute order
    double minVolumeForExecution = 0.1;
    
    // Spread (chênh lệch giá bid/ask)
    double spreadBase = 0.0001;  // 0.01%
    double spreadVolatility = 0.00005;
    
    // Leverage
    int maxLeverage = 100;
    int defaultLeverage = 10;
    
    // Margin
    double maintenanceMarginRate = 0.005;  // 0.5%
    double initialMarginRate = 0.01;  // 1%
    
    // Both sides hit penalty (nếu stop loss và take profit đều bị quét)
    bool penalizeBothSidesHit = true;
    double bothSidesHitPenalty = 0.001;  // Phạt thêm 0.1% nếu quét 2 đầu
    
    // Market impact (ảnh hưởng của order lớn đến giá)
    double marketImpactFactor = 0.0001;  // 0.01% per 1% of volume
    
    // Random seed (0 = use time, >0 = fixed seed for deterministic results)
    unsigned int randomSeed = 0;  // Default: non-deterministic
};

class BybitSimulator {
    private:
        SimulatorConfig config;
        std::vector<Candle> candles;
        std::map<std::string, double> volumeMA;  // Volume moving average cho mỗi symbol
        std::map<std::string, double> lastFundingRate;  // Funding rate cuối cùng
        std::map<std::string, long long> lastFundingTime;  // Thời gian funding cuối cùng
        
        // Helper functions
        double calculateSlippage(const Candle& candle, double orderSize, bool isBuy);
        double calculateFundingRate(const std::string& symbol, long long timestamp);
        double calculateSpread(const Candle& candle);
        double calculateMarketImpact(double orderSize, double volume);
        bool isLowVolume(const std::string& symbol, const Candle& candle);
        void updateVolumeMA(const std::string& symbol, const Candle& candle);
        
    public:
        BybitSimulator(const SimulatorConfig& cfg = SimulatorConfig());
        
        // Load dữ liệu candles
        void loadCandles(const std::vector<Candle>& candleData);
        void addCandle(const Candle& candle);
        
        // Execute order
        Trade executeOrder(const Order& order, const Candle& currentCandle, long long timestamp);
        
        // Calculate fees
        double calculateTradingFee(double quantity, double price, bool isMaker);
        double calculateFundingFee(const Position& position, const Candle& currentCandle, long long timestamp);
        
        // Get current market data
        double getBidPrice(const Candle& candle);
        double getAskPrice(const Candle& candle);
        double getMarkPrice(const Candle& candle);
        
        // Check if order can be filled
        bool canFillOrder(const Order& order, const Candle& candle);
        
        // Check if both sides hit (stop loss và take profit cùng lúc)
        bool checkBothSidesHit(const Order& order, const Candle& candle);
        
        // Update funding rate
        void updateFundingRate(const std::string& symbol, long long timestamp);
        
        // Get config
        SimulatorConfig getConfig() const;
        void setConfig(const SimulatorConfig& cfg);
};

#endif

