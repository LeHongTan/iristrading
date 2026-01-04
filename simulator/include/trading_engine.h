#ifndef TRADING_ENGINE_H
#define TRADING_ENGINE_H

#include <vector>
#include <string>
#include <map>
#include "../../include/candle.h"
#include "order.h"
#include "bybit_simulator.h"

class TradingEngine {
    private:
        Account account;
        BybitSimulator simulator;
        std::map<std::string, std::vector<Candle>> candleData;  // symbol -> candles
        std::map<std::string, int> currentCandleIndex;  // symbol -> current index
        
        // Helper functions
        void updatePositionPnL(const std::string& symbol, const Candle& candle, long long timestamp);
        void applyFunding(const std::string& symbol, const Candle& candle, long long timestamp);
        void checkLiquidation(const std::string& symbol, const Candle& candle);
        bool canOpenPosition(const std::string& symbol, double size, double price, PositionSide side);
        void closePosition(const std::string& symbol, double quantity, double price, long long timestamp);
        
    public:
        TradingEngine(const SimulatorConfig& config = SimulatorConfig());
        
        // Load dữ liệu
        void loadCandleData(const std::string& symbol, const std::vector<Candle>& candles);
        
        // Account management
        void setInitialBalance(double balance);
        Account getAccount() const;
        
        // Order management
        std::string placeOrder(const std::string& symbol, OrderType type, OrderSide side, 
                              PositionSide positionSide, double quantity, double price = 0.0,
                              double stopLoss = 0.0, double takeProfit = 0.0);
        bool cancelOrder(const std::string& orderId);
        std::vector<Order> getOpenOrders() const;
        std::vector<Order> getOrderHistory() const;
        
        // Position management
        std::vector<Position> getPositions() const;
        Position getPosition(const std::string& symbol) const;
        bool hasPosition(const std::string& symbol) const;
        
        // Process next candle
        void processNextCandle(const std::string& symbol, const Candle& candle, long long timestamp);
        void processNextCandle(const std::string& symbol);  // Auto increment index
        
        // Check stop loss and take profit for all open positions
        void checkStopLossTakeProfit(const std::string& symbol, const Candle& candle, long long timestamp);
        
        // Trade execution
        void executePendingOrders(const std::string& symbol, const Candle& candle, long long timestamp);
        
        // Statistics
        double getTotalPnL() const;
        double getRealizedPnL() const;
        double getUnrealizedPnL() const;
        double getTotalFees() const;
        int getTotalTrades() const;
        double getWinRate() const;
        
        // Reset
        void reset();
};

#endif

