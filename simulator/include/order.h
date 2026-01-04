#ifndef ORDER_H
#define ORDER_H

#include <string>
#include <vector>
#include "../../include/candle.h"

enum OrderType {
    MARKET,
    LIMIT,
    STOP_MARKET,
    STOP_LIMIT
};

enum OrderSide {
    BUY,
    SELL
};

enum OrderStatus {
    PENDING,
    FILLED,
    PARTIALLY_FILLED,
    CANCELLED,
    REJECTED
};

enum PositionSide {
    LONG,
    SHORT,
    BOTH
};

struct Order {
    std::string orderId;
    std::string symbol;
    OrderType type;
    OrderSide side;
    PositionSide positionSide;
    double quantity;
    double price;  // Limit price hoặc trigger price
    double stopLoss;
    double takeProfit;
    OrderStatus status;
    long long timestamp;
    double filledQuantity;
    double avgFillPrice;
    double fee;  // Phí đã trả
    bool isBothSidesHit;  // true nếu cả stop loss và take profit đều bị quét
};

struct Position {
    std::string symbol;
    PositionSide side;
    double size;
    double entryPrice;
    double markPrice;  // Giá mark hiện tại
    double liquidationPrice;
    double unrealizedPnl;
    double realizedPnl;
    double margin;
    double leverage;
    long long openTime;
    long long lastUpdateTime;
    bool isOpen;
    double totalFees;  // Tổng phí đã trả (trading fee + funding)
    double stopLoss;  // Stop loss price
    double takeProfit;  // Take profit price
    std::string entryOrderId;  // Order ID tạo ra position này
};

struct Trade {
    std::string tradeId;
    std::string orderId;
    std::string symbol;
    OrderSide side;
    double quantity;
    double price;
    double fee;
    long long timestamp;
    bool isMaker;  // true nếu là maker order
};

struct Account {
    double balance;  // Số dư tài khoản
    double availableBalance;  // Số dư khả dụng
    double marginUsed;  // Margin đã sử dụng
    double unrealizedPnl;  // PnL chưa thực hiện
    double realizedPnl;  // PnL đã thực hiện
    double totalFees;  // Tổng phí đã trả
    std::vector<Position> positions;
    std::vector<Order> openOrders;
    std::vector<Trade> tradeHistory;
};

#endif

