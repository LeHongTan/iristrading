#include "../include/trading_engine.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <ctime>

TradingEngine::TradingEngine(const SimulatorConfig& config) : simulator(config) {
    account.balance = 10000.0;  // Default balance
    account.availableBalance = 10000.0;
    account.marginUsed = 0.0;
    account.unrealizedPnl = 0.0;
    account.realizedPnl = 0.0;
    account.totalFees = 0.0;
}

void TradingEngine::setInitialBalance(double balance) {
    account.balance = balance;
    account.availableBalance = balance;
    account.marginUsed = 0.0;
    account.unrealizedPnl = 0.0;
    account.realizedPnl = 0.0;
    account.totalFees = 0.0;
    account.positions.clear();
    account.openOrders.clear();
    account.tradeHistory.clear();
}

void TradingEngine::loadCandleData(const std::string& symbol, const std::vector<Candle>& candles) {
    candleData[symbol] = candles;
    currentCandleIndex[symbol] = 0;
    simulator.loadCandles(candles);
}

Account TradingEngine::getAccount() const {
    return account;
}

std::string TradingEngine::placeOrder(const std::string& symbol, OrderType type, OrderSide side,
                                     PositionSide positionSide, double quantity, double price,
                                     double stopLoss, double takeProfit) {
    Order order;
    order.orderId = "ORDER_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(account.openOrders.size());
    order.symbol = symbol;
    order.type = type;
    order.side = side;
    order.positionSide = positionSide;
    order.quantity = quantity;
    order.price = price;
    order.stopLoss = stopLoss;
    order.takeProfit = takeProfit;
    order.status = PENDING;
    
    // Sử dụng current candle timestamp nếu có, hoặc current time
    if (candleData.find(symbol) != candleData.end() && currentCandleIndex[symbol] < static_cast<int>(candleData[symbol].size())) {
        const Candle& currentCandle = candleData[symbol][currentCandleIndex[symbol]];
        order.timestamp = currentCandle.getTimeOpen();  // Dùng timestamp của current candle
    } else {
        order.timestamp = std::time(nullptr) * 1000;
    }
    
    order.filledQuantity = 0.0;
    order.avgFillPrice = 0.0;
    order.fee = 0.0;
    order.isBothSidesHit = false;
    
    // Validation: Chặn Limit order với timestamp cũ (look-ahead bias)
    // Nếu là Limit order và có current candle data, kiểm tra timestamp
    if (type == LIMIT && candleData.find(symbol) != candleData.end() && 
        currentCandleIndex[symbol] < static_cast<int>(candleData[symbol].size())) {
        const Candle& currentCandle = candleData[symbol][currentCandleIndex[symbol]];
        long long currentCandleTime = currentCandle.getTimeOpen();
        
        // Nếu order timestamp < current candle time, có thể là look-ahead bias
        // (User dùng candle.close để đặt Limit order với timestamp cũ)
        if (order.timestamp < currentCandleTime) {
            order.status = REJECTED;
            return order.orderId;
        }
        
        // Nếu Limit price nằm trong range của current candle (đã đóng), reject
        // Vì không thể đặt Limit order trong nến đã đóng
        if (order.timestamp == currentCandleTime) {
            if ((side == BUY && order.price >= currentCandle.getLow() && order.price <= currentCandle.getHigh()) ||
                (side == SELL && order.price >= currentCandle.getLow() && order.price <= currentCandle.getHigh())) {
                // Limit price nằm trong range của nến đã đóng → có thể là look-ahead bias
                // Chỉ reject nếu giá quá gần với close (ăn gian)
                double closePrice = currentCandle.getClose();
                double priceDiff = std::abs(order.price - closePrice) / closePrice;
                if (priceDiff < 0.001) {  // Nếu cách close < 0.1%, có thể là ăn gian
                    order.status = REJECTED;
                    return order.orderId;
                }
            }
        }
    }
    
    // Kiểm tra margin
    if (candleData.find(symbol) != candleData.end() && currentCandleIndex[symbol] < static_cast<int>(candleData[symbol].size())) {
        const Candle& currentCandle = candleData[symbol][currentCandleIndex[symbol]];
        double orderValue = quantity * (price > 0 ? price : currentCandle.getClose());
        
        if (!canOpenPosition(symbol, quantity, orderValue, positionSide)) {
            order.status = REJECTED;
            return order.orderId;
        }
    }
    
    account.openOrders.push_back(order);
    return order.orderId;
}

bool TradingEngine::cancelOrder(const std::string& orderId) {
    auto it = std::find_if(account.openOrders.begin(), account.openOrders.end(),
                          [&orderId](const Order& o) { return o.orderId == orderId; });
    
    if (it != account.openOrders.end() && it->status == PENDING) {
        it->status = CANCELLED;
        account.openOrders.erase(it);
        return true;
    }
    
    return false;
}

std::vector<Order> TradingEngine::getOpenOrders() const {
    std::vector<Order> open;
    for (const auto& order : account.openOrders) {
        if (order.status == PENDING || order.status == PARTIALLY_FILLED) {
            open.push_back(order);
        }
    }
    return open;
}

std::vector<Order> TradingEngine::getOrderHistory() const {
    return account.openOrders;  // In real implementation, separate history
}

std::vector<Position> TradingEngine::getPositions() const {
    return account.positions;
}

Position TradingEngine::getPosition(const std::string& symbol) const {
    for (const auto& pos : account.positions) {
        if (pos.symbol == symbol && pos.isOpen) {
            return pos;
        }
    }
    
    // Return empty position
    Position empty;
    empty.symbol = symbol;
    empty.size = 0.0;
    empty.isOpen = false;
    return empty;
}

bool TradingEngine::hasPosition(const std::string& symbol) const {
    for (const auto& pos : account.positions) {
        if (pos.symbol == symbol && pos.isOpen) {
            return true;
        }
    }
    return false;
}

bool TradingEngine::canOpenPosition(const std::string& symbol, double size, double price, PositionSide side) {
    double leverage = simulator.getConfig().defaultLeverage;
    double marginRequired = (size * price) / leverage;
    
    // Kiểm tra margin available
    double availableMargin = account.availableBalance - account.marginUsed;
    
    return availableMargin >= marginRequired;
}

void TradingEngine::closePosition(const std::string& symbol, double quantity, double price, long long timestamp) {
    for (auto& pos : account.positions) {
        if (pos.symbol == symbol && pos.isOpen) {
            double closeQuantity = std::min(quantity, pos.size);
            double pnl = 0.0;
            
            if (pos.side == LONG) {
                pnl = (price - pos.entryPrice) * closeQuantity;
            } else {
                pnl = (pos.entryPrice - price) * closeQuantity;
            }
            
            // Tính phí đóng
            double fee = simulator.calculateTradingFee(closeQuantity, price, false);
            pnl -= fee;
            
            account.realizedPnl += pnl;
            account.totalFees += fee;
            account.balance += pnl;
            account.availableBalance += pnl;
            
            pos.size -= closeQuantity;
            if (pos.size <= 0.0001) {  // Close position
                pos.isOpen = false;
                account.marginUsed -= pos.margin;
            } else {
                // Partial close
                pos.margin = (pos.size * pos.entryPrice) / pos.leverage;
            }
            
            break;
        }
    }
}

void TradingEngine::updatePositionPnL(const std::string& symbol, const Candle& candle, long long timestamp) {
    for (auto& pos : account.positions) {
        if (pos.symbol == symbol && pos.isOpen) {
            pos.markPrice = simulator.getMarkPrice(candle);
            pos.lastUpdateTime = timestamp;
            
            // Tính unrealized PnL
            if (pos.side == LONG) {
                pos.unrealizedPnl = (pos.markPrice - pos.entryPrice) * pos.size;
            } else {
                pos.unrealizedPnl = (pos.entryPrice - pos.markPrice) * pos.size;
            }
            
            // Cập nhật account unrealized PnL
            account.unrealizedPnl = 0.0;
            for (const auto& p : account.positions) {
                if (p.isOpen) {
                    account.unrealizedPnl += p.unrealizedPnl;
                }
            }
        }
    }
}

void TradingEngine::applyFunding(const std::string& symbol, const Candle& candle, long long timestamp) {
    for (auto& pos : account.positions) {
        if (pos.symbol == symbol && pos.isOpen) {
            double fundingFee = simulator.calculateFundingFee(pos, candle, timestamp);
            pos.unrealizedPnl += fundingFee;
            account.totalFees += std::abs(fundingFee);
            account.balance += fundingFee;
            account.availableBalance += fundingFee;
        }
    }
}

void TradingEngine::checkLiquidation(const std::string& symbol, const Candle& candle) {
    // Cross Margin: Kiểm tra tổng account, không phải từng position riêng
    // Tính tổng margin và PnL của tất cả positions
    double totalMargin = 0.0;
    double totalUnrealizedPnl = 0.0;
    
    for (const auto& pos : account.positions) {
        if (pos.isOpen) {
            totalMargin += pos.margin;
            totalUnrealizedPnl += pos.unrealizedPnl;
        }
    }
    
    // Tính maintenance margin requirement
    double totalMaintenanceMargin = 0.0;
    for (const auto& pos : account.positions) {
        if (pos.isOpen) {
            totalMaintenanceMargin += pos.size * pos.entryPrice * simulator.getConfig().maintenanceMarginRate;
        }
    }
    
    // Cross Margin: Account balance + unrealized PnL phải >= maintenance margin
    double availableForMargin = account.balance + totalUnrealizedPnl;
    
    if (availableForMargin < totalMaintenanceMargin) {
        // Liquidation: Đóng tất cả positions (Cross Margin)
        // Lưu danh sách positions cần đóng trước để tránh iterator invalidation
        std::vector<std::pair<std::string, double>> positionsToClose;
        for (const auto& pos : account.positions) {
            if (pos.isOpen) {
                positionsToClose.push_back({pos.symbol, pos.size});
            }
        }
        
        // Đóng tất cả positions
        for (const auto& [posSymbol, posSize] : positionsToClose) {
            // Tìm position để lấy markPrice
            for (auto& pos : account.positions) {
                if (pos.symbol == posSymbol && pos.isOpen) {
                    closePosition(posSymbol, posSize, pos.markPrice, candle.getTimeClose());
                    break;
                }
            }
        }
    }
}

void TradingEngine::executePendingOrders(const std::string& symbol, const Candle& candle, long long timestamp) {
    std::vector<Order> ordersToProcess = account.openOrders;
    
    for (auto& order : ordersToProcess) {
        if (order.symbol != symbol || order.status != PENDING) continue;
        
        // Kiểm tra cả 2 đầu có bị quét không
        bool bothSidesHit = simulator.checkBothSidesHit(order, candle);
        if (bothSidesHit && simulator.getConfig().penalizeBothSidesHit) {
            order.isBothSidesHit = true;
            order.status = REJECTED;
            continue;
        }
        
        // Kiểm tra có thể fill không
        if (simulator.canFillOrder(order, candle)) {
            Trade trade = simulator.executeOrder(order, candle, timestamp);
            
            // Nếu trade không fill được (quantity = 0), bỏ qua
            if (trade.quantity == 0.0) {
                continue;
            }
            
            // Update order
            order.filledQuantity = trade.quantity;
            order.avgFillPrice = trade.price;
            order.fee = trade.fee;
            order.status = FILLED;
            
            // Update account
            account.totalFees += trade.fee;
            account.tradeHistory.push_back(trade);
            
            // Open or update position - Dùng index thay vì pointer để tránh memory safety risk
            int positionIndex = -1;
            for (size_t i = 0; i < account.positions.size(); ++i) {
                if (account.positions[i].symbol == symbol && 
                    account.positions[i].isOpen && 
                    account.positions[i].side == (order.side == BUY ? LONG : SHORT)) {
                    positionIndex = i;
                    break;
                }
            }
            
            if (positionIndex == -1) {
                // Open new position
                Position newPos;
                newPos.symbol = symbol;
                newPos.side = (order.side == BUY) ? LONG : SHORT;
                newPos.size = trade.quantity;
                newPos.entryPrice = trade.price;
                newPos.markPrice = trade.price;
                newPos.leverage = simulator.getConfig().defaultLeverage;
                newPos.margin = (trade.quantity * trade.price) / newPos.leverage;
                newPos.openTime = timestamp;
                newPos.lastUpdateTime = timestamp;
                newPos.isOpen = true;
                newPos.unrealizedPnl = 0.0;
                newPos.realizedPnl = 0.0;
                newPos.totalFees = trade.fee;
                newPos.stopLoss = order.stopLoss;
                newPos.takeProfit = order.takeProfit;
                newPos.entryOrderId = order.orderId;
                newPos.liquidationPrice = 0.0;  // Initialize
                
                account.positions.push_back(newPos);
                account.marginUsed += newPos.margin;
                account.availableBalance -= newPos.margin;
            } else {
                // Update existing position - Dùng index để tránh pointer invalidation
                double totalValue = account.positions[positionIndex].size * account.positions[positionIndex].entryPrice + trade.quantity * trade.price;
                double totalSize = account.positions[positionIndex].size + trade.quantity;
                account.positions[positionIndex].entryPrice = totalValue / totalSize;
                account.positions[positionIndex].size = totalSize;
                account.positions[positionIndex].margin = (totalSize * account.positions[positionIndex].entryPrice) / account.positions[positionIndex].leverage;
                account.positions[positionIndex].totalFees += trade.fee;
                // Update SL/TP nếu order mới có
                if (order.stopLoss > 0) account.positions[positionIndex].stopLoss = order.stopLoss;
                if (order.takeProfit > 0) account.positions[positionIndex].takeProfit = order.takeProfit;
            }
        }
    }
    
    // Remove filled orders
    account.openOrders.erase(
        std::remove_if(account.openOrders.begin(), account.openOrders.end(),
                      [](const Order& o) { return o.status == FILLED || o.status == REJECTED; }),
        account.openOrders.end()
    );
}

void TradingEngine::checkStopLossTakeProfit(const std::string& symbol, const Candle& candle, long long timestamp) {
    for (auto& pos : account.positions) {
        if (pos.symbol != symbol || !pos.isOpen) continue;
        
        // QUAN TRỌNG: Không check SL/TP trong cùng nến khi position vừa được tạo
        // Nếu position được mở trong nến này (openTime == timestamp), bỏ qua check
        // Vì nếu Buy Limit khớp tại Low, không thể có SL thấp hơn trong cùng nến
        if (pos.openTime == timestamp) {
            continue;  // Skip check trong nến đầu tiên
        }
        
        // Kiểm tra cả 2 đầu có bị quét không
        bool stopLossHit = false;
        bool takeProfitHit = false;
        double closePrice = 0.0;
        
        if (pos.stopLoss > 0) {
            if (pos.side == LONG && candle.getLow() <= pos.stopLoss) {
                stopLossHit = true;
                closePrice = pos.stopLoss;  // Fill tại SL price
            } else if (pos.side == SHORT && candle.getHigh() >= pos.stopLoss) {
                stopLossHit = true;
                closePrice = pos.stopLoss;
            }
        }
        
        if (pos.takeProfit > 0) {
            if (pos.side == LONG && candle.getHigh() >= pos.takeProfit) {
                takeProfitHit = true;
                if (!stopLossHit) closePrice = pos.takeProfit;  // Fill tại TP price
            } else if (pos.side == SHORT && candle.getLow() <= pos.takeProfit) {
                takeProfitHit = true;
                if (!stopLossHit) closePrice = pos.takeProfit;
            }
        }
        
        // Nếu cả 2 đầu đều bị quét → tính là thua
        if (stopLossHit && takeProfitHit && simulator.getConfig().penalizeBothSidesHit) {
            // Đóng tại giá xấu hơn (SL cho long, TP cho short)
            if (pos.side == LONG) {
                closePrice = pos.stopLoss;  // Đóng tại SL (giá xấu hơn)
            } else {
                closePrice = pos.takeProfit;  // Đóng tại TP (giá xấu hơn cho short)
            }
            closePosition(symbol, pos.size, closePrice, timestamp);
        } else if (stopLossHit || takeProfitHit) {
            // Chỉ một trong hai bị quét
            closePosition(symbol, pos.size, closePrice, timestamp);
        }
    }
}

void TradingEngine::processNextCandle(const std::string& symbol, const Candle& candle, long long timestamp) {
    // Execute pending orders
    executePendingOrders(symbol, candle, timestamp);
    
    // Update positions PnL
    updatePositionPnL(symbol, candle, timestamp);
    
    // Check stop loss and take profit (QUAN TRỌNG: Kiểm tra mỗi nến, không chỉ khi entry)
    checkStopLossTakeProfit(symbol, candle, timestamp);
    
    // Apply funding
    applyFunding(symbol, candle, timestamp);
    
    // Check liquidation
    checkLiquidation(symbol, candle);
    
    // Update available balance
    account.availableBalance = account.balance - account.marginUsed + account.unrealizedPnl;
}

void TradingEngine::processNextCandle(const std::string& symbol) {
    if (candleData.find(symbol) == candleData.end()) return;
    if (currentCandleIndex[symbol] >= static_cast<int>(candleData[symbol].size())) return;
    
    const Candle& candle = candleData[symbol][currentCandleIndex[symbol]];
    processNextCandle(symbol, candle, candle.getTimeOpen());
    
    currentCandleIndex[symbol]++;
}

double TradingEngine::getTotalPnL() const {
    return account.realizedPnl + account.unrealizedPnl;
}

double TradingEngine::getRealizedPnL() const {
    return account.realizedPnl;
}

double TradingEngine::getUnrealizedPnL() const {
    return account.unrealizedPnl;
}

double TradingEngine::getTotalFees() const {
    return account.totalFees;
}

int TradingEngine::getTotalTrades() const {
    return account.tradeHistory.size();
}

double TradingEngine::getWinRate() const {
    if (account.tradeHistory.empty()) return 0.0;
    
    int wins = 0;
    // Simplified: count profitable trades
    // In real implementation, need to track entry/exit pairs
    return static_cast<double>(wins) / account.tradeHistory.size();
}

void TradingEngine::reset() {
    account.balance = 10000.0;
    account.availableBalance = 10000.0;
    account.marginUsed = 0.0;
    account.unrealizedPnl = 0.0;
    account.realizedPnl = 0.0;
    account.totalFees = 0.0;
    account.positions.clear();
    account.openOrders.clear();
    account.tradeHistory.clear();
    
    for (auto& pair : currentCandleIndex) {
        pair.second = 0;
    }
}

