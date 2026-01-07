#include "../include/bybit_simulator.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>

BybitSimulator::BybitSimulator(const SimulatorConfig& cfg) : config(cfg) {
    // Initialize random seed
    // Use fixed seed if provided, otherwise use time for randomness
    if (config.randomSeed > 0) {
        std::srand(config.randomSeed);  // Fixed seed for deterministic results
    } else {
        std::srand(std::time(nullptr));  // Time-based seed (non-deterministic)
    }
}

void BybitSimulator::loadCandles(const std::vector<Candle>& candleData) {
    candles = candleData;
    // Initialize volume MA for each unique symbol (if we have symbol info)
    // For now, we'll calculate based on all candles
    if (!candles.empty()) {
        double sumVolume = 0.0;
        int count = std::min(20, static_cast<int>(candles.size()));
        for (int i = candles.size() - count; i < static_cast<int>(candles.size()); ++i) {
            sumVolume += candles[i].getVolume();
        }
        volumeMA["DEFAULT"] = sumVolume / count;
    }
}

void BybitSimulator::addCandle(const Candle& candle) {
    candles.push_back(candle);
    updateVolumeMA("DEFAULT", candle);
}

void BybitSimulator::updateVolumeMA(const std::string& symbol, const Candle& candle) {
    if (volumeMA.find(symbol) == volumeMA.end()) {
        volumeMA[symbol] = candle.getVolume();
    } else {
        // Exponential moving average
        volumeMA[symbol] = volumeMA[symbol] * 0.9 + candle.getVolume() * 0.1;
    }
}

bool BybitSimulator::isLowVolume(const std::string& symbol, const Candle& candle) {
    double avgVolume = volumeMA.find(symbol) != volumeMA.end() ? volumeMA[symbol] : candle.getVolume();
    return candle.getVolume() < avgVolume * config.lowVolumeThreshold;
}

double BybitSimulator::calculateSlippage(const Candle& candle, double orderSize, bool isBuy) {
    double baseSlippage = config.slippageBase;
    
    // Thêm biến động ngẫu nhiên
    double volatility = config.slippageVolatility * (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0;
    
    // Nếu volume thấp, tăng slippage
    if (isLowVolume("DEFAULT", candle)) {
        baseSlippage *= config.lowVolumeSlippageMultiplier;
    }
    
    // Market impact - order lớn hơn sẽ có slippage cao hơn
    double volumeRatio = orderSize / std::max(candle.getVolume(), config.minVolumeForExecution);
    double marketImpact = std::min(volumeRatio * config.marketImpactFactor, 0.01);  // Max 1%
    
    return baseSlippage + volatility + marketImpact;
}

double BybitSimulator::calculateFundingRate(const std::string& symbol, long long timestamp) {
    // Kiểm tra xem đã đến thời gian funding chưa (mỗi 8 giờ)
    if (lastFundingTime.find(symbol) != lastFundingTime.end()) {
        long long timeSinceLastFunding = timestamp - lastFundingTime[symbol];
        long long fundingIntervalMs = config.fundingIntervalHours * 3600 * 1000;
        
        if (timeSinceLastFunding < fundingIntervalMs) {
            return lastFundingRate[symbol];
        }
    }
    
    // Tính funding rate mới (cao hơn thực tế)
    double baseRate = config.fundingRateBase;
    double volatility = config.fundingRateVolatility * (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0;
    double fundingRate = baseRate + volatility;
    
    // Cập nhật
    lastFundingRate[symbol] = fundingRate;
    lastFundingTime[symbol] = timestamp;
    
    return fundingRate;
}

double BybitSimulator::calculateSpread(const Candle& candle) {
    double baseSpread = config.spreadBase;
    double volatility = config.spreadVolatility * (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0;
    return baseSpread + volatility;
}

double BybitSimulator::calculateMarketImpact(double orderSize, double volume) {
    if (volume == 0) return 0.0;
    double volumeRatio = orderSize / volume;
    return volumeRatio * config.marketImpactFactor;
}

double BybitSimulator::getBidPrice(const Candle& candle) {
    double spread = calculateSpread(candle);
    return candle.getClose() * (1.0 - spread / 2.0);
}

double BybitSimulator::getAskPrice(const Candle& candle) {
    double spread = calculateSpread(candle);
    return candle.getClose() * (1.0 + spread / 2.0);
}

double BybitSimulator::getMarkPrice(const Candle& candle) {
    // Mark price thường là giá index hoặc giá trung bình
    return candle.getClose();
}

double BybitSimulator::calculateTradingFee(double quantity, double price, bool isMaker) {
    double feeRate = isMaker ? config.makerFeeRate : config.takerFeeRate;
    return quantity * price * feeRate;
}

double BybitSimulator::calculateFundingFee(const Position& position, const Candle& currentCandle, long long timestamp) {
    if (!position.isOpen || position.size == 0) return 0.0;
    
    double fundingRate = calculateFundingRate(position.symbol, timestamp);
    double positionValue = position.size * position.markPrice;
    
    // Funding fee = position value * funding rate
    // Long position trả funding nếu rate dương, nhận nếu rate âm
    // Short position ngược lại
    double fundingFee = positionValue * fundingRate;
    
    if (position.side == LONG) {
        return -fundingFee;  // Long trả phí nếu rate dương
    } else {
        return fundingFee;  // Short trả phí nếu rate âm
    }
}

bool BybitSimulator::canFillOrder(const Order& order, const Candle& candle) {
    if (order.type == MARKET) {
        return true;  // Market order luôn có thể fill
    }
    
    if (order.type == LIMIT) {
        if (order.side == BUY) {
            // Buy limit: chỉ fill khi giá <= limit price
            return candle.getLow() <= order.price;
        } else {
            // Sell limit: chỉ fill khi giá >= limit price
            return candle.getHigh() >= order.price;
        }
    }
    
    if (order.type == STOP_MARKET || order.type == STOP_LIMIT) {
        if (order.side == BUY) {
            // Buy stop: trigger khi giá >= stop price
            return candle.getHigh() >= order.price;
        } else {
            // Sell stop: trigger khi giá <= stop price
            return candle.getLow() <= order.price;
        }
    }
    
    return false;
}

bool BybitSimulator::checkBothSidesHit(const Order& order, const Candle& candle) {
    // Kiểm tra xem cả stop loss và take profit có bị quét trong cùng một candle không
    if (order.stopLoss > 0 && order.takeProfit > 0) {
        bool stopLossHit = false;
        bool takeProfitHit = false;
        
        if (order.side == BUY) {
            // Long position
            stopLossHit = (candle.getLow() <= order.stopLoss);
            takeProfitHit = (candle.getHigh() >= order.takeProfit);
        } else {
            // Short position
            stopLossHit = (candle.getHigh() >= order.stopLoss);
            takeProfitHit = (candle.getLow() <= order.takeProfit);
        }
        
        return stopLossHit && takeProfitHit;
    }
    
    return false;
}

Trade BybitSimulator::executeOrder(const Order& order, const Candle& currentCandle, long long timestamp) {
    Trade trade;
    trade.tradeId = "TRADE_" + std::to_string(timestamp) + "_" + std::to_string(std::rand());
    trade.orderId = order.orderId;
    trade.symbol = order.symbol;
    trade.side = order.side;
    trade.quantity = order.quantity;
    trade.timestamp = timestamp;
    
    // Kiểm tra cả 2 đầu có bị quét không
    bool bothSidesHit = checkBothSidesHit(order, currentCandle);
    
    // Xác định giá fill
    double fillPrice = 0.0;
    bool isMaker = false;
    
    if (order.type == MARKET) {
        // Market order: fill tại giá thị trường + slippage
        double slippage = calculateSlippage(currentCandle, order.quantity, order.side == BUY);
        if (order.side == BUY) {
            fillPrice = getAskPrice(currentCandle) * (1.0 + slippage);
            isMaker = false;  // Market order là taker
        } else {
            fillPrice = getBidPrice(currentCandle) * (1.0 - slippage);
            isMaker = false;
        }
    } else if (order.type == LIMIT) {
        // Limit order: Xử lý gap giá
        // Nếu có gap (giá mở cửa tốt hơn limit price), fill tại giá tốt hơn
        if (order.side == BUY) {
            // Buy limit: Nếu giá mở cửa thấp hơn limit price → fill tại open (giá tốt hơn)
            if (currentCandle.getOpen() < order.price) {
                fillPrice = currentCandle.getOpen();  // Gap down → fill tại open
                isMaker = false;  // Taker vì fill ngay tại open
            } else if (currentCandle.getLow() <= order.price && currentCandle.getHigh() >= order.price) {
                // Giá chạm limit trong nến → fill tại limit price (có thể fill)
                fillPrice = order.price;
                isMaker = true;
            } else {
                // Không fill được - return empty trade
                trade.quantity = 0.0;
                return trade;
            }
        } else {
            // Sell limit: Nếu giá mở cửa cao hơn limit price → fill tại open (giá tốt hơn)
            if (currentCandle.getOpen() > order.price) {
                fillPrice = currentCandle.getOpen();  // Gap up → fill tại open
                isMaker = false;  // Taker vì fill ngay tại open
            } else if (currentCandle.getHigh() >= order.price && currentCandle.getLow() <= order.price) {
                // Giá chạm limit trong nến → fill tại limit price (có thể fill)
                fillPrice = order.price;
                isMaker = true;
            } else {
                // Không fill được - return empty trade
                trade.quantity = 0.0;
                return trade;
            }
        }
    } else if (order.type == STOP_MARKET) {
        // Stop market: trigger tại stop price, fill tại market
        double slippage = calculateSlippage(currentCandle, order.quantity, order.side == BUY);
        if (order.side == BUY) {
            fillPrice = std::max(order.price, getAskPrice(currentCandle)) * (1.0 + slippage);
        } else {
            fillPrice = std::min(order.price, getBidPrice(currentCandle)) * (1.0 - slippage);
        }
        isMaker = false;
    } else if (order.type == STOP_LIMIT) {
        // Stop limit: trigger tại stop price, fill tại limit price
        fillPrice = order.price;
        isMaker = true;
    }
    
    // Tính phí
    trade.fee = calculateTradingFee(order.quantity, fillPrice, isMaker);
    trade.isMaker = isMaker;
    trade.price = fillPrice;
    
    // Nếu quét 2 đầu, thêm phí phạt
    if (bothSidesHit && config.penalizeBothSidesHit) {
        trade.fee += order.quantity * fillPrice * config.bothSidesHitPenalty;
    }
    
    return trade;
}

void BybitSimulator::updateFundingRate(const std::string& symbol, long long timestamp) {
    calculateFundingRate(symbol, timestamp);  // Update funding rate
}

SimulatorConfig BybitSimulator::getConfig() const {
    return config;
}

void BybitSimulator::setConfig(const SimulatorConfig& cfg) {
    config = cfg;
}

