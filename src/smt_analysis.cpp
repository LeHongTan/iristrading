#include "smt_analysis.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// Helper: Calculate RSI
std::vector<double> SMTAnalysis::calculateRSI(const std::vector<Candle>& candles, int period) {
    std::vector<double> rsi;
    if (candles.size() < period + 1) return rsi;
    
    rsi.resize(candles.size(), 50.0); // Default RSI = 50
    
    for (size_t i = period; i < candles.size(); ++i) {
        double gains = 0.0;
        double losses = 0.0;
        
        for (int j = i - period + 1; j <= static_cast<int>(i); ++j) {
            double change = candles[j].getClose() - candles[j - 1].getClose();
            if (change > 0) {
                gains += change;
            } else {
                losses += std::abs(change);
            }
        }
        
        double avgGain = gains / period;
        double avgLoss = losses / period;
        
        if (avgLoss == 0.0) {
            rsi[i] = 100.0;
        } else {
            double rs = avgGain / avgLoss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    return rsi;
}

// Helper: Calculate Momentum
std::vector<double> SMTAnalysis::calculateMomentum(const std::vector<Candle>& candles, int period) {
    std::vector<double> momentum;
    if (candles.size() < period) return momentum;
    
    momentum.resize(candles.size(), 0.0);
    
    for (size_t i = period; i < candles.size(); ++i) {
        momentum[i] = candles[i].getClose() - candles[i - period].getClose();
    }
    
    return momentum;
}

// Helper: Calculate Volume Moving Average
std::vector<double> SMTAnalysis::calculateVolumeMA(const std::vector<Candle>& candles, int period) {
    std::vector<double> volumeMA;
    if (candles.size() < period) return volumeMA;
    
    volumeMA.resize(candles.size(), 0.0);
    
    for (size_t i = period - 1; i < candles.size(); ++i) {
        double sum = 0.0;
        for (int j = i - period + 1; j <= static_cast<int>(i); ++j) {
            sum += candles[j].getVolume();
        }
        volumeMA[i] = sum / period;
    }
    
    return volumeMA;
}

// Helper: Get Price Values (Close prices)
std::vector<double> SMTAnalysis::calculatePriceValues(const std::vector<Candle>& candles) {
    std::vector<double> prices;
    prices.reserve(candles.size());
    
    for (const auto& candle : candles) {
        prices.push_back(candle.getClose());
    }
    
    return prices;
}

// Find Swing Highs
std::vector<MomentumPoint> SMTAnalysis::findSwingHighs(const std::vector<double>& values, const std::vector<Candle>& candles, int lookback) {
    std::vector<MomentumPoint> swingHighs;
    if (values.size() != candles.size() || values.size() < lookback * 2 + 1) return swingHighs;
    
    for (size_t i = lookback; i < values.size() - lookback; ++i) {
        bool isSwingHigh = true;
        
        // Kiểm tra xem giá trị tại i có cao hơn các giá trị xung quanh không
        for (int j = i - lookback; j <= static_cast<int>(i) + lookback; ++j) {
            if (j != static_cast<int>(i) && values[j] >= values[i]) {
                isSwingHigh = false;
                break;
            }
        }
        
        if (isSwingHigh) {
            MomentumPoint point;
            point.time = candles[i].getTimeOpen();
            point.value = values[i];
            point.candleIndex = i;
            swingHighs.push_back(point);
        }
    }
    
    return swingHighs;
}

// Find Swing Lows
std::vector<MomentumPoint> SMTAnalysis::findSwingLows(const std::vector<double>& values, const std::vector<Candle>& candles, int lookback) {
    std::vector<MomentumPoint> swingLows;
    if (values.size() != candles.size() || values.size() < lookback * 2 + 1) return swingLows;
    
    for (size_t i = lookback; i < values.size() - lookback; ++i) {
        bool isSwingLow = true;
        
        // Kiểm tra xem giá trị tại i có thấp hơn các giá trị xung quanh không
        for (int j = i - lookback; j <= static_cast<int>(i) + lookback; ++j) {
            if (j != static_cast<int>(i) && values[j] <= values[i]) {
                isSwingLow = false;
                break;
            }
        }
        
        if (isSwingLow) {
            MomentumPoint point;
            point.time = candles[i].getTimeOpen();
            point.value = values[i];
            point.candleIndex = i;
            swingLows.push_back(point);
        }
    }
    
    return swingLows;
}

// Calculate Divergence Strength
double SMTAnalysis::calculateDivergenceStrength(const Divergence& divergence) {
    double priceChange = std::abs(divergence.priceEnd - divergence.priceStart) / divergence.priceStart;
    double indicatorChange = std::abs(divergence.indicatorEnd - divergence.indicatorStart);
    
    if (divergence.indicatorStart == 0) return 0.0;
    
    double indicatorChangePercent = indicatorChange / std::abs(divergence.indicatorStart);
    
    // Độ mạnh = độ khác biệt giữa price và indicator movement
    double strength = std::abs(priceChange - indicatorChangePercent);
    
    // Normalize về 0-1
    return std::min(1.0, strength * 10.0);
}

// Confirm Divergence
bool SMTAnalysis::confirmDivergence(const Divergence& divergence, const std::vector<Candle>& candles, int confirmationBars) {
    if (divergence.candleIndexEnd + confirmationBars >= static_cast<int>(candles.size())) {
        return false;
    }
    
    // Kiểm tra xem sau divergence có reversal không
    if (divergence.type == BULLISH_DIVERGENCE || divergence.type == HIDDEN_BULLISH_DIVERGENCE) {
        // Bullish divergence: Giá nên tăng sau đó
        for (int i = divergence.candleIndexEnd + 1; i <= divergence.candleIndexEnd + confirmationBars && i < static_cast<int>(candles.size()); ++i) {
            if (candles[i].getClose() > divergence.priceEnd) {
                return true;
            }
        }
    } else {
        // Bearish divergence: Giá nên giảm sau đó
        for (int i = divergence.candleIndexEnd + 1; i <= divergence.candleIndexEnd + confirmationBars && i < static_cast<int>(candles.size()); ++i) {
            if (candles[i].getClose() < divergence.priceEnd) {
                return true;
            }
        }
    }
    
    return false;
}

// ========== SMT DIVERGENCE IMPLEMENTATION ==========

// Sync candles của 2 tài sản theo timestamp
std::pair<std::vector<Candle>, std::vector<Candle>> SMTAnalysis::syncCandlesByTime(
    const std::vector<Candle>& asset1Candles,
    const std::vector<Candle>& asset2Candles,
    double timeTolerance) {
    
    std::vector<Candle> syncedAsset1;
    std::vector<Candle> syncedAsset2;
    
    for (const auto& candle1 : asset1Candles) {
        long long time1 = candle1.getTimeOpen();
        
        // Tìm candle gần nhất trong asset2
        for (const auto& candle2 : asset2Candles) {
            long long time2 = candle2.getTimeOpen();
            long long diff = std::abs(time1 - time2);
            
            if (diff <= static_cast<long long>(timeTolerance)) {
                syncedAsset1.push_back(candle1);
                syncedAsset2.push_back(candle2);
                break;
            }
        }
    }
    
    return {syncedAsset1, syncedAsset2};
}

// Tìm swing highs từ candles trực tiếp
std::vector<MomentumPoint> SMTAnalysis::findSwingHighsFromCandles(const std::vector<Candle>& candles, int lookback) {
    std::vector<MomentumPoint> swingHighs;
    if (candles.size() < lookback * 2 + 1) return swingHighs;
    
    for (size_t i = lookback; i < candles.size() - lookback; ++i) {
        bool isSwingHigh = true;
        double high = candles[i].getHigh();
        
        for (int j = i - lookback; j <= static_cast<int>(i) + lookback; ++j) {
            if (j != static_cast<int>(i) && candles[j].getHigh() >= high) {
                isSwingHigh = false;
                break;
            }
        }
        
        if (isSwingHigh) {
            MomentumPoint point;
            point.time = candles[i].getTimeOpen();
            point.value = high;
            point.candleIndex = i;
            swingHighs.push_back(point);
        }
    }
    
    return swingHighs;
}

// Tìm swing lows từ candles trực tiếp
std::vector<MomentumPoint> SMTAnalysis::findSwingLowsFromCandles(const std::vector<Candle>& candles, int lookback) {
    std::vector<MomentumPoint> swingLows;
    if (candles.size() < lookback * 2 + 1) return swingLows;
    
    for (size_t i = lookback; i < candles.size() - lookback; ++i) {
        bool isSwingLow = true;
        double low = candles[i].getLow();
        
        for (int j = i - lookback; j <= static_cast<int>(i) + lookback; ++j) {
            if (j != static_cast<int>(i) && candles[j].getLow() <= low) {
                isSwingLow = false;
                break;
            }
        }
        
        if (isSwingLow) {
            MomentumPoint point;
            point.time = candles[i].getTimeOpen();
            point.value = low;
            point.candleIndex = i;
            swingLows.push_back(point);
        }
    }
    
    return swingLows;
}

// SMT Divergence - So sánh giá của 2 tài sản khác nhau
std::vector<Divergence> SMTAnalysis::detectSMTDivergence(
    const std::vector<Candle>& asset1Candles,
    const std::vector<Candle>& asset2Candles,
    const std::string& asset1Name,
    const std::string& asset2Name,
    bool isInverseCorrelation,
    int lookbackPeriod) {
    
    std::vector<Divergence> divergences;
    
    if (asset1Candles.size() < lookbackPeriod * 2 || asset2Candles.size() < lookbackPeriod * 2) {
        return divergences;
    }
    
    // Sync candles theo timestamp
    auto [syncedAsset1, syncedAsset2] = syncCandlesByTime(asset1Candles, asset2Candles);
    
    if (syncedAsset1.size() < lookbackPeriod * 2 || syncedAsset2.size() < lookbackPeriod * 2) {
        return divergences;
    }
    
    // Tìm swing highs và lows cho cả 2 tài sản
    std::vector<MomentumPoint> asset1Highs = findSwingHighsFromCandles(syncedAsset1, lookbackPeriod / 2);
    std::vector<MomentumPoint> asset1Lows = findSwingLowsFromCandles(syncedAsset1, lookbackPeriod / 2);
    std::vector<MomentumPoint> asset2Highs = findSwingHighsFromCandles(syncedAsset2, lookbackPeriod / 2);
    std::vector<MomentumPoint> asset2Lows = findSwingLowsFromCandles(syncedAsset2, lookbackPeriod / 2);
    
    if (asset1Highs.size() < 2 || asset1Lows.size() < 2 || asset2Highs.size() < 2 || asset2Lows.size() < 2) {
        return divergences;
    }
    
    // Phát hiện Bearish SMT Divergence
    // Asset1 tạo higher high, nhưng Asset2 tạo lower high (tương quan thuận)
    // Hoặc Asset1 tạo higher high, Asset2 tạo higher high (tương quan nghịch - DXY case)
    for (size_t i = 1; i < asset1Highs.size(); ++i) {
        const MomentumPoint& prevHigh1 = asset1Highs[i - 1];
        const MomentumPoint& currHigh1 = asset1Highs[i];
        
        // Asset1 tạo higher high
        if (currHigh1.value > prevHigh1.value) {
            // Tìm swing high tương ứng trong Asset2
            for (size_t j = 1; j < asset2Highs.size(); ++j) {
                const MomentumPoint& prevHigh2 = asset2Highs[j - 1];
                const MomentumPoint& currHigh2 = asset2Highs[j];
                
                // Kiểm tra xem có overlap về thời gian không
                if (currHigh2.time >= prevHigh1.time && currHigh2.time <= currHigh1.time) {
                    bool isDivergence = false;
                    
                    if (!isInverseCorrelation) {
                        // Tương quan thuận: Asset2 nên tạo higher high nhưng lại tạo lower high
                        isDivergence = (currHigh2.value < prevHigh2.value);
                    } else {
                        // Tương quan nghịch: Asset2 nên tạo lower high nhưng lại tạo higher high
                        isDivergence = (currHigh2.value > prevHigh2.value);
                    }
                    
                    if (isDivergence) {
                        Divergence div;
                        div.type = BEARISH_DIVERGENCE;
                        div.timeStart = prevHigh1.time;
                        div.timeEnd = currHigh1.time;
                        div.priceStart = prevHigh1.value;
                        div.priceEnd = currHigh1.value;
                        div.indicatorStart = prevHigh2.value;
                        div.indicatorEnd = currHigh2.value;
                        div.candleIndexStart = prevHigh1.candleIndex;
                        div.candleIndexEnd = currHigh1.candleIndex;
                        div.entity1Name = asset1Name;
                        div.entity2Name = asset2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    // Phát hiện Bullish SMT Divergence
    // Asset1 tạo lower low, nhưng Asset2 tạo higher low (tương quan thuận)
    // Hoặc Asset1 tạo lower low, Asset2 tạo lower low (tương quan nghịch)
    for (size_t i = 1; i < asset1Lows.size(); ++i) {
        const MomentumPoint& prevLow1 = asset1Lows[i - 1];
        const MomentumPoint& currLow1 = asset1Lows[i];
        
        // Asset1 tạo lower low
        if (currLow1.value < prevLow1.value) {
            // Tìm swing low tương ứng trong Asset2
            for (size_t j = 1; j < asset2Lows.size(); ++j) {
                const MomentumPoint& prevLow2 = asset2Lows[j - 1];
                const MomentumPoint& currLow2 = asset2Lows[j];
                
                // Kiểm tra xem có overlap về thời gian không
                if (currLow2.time >= prevLow1.time && currLow2.time <= currLow1.time) {
                    bool isDivergence = false;
                    
                    if (!isInverseCorrelation) {
                        // Tương quan thuận: Asset2 nên tạo lower low nhưng lại tạo higher low
                        isDivergence = (currLow2.value > prevLow2.value);
                    } else {
                        // Tương quan nghịch: Asset2 nên tạo higher low nhưng lại tạo lower low
                        isDivergence = (currLow2.value < prevLow2.value);
                    }
                    
                    if (isDivergence) {
                        Divergence div;
                        div.type = BULLISH_DIVERGENCE;
                        div.timeStart = prevLow1.time;
                        div.timeEnd = currLow1.time;
                        div.priceStart = prevLow1.value;
                        div.priceEnd = currLow1.value;
                        div.indicatorStart = prevLow2.value;
                        div.indicatorEnd = currLow2.value;
                        div.candleIndexStart = prevLow1.candleIndex;
                        div.candleIndexEnd = currLow1.candleIndex;
                        div.entity1Name = asset1Name;
                        div.entity2Name = asset2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    return divergences;
}

// SMT Divergence giữa BTC và ETH
std::vector<Divergence> SMTAnalysis::detectBTCETHDivergence(
    const std::vector<Candle>& btcCandles,
    const std::vector<Candle>& ethCandles,
    int lookbackPeriod) {
    
    return detectSMTDivergence(btcCandles, ethCandles, "BTC", "ETH", false, lookbackPeriod);
}

// SMT Divergence giữa Currency và DXY (tương quan nghịch)
std::vector<Divergence> SMTAnalysis::detectCurrencyDXYDivergence(
    const std::vector<Candle>& currencyCandles,
    const std::vector<Candle>& dxyCandles,
    const std::string& currencyName,
    int lookbackPeriod) {
    
    return detectSMTDivergence(currencyCandles, dxyCandles, currencyName, "DXY", true, lookbackPeriod);
}

// ========== INDICATOR DIVERGENCE IMPLEMENTATION ==========

// Main Indicator Divergence Detection Function (đổi tên từ detectDivergence)
std::vector<Divergence> SMTAnalysis::detectIndicatorDivergence(
    const std::vector<Candle>& candles,
    const std::vector<double>& entity1Values,
    const std::vector<double>& entity2Values,
    const std::string& entity1Name,
    const std::string& entity2Name,
    int lookbackPeriod) {
    
    std::vector<Divergence> divergences;
    
    if (candles.size() != entity1Values.size() || candles.size() != entity2Values.size()) {
        return divergences;
    }
    
    if (candles.size() < lookbackPeriod * 2) {
        return divergences;
    }
    
    // Tìm swing highs và lows cho cả 2 entities
    std::vector<MomentumPoint> entity1Highs = findSwingHighs(entity1Values, candles, lookbackPeriod / 2);
    std::vector<MomentumPoint> entity1Lows = findSwingLows(entity1Values, candles, lookbackPeriod / 2);
    std::vector<MomentumPoint> entity2Highs = findSwingHighs(entity2Values, candles, lookbackPeriod / 2);
    std::vector<MomentumPoint> entity2Lows = findSwingLows(entity2Values, candles, lookbackPeriod / 2);
    
    // Phát hiện Bearish Divergence (Higher High in price, Lower High in indicator)
    for (size_t i = 1; i < entity1Highs.size(); ++i) {
        const MomentumPoint& prevHigh1 = entity1Highs[i - 1];
        const MomentumPoint& currHigh1 = entity1Highs[i];
        
        // Price tạo higher high
        if (currHigh1.value > prevHigh1.value) {
            // Tìm indicator high tương ứng
            for (size_t j = 0; j < entity2Highs.size(); ++j) {
                if (entity2Highs[j].candleIndex >= prevHigh1.candleIndex && 
                    entity2Highs[j].candleIndex <= currHigh1.candleIndex) {
                    
                    // Kiểm tra xem có lower high trong indicator không
                    if (j > 0 && entity2Highs[j].value < entity2Highs[j - 1].value) {
                        Divergence div;
                        div.type = BEARISH_DIVERGENCE;
                        div.timeStart = prevHigh1.time;
                        div.timeEnd = currHigh1.time;
                        div.priceStart = prevHigh1.value;
                        div.priceEnd = currHigh1.value;
                        div.indicatorStart = entity2Highs[j - 1].value;
                        div.indicatorEnd = entity2Highs[j].value;
                        div.candleIndexStart = prevHigh1.candleIndex;
                        div.candleIndexEnd = currHigh1.candleIndex;
                        div.entity1Name = entity1Name;
                        div.entity2Name = entity2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    // Phát hiện Bullish Divergence (Lower Low in price, Higher Low in indicator)
    for (size_t i = 1; i < entity1Lows.size(); ++i) {
        const MomentumPoint& prevLow1 = entity1Lows[i - 1];
        const MomentumPoint& currLow1 = entity1Lows[i];
        
        // Price tạo lower low
        if (currLow1.value < prevLow1.value) {
            // Tìm indicator low tương ứng
            for (size_t j = 0; j < entity2Lows.size(); ++j) {
                if (entity2Lows[j].candleIndex >= prevLow1.candleIndex && 
                    entity2Lows[j].candleIndex <= currLow1.candleIndex) {
                    
                    // Kiểm tra xem có higher low trong indicator không
                    if (j > 0 && entity2Lows[j].value > entity2Lows[j - 1].value) {
                        Divergence div;
                        div.type = BULLISH_DIVERGENCE;
                        div.timeStart = prevLow1.time;
                        div.timeEnd = currLow1.time;
                        div.priceStart = prevLow1.value;
                        div.priceEnd = currLow1.value;
                        div.indicatorStart = entity2Lows[j - 1].value;
                        div.indicatorEnd = entity2Lows[j].value;
                        div.candleIndexStart = prevLow1.candleIndex;
                        div.candleIndexEnd = currLow1.candleIndex;
                        div.entity1Name = entity1Name;
                        div.entity2Name = entity2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    // Phát hiện Hidden Bullish Divergence (Higher Low in price, Lower Low in indicator)
    for (size_t i = 1; i < entity1Lows.size(); ++i) {
        const MomentumPoint& prevLow1 = entity1Lows[i - 1];
        const MomentumPoint& currLow1 = entity1Lows[i];
        
        // Price tạo higher low
        if (currLow1.value > prevLow1.value) {
            // Tìm indicator low tương ứng
            for (size_t j = 0; j < entity2Lows.size(); ++j) {
                if (entity2Lows[j].candleIndex >= prevLow1.candleIndex && 
                    entity2Lows[j].candleIndex <= currLow1.candleIndex) {
                    
                    // Kiểm tra xem có lower low trong indicator không
                    if (j > 0 && entity2Lows[j].value < entity2Lows[j - 1].value) {
                        Divergence div;
                        div.type = HIDDEN_BULLISH_DIVERGENCE;
                        div.timeStart = prevLow1.time;
                        div.timeEnd = currLow1.time;
                        div.priceStart = prevLow1.value;
                        div.priceEnd = currLow1.value;
                        div.indicatorStart = entity2Lows[j - 1].value;
                        div.indicatorEnd = entity2Lows[j].value;
                        div.candleIndexStart = prevLow1.candleIndex;
                        div.candleIndexEnd = currLow1.candleIndex;
                        div.entity1Name = entity1Name;
                        div.entity2Name = entity2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    // Phát hiện Hidden Bearish Divergence (Lower High in price, Higher High in indicator)
    for (size_t i = 1; i < entity1Highs.size(); ++i) {
        const MomentumPoint& prevHigh1 = entity1Highs[i - 1];
        const MomentumPoint& currHigh1 = entity1Highs[i];
        
        // Price tạo lower high
        if (currHigh1.value < prevHigh1.value) {
            // Tìm indicator high tương ứng
            for (size_t j = 0; j < entity2Highs.size(); ++j) {
                if (entity2Highs[j].candleIndex >= prevHigh1.candleIndex && 
                    entity2Highs[j].candleIndex <= currHigh1.candleIndex) {
                    
                    // Kiểm tra xem có higher high trong indicator không
                    if (j > 0 && entity2Highs[j].value > entity2Highs[j - 1].value) {
                        Divergence div;
                        div.type = HIDDEN_BEARISH_DIVERGENCE;
                        div.timeStart = prevHigh1.time;
                        div.timeEnd = currHigh1.time;
                        div.priceStart = prevHigh1.value;
                        div.priceEnd = currHigh1.value;
                        div.indicatorStart = entity2Highs[j - 1].value;
                        div.indicatorEnd = entity2Highs[j].value;
                        div.candleIndexStart = prevHigh1.candleIndex;
                        div.candleIndexEnd = currHigh1.candleIndex;
                        div.entity1Name = entity1Name;
                        div.entity2Name = entity2Name;
                        div.strength = calculateDivergenceStrength(div);
                        divergences.push_back(div);
                    }
                    break;
                }
            }
        }
    }
    
    return divergences;
}

// Price vs Volume Divergence
std::vector<Divergence> SMTAnalysis::detectPriceVolumeDivergence(const std::vector<Candle>& candles, int lookbackPeriod) {
    std::vector<double> prices = calculatePriceValues(candles);
    std::vector<double> volumes;
    volumes.reserve(candles.size());
    
    for (const auto& candle : candles) {
        volumes.push_back(candle.getVolume());
    }
    
    return detectIndicatorDivergence(candles, prices, volumes, "Price", "Volume", lookbackPeriod);
}

// Price vs Momentum Divergence
std::vector<Divergence> SMTAnalysis::detectPriceMomentumDivergence(const std::vector<Candle>& candles, int momentumPeriod, int lookbackPeriod) {
    std::vector<double> prices = calculatePriceValues(candles);
    std::vector<double> momentum = calculateMomentum(candles, momentumPeriod);
    
    return detectIndicatorDivergence(candles, prices, momentum, "Price", "Momentum", lookbackPeriod);
}

// Price vs RSI Divergence
std::vector<Divergence> SMTAnalysis::detectPriceRSIDivergence(const std::vector<Candle>& candles, int rsiPeriod, int lookbackPeriod) {
    std::vector<double> prices = calculatePriceValues(candles);
    std::vector<double> rsi = calculateRSI(candles, rsiPeriod);
    
    return detectIndicatorDivergence(candles, prices, rsi, "Price", "RSI", lookbackPeriod);
}

// Price vs Order Flow Divergence
std::vector<Divergence> SMTAnalysis::detectPriceOrderFlowDivergence(const std::vector<Candle>& candles, int lookbackPeriod) {
    std::vector<double> prices = calculatePriceValues(candles);
    
    // Order Flow = Volume * Price Change (buying pressure vs selling pressure)
    std::vector<double> orderFlow;
    orderFlow.reserve(candles.size());
    orderFlow.push_back(0.0);
    
    for (size_t i = 1; i < candles.size(); ++i) {
        double priceChange = candles[i].getClose() - candles[i - 1].getClose();
        double volume = candles[i].getVolume();
        // Positive order flow = buying pressure, Negative = selling pressure
        orderFlow.push_back(priceChange * volume);
    }
    
    return detectIndicatorDivergence(candles, prices, orderFlow, "Price", "OrderFlow", lookbackPeriod);
}

