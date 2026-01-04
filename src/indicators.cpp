#include "indicators.h"
#include <algorithm>
#include <cmath>
#include <ctime>

// For thread-safe gmtime_r on Unix/macOS
#ifdef __APPLE__
    #include <time.h>
#elif defined(__unix__) || defined(__linux__)
    #include <time.h>
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200112L
    #endif
#endif

// Helper functions
bool Indicators::isBullishCandle(const Candle& candle) {
    return candle.getClose() > candle.getOpen();
}

bool Indicators::isBearishCandle(const Candle& candle) {
    return candle.getClose() < candle.getOpen();
}

double Indicators::getCandleBody(const Candle& candle) {
    return std::abs(candle.getClose() - candle.getOpen());
}

bool Indicators::hasWick(const Candle& candle, double minWickRatio) {
    double body = getCandleBody(candle);
    double totalRange = candle.getHigh() - candle.getLow();
    if (totalRange == 0) return false;
    
    double upperWick = isBullishCandle(candle) 
        ? candle.getHigh() - candle.getClose() 
        : candle.getHigh() - candle.getOpen();
    double lowerWick = isBullishCandle(candle)
        ? candle.getOpen() - candle.getLow()
        : candle.getClose() - candle.getLow();
    
    return (upperWick / totalRange >= minWickRatio) || (lowerWick / totalRange >= minWickRatio);
}

// FVG Detection
std::vector<FVG> Indicators::detectFVG(const std::vector<Candle>& candles) {
    std::vector<FVG> fvgs;
    if (candles.size() < 3) return fvgs;
    
    for (size_t i = 1; i < candles.size() - 1; ++i) {
        const Candle& prev = candles[i - 1];
        const Candle& curr = candles[i];
        const Candle& next = candles[i + 1];
        
        // Bullish FVG: gap tăng giá - previous candle high < next candle low
        // Có khoảng trống giữa high của nến trước và low của nến sau
        if (prev.getHigh() < next.getLow()) {
            FVG fvg;
            fvg.timeStart = curr.getTimeOpen();
            fvg.timeEnd = curr.getTimeClose();
            fvg.top = next.getLow();  // Top của gap
            fvg.bottom = prev.getHigh();  // Bottom của gap
            fvg.isBullish = true;
            fvg.isBroken = false;
            fvg.candleIndex = i;
            fvgs.push_back(fvg);
        }
        // Bearish FVG: gap giảm giá - previous candle low > next candle high
        // Có khoảng trống giữa low của nến trước và high của nến sau
        else if (prev.getLow() > next.getHigh()) {
            FVG fvg;
            fvg.timeStart = curr.getTimeOpen();
            fvg.timeEnd = curr.getTimeClose();
            fvg.top = prev.getLow();  // Top của gap
            fvg.bottom = next.getHigh();  // Bottom của gap
            fvg.isBullish = false;
            fvg.isBroken = false;
            fvg.candleIndex = i;
            fvgs.push_back(fvg);
        }
    }
    
    return fvgs;
}

std::vector<FVG> Indicators::detectBullishFVG(const std::vector<Candle>& candles) {
    std::vector<FVG> allFvgs = detectFVG(candles);
    std::vector<FVG> bullishFvgs;
    for (const auto& fvg : allFvgs) {
        if (fvg.isBullish) {
            bullishFvgs.push_back(fvg);
        }
    }
    return bullishFvgs;
}

std::vector<FVG> Indicators::detectBearishFVG(const std::vector<Candle>& candles) {
    std::vector<FVG> allFvgs = detectFVG(candles);
    std::vector<FVG> bearishFvgs;
    for (const auto& fvg : allFvgs) {
        if (!fvg.isBullish) {
            bearishFvgs.push_back(fvg);
        }
    }
    return bearishFvgs;
}

bool Indicators::isFVGBroken(const FVG& fvg, const std::vector<Candle>& candles, int startIndex) {
    if (startIndex >= static_cast<int>(candles.size())) return false;
    
    for (int i = startIndex; i < static_cast<int>(candles.size()); ++i) {
        const Candle& candle = candles[i];
        
        if (fvg.isBullish) {
            // Bullish FVG is broken when price closes below the bottom
            if (candle.getClose() < fvg.bottom) {
                return true;
            }
        } else {
            // Bearish FVG is broken when price closes above the top
            if (candle.getClose() > fvg.top) {
                return true;
            }
        }
    }
    
    return false;
}

void Indicators::updateFVGStatus(std::vector<FVG>& fvgs, const std::vector<Candle>& candles) {
    for (auto& fvg : fvgs) {
        if (!fvg.isBroken) {
            fvg.isBroken = isFVGBroken(fvg, candles, fvg.candleIndex + 1);
        }
    }
}

// Order Block Detection
std::vector<OrderBlock> Indicators::detectOrderBlocks(const std::vector<Candle>& candles) {
    std::vector<OrderBlock> obs;
    if (candles.size() < 3) return obs; // Cần ít nhất 3 nến để xác nhận structural shift
    
    for (size_t i = 1; i < candles.size() - 1; ++i) {
        const Candle& prev = candles[i - 1];
        const Candle& curr = candles[i];
        const Candle& next = candles[i + 1];
        
        // Bullish OB: Nến bearish cuối cùng trước khi có move mạnh lên
        // Cần xác nhận: nến sau tạo ra structural shift hoặc break price zone
        if (isBearishCandle(prev) && isBullishCandle(curr)) {
            double body = getCandleBody(prev);
            double totalRange = prev.getHigh() - prev.getLow();
            
            // Check if previous candle has significant body (at least 60% of range)
            if (totalRange > 0 && (body / totalRange) >= 0.6) {
                // Xác nhận: nến sau tạo ra move mạnh (break high của prev hoặc tạo FVG)
                bool confirmed = false;
                
                // Kiểm tra break high của prev
                if (curr.getHigh() > prev.getHigh() || next.getHigh() > prev.getHigh()) {
                    confirmed = true;
                }
                
                // Hoặc kiểm tra có FVG sau đó (dấu hiệu move mạnh)
                if (i + 1 < candles.size() - 1) {
                    const Candle& next2 = candles[i + 2];
                    if (curr.getHigh() < next2.getLow() || curr.getLow() > next2.getHigh()) {
                        confirmed = true;
                    }
                }
                
                // Hoặc kiểm tra structural shift
                if (detectMSS(candles, i + 1) || detectBOS(candles, i + 1)) {
                    confirmed = true;
                }
                
                if (confirmed) {
                    OrderBlock ob;
                    ob.timeStart = prev.getTimeOpen();
                    ob.timeEnd = prev.getTimeClose();
                    ob.top = prev.getOpen();
                    ob.bottom = prev.getClose();
                    ob.isBullish = true;
                    ob.isBroken = false;
                    ob.candleIndex = i - 1;
                    ob.volume = prev.getVolume();
                    obs.push_back(ob);
                }
            }
        }
        // Bearish OB: Nến bullish cuối cùng trước khi có move mạnh xuống
        // Cần xác nhận: nến sau tạo ra structural shift hoặc break price zone
        else if (isBullishCandle(prev) && isBearishCandle(curr)) {
            double body = getCandleBody(prev);
            double totalRange = prev.getHigh() - prev.getLow();
            
            // Check if previous candle has significant body (at least 60% of range)
            if (totalRange > 0 && (body / totalRange) >= 0.6) {
                // Xác nhận: nến sau tạo ra move mạnh (break low của prev hoặc tạo FVG)
                bool confirmed = false;
                
                // Kiểm tra break low của prev
                if (curr.getLow() < prev.getLow() || next.getLow() < prev.getLow()) {
                    confirmed = true;
                }
                
                // Hoặc kiểm tra có FVG sau đó (dấu hiệu move mạnh)
                if (i + 1 < candles.size() - 1) {
                    const Candle& next2 = candles[i + 2];
                    if (curr.getHigh() < next2.getLow() || curr.getLow() > next2.getHigh()) {
                        confirmed = true;
                    }
                }
                
                // Hoặc kiểm tra structural shift
                if (detectMSS(candles, i + 1) || detectBOS(candles, i + 1)) {
                    confirmed = true;
                }
                
                if (confirmed) {
                    OrderBlock ob;
                    ob.timeStart = prev.getTimeOpen();
                    ob.timeEnd = prev.getTimeClose();
                    ob.top = prev.getClose();
                    ob.bottom = prev.getOpen();
                    ob.isBullish = false;
                    ob.isBroken = false;
                    ob.candleIndex = i - 1;
                    ob.volume = prev.getVolume();
                    obs.push_back(ob);
                }
            }
        }
    }
    
    return obs;
}

std::vector<OrderBlock> Indicators::detectBullishOB(const std::vector<Candle>& candles) {
    std::vector<OrderBlock> allOBs = detectOrderBlocks(candles);
    std::vector<OrderBlock> bullishOBs;
    for (const auto& ob : allOBs) {
        if (ob.isBullish) {
            bullishOBs.push_back(ob);
        }
    }
    return bullishOBs;
}

std::vector<OrderBlock> Indicators::detectBearishOB(const std::vector<Candle>& candles) {
    std::vector<OrderBlock> allOBs = detectOrderBlocks(candles);
    std::vector<OrderBlock> bearishOBs;
    for (const auto& ob : allOBs) {
        if (!ob.isBullish) {
            bearishOBs.push_back(ob);
        }
    }
    return bearishOBs;
}

bool Indicators::isOBBroken(const OrderBlock& ob, const std::vector<Candle>& candles, int startIndex) {
    if (startIndex >= static_cast<int>(candles.size())) return false;
    
    for (int i = startIndex; i < static_cast<int>(candles.size()); ++i) {
        const Candle& candle = candles[i];
        
        if (ob.isBullish) {
            // Bullish OB is broken when price closes below the bottom
            if (candle.getClose() < ob.bottom) {
                return true;
            }
        } else {
            // Bearish OB is broken when price closes above the top
            if (candle.getClose() > ob.top) {
                return true;
            }
        }
    }
    
    return false;
}

void Indicators::updateOBStatus(std::vector<OrderBlock>& obs, const std::vector<Candle>& candles) {
    for (auto& ob : obs) {
        if (!ob.isBroken) {
            ob.isBroken = isOBBroken(ob, candles, ob.candleIndex + 1);
        }
    }
}

// Liquidity Detection
std::vector<LiquidityZone> Indicators::detectLiquidity(const std::vector<Candle>& candles) {
    std::vector<LiquidityZone> zones;
    if (candles.size() < 3) return zones;
    
    for (size_t i = 1; i < candles.size() - 1; ++i) {
        const Candle& prev = candles[i - 1];
        const Candle& curr = candles[i];
        const Candle& next = candles[i + 1];
        
        // Sellside liquidity: wick above that gets taken out
        if (curr.getHigh() > prev.getHigh() && curr.getHigh() > next.getHigh()) {
            LiquidityZone zone;
            zone.timeStart = curr.getTimeOpen();
            zone.timeEnd = curr.getTimeClose();
            zone.price = curr.getHigh();
            zone.isSellside = true;
            zone.isTaken = false;
            zone.candleIndex = i;
            zones.push_back(zone);
        }
        
        // Buyside liquidity: wick below that gets taken out
        if (curr.getLow() < prev.getLow() && curr.getLow() < next.getLow()) {
            LiquidityZone zone;
            zone.timeStart = curr.getTimeOpen();
            zone.timeEnd = curr.getTimeClose();
            zone.price = curr.getLow();
            zone.isSellside = false;
            zone.isTaken = false;
            zone.candleIndex = i;
            zones.push_back(zone);
        }
    }
    
    return zones;
}

std::vector<LiquidityZone> Indicators::detectSellsideLiquidity(const std::vector<Candle>& candles) {
    std::vector<LiquidityZone> allZones = detectLiquidity(candles);
    std::vector<LiquidityZone> sellsideZones;
    for (const auto& zone : allZones) {
        if (zone.isSellside) {
            sellsideZones.push_back(zone);
        }
    }
    return sellsideZones;
}

std::vector<LiquidityZone> Indicators::detectBuysideLiquidity(const std::vector<Candle>& candles) {
    std::vector<LiquidityZone> allZones = detectLiquidity(candles);
    std::vector<LiquidityZone> buysideZones;
    for (const auto& zone : allZones) {
        if (!zone.isSellside) {
            buysideZones.push_back(zone);
        }
    }
    return buysideZones;
}

bool Indicators::isLiquidityTaken(const LiquidityZone& zone, const std::vector<Candle>& candles, int startIndex) {
    if (startIndex >= static_cast<int>(candles.size())) return false;
    
    for (int i = startIndex; i < static_cast<int>(candles.size()); ++i) {
        const Candle& candle = candles[i];
        
        if (zone.isSellside) {
            // Sellside liquidity is taken when price breaks above
            if (candle.getHigh() > zone.price) {
                return true;
            }
        } else {
            // Buyside liquidity is taken when price breaks below
            if (candle.getLow() < zone.price) {
                return true;
            }
        }
    }
    
    return false;
}

void Indicators::updateLiquidityStatus(std::vector<LiquidityZone>& zones, const std::vector<Candle>& candles) {
    for (auto& zone : zones) {
        if (!zone.isTaken) {
            zone.isTaken = isLiquidityTaken(zone, candles, zone.candleIndex + 1);
        }
    }
}

// Market Structure Detection
MarketStructure Indicators::detectMarketStructure(const std::vector<Candle>& candles, int lookback) {
    if (candles.size() < static_cast<size_t>(lookback)) {
        return NEUTRAL;
    }
    
    // Find highest high and lowest low in lookback period
    double highestHigh = candles[candles.size() - lookback].getHigh();
    double lowestLow = candles[candles.size() - lookback].getLow();
    
    for (size_t i = candles.size() - lookback; i < candles.size(); ++i) {
        if (candles[i].getHigh() > highestHigh) {
            highestHigh = candles[i].getHigh();
        }
        if (candles[i].getLow() < lowestLow) {
            lowestLow = candles[i].getLow();
        }
    }
    
    // Determine structure based on recent price action
    double recentClose = candles.back().getClose();
    double midPoint = (highestHigh + lowestLow) / 2.0;
    
    if (recentClose > midPoint) {
        return BULLISH;
    } else if (recentClose < midPoint) {
        return BEARISH;
    }
    
    return NEUTRAL;
}

bool Indicators::detectMSS(const std::vector<Candle>& candles, int index) {
    if (index < 2 || index >= static_cast<int>(candles.size())) return false;
    
    const Candle& curr = candles[index];
    const Candle& prev = candles[index - 1];
    const Candle& prev2 = candles[index - 2];
    
    // MSS: Change from bearish to bullish structure
    // Previous structure was bearish (lower lows), now breaking higher
    if (prev2.getLow() > prev.getLow() && curr.getLow() > prev.getLow() && curr.getClose() > prev.getHigh()) {
        return true;
    }
    
    // MSS: Change from bullish to bearish structure
    // Previous structure was bullish (higher highs), now breaking lower
    if (prev2.getHigh() < prev.getHigh() && curr.getHigh() < prev.getHigh() && curr.getClose() < prev.getLow()) {
        return true;
    }
    
    return false;
}

bool Indicators::detectBOS(const std::vector<Candle>& candles, int index) {
    if (index < 1 || index >= static_cast<int>(candles.size())) return false;
    
    const Candle& curr = candles[index];
    const Candle& prev = candles[index - 1];
    
    // BOS in bullish trend: Breaking above previous high
    if (curr.getClose() > prev.getHigh()) {
        // Check if we're in an uptrend
        bool inUptrend = true;
        for (int i = index - 1; i >= std::max(0, index - 10); --i) {
            if (candles[i].getLow() < candles[std::max(0, i - 1)].getLow()) {
                inUptrend = false;
                break;
            }
        }
        if (inUptrend) return true;
    }
    
    // BOS in bearish trend: Breaking below previous low
    if (curr.getClose() < prev.getLow()) {
        // Check if we're in a downtrend
        bool inDowntrend = true;
        for (int i = index - 1; i >= std::max(0, index - 10); --i) {
            if (candles[i].getHigh() > candles[std::max(0, i - 1)].getHigh()) {
                inDowntrend = false;
                break;
            }
        }
        if (inDowntrend) return true;
    }
    
    return false;
}

// Time Killzone
std::vector<Killzone> Indicators::getKillzones() {
    std::vector<Killzone> killzones;
    
    // London Killzone: 8:00 AM - 12:00 PM GMT
    Killzone london;
    london.startHour = 8;
    london.startMinute = 0;
    london.endHour = 12;
    london.endMinute = 0;
    london.name = "London Killzone";
    killzones.push_back(london);
    
    // New York Killzone: 1:00 PM - 4:00 PM GMT (8:00 AM - 11:00 AM EST)
    Killzone ny;
    ny.startHour = 13;
    ny.startMinute = 0;
    ny.endHour = 16;
    ny.endMinute = 0;
    ny.name = "New York Killzone";
    killzones.push_back(ny);
    
    // Asian Killzone: 11:00 PM - 2:00 AM GMT (Tokyo session)
    Killzone asian;
    asian.startHour = 23;
    asian.startMinute = 0;
    asian.endHour = 2;
    asian.endMinute = 0;
    asian.name = "Asian Killzone";
    killzones.push_back(asian);
    
    return killzones;
}

bool Indicators::isInKillzone(long long timestamp) {
    std::time_t time = timestamp / 1000; // Convert milliseconds to seconds
    std::tm tm_buf;
    
    // Use gmtime_r for thread-safety on macOS/Unix
    #ifdef __APPLE__
        std::tm* tm_ptr = gmtime_r(&time, &tm_buf);
    #else
        std::tm* tm_ptr = std::gmtime(&time);
        if (tm_ptr) {
            tm_buf = *tm_ptr;
            tm_ptr = &tm_buf;
        }
    #endif
    
    if (!tm_ptr) return false;
    
    int hour = tm_ptr->tm_hour;
    int minute = tm_ptr->tm_min;
    
    std::vector<Killzone> killzones = getKillzones();
    
    for (const auto& zone : killzones) {
        int startTime = zone.startHour * 60 + zone.startMinute;
        int endTime = zone.endHour * 60 + zone.endMinute;
        int currentTime = hour * 60 + minute;
        
        // Handle overnight zones (e.g., 23:00 - 02:00)
        if (startTime > endTime) {
            if (currentTime >= startTime || currentTime < endTime) {
                return true;
            }
        } else {
            if (currentTime >= startTime && currentTime < endTime) {
                return true;
            }
        }
    }
    
    return false;
}

std::string Indicators::getCurrentKillzone(long long timestamp) {
    std::time_t time = timestamp / 1000;
    std::tm tm_buf;
    
    // Use gmtime_r for thread-safety on macOS/Unix
    #ifdef __APPLE__
        std::tm* tm_ptr = gmtime_r(&time, &tm_buf);
    #else
        std::tm* tm_ptr = std::gmtime(&time);
        if (tm_ptr) {
            tm_buf = *tm_ptr;
            tm_ptr = &tm_buf;
        }
    #endif
    
    if (!tm_ptr) return "No Killzone";
    
    int hour = tm_ptr->tm_hour;
    int minute = tm_ptr->tm_min;
    
    std::vector<Killzone> killzones = getKillzones();
    
    for (const auto& zone : killzones) {
        int startTime = zone.startHour * 60 + zone.startMinute;
        int endTime = zone.endHour * 60 + zone.endMinute;
        int currentTime = hour * 60 + minute;
        
        // Handle overnight zones
        if (startTime > endTime) {
            if (currentTime >= startTime || currentTime < endTime) {
                return zone.name;
            }
        } else {
            if (currentTime >= startTime && currentTime < endTime) {
                return zone.name;
            }
        }
    }
    
    return "No Killzone";
}

// Additional Helper Functions
double Indicators::calculateRange(const std::vector<Candle>& candles, int startIndex, int endIndex) {
    if (startIndex < 0 || endIndex >= static_cast<int>(candles.size()) || startIndex > endIndex) {
        return 0.0;
    }
    
    double highest = candles[startIndex].getHigh();
    double lowest = candles[startIndex].getLow();
    
    for (int i = startIndex; i <= endIndex; ++i) {
        if (candles[i].getHigh() > highest) highest = candles[i].getHigh();
        if (candles[i].getLow() < lowest) lowest = candles[i].getLow();
    }
    
    return highest - lowest;
}

double Indicators::findHighestHigh(const std::vector<Candle>& candles, int startIndex, int endIndex) {
    if (startIndex < 0 || endIndex >= static_cast<int>(candles.size()) || startIndex > endIndex) {
        return 0.0;
    }
    
    double highest = candles[startIndex].getHigh();
    for (int i = startIndex; i <= endIndex; ++i) {
        if (candles[i].getHigh() > highest) highest = candles[i].getHigh();
    }
    
    return highest;
}

double Indicators::findLowestLow(const std::vector<Candle>& candles, int startIndex, int endIndex) {
    if (startIndex < 0 || endIndex >= static_cast<int>(candles.size()) || startIndex > endIndex) {
        return 0.0;
    }
    
    double lowest = candles[startIndex].getLow();
    for (int i = startIndex; i <= endIndex; ++i) {
        if (candles[i].getLow() < lowest) lowest = candles[i].getLow();
    }
    
    return lowest;
}

// SMT Indicators - Change of Character (CHoCH)
bool Indicators::isCHoCH(const std::vector<Candle>& candles, int index) {
    if (index < 2 || index >= static_cast<int>(candles.size())) return false;
    
    const Candle& curr = candles[index];
    const Candle& prev = candles[index - 1];
    const Candle& prev2 = candles[index - 2];
    
    // Bullish CHoCH: Từ downtrend chuyển sang uptrend
    // Kiểm tra: prev2 có lower low, nhưng curr tạo higher low và break high
    if (prev2.getLow() > prev.getLow() && curr.getLow() > prev.getLow() && curr.getClose() > prev.getHigh()) {
        return true;
    }
    
    // Bearish CHoCH: Từ uptrend chuyển sang downtrend
    // Kiểm tra: prev2 có higher high, nhưng curr tạo lower high và break low
    if (prev2.getHigh() < prev.getHigh() && curr.getHigh() < prev.getHigh() && curr.getClose() < prev.getLow()) {
        return true;
    }
    
    return false;
}

std::vector<ChangeOfCharacter> Indicators::detectCHoCH(const std::vector<Candle>& candles) {
    std::vector<ChangeOfCharacter> chochs;
    if (candles.size() < 3) return chochs;
    
    MarketStructure currentStructure = NEUTRAL;
    
    for (size_t i = 2; i < candles.size(); ++i) {
        if (isCHoCH(candles, i)) {
            ChangeOfCharacter choch;
            choch.time = candles[i].getTimeOpen();
            choch.price = candles[i].getClose();
            choch.candleIndex = i;
            choch.previousStructure = currentStructure;
            
            // Xác định structure mới
            if (candles[i].getClose() > candles[i - 1].getHigh()) {
                choch.isBullish = true;
                choch.newStructure = BULLISH;
                currentStructure = BULLISH;
            } else {
                choch.isBullish = false;
                choch.newStructure = BEARISH;
                currentStructure = BEARISH;
            }
            
            chochs.push_back(choch);
        } else {
            // Cập nhật current structure
            if (candles[i].getClose() > candles[i - 1].getHigh()) {
                currentStructure = BULLISH;
            } else if (candles[i].getClose() < candles[i - 1].getLow()) {
                currentStructure = BEARISH;
            }
        }
    }
    
    return chochs;
}

// SMT Indicators - Premium/Discount Zones
PremiumDiscountZone Indicators::calculatePremiumDiscount(const std::vector<Candle>& candles, int lookback) {
    PremiumDiscountZone zone;
    
    if (candles.size() < static_cast<size_t>(lookback)) {
        zone.isPremium = false;
        zone.equilibrium = 0.0;
        return zone;
    }
    
    int startIndex = candles.size() - lookback;
    double highest = findHighestHigh(candles, startIndex, candles.size() - 1);
    double lowest = findLowestLow(candles, startIndex, candles.size() - 1);
    
    zone.equilibrium = (highest + lowest) / 2.0;
    zone.top = highest;
    zone.bottom = lowest;
    zone.timeStart = candles[startIndex].getTimeOpen();
    zone.timeEnd = candles.back().getTimeClose();
    
    // Premium = trên equilibrium, Discount = dưới equilibrium
    double currentPrice = candles.back().getClose();
    zone.isPremium = (currentPrice > zone.equilibrium);
    
    return zone;
}

bool Indicators::isInPremiumZone(double price, const PremiumDiscountZone& zone) {
    return price > zone.equilibrium && price <= zone.top;
}

bool Indicators::isInDiscountZone(double price, const PremiumDiscountZone& zone) {
    return price < zone.equilibrium && price >= zone.bottom;
}

// SMT Indicators - Equilibrium/Imbalance
Equilibrium Indicators::calculateEquilibrium(const std::vector<Candle>& candles, int lookback) {
    Equilibrium eq;
    
    if (candles.size() < static_cast<size_t>(lookback)) {
        eq.isValid = false;
        return eq;
    }
    
    int startIndex = candles.size() - lookback;
    double highest = findHighestHigh(candles, startIndex, candles.size() - 1);
    double lowest = findLowestLow(candles, startIndex, candles.size() - 1);
    
    eq.price = (highest + lowest) / 2.0;
    eq.upperBound = highest;
    eq.lowerBound = lowest;
    eq.timeStart = candles[startIndex].getTimeOpen();
    eq.timeEnd = candles.back().getTimeClose();
    eq.isValid = true;
    
    return eq;
}

bool Indicators::isInEquilibrium(double price, const Equilibrium& eq, double tolerance) {
    if (!eq.isValid) return false;
    
    double range = eq.upperBound - eq.lowerBound;
    double toleranceValue = range * tolerance;
    
    return std::abs(price - eq.price) <= toleranceValue;
}

bool Indicators::isImbalanced(const std::vector<Candle>& candles, int index, const Equilibrium& eq) {
    if (!eq.isValid || index < 0 || index >= static_cast<int>(candles.size())) {
        return false;
    }
    
    const Candle& candle = candles[index];
    double price = candle.getClose();
    
    // Imbalance: giá cách xa equilibrium
    double distanceFromEq = std::abs(price - eq.price);
    double range = eq.upperBound - eq.lowerBound;
    
    // Nếu giá cách equilibrium > 30% của range, coi là imbalanced
    return (distanceFromEq / range) > 0.3;
}

// SMT Indicators - Liquidity Sweep
std::vector<LiquiditySweep> Indicators::detectLiquiditySweeps(const std::vector<Candle>& candles) {
    std::vector<LiquiditySweep> sweeps;
    if (candles.size() < 5) return sweeps;
    
    // Tìm các liquidity zones trước
    std::vector<LiquidityZone> liquidityZones = detectLiquidity(candles);
    
    for (size_t i = 2; i < candles.size() - 2; ++i) {
        const Candle& curr = candles[i];
        
        // Kiểm tra sellside sweep: break high rồi reverse
        for (const auto& zone : liquidityZones) {
            if (zone.isSellside && !zone.isTaken) {
                // Price break above zone rồi reverse
                if (curr.getHigh() > zone.price && curr.getClose() < zone.price) {
                    LiquiditySweep sweep;
                    sweep.time = curr.getTimeOpen();
                    sweep.price = zone.price;
                    sweep.sweptPrice = zone.price;
                    sweep.isSellside = true;
                    sweep.candleIndex = i;
                    sweep.isValid = isValidSweep(sweep, candles, 5);
                    sweeps.push_back(sweep);
                }
            }
        }
        
        // Kiểm tra buyside sweep: break low rồi reverse
        for (const auto& zone : liquidityZones) {
            if (!zone.isSellside && !zone.isTaken) {
                // Price break below zone rồi reverse
                if (curr.getLow() < zone.price && curr.getClose() > zone.price) {
                    LiquiditySweep sweep;
                    sweep.time = curr.getTimeOpen();
                    sweep.price = zone.price;
                    sweep.sweptPrice = zone.price;
                    sweep.isSellside = false;
                    sweep.candleIndex = i;
                    sweep.isValid = isValidSweep(sweep, candles, 5);
                    sweeps.push_back(sweep);
                }
            }
        }
    }
    
    return sweeps;
}

std::vector<LiquiditySweep> Indicators::detectSellsideSweeps(const std::vector<Candle>& candles) {
    std::vector<LiquiditySweep> allSweeps = detectLiquiditySweeps(candles);
    std::vector<LiquiditySweep> sellsideSweeps;
    for (const auto& sweep : allSweeps) {
        if (sweep.isSellside) {
            sellsideSweeps.push_back(sweep);
        }
    }
    return sellsideSweeps;
}

std::vector<LiquiditySweep> Indicators::detectBuysideSweeps(const std::vector<Candle>& candles) {
    std::vector<LiquiditySweep> allSweeps = detectLiquiditySweeps(candles);
    std::vector<LiquiditySweep> buysideSweeps;
    for (const auto& sweep : allSweeps) {
        if (!sweep.isSellside) {
            buysideSweeps.push_back(sweep);
        }
    }
    return buysideSweeps;
}

bool Indicators::isValidSweep(const LiquiditySweep& sweep, const std::vector<Candle>& candles, int lookback) {
    if (sweep.candleIndex + lookback >= static_cast<int>(candles.size())) {
        return false;
    }
    
    // Kiểm tra xem sau khi sweep có reversal không
    const Candle& sweepCandle = candles[sweep.candleIndex];
    
    if (sweep.isSellside) {
        // Sau sellside sweep, giá nên reverse xuống
        for (int i = sweep.candleIndex + 1; i < sweep.candleIndex + lookback && i < static_cast<int>(candles.size()); ++i) {
            if (candles[i].getClose() < sweepCandle.getClose()) {
                return true;
            }
        }
    } else {
        // Sau buyside sweep, giá nên reverse lên
        for (int i = sweep.candleIndex + 1; i < sweep.candleIndex + lookback && i < static_cast<int>(candles.size()); ++i) {
            if (candles[i].getClose() > sweepCandle.getClose()) {
                return true;
            }
        }
    }
    
    return false;
}

// SMT Indicators - Inducement (Fake Breakout)
std::vector<Inducement> Indicators::detectInducements(const std::vector<Candle>& candles) {
    std::vector<Inducement> inducements;
    if (candles.size() < 5) return inducements;
    
    for (size_t i = 2; i < candles.size() - 2; ++i) {
        const Candle& curr = candles[i];
        const Candle& prev = candles[i - 1];
        
        // Tìm previous high/low
        double prevHigh = findHighestHigh(candles, std::max(0, static_cast<int>(i) - 10), i - 1);
        double prevLow = findLowestLow(candles, std::max(0, static_cast<int>(i) - 10), i - 1);
        
        // Bullish inducement: Break above previous high rồi reverse
        if (curr.getHigh() > prevHigh && curr.getClose() < prevHigh) {
            Inducement ind;
            ind.time = curr.getTimeOpen();
            ind.price = prevHigh;
            ind.breakoutPrice = curr.getHigh();
            ind.isBullish = true;
            ind.candleIndex = i;
            ind.isConfirmed = false;
            inducements.push_back(ind);
        }
        
        // Bearish inducement: Break below previous low rồi reverse
        if (curr.getLow() < prevLow && curr.getClose() > prevLow) {
            Inducement ind;
            ind.time = curr.getTimeOpen();
            ind.price = prevLow;
            ind.breakoutPrice = curr.getLow();
            ind.isBullish = false;
            ind.candleIndex = i;
            ind.isConfirmed = false;
            inducements.push_back(ind);
        }
    }
    
    // Confirm inducements
    confirmInducements(inducements, candles);
    
    return inducements;
}

bool Indicators::isInducement(const std::vector<Candle>& candles, int index, double threshold) {
    if (index < 2 || index >= static_cast<int>(candles.size()) - 2) return false;
    
    const Candle& curr = candles[index];
    double prevHigh = findHighestHigh(candles, std::max(0, index - 10), index - 1);
    double prevLow = findLowestLow(candles, std::max(0, index - 10), index - 1);
    
    // Breakout rồi reverse trong cùng nến hoặc nến sau
    if (curr.getHigh() > prevHigh * (1 + threshold) && curr.getClose() < prevHigh) {
        return true;
    }
    
    if (curr.getLow() < prevLow * (1 - threshold) && curr.getClose() > prevLow) {
        return true;
    }
    
    return false;
}

void Indicators::confirmInducements(std::vector<Inducement>& inducements, const std::vector<Candle>& candles) {
    for (auto& ind : inducements) {
        if (ind.candleIndex + 3 >= static_cast<int>(candles.size())) continue;
        
        const Candle& breakoutCandle = candles[ind.candleIndex];
        
        if (ind.isBullish) {
            // Bullish inducement: Breakout lên rồi reverse xuống
            // Kiểm tra 3 nến sau có close dưới breakout price không
            bool reversed = false;
            for (int i = ind.candleIndex + 1; i < ind.candleIndex + 4 && i < static_cast<int>(candles.size()); ++i) {
                if (candles[i].getClose() < ind.breakoutPrice) {
                    reversed = true;
                    ind.reversalPrice = candles[i].getClose();
                    break;
                }
            }
            ind.isConfirmed = reversed;
        } else {
            // Bearish inducement: Breakout xuống rồi reverse lên
            // Kiểm tra 3 nến sau có close trên breakout price không
            bool reversed = false;
            for (int i = ind.candleIndex + 1; i < ind.candleIndex + 4 && i < static_cast<int>(candles.size()); ++i) {
                if (candles[i].getClose() > ind.breakoutPrice) {
                    reversed = true;
                    ind.reversalPrice = candles[i].getClose();
                    break;
                }
            }
            ind.isConfirmed = reversed;
        }
    }
}

