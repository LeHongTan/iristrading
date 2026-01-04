#ifndef INDICATORS_H
#define INDICATORS_H

#include <vector>
#include "candle.h"

// Structure for Fair Value Gap (FVG)
struct FVG {
    long long timeStart;
    long long timeEnd;
    double top;
    double bottom;
    bool isBullish; // true for bullish FVG, false for bearish FVG
    bool isBroken;
    int candleIndex; // Index of the candle that created this FVG
};

// Structure for Order Block (OB)
struct OrderBlock {
    long long timeStart;
    long long timeEnd;
    double top;
    double bottom;
    bool isBullish; // true for bullish OB, false for bearish OB
    bool isBroken;
    int candleIndex;
    double volume;
};

// Structure for Liquidity Zone
struct LiquidityZone {
    long long timeStart;
    long long timeEnd;
    double price;
    bool isSellside; // true for sellside (above), false for buyside (below)
    bool isTaken;
    int candleIndex;
};

// Structure for Market Structure
enum MarketStructure {
    BULLISH,
    BEARISH,
    NEUTRAL
};

// Structure for Time Killzone
struct Killzone {
    int startHour;
    int startMinute;
    int endHour;
    int endMinute;
    std::string name;
};

// Structure for Change of Character (CHoCH)
struct ChangeOfCharacter {
    long long time;
    double price;
    bool isBullish; // true for bullish CHoCH, false for bearish CHoCH
    int candleIndex;
    MarketStructure previousStructure;
    MarketStructure newStructure;
};

// Structure for Premium/Discount Zone
struct PremiumDiscountZone {
    long long timeStart;
    long long timeEnd;
    double top;
    double bottom;
    bool isPremium; // true for premium zone, false for discount zone
    double equilibrium; // Equilibrium price (midpoint)
};

// Structure for Equilibrium/Imbalance
struct Equilibrium {
    double price;
    long long timeStart;
    long long timeEnd;
    bool isValid;
    double upperBound;
    double lowerBound;
};

// Structure for Liquidity Sweep
struct LiquiditySweep {
    long long time;
    double price;
    bool isSellside; // true for sellside sweep, false for buyside sweep
    int candleIndex;
    bool isValid; // true if followed by reversal
    double sweptPrice; // Price that was swept
};

// Structure for Inducement (Fake Breakout)
struct Inducement {
    long long time;
    double price;
    bool isBullish; // true for bullish inducement (fake breakout up), false for bearish
    int candleIndex;
    bool isConfirmed; // true if price reversed after breakout
    double breakoutPrice;
    double reversalPrice;
};

class Indicators {
    public:
        // FVG (Fair Value Gap) detection
        static std::vector<FVG> detectFVG(const std::vector<Candle>& candles);
        static std::vector<FVG> detectBullishFVG(const std::vector<Candle>& candles);
        static std::vector<FVG> detectBearishFVG(const std::vector<Candle>& candles);
        static bool isFVGBroken(const FVG& fvg, const std::vector<Candle>& candles, int startIndex);
        static void updateFVGStatus(std::vector<FVG>& fvgs, const std::vector<Candle>& candles);
        
        // Order Block detection
        static std::vector<OrderBlock> detectOrderBlocks(const std::vector<Candle>& candles);
        static std::vector<OrderBlock> detectBullishOB(const std::vector<Candle>& candles);
        static std::vector<OrderBlock> detectBearishOB(const std::vector<Candle>& candles);
        static bool isOBBroken(const OrderBlock& ob, const std::vector<Candle>& candles, int startIndex);
        static void updateOBStatus(std::vector<OrderBlock>& obs, const std::vector<Candle>& candles);
        
        // Liquidity detection
        static std::vector<LiquidityZone> detectLiquidity(const std::vector<Candle>& candles);
        static std::vector<LiquidityZone> detectSellsideLiquidity(const std::vector<Candle>& candles);
        static std::vector<LiquidityZone> detectBuysideLiquidity(const std::vector<Candle>& candles);
        static bool isLiquidityTaken(const LiquidityZone& zone, const std::vector<Candle>& candles, int startIndex);
        static void updateLiquidityStatus(std::vector<LiquidityZone>& zones, const std::vector<Candle>& candles);
        
        // Market Structure
        static MarketStructure detectMarketStructure(const std::vector<Candle>& candles, int lookback = 20);
        static bool detectMSS(const std::vector<Candle>& candles, int index); // Market Structure Shift
        static bool detectBOS(const std::vector<Candle>& candles, int index); // Break of Structure
        
        // Time Killzone
        static std::vector<Killzone> getKillzones();
        static bool isInKillzone(long long timestamp);
        static std::string getCurrentKillzone(long long timestamp);
        
        // SMT (Smart Money Tools) Indicators
        // Change of Character (CHoCH)
        static std::vector<ChangeOfCharacter> detectCHoCH(const std::vector<Candle>& candles);
        static bool isCHoCH(const std::vector<Candle>& candles, int index);
        
        // Premium/Discount Zones
        static PremiumDiscountZone calculatePremiumDiscount(const std::vector<Candle>& candles, int lookback = 50);
        static bool isInPremiumZone(double price, const PremiumDiscountZone& zone);
        static bool isInDiscountZone(double price, const PremiumDiscountZone& zone);
        
        // Equilibrium/Imbalance
        static Equilibrium calculateEquilibrium(const std::vector<Candle>& candles, int lookback = 20);
        static bool isInEquilibrium(double price, const Equilibrium& eq, double tolerance = 0.001);
        static bool isImbalanced(const std::vector<Candle>& candles, int index, const Equilibrium& eq);
        
        // Liquidity Sweep
        static std::vector<LiquiditySweep> detectLiquiditySweeps(const std::vector<Candle>& candles);
        static std::vector<LiquiditySweep> detectSellsideSweeps(const std::vector<Candle>& candles);
        static std::vector<LiquiditySweep> detectBuysideSweeps(const std::vector<Candle>& candles);
        static bool isValidSweep(const LiquiditySweep& sweep, const std::vector<Candle>& candles, int lookback = 5);
        
        // Inducement (Fake Breakout)
        static std::vector<Inducement> detectInducements(const std::vector<Candle>& candles);
        static bool isInducement(const std::vector<Candle>& candles, int index, double threshold = 0.002);
        static void confirmInducements(std::vector<Inducement>& inducements, const std::vector<Candle>& candles);
        
        // Helper functions
        static bool isBullishCandle(const Candle& candle);
        static bool isBearishCandle(const Candle& candle);
        static double getCandleBody(const Candle& candle);
        static bool hasWick(const Candle& candle, double minWickRatio = 0.3);
        static double calculateRange(const std::vector<Candle>& candles, int startIndex, int endIndex);
        static double findHighestHigh(const std::vector<Candle>& candles, int startIndex, int endIndex);
        static double findLowestLow(const std::vector<Candle>& candles, int startIndex, int endIndex);
};

#endif