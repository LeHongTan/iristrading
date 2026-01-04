#ifndef SMT_ANALYSIS_H
#define SMT_ANALYSIS_H

#include <vector>
#include <string>
#include "candle.h"

// Structure for Divergence Analysis
enum DivergenceType {
    BULLISH_DIVERGENCE,  // Price tạo lower low nhưng indicator tạo higher low
    BEARISH_DIVERGENCE,  // Price tạo higher high nhưng indicator tạo lower high
    HIDDEN_BULLISH_DIVERGENCE,  // Price tạo higher low nhưng indicator tạo lower low
    HIDDEN_BEARISH_DIVERGENCE   // Price tạo lower high nhưng indicator tạo higher high
};

struct Divergence {
    DivergenceType type;
    long long timeStart;
    long long timeEnd;
    double priceStart;
    double priceEnd;
    double indicatorStart;
    double indicatorEnd;
    int candleIndexStart;
    int candleIndexEnd;
    std::string entity1Name;  // Tên thực thể 1 (thường là Price)
    std::string entity2Name;  // Tên thực thể 2 (Volume, Momentum, RSI, etc.)
    double strength;  // Độ mạnh của divergence (0-1)
};

// Structure for Momentum data point
struct MomentumPoint {
    long long time;
    double value;
    int candleIndex;
};

// Structure for Volume Profile
struct VolumeProfile {
    long long time;
    double volume;
    double price;
    int candleIndex;
};

class SMTAnalysis {
    public:
        // ========== SMT DIVERGENCE (Phân kỳ liên thị trường) ==========
        // So sánh giá của 2 tài sản khác nhau có tương quan thuận/nghịch
        
        // SMT Divergence - So sánh giá của 2 tài sản (có tương quan thuận)
        static std::vector<Divergence> detectSMTDivergence(
            const std::vector<Candle>& asset1Candles,  // Tài sản 1 (ví dụ: BTC)
            const std::vector<Candle>& asset2Candles,   // Tài sản 2 (ví dụ: ETH)
            const std::string& asset1Name = "Asset1",
            const std::string& asset2Name = "Asset2",
            bool isInverseCorrelation = false,  // true nếu tương quan nghịch (như EURUSD vs DXY)
            int lookbackPeriod = 20
        );
        
        // SMT Divergence giữa BTC và ETH (ví dụ phổ biến)
        static std::vector<Divergence> detectBTCETHDivergence(
            const std::vector<Candle>& btcCandles,
            const std::vector<Candle>& ethCandles,
            int lookbackPeriod = 20
        );
        
        // SMT Divergence giữa Currency Pair và Dollar Index (tương quan nghịch)
        static std::vector<Divergence> detectCurrencyDXYDivergence(
            const std::vector<Candle>& currencyCandles,  // EURUSD, GBPUSD, etc.
            const std::vector<Candle>& dxyCandles,       // DXY
            const std::string& currencyName = "Currency",
            int lookbackPeriod = 20
        );
        
        // ========== INDICATOR DIVERGENCE (Phân kỳ chỉ báo) ==========
        // So sánh Giá vs Indicator của cùng một tài sản
        
        // Divergence Detection - Phân kỳ giữa 2 thực thể có tương quan thuận (cho indicators)
        static std::vector<Divergence> detectIndicatorDivergence(
            const std::vector<Candle>& candles,
            const std::vector<double>& entity1Values,  // Thực thể 1 (thường là Price)
            const std::vector<double>& entity2Values,  // Thực thể 2 (Volume, Momentum, etc.)
            const std::string& entity1Name = "Price",
            const std::string& entity2Name = "Indicator",
            int lookbackPeriod = 20
        );
        
        // Price vs Volume Divergence (Indicator Divergence)
        static std::vector<Divergence> detectPriceVolumeDivergence(
            const std::vector<Candle>& candles,
            int lookbackPeriod = 20
        );
        
        // Price vs Momentum Divergence (Indicator Divergence)
        static std::vector<Divergence> detectPriceMomentumDivergence(
            const std::vector<Candle>& candles,
            int momentumPeriod = 14,
            int lookbackPeriod = 20
        );
        
        // Price vs RSI Divergence (Indicator Divergence)
        static std::vector<Divergence> detectPriceRSIDivergence(
            const std::vector<Candle>& candles,
            int rsiPeriod = 14,
            int lookbackPeriod = 20
        );
        
        // Price vs Order Flow Divergence (Indicator Divergence)
        static std::vector<Divergence> detectPriceOrderFlowDivergence(
            const std::vector<Candle>& candles,
            int lookbackPeriod = 20
        );
        
        // Helper functions để tính các indicators
        static std::vector<double> calculateRSI(const std::vector<Candle>& candles, int period = 14);
        static std::vector<double> calculateMomentum(const std::vector<Candle>& candles, int period = 14);
        static std::vector<double> calculateVolumeMA(const std::vector<Candle>& candles, int period = 20);
        static std::vector<double> calculatePriceValues(const std::vector<Candle>& candles); // Close prices
        
        // Tìm swing highs và lows
        static std::vector<MomentumPoint> findSwingHighs(const std::vector<double>& values, const std::vector<Candle>& candles, int lookback = 5);
        static std::vector<MomentumPoint> findSwingLows(const std::vector<double>& values, const std::vector<Candle>& candles, int lookback = 5);
        
        // Tìm swing highs và lows từ candles trực tiếp (cho SMT Divergence)
        static std::vector<MomentumPoint> findSwingHighsFromCandles(const std::vector<Candle>& candles, int lookback = 5);
        static std::vector<MomentumPoint> findSwingLowsFromCandles(const std::vector<Candle>& candles, int lookback = 5);
        
        // Sync candles của 2 tài sản theo timestamp (để so sánh)
        static std::pair<std::vector<Candle>, std::vector<Candle>> syncCandlesByTime(
            const std::vector<Candle>& asset1Candles,
            const std::vector<Candle>& asset2Candles,
            double timeTolerance = 60000.0  // Tolerance in milliseconds (default 1 minute)
        );
        
        // Tính độ mạnh của divergence
        static double calculateDivergenceStrength(const Divergence& divergence);
        
        // Xác nhận divergence
        static bool confirmDivergence(const Divergence& divergence, const std::vector<Candle>& candles, int confirmationBars = 3);
};

#endif

