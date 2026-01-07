¬# Bybit Trading Simulator

Môi trường giả lập giao dịch Bybit với các điều kiện khắc nghiệt hơn thực tế để training và backtesting strategies.

## Tính năng

### 1. **Phí giao dịch (Trading Fees)**
- Maker fee: 0.02% (cao hơn thực tế 0.01%)
- Taker fee: 0.08% (cao hơn thực tế 0.06%)

### 2. **Funding Rate**
- Base funding rate: 0.03% (cao hơn thực tế -0.01% đến 0.01%)
- Funding mỗi 8 giờ
- Có biến động ngẫu nhiên

### 3. **Slippage (Trượt giá)**
- Base slippage: 0.05% (cao hơn thực tế)
- Tăng 3x khi volume thấp
- Market impact cho orders lớn

### 4. **Quét 2 đầu (Both Sides Hit)**
- Nếu stop loss và take profit đều bị quét trong cùng một candle → tính là thua
- Phạt thêm 0.1% phí

### 5. **Liquidation**
- Maintenance margin: 0.5%
- Initial margin: 1%
- Tự động liquidate khi margin không đủ

## Cấu trúc

```
simulator/
├── include/
│   ├── order.h          # Định nghĩa Order, Position, Account
│   ├── bybit_simulator.h # Mô phỏng môi trường Bybit
│   ├── trading_engine.h  # Xử lý orders và positions
│   └── backtest_engine.h # Engine chạy backtest
└── src/
    ├── bybit_simulator.cpp
    ├── trading_engine.cpp
    └── backtest_engine.cpp
```

## Cách sử dụng

### 1. Tạo Strategy

```cpp
#include "backtest_engine.h"
#include "indicators.h"

void myStrategy(TradingEngine& engine, const std::string& symbol,
                const Candle& candle, long long timestamp,
                const std::vector<Candle>& history) {
    
    if (history.size() < 20) return;
    
    // Tính indicators
    std::vector<FVG> fvgs = Indicators::detectFVG(history);
    std::vector<OrderBlock> obs = Indicators::detectOrderBlocks(history);
    
    // Logic trading
    if (!engine.hasPosition(symbol)) {
        // Tìm entry signal
        if (!obs.empty() && obs.back().isBullish && !obs.back().isBroken) {
            double entryPrice = candle.getClose();
            double stopLoss = obs.back().bottom * 0.99;
            double takeProfit = entryPrice + (entryPrice - stopLoss) * 2.0;
            
            engine.placeOrder(symbol, MARKET, BUY, LONG, 0.01, 
                            entryPrice, stopLoss, takeProfit);
        }
    }
}
```

### 2. Chạy Backtest

```cpp
#include "backtest_engine.h"

int main() {
    // Tạo config (có thể tùy chỉnh)
    SimulatorConfig config;
    config.makerFeeRate = 0.0002;
    config.takerFeeRate = 0.0008;
    config.fundingRateBase = 0.0003;
    
    // Tạo backtest engine
    BacktestEngine engine(config);
    engine.setInitialBalance(10000.0);
    
    // Load dữ liệu
    engine.loadDataFromCSV("data/BTCUSDT_15.csv");
    
    // Set strategy
    engine.setStrategy(myStrategy);
    
    // Chạy backtest
    BacktestResult result = engine.runAll("BTCUSDT_15");
    
    // Xem kết quả
    std::cout << "Total Return: " << result.totalReturnPercent << "%" << std::endl;
    std::cout << "Win Rate: " << result.winRate << "%" << std::endl;
    std::cout << "Max Drawdown: " << result.maxDrawdownPercent << "%" << std::endl;
    std::cout << "Both Sides Hit: " << result.bothSidesHitCount << std::endl;
    
    return 0;
}
```

### 3. Tùy chỉnh Config

```cpp
SimulatorConfig config;

// Tăng độ khắc nghiệt
config.makerFeeRate = 0.0003;  // 0.03%
config.takerFeeRate = 0.001;   // 0.1%
config.fundingRateBase = 0.0005;  // 0.05%
config.slippageBase = 0.001;   // 0.1%
config.lowVolumeSlippageMultiplier = 5.0;  // 5x khi volume thấp
config.penalizeBothSidesHit = true;
config.bothSidesHitPenalty = 0.002;  // Phạt 0.2%
```

## Kết quả Backtest

`BacktestResult` bao gồm:
- `totalReturn`, `totalReturnPercent` - Tổng lợi nhuận
- `realizedPnL`, `unrealizedPnL` - PnL đã thực hiện/chưa thực hiện
- `totalFees` - Tổng phí đã trả
- `totalTrades`, `winningTrades`, `losingTrades` - Số lượng trades
- `winRate` - Tỷ lệ thắng
- `profitFactor` - Hệ số lợi nhuận
- `maxDrawdown`, `maxDrawdownPercent` - Drawdown tối đa
- `sharpeRatio` - Sharpe ratio
- `bothSidesHitCount` - Số lần quét 2 đầu
- `equityCurve` - Đường cong vốn

## Lưu ý

1. **Môi trường khắc nghiệt hơn thực tế**: Phí cao hơn, slippage dễ hơn, funding cao hơn
2. **Quét 2 đầu = Thua**: Nếu stop loss và take profit đều bị quét → tính là thua
3. **Volume thấp**: Slippage tăng 3-5x khi volume thấp
4. **Market impact**: Orders lớn sẽ ảnh hưởng đến giá

## Training Model

Simulator này có thể được sử dụng để:
- Backtest strategies
- Training reinforcement learning models
- Paper trading với điều kiện khắc nghiệt
- Stress testing strategies

