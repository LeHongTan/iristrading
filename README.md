# IrisTrading

Production-grade RL trading system for multi-position portfolio trading across Bybit perpetuals using reinforcement learning with PPO.

## Features

### 🎯 Core Capabilities
- **Multi-Symbol Portfolio Trading**: Simultaneous positions across 5 Bybit perpetuals (BTC, ETH, SOL, BNB, XRP)
- **No-Leak Execution**: Time flows forward - decisions at step `t` execute at `open(t+1)`
- **Dynamic Position Sizing**: Scales with equity growth, capped to mitigate slippage
- **Production-Grade RL**: Sequence-based PPO with GRU encoder for temporal patterns
- **Mark-to-Market Accounting**: Real-time equity tracking with taker fees and slippage

### 🧠 Machine Learning Model
- **Architecture**: Multi-symbol actor-critic with GRU-based temporal encoding
- **Input**: 256-step sequences per symbol (configurable to 512)
- **Output**: 
  - Direction: Categorical (HOLD/LONG/SHORT) per symbol
  - Size: Continuous Beta distribution [0,1] per symbol
- **Training**: PPO with GAE, supports MPS/CUDA/CPU

### 💰 Position Sizing Strategy
Dynamic sizing schedule that:
- Grows aggressively from small capital (2% risk)
- Reduces percentage sizing as equity increases
- Hard caps notional beyond thresholds to limit slippage

Example schedule:
```
1x equity  -> 2.0% risk, $2k max per position
2x equity  -> 1.5% risk, $5k max per position
5x equity  -> 1.0% risk, $10k max per position
10x equity -> 0.5% risk, $15k max per position (hard cap)
```

## Installation

### Prerequisites
- Rust 1.70+ 
- Python 3.12+
- PyTorch (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/LeHongTan/iristrading.git
cd iristrading
```

2. **Set up Python environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch numpy
```

3. **Build Rust binary**
```bash
cargo build --release
```

## Usage

### Training Mode

Train the RL agent on historical data:

```bash
# Activate Python venv first (required for PyTorch)
source .venv/bin/activate

# Run training
VIRTUAL_ENV=$PWD/.venv cargo run --release -- --mode train
```

**Training Configuration**: Edit `config.toml` to customize:
- Symbols list and timeframe
- Sequence length (256 or 512)
- Model architecture (hidden dims, layers)
- Training hyperparameters (learning rate, PPO clips, etc.)
- Position sizing schedule
- Fees and slippage parameters

**Checkpoints**: Saved to `checkpoints/` directory every N episodes (configurable).

### Backtest Mode

Test trading strategy on historical data:

```bash
cargo run --release -- --mode backtest --symbol BTCUSDT --balance 1000
```

### Live Trading Mode

Connect to Bybit WebSocket for live trading (single symbol):

```bash
cargo run --release -- --mode live --symbol BTCUSDT --balance 1000
```

## Configuration

### Main Config File: `config.toml`

```toml
[symbols]
list = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
timeframe = "5m"

[model]
sequence_length = 256  # Can increase to 512
hidden_dim = 256
features_per_symbol = 20

[training]
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
epochs_per_update = 4
batch_size = 64
update_interval = 2048  # Steps between PPO updates
max_episodes = 1000
save_interval = 10

[execution]
taker_fee = 0.00055  # 0.055% Bybit taker fee
base_slippage = 0.0001
size_impact = 0.00001  # Per $1000 notional

[position_sizing]
base_risk_pct = 0.02
sizing_schedule = [
    [1.0, 0.020, 2000.0],
    [2.0, 0.015, 5000.0],
    [5.0, 0.010, 10000.0],
    [10.0, 0.005, 15000.0],
]

[backtest]
initial_balance = 1000.0
max_candles_per_symbol = 5000
```

## Architecture

### No-Leak Execution Model

The system enforces strict temporal ordering to prevent look-ahead bias:

1. **Observation (t)**: Agent observes data up to and including candle `t`
2. **Decision (t)**: Policy outputs action based on observation at `t`
3. **Execution (t+1)**: Action executes at `open(t+1)` 
4. **Mark-to-Market (t+1)**: Reward calculated using `close(t+1)`

This ensures the agent never has access to future information.

### Training Loop

```
For each episode:
  1. Reset portfolio
  2. For each timestep t (after warmup):
     a. Build state: sequences up to t for all symbols
     b. Get action from policy (direction + size per symbol)
     c. Execute at open(t+1) with fees/slippage
     d. Calculate reward: Δequity from t to t+1
     e. Store transition
     f. Periodic PPO update (every N steps)
  3. Close all positions at episode end
  4. Save checkpoint
```

### Multi-Symbol State Representation

Each state consists of:
- **Symbol sequences**: 256 x 20 features per symbol (5 symbols total)
- **Portfolio features**: Equity ratio, position utilization, absolute equity

Features per timestep include: OHLCV, momentum, volatility, ICT patterns (FVG, OB), RSI, Bollinger Bands.

## Development

### Running Tests

```bash
cargo test
```

### Project Structure

```
iristrading/
├── src/
│   ├── main.rs           # CLI and mode dispatch
│   ├── config.rs         # Configuration management
│   ├── data_loader.rs    # Multi-symbol data alignment
│   ├── portfolio.rs      # Portfolio management
│   ├── training.rs       # Training engine
│   ├── ict.rs            # ICT pattern detection
│   ├── risk.rs           # Risk management
│   └── database.rs       # SQLite persistence
├── python/
│   ├── brain.py          # Legacy single-symbol agent
│   └── brain_multisymbol.py  # Multi-symbol PPO agent
├── config.toml           # Main configuration
└── Cargo.toml            # Rust dependencies
```

## Monitoring Training

Training logs show:
- Episode number and progress
- Return % and final equity per episode
- Number of trades executed
- Policy/value loss metrics during updates
- Checkpoint save notifications

Example output:
```
[INFO] Episode 1/1000: Return=5.23%, Final Equity=$1052.30, Trades=12
[DEBUG] Training update at step 2048: policy_loss=0.0234, value_loss=0.0156
[INFO] Checkpoint saved: checkpoints/model_episode_10.pt
```

## Performance Considerations

- **Sequence Length**: Longer sequences (512) capture more temporal context but require more memory
- **Batch Size**: Larger batches (128) improve stability but slow down updates
- **Update Interval**: More frequent updates (1024) provide faster learning but may be less stable
- **Device**: MPS (Apple Silicon) or CUDA provide 10-100x speedup over CPU

## Troubleshooting

### "VIRTUAL_ENV not set" Warning
Make sure to activate the Python virtual environment before running:
```bash
source .venv/bin/activate
VIRTUAL_ENV=$PWD/.venv cargo run --release -- --mode train
```

### PyTorch Import Errors
Ensure PyTorch is installed in the activated venv:
```bash
source .venv/bin/activate
pip install torch
```

### Out of Memory
Reduce:
- `sequence_length` (256 -> 128)
- `batch_size` (64 -> 32)
- `hidden_dim` (256 -> 128)

## Future Enhancements

- [ ] Higher timeframe (HTF) features with proper no-leak alignment
- [ ] Multi-agent portfolio (different strategies per symbol)
- [ ] Live data ingestion from Bybit API
- [ ] Real-time dashboard for monitoring
- [ ] Advanced order types (limit, stop-loss)
- [ ] Walk-forward optimization

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome! Please ensure:
1. Code builds without warnings
2. Tests pass
3. No-leak invariants maintained
4. Documentation updated

## Contact

For questions or support, open an issue on GitHub.

---

**User Request**: "AI chạy từ nền lên nóc" ✅
This system implements a complete ground-up ML training pipeline for portfolio trading with production-grade execution semantics.
