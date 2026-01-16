# Test Plan for IrisTrading RL Training Pipeline

## No-Leak Invariant Tests

### Test 1: Temporal Ordering Verification
**Objective**: Ensure decisions at step `t` only use data up to `t`, and execution happens at `open(t+1)`.

**Test Procedure**:
1. Load historical data for all 5 symbols
2. Run training for 1 episode with verbose logging
3. For each step, verify:
   - State observation includes only candles with `timestamp <= t`
   - Action execution price is `open(t+1)`
   - Reward calculation uses `close(t+1)`

**Expected Result**: No candle from step `t+1` or later is accessible when building state at step `t`.

---

### Test 2: Sequence Window Validation
**Objective**: Verify that sequence windows never extend beyond current timestamp.

**Test Procedure**:
1. Set sequence_length = 256
2. At step `t = 100` (< 256), verify sequence is zero-padded at the start
3. At step `t = 300` (> 256), verify sequence contains only candles from `[t-255, t]`
4. Inspect timestamps in sequence to confirm no future leakage

**Expected Result**: Sequence always bounded by current timestamp.

---

### Test 3: HTF (Higher TimeFrame) Alignment
**Objective**: If HTF features added in future, ensure they use only completed candles.

**Test Procedure**:
1. Implement HTF aggregation (e.g., 5m -> 1h)
2. At step `t` on 5m timeframe, verify HTF candle uses only completed lower timeframe candles
3. Example: 1h candle closing at 10:00 should not be accessible at 9:55 on 5m

**Expected Result**: HTF features lag appropriately to prevent look-ahead.

---

## Position Sizing Tests

### Test 4: Dynamic Sizing Schedule
**Objective**: Validate position sizing scales correctly with equity.

**Test Procedure**:
1. Initialize portfolio with $1000
2. Test sizing at different equity levels:
   - $1000 (1x): Should use 2% risk, max $2000 notional
   - $2000 (2x): Should use 1.5% risk, max $5000 notional  
   - $5000 (5x): Should use 1.0% risk, max $10000 notional
   - $20000 (20x): Should use 0.5% risk, max $15000 notional (hard cap)
3. Verify `size_fraction = 1.0` respects the cap
4. Verify `size_fraction = 0.5` reduces notional proportionally

**Expected Result**: Position notional follows sizing schedule and respects caps.

---

### Test 5: Fee and Slippage Calculation
**Objective**: Ensure costs are correctly calculated and deducted from equity.

**Test Procedure**:
1. Open position with notional = $10000
2. Calculate expected fees: $10000 * 0.00055 = $5.50
3. Calculate expected slippage: $10000 * (0.0001 + 0.00001 * 10) = $2.00
4. Total cost = $7.50
5. Verify equity reduces by this amount
6. Close position and verify costs again

**Expected Result**: Equity reflects all transaction costs accurately.

---

## Multi-Symbol Portfolio Tests

### Test 6: Simultaneous Positions
**Objective**: Verify portfolio can hold multiple positions across symbols.

**Test Procedure**:
1. Execute actions to open positions in all 5 symbols
2. Verify all positions exist in portfolio
3. Update prices for all symbols
4. Verify mark-to-market calculates unrealized PnL correctly for all
5. Close one position and verify others remain

**Expected Result**: Portfolio manages up to 5 concurrent positions correctly.

---

### Test 7: Position Direction Changes
**Objective**: Test closing and reversing positions.

**Test Procedure**:
1. Open LONG position in BTCUSDT
2. Issue SHORT action for BTCUSDT
3. Verify:
   - LONG position closed with PnL
   - SHORT position opened
   - Equity updated correctly
   - Fees/slippage applied for both close and open

**Expected Result**: Position reversal executes as close + open with proper accounting.

---

## Training Loop Tests

### Test 8: Episode Completion
**Objective**: Verify training episode runs to completion without errors.

**Test Procedure**:
1. Set max_episodes = 1
2. Run training mode
3. Monitor for:
   - Episode starts
   - Steps progress through timeline
   - Actions sampled from policy
   - Transitions stored
   - Episode completes
   - Final equity reported

**Expected Result**: Episode completes successfully with logged statistics.

---

### Test 9: PPO Update Trigger
**Objective**: Ensure PPO updates occur at correct intervals.

**Test Procedure**:
1. Set update_interval = 100
2. Run training for 1 episode
3. Log when `train_step()` is called
4. Verify calls occur at steps: 100, 200, 300, etc.
5. Verify GAE computation uses correct next_value

**Expected Result**: PPO updates triggered at configured intervals.

---

### Test 10: Checkpoint Save/Load
**Objective**: Verify model checkpoints save and load correctly.

**Test Procedure**:
1. Train for 10 episodes with save_interval = 5
2. Verify checkpoints saved at episodes 5 and 10
3. Load checkpoint from episode 5
4. Verify training_step counter matches
5. Continue training and verify learning resumes

**Expected Result**: Checkpoints restore full training state.

---

## Integration Tests

### Test 11: End-to-End Training Run
**Objective**: Complete training run on sample data.

**Test Procedure**:
1. Generate sample data for all 5 symbols (5000 candles each)
2. Run training with:
   - max_episodes = 3
   - sequence_length = 256
   - warmup_steps = 30
3. Verify:
   - All episodes complete
   - Equity varies across episodes
   - Checkpoints saved
   - No errors or panics

**Expected Result**: Training completes 3 episodes successfully.

---

### Test 12: Config Loading
**Objective**: Verify configuration loads from file correctly.

**Test Procedure**:
1. Create custom config.toml with modified values
2. Set CONFIG_PATH environment variable
3. Run training
4. Verify custom values are used (check logs)

**Expected Result**: Custom configuration loads and applies.

---

## Performance Tests

### Test 13: Memory Usage
**Objective**: Ensure training doesn't leak memory.

**Test Procedure**:
1. Monitor memory usage during training
2. Run for 10 episodes
3. Verify memory stabilizes (doesn't grow unbounded)

**Expected Result**: Memory usage stays bounded.

---

### Test 14: Training Speed
**Objective**: Measure training throughput.

**Test Procedure**:
1. Run training for 1 episode
2. Measure:
   - Steps per second
   - Episodes per hour (estimated)
3. Compare CPU vs MPS/CUDA

**Expected Result**: Reasonable training speed (at least 10 steps/sec on CPU).

---

## Manual Validation Checklist

Before deployment:
- [ ] Trained model achieves positive returns in backtest
- [ ] Position sizing behaves as expected across equity ranges
- [ ] No-leak execution verified by manual inspection of logs
- [ ] Checkpoint save/load tested manually
- [ ] All 5 symbols can be traded simultaneously
- [ ] Fees and slippage costs are realistic
- [ ] Training runs stably for 100+ episodes
- [ ] README instructions are accurate and complete

---

## How to Run Tests

Since unit tests are minimal (Rust binary-only project), most validation is integration-style:

### Validation Run
```bash
# Activate Python environment
source .venv/bin/activate

# Run short training to validate pipeline
VIRTUAL_ENV=$PWD/.venv cargo run --release -- --mode train

# Check logs for:
# - Episode completion
# - Positive and negative returns (agent is learning variance)
# - Checkpoint saves
# - No errors
```

### Inspect Execution Logs
```bash
# Run with RUST_LOG=debug for detailed logging
RUST_LOG=debug VIRTUAL_ENV=$PWD/.venv cargo run --release -- --mode train 2>&1 | tee training.log

# Grep for specific events
grep "Step" training.log | head -20
grep "Episode.*complete" training.log
grep "Checkpoint saved" training.log
```

### Verify No-Leak Manually
Add debug prints to `training.rs` showing:
- Current step `t`
- Max timestamp in state sequences
- Execution timestamp (should be `t+1`)
- Reward timestamp (should be `t+1`)

Confirm execution timestamp always > state timestamp.

---

## Future Test Additions

Once codebase stabilizes:
1. Add proper unit tests by refactoring into library crate
2. Add property-based tests for portfolio invariants
3. Add benchmark suite for performance tracking
4. Add CI/CD pipeline with automated testing
5. Add fuzzing for edge cases in position sizing

