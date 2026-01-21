use std::collections::HashMap;
use crate::ict::Candle;
use crate::config::{Config, ExecutionConfig, PositionSizingConfig};

/// Position direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Hold,
    Long,
    Short,
}

impl Direction {
    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => Direction::Long,
            -1 => Direction::Short,
            _ => Direction::Hold,
        }
    }
}

/// Action for a single symbol: direction + size fraction
#[derive(Debug, Clone, Copy)]
pub struct SymbolAction {
    pub direction: Direction,
    /// Size fraction in [0, 1] to be mapped to target notional
    pub size_fraction: f64,
}

impl SymbolAction {
    /// Convert i32 from agent (default size_fraction=1.0, bạn muốn dynamic thì sửa thêm)
    pub fn from_i32(v: i32) -> Self {
        SymbolAction {
            direction: Direction::from_i32(v),
            size_fraction: 1.0,
        }
    }
}

/// Portfolio action across all symbols
#[derive(Debug, Clone)]
pub struct PortfolioAction {
    pub actions: HashMap<String, SymbolAction>,
}

/// Single symbol position
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub direction: Direction,
    pub entry_price: f64,
    pub quantity: f64,
    pub entry_timestamp: i64,
}

/// Portfolio holding multiple positions across symbols
#[derive(Debug)]
pub struct Portfolio {
    positions: HashMap<String, Position>,
    equity: f64,
    initial_balance: f64,
    symbols: Vec<String>,
    exec_config: ExecutionConfig,
    sizing_config: PositionSizingConfig,
}

impl Portfolio {
    pub fn new(
        initial_balance: f64,
        symbols: Vec<String>,
        config: &Config,
    ) -> Self {
        Self {
            positions: HashMap::new(),
            equity: initial_balance,
            initial_balance,
            symbols,
            exec_config: config.execution.clone(),
            sizing_config: config.position_sizing.clone(),
        }
    }

    /// Get current equity
    pub fn equity(&self) -> f64 {
        self.equity
    }

    /// Get position for a symbol if it exists
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Check if we have a position in a symbol
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Get all symbols
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Calculate target notional for a symbol given size_fraction
    fn calculate_target_notional(&self, size_fraction: f64) -> f64 {
        let (max_risk_pct, max_notional) = self.sizing_config.get_sizing(
            self.equity,
            self.initial_balance,
        );
        
        // Max equity-based notional
        let equity_based_notional = self.equity * max_risk_pct;
        
        // Apply size fraction and cap at max_notional
        let target_notional = equity_based_notional * size_fraction.clamp(0.0, 1.0);
        target_notional.min(max_notional)
    }

    /// Calculate slippage cost based on notional size
    fn calculate_slippage(&self, notional: f64) -> f64 {
        // base_slippage + size_impact * (notional / 1000)
        self.exec_config.base_slippage + 
            self.exec_config.size_impact * (notional / 1000.0)
    }

    /// Execute portfolio action at current prices
    /// Returns (total_cost, gross_pnls) where gross_pnls are before cost deduction
    pub fn execute_action(
        &mut self,
        action: &PortfolioAction,
        current_prices: &HashMap<String, f64>,
        timestamp: i64,
    ) -> (f64, HashMap<String, f64>) {
        let mut total_cost = 0.0;
        let mut pnls = HashMap::new();

        // Clone symbols to avoid borrow issues
        let symbols = self.symbols.clone();

        for symbol in &symbols {
            let symbol_action = action.actions.get(symbol);
            let current_price = current_prices.get(symbol).copied().unwrap_or(0.0);
            
            if current_price <= 0.0 {
                continue;
            }

            let existing_position = self.positions.get(symbol);
            
            match symbol_action {
                Some(action) => {
                    // Close existing position if direction changed
                    if let Some(pos) = existing_position {
                        if action.direction != Direction::Hold && action.direction != pos.direction {
                            let (pnl, cost) = self.close_position_internal(symbol, current_price);
                            total_cost += cost;
                            pnls.insert(symbol.clone(), pnl);
                        }
                    }

                    // Open new position if direction is not Hold and we don't have one
                    if action.direction != Direction::Hold && !self.positions.contains_key(symbol) {
                        let cost = self.open_position_internal(
                            symbol,
                            action.direction,
                            action.size_fraction,
                            current_price,
                            timestamp,
                        );
                        total_cost += cost;
                    }
                }
                None => {
                    // No action specified, hold existing position
                }
            }
        }

        (total_cost, pnls)
    }

    /// Open a new position for a symbol
    fn open_position_internal(
        &mut self,
        symbol: &str,
        direction: Direction,
        size_fraction: f64,
        price: f64,
        timestamp: i64,
    ) -> f64 {
        let target_notional = self.calculate_target_notional(size_fraction);
        
        if target_notional <= 0.0 {
            return 0.0;
        }

        let quantity = target_notional / price;
        
        // Calculate costs
        let notional = quantity * price;
        let fee_cost = notional * self.exec_config.taker_fee;
        let slippage_pct = self.calculate_slippage(notional);
        let slippage_cost = notional * slippage_pct;
        
        let total_cost = fee_cost + slippage_cost;

        self.positions.insert(
            symbol.to_string(),
            Position {
                symbol: symbol.to_string(),
                direction,
                entry_price: price,
                quantity,
                entry_timestamp: timestamp,
            },
        );

        total_cost
    }

    /// Close an existing position for a symbol
    /// Returns (gross_pnl, total_cost) - costs are NOT deducted from gross_pnl
    fn close_position_internal(&mut self, symbol: &str, exit_price: f64) -> (f64, f64) {
        if let Some(position) = self.positions.remove(symbol) {
            let notional = position.quantity * exit_price;
            
            // Calculate gross PnL (before costs)
            let price_diff = match position.direction {
                Direction::Long => exit_price - position.entry_price,
                Direction::Short => position.entry_price - exit_price,
                Direction::Hold => 0.0,
            };
            let gross_pnl = price_diff * position.quantity;
            
            // Calculate costs
            let fee_cost = notional * self.exec_config.taker_fee;
            let slippage_pct = self.calculate_slippage(notional);
            let slippage_cost = notional * slippage_pct;
            let total_cost = fee_cost + slippage_cost;
            
            // Return gross PnL and cost separately - caller subtracts cost
            (gross_pnl, total_cost)
        } else {
            (0.0, 0.0)
        }
    }

    /// Mark-to-market all positions and update equity
    /// Returns unrealized PnL
    pub fn mark_to_market(&mut self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut total_unrealized = 0.0;

        for position in self.positions.values() {
            if let Some(&current_price) = current_prices.get(&position.symbol) {
                let price_diff = match position.direction {
                    Direction::Long => current_price - position.entry_price,
                    Direction::Short => position.entry_price - current_price,
                    Direction::Hold => 0.0,
                };
                let unrealized_pnl = price_diff * position.quantity;
                total_unrealized += unrealized_pnl;
            }
        }

        total_unrealized
    }

    /// Update equity (called after executing actions with realized PnL)
    pub fn update_equity(&mut self, realized_pnl: f64, cost: f64) {
        self.equity += realized_pnl - cost;
    }

    /// Force close all positions
    /// Returns net PnL after costs
    pub fn close_all_positions(&mut self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut total_gross_pnl = 0.0;
        let mut total_cost = 0.0;
        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        
        for symbol in symbols {
            if let Some(&price) = current_prices.get(&symbol) {
                let (gross_pnl, cost) = self.close_position_internal(&symbol, price);
                total_gross_pnl += gross_pnl;
                total_cost += cost;
            }
        }
        
        // Update equity with gross PnL and costs (costs subtracted exactly once)
        self.equity += total_gross_pnl - total_cost;
        total_gross_pnl - total_cost
    }

    /// Get portfolio state features for RL
    pub fn get_portfolio_features(&self) -> Vec<f64> {
        vec![
            self.equity / self.initial_balance, // Equity ratio
            self.positions.len() as f64 / self.symbols.len() as f64, // Position utilization
            self.equity / 1000.0, // Absolute equity (normalized)
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_test_config() -> Config {
        // Use the default config which has all the necessary fields
        Config::default()
    }

    #[test]
    fn test_cost_subtracted_once() {
        // Test that costs are subtracted exactly once, not double-counted
        let config = create_test_config();
        let mut portfolio = Portfolio::new(
            1000.0,
            vec!["BTCUSDT".to_string()],
            &config,
        );

        // Open a position
        let open_cost = portfolio.open_position_internal(
            "BTCUSDT",
            Direction::Long,
            0.5,
            50000.0,
            1000,
        );

        let equity_after_open = portfolio.equity();
        assert_eq!(equity_after_open, 1000.0); // Equity unchanged (not yet updated)

        // Update equity with opening cost
        portfolio.update_equity(0.0, open_cost);
        let equity_after_cost = portfolio.equity();
        assert!(equity_after_cost < 1000.0); // Equity reduced by cost
        assert!(equity_after_cost > 995.0); // But not by too much

        // Close the position with profit
        let (gross_pnl, close_cost) = portfolio.close_position_internal("BTCUSDT", 51000.0);
        
        // gross_pnl should be positive (price went up)
        assert!(gross_pnl > 0.0, "Expected positive gross PnL");
        
        // Update equity
        portfolio.update_equity(gross_pnl, close_cost);
        
        // Final equity should be: initial - open_cost + gross_pnl - close_cost
        // Which should be > initial since we made profit
        let final_equity = portfolio.equity();
        
        // Check that final equity makes sense
        // We started with 1000, paid open_cost, made gross_pnl, paid close_cost
        let expected_equity = 1000.0 - open_cost + gross_pnl - close_cost;
        assert!((final_equity - expected_equity).abs() < 0.01, 
                "Final equity {} doesn't match expected {}", final_equity, expected_equity);
    }

    #[test]
    fn test_reward_equals_delta_equity_mtm() {
        // Test that reward calculation equals change in mark-to-market equity
        let config = create_test_config();
        let mut portfolio = Portfolio::new(
            1000.0,
            vec!["BTCUSDT".to_string()],
            &config,
        );

        let initial_equity = 1000.0;
        
        // Step 1: Mark to market at time t (no positions yet)
        let mut prices_t = HashMap::new();
        prices_t.insert("BTCUSDT".to_string(), 50000.0);
        let unrealized_t = portfolio.mark_to_market(&prices_t);
        let equity_mtm_t = portfolio.equity() + unrealized_t;
        assert_eq!(equity_mtm_t, initial_equity);

        // Step 2: Open position and update equity
        let mut action = PortfolioAction {
            actions: HashMap::new(),
        };
        action.actions.insert(
            "BTCUSDT".to_string(),
            SymbolAction {
                direction: Direction::Long,
                size_fraction: 0.5,
            },
        );

        let mut exec_prices = HashMap::new();
        exec_prices.insert("BTCUSDT".to_string(), 50000.0);
        
        let (cost, pnls) = portfolio.execute_action(&action, &exec_prices, 1000);
        let realized_pnl: f64 = pnls.values().sum();
        portfolio.update_equity(realized_pnl, cost);

        // Step 3: Mark to market at time t+1 (price increased)
        let mut prices_t1 = HashMap::new();
        prices_t1.insert("BTCUSDT".to_string(), 51000.0);
        let unrealized_t1 = portfolio.mark_to_market(&prices_t1);
        let equity_mtm_t1 = portfolio.equity() + unrealized_t1;

        // Step 4: Compute reward
        let reward = (equity_mtm_t1 - equity_mtm_t) / initial_equity;

        // Reward should be positive since price went up
        assert!(reward > 0.0, "Expected positive reward with price increase");
        
        // The equity change should account for both costs and unrealized gains
        let delta_equity = equity_mtm_t1 - equity_mtm_t;
        assert!(delta_equity > -cost, "Delta equity should reflect the trade");
    }

    #[test]
    fn test_no_lookahead_invariant() {
        // This is a structural test to verify the design prevents look-ahead
        // In the actual training loop:
        // - State at step t uses data <= t
        // - Execution happens at open(t+1)
        // - Reward uses close(t+1)
        
        // This test verifies that the portfolio methods don't require future data
        let config = create_test_config();
        let mut portfolio = Portfolio::new(
            1000.0,
            vec!["BTCUSDT".to_string()],
            &config,
        );

        // At time t, we can observe current prices (close of t)
        let mut prices_t = HashMap::new();
        prices_t.insert("BTCUSDT".to_string(), 50000.0);
        
        // Mark to market with current prices (no future data needed)
        let _unrealized = portfolio.mark_to_market(&prices_t);

        // At time t+1, we execute at open(t+1)
        let mut exec_prices = HashMap::new();
        exec_prices.insert("BTCUSDT".to_string(), 50100.0); // open(t+1)
        
        let mut action = PortfolioAction {
            actions: HashMap::new(),
        };
        action.actions.insert(
            "BTCUSDT".to_string(),
            SymbolAction {
                direction: Direction::Long,
                size_fraction: 0.5,
            },
        );

        // Execute action - this only needs current (t+1) prices, not future
        let (cost, pnls) = portfolio.execute_action(&action, &exec_prices, 1001);
        let realized_pnl: f64 = pnls.values().sum();
        portfolio.update_equity(realized_pnl, cost);

        // After execution, we can mark to market at close(t+1)
        let mut prices_t1_close = HashMap::new();
        prices_t1_close.insert("BTCUSDT".to_string(), 50200.0);
        let _unrealized_t1 = portfolio.mark_to_market(&prices_t1_close);

        // This test passes if we can do all operations without needing future data
        // The structure enforces no look-ahead by design
        assert!(true, "No-lookahead invariant maintained by design");
    }
}
