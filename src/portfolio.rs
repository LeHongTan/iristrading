use std::collections::HashMap;
use crate::ict::Candle;
use crate::config::{Config, ExecutionConfig, PositionSizingConfig};

/// Position direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Hold = 0,
    Long = 1,
    Short = 2,
}

impl Direction {
    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => Direction::Long,
            2 => Direction::Short,
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
    /// Returns total cost (fees + slippage) and individual PnLs
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
    fn close_position_internal(&mut self, symbol: &str, exit_price: f64) -> (f64, f64) {
        if let Some(position) = self.positions.remove(symbol) {
            let notional = position.quantity * exit_price;
            
            // Calculate PnL
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
            
            let net_pnl = gross_pnl - total_cost;
            
            (net_pnl, total_cost)
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
    pub fn close_all_positions(&mut self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut total_pnl = 0.0;
        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        
        for symbol in symbols {
            if let Some(&price) = current_prices.get(&symbol) {
                let (pnl, cost) = self.close_position_internal(&symbol, price);
                total_pnl += pnl;
                self.equity -= cost; // Deduct closing costs
            }
        }
        
        self.equity += total_pnl;
        total_pnl
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
