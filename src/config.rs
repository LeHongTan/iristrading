use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub symbols: SymbolsConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub execution: ExecutionConfig,
    pub position_sizing: PositionSizingConfig,
    pub backtest: BacktestConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SymbolsConfig {
    pub list: Vec<String>,
    pub timeframes: Vec<String>, 
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub sequence_length: usize,
    pub hidden_dim: usize,
    pub features_per_symbol: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub clip_epsilon: f64,
    pub entropy_coef: f64,
    pub value_coef: f64,
    pub epochs_per_update: usize,
    pub batch_size: usize,
    pub update_interval: usize,
    pub max_episodes: usize,
    pub warmup_steps: usize,
    pub save_interval: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExecutionConfig {
    pub taker_fee: f64,
    pub base_slippage: f64,
    pub size_impact: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PositionSizingConfig {
    pub base_risk_pct: f64,
    /// Vec of [equity_multiple, max_risk_pct, max_notional_usd]
    pub sizing_schedule: Vec<[f64; 3]>,
    pub max_leverage: f64,
}

impl PositionSizingConfig {
    /// Get max risk percentage and max notional for a given equity
    pub fn get_sizing(&self, equity: f64, initial_balance: f64) -> (f64, f64) {
        let equity_multiple = equity / initial_balance.max(1.0);
        
        // Find the appropriate tier in the schedule
        let mut max_risk_pct = self.base_risk_pct;
        let mut max_notional = 2000.0; // Default
        
        for schedule_item in &self.sizing_schedule {
            let [threshold, risk, notional] = schedule_item;
            if equity_multiple >= *threshold {
                max_risk_pct = *risk;
                max_notional = *notional;
            } else {
                break;
            }
        }
        
        (max_risk_pct, max_notional)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BacktestConfig {
    pub initial_balance: f64,
    pub max_candles_per_symbol: usize,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read config file")?;
        let config: Config = toml::from_str(&content)
            .context("Failed to parse config file")?;
        Ok(config)
    }

    pub fn default_path() -> &'static str {
        "config.toml"
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            symbols: SymbolsConfig {
                list: vec![
                    "BTCUSDT".to_string(),
                    "ETHUSDT".to_string(),
                    "SOLUSDT".to_string(),
                    "BNBUSDT".to_string(),
                    "XRPUSDT".to_string(),
                ],
                timeframes: vec!["1m".to_string(), "5m".to_string(), "15m".to_string()],
            },
            model: ModelConfig {
                sequence_length: 256,
                hidden_dim: 256,
                features_per_symbol: 20,
            },
            training: TrainingConfig {
                learning_rate: 3e-4,
                gamma: 0.99,
                gae_lambda: 0.95,
                clip_epsilon: 0.2,
                entropy_coef: 0.01,
                value_coef: 0.5,
                epochs_per_update: 4,
                batch_size: 64,
                update_interval: 2048,
                max_episodes: 1000,
                warmup_steps: 30,
                save_interval: 10,
            },
            execution: ExecutionConfig {
                taker_fee: 0.00055,
                base_slippage: 0.0001,
                size_impact: 0.00001,
            },
            position_sizing: PositionSizingConfig {
                base_risk_pct: 0.02,
                sizing_schedule: vec![
                    [1.0, 0.020, 2000.0],
                    [2.0, 0.015, 5000.0],
                    [5.0, 0.010, 10000.0],
                    [10.0, 0.005, 15000.0],
                ],
                max_leverage: 10.0,
            },
            backtest: BacktestConfig {
                initial_balance: 1000.0,
                max_candles_per_symbol: 5000,
            },
        }
    }
}
