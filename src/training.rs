use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::data_loader::MultiSymbolData;
use crate::ict::calculate_ict_features;
use crate::portfolio::{Direction, Portfolio, PortfolioAction, SymbolAction};

/// Training engine that orchestrates the RL training loop
pub struct TrainingEngine {
    config: Config,
    data: MultiSymbolData,
    portfolio: Portfolio,
    current_step: usize,
    episode_count: usize,
    warmup_steps: usize,
    step_count: usize,  // Total steps across all episodes
}

impl TrainingEngine {
    pub fn new(config: Config, data: MultiSymbolData) -> Self {
        let symbols = config.symbols.list.clone();
        let portfolio = Portfolio::new(
            config.backtest.initial_balance,
            symbols,
            &config,
        );
        
        let warmup_steps = config.training.warmup_steps;
        
        Self {
            config,
            data,
            portfolio,
            current_step: 0,
            episode_count: 0,
            warmup_steps,
            step_count: 0,
        }
    }

    /// Run a single training episode
    /// Returns episode statistics
    pub fn run_episode(&mut self, python_brain: &PythonBrain) -> Result<EpisodeStats> {
        info!("Starting episode {}", self.episode_count + 1);
        
        // Reset portfolio for new episode
        let symbols = self.config.symbols.list.clone();
        self.portfolio = Portfolio::new(
            self.config.backtest.initial_balance,
            symbols,
            &self.config,
        );
        
        let timeline_len = self.data.len();
        if timeline_len < self.warmup_steps + 10 {
            return Err(anyhow!("Not enough data for training"));
        }
        
        let mut episode_reward = 0.0;
        let mut num_trades = 0;
        
        // Start from warmup_steps to have enough history for sequences
        for step in self.warmup_steps..timeline_len - 1 {
            self.current_step = step;
            
            // NO-LEAK PRINCIPLE:
            // At step t, we can only observe data up to t
            // Decision is made at t, but execution happens at open(t+1)
            
            // 1. Build observation state using data up to step t
            let state = self.build_state(step)?;
            
            // 2. Get action from policy
            let action_info = python_brain.get_action(state.clone())?;
            
            // 3. Execute action at open(t+1) - NO LEAK
            let next_step = step + 1;
            let next_timestamp = self.data.timeline()[next_step];
            
            // Get execution prices (open of next candle)
            let mut execution_prices = HashMap::new();
            for symbol in self.portfolio.symbols() {
                if let Some(candle) = self.data.get_candle(symbol, next_timestamp) {
                    execution_prices.insert(symbol.clone(), candle.open);
                }
            }
            
            // Build portfolio action from brain output
            let portfolio_action = self.build_portfolio_action(&action_info);
            
            // Execute the action
            let (cost, pnls) = self.portfolio.execute_action(
                &portfolio_action,
                &execution_prices,
                next_timestamp,
            );
            
            let realized_pnl: f64 = pnls.values().sum();
            num_trades += pnls.len();
            
            // Update equity
            self.portfolio.update_equity(realized_pnl, cost);
            
            // 4. Mark-to-market at close(t+1) for reward calculation
            let mut mtm_prices = HashMap::new();
            for symbol in self.portfolio.symbols() {
                if let Some(candle) = self.data.get_candle(symbol, next_timestamp) {
                    mtm_prices.insert(symbol.clone(), candle.close);
                }
            }
            
            let unrealized_pnl = self.portfolio.mark_to_market(&mtm_prices);
            let current_equity = self.portfolio.equity() + unrealized_pnl;
            
            // 5. Compute reward: change in equity minus costs
            // Reward is based on equity at close(t+1)
            let prev_equity = if step == self.warmup_steps {
                self.config.backtest.initial_balance
            } else {
                // We would need to track this, but for simplicity use portfolio equity
                self.portfolio.equity() - realized_pnl + cost
            };
            
            let reward = (current_equity - prev_equity) / prev_equity.max(1.0);
            episode_reward += reward;
            
            // 6. Store transition
            let done = (next_step >= timeline_len - 1) as i32;
            python_brain.store_transition(
                state.clone(),
                action_info.clone(),
                reward,
                done,
            )?;
            
            // 7. Periodic training update
            if self.step_count > 0 && self.step_count % self.config.training.update_interval == 0 {
                // Get next state for bootstrapping
                let next_state = if next_step < timeline_len - 1 {
                    self.build_state(next_step)?
                } else {
                    state.clone()
                };
                
                let next_value = python_brain.get_value(&next_state)?;
                let train_stats = python_brain.train(next_value)?;
                
                debug!(
                    "Training update at step {}: policy_loss={:.4}, value_loss={:.4}",
                    self.step_count,
                    train_stats.get("policy_loss").unwrap_or(&0.0),
                    train_stats.get("value_loss").unwrap_or(&0.0),
                );
            }
            
            self.step_count += 1;
            
            if step % 100 == 0 {
                debug!(
                    "Step {}/{}: Equity=${:.2}, Reward={:.6}",
                    step, timeline_len, current_equity, reward
                );
            }
        }
        
        // Close all positions at end of episode
        let final_timestamp = self.data.timeline()[timeline_len - 1];
        let mut final_prices = HashMap::new();
        for symbol in self.portfolio.symbols() {
            if let Some(candle) = self.data.get_candle(symbol, final_timestamp) {
                final_prices.insert(symbol.clone(), candle.close);
            }
        }
        
        let _final_pnl = self.portfolio.close_all_positions(&final_prices);
        let final_equity = self.portfolio.equity();
        
        self.episode_count += 1;
        
        info!(
            "Episode {} complete: Initial=${:.2}, Final=${:.2}, Return={:.2}%, Trades={}",
            self.episode_count,
            self.config.backtest.initial_balance,
            final_equity,
            ((final_equity / self.config.backtest.initial_balance) - 1.0) * 100.0,
            num_trades,
        );
        
        Ok(EpisodeStats {
            episode: self.episode_count,
            initial_equity: self.config.backtest.initial_balance,
            final_equity,
            total_reward: episode_reward,
            num_trades,
        })
    }

    /// Build state observation for step t (using only data up to t)
    fn build_state(&self, step: usize) -> Result<State> {
        let sequence_length = self.config.model.sequence_length;
        let symbols = &self.config.symbols.list;
        
        // Compute start index for sequence (ensure we don't go negative)
        let start_idx = if step + 1 >= sequence_length {
            step + 1 - sequence_length
        } else {
            0
        };
        
        let mut symbol_sequences = Vec::new();
        
        // Build sequence for each symbol
        for symbol in symbols {
            let candle_options = self.data.get_sequence(symbol, start_idx, step);
            
            // Convert candles to feature vectors
            let mut features = Vec::new();
            for candle_opt in candle_options {
                if let Some(candle) = candle_opt {
                    // Build features from single candle + context
                    // For now, simplified: just use candle data
                    // In production, you'd want to build ICT features from sliding window
                    let candle_features = vec![
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                        (candle.close - candle.open) / candle.open.max(0.0001),
                        (candle.high - candle.low) / candle.low.max(0.0001),
                    ];
                    
                    // Pad to features_per_symbol
                    let mut padded = candle_features;
                    while padded.len() < self.config.model.features_per_symbol {
                        padded.push(0.0);
                    }
                    padded.truncate(self.config.model.features_per_symbol);
                    
                    features.push(padded);
                } else {
                    // Missing data - use zeros
                    features.push(vec![0.0; self.config.model.features_per_symbol]);
                }
            }
            
            // Pad sequence if needed
            while features.len() < sequence_length {
                features.insert(0, vec![0.0; self.config.model.features_per_symbol]);
            }
            
            symbol_sequences.push(features);
        }
        
        // Portfolio features
        let portfolio_features = self.portfolio.get_portfolio_features();
        
        Ok(State {
            symbol_sequences,
            portfolio_features,
        })
    }

    /// Build portfolio action from brain output
    fn build_portfolio_action(&self, action_info: &ActionInfo) -> PortfolioAction {
        let mut actions = HashMap::new();
        
        for (i, symbol) in self.config.symbols.list.iter().enumerate() {
            if i < action_info.directions.len() && i < action_info.sizes.len() {
                let direction = Direction::from_i32(action_info.directions[i]);
                let size_fraction = action_info.sizes[i];
                
                actions.insert(
                    symbol.clone(),
                    SymbolAction {
                        direction,
                        size_fraction,
                    },
                );
            }
        }
        
        PortfolioAction { actions }
    }

    pub fn episode_count(&self) -> usize {
        self.episode_count
    }
}

/// State representation for RL
#[derive(Debug, Clone)]
pub struct State {
    /// Sequence per symbol: Vec of length num_symbols, each containing sequence of features
    pub symbol_sequences: Vec<Vec<Vec<f64>>>,
    /// Portfolio-level features
    pub portfolio_features: Vec<f64>,
}

/// Action information from brain
#[derive(Debug, Clone)]
pub struct ActionInfo {
    pub directions: Vec<i32>,
    pub sizes: Vec<f64>,
    pub log_prob: f64,
    pub value: f64,
}

/// Episode statistics
#[derive(Debug)]
pub struct EpisodeStats {
    pub episode: usize,
    pub initial_equity: f64,
    pub final_equity: f64,
    pub total_reward: f64,
    pub num_trades: usize,
}

/// Python brain interface
pub struct PythonBrain {
    get_action_func: Py<PyAny>,
    store_transition_func: Py<PyAny>,
    train_func: Py<PyAny>,
    save_func: Py<PyAny>,
    load_func: Py<PyAny>,
}

impl PythonBrain {
    pub fn new(config: &Config) -> Result<Self> {
        Python::with_gil(|py| {
            let sys = PyModule::import_bound(py, "sys")
                .map_err(|e| anyhow!("Failed to import sys: {}", e))?;
            let path = sys.getattr("path")
                .map_err(|e| anyhow!("Failed to get sys.path: {}", e))?;

            path.call_method1("insert", (0, "./python"))
                .map_err(|e| anyhow!("Failed to insert python path: {}", e))?;

            if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
                let site_packages = format!("{}/lib/python3.12/site-packages", venv);
                path.call_method1("insert", (0, site_packages))
                    .map_err(|e| anyhow!("Failed to insert venv path: {}", e))?;
                info!("Using VIRTUAL_ENV for Python imports: {}", venv);
            } else {
                warn!("VIRTUAL_ENV not set; embedded Python may not see venv packages");
            }

            let brain_module = PyModule::import_bound(py, "brain_multisymbol")
                .map_err(|e| anyhow!("Failed to import brain_multisymbol: {}", e))?;
            
            // Initialize agent with config parameters
            let init_kwargs = PyDict::new_bound(py);
            init_kwargs.set_item("num_symbols", config.symbols.list.len())
                .map_err(|e| anyhow!("Failed to set num_symbols: {}", e))?;
            init_kwargs.set_item("sequence_length", config.model.sequence_length)
                .map_err(|e| anyhow!("Failed to set sequence_length: {}", e))?;
            init_kwargs.set_item("features_per_timestep", config.model.features_per_symbol)
                .map_err(|e| anyhow!("Failed to set features_per_timestep: {}", e))?;
            init_kwargs.set_item("hidden_dim", config.model.hidden_dim)
                .map_err(|e| anyhow!("Failed to set hidden_dim: {}", e))?;
            
            brain_module.getattr("initialize_agent")
                .map_err(|e| anyhow!("Failed to get initialize_agent: {}", e))?
                .call((), Some(&init_kwargs))
                .map_err(|e| anyhow!("Failed to call initialize_agent: {}", e))?;

            let get_action_func = brain_module.getattr("get_action_with_info")
                .map_err(|e| anyhow!("Failed to get get_action_with_info: {}", e))?
                .into_py(py);
            let store_transition_func = brain_module.getattr("store_transition")
                .map_err(|e| anyhow!("Failed to get store_transition: {}", e))?
                .into_py(py);
            let train_func = brain_module.getattr("train_step")
                .map_err(|e| anyhow!("Failed to get train_step: {}", e))?
                .into_py(py);
            let save_func = brain_module.getattr("save_model")
                .map_err(|e| anyhow!("Failed to get save_model: {}", e))?
                .into_py(py);
            let load_func = brain_module.getattr("load_model")
                .map_err(|e| anyhow!("Failed to get load_model: {}", e))?
                .into_py(py);

            info!("Python Multi-Symbol Brain initialized successfully");
            Ok(Self {
                get_action_func,
                store_transition_func,
                train_func,
                save_func,
                load_func,
            })
        })
    }

    pub fn get_action(&self, state: State) -> Result<ActionInfo> {
        Python::with_gil(|py| {
            // Convert symbol sequences to Python lists
            let py_symbol_seqs = PyList::new_bound(py, 
                state.symbol_sequences.iter().map(|seq| {
                    PyList::new_bound(py, seq.iter().map(|features| {
                        PyList::new_bound(py, features)
                    }))
                })
            );
            
            let py_portfolio_feats = PyList::new_bound(py, &state.portfolio_features);
            
            let result = self.get_action_func.bind(py).call1((py_symbol_seqs, py_portfolio_feats))
                .map_err(|e| anyhow!("Python error: {}", e))?;
            
            // Parse result dictionary
            let dict = result.downcast::<PyDict>()
                .map_err(|e| anyhow!("Failed to downcast: {}", e))?;
            
            let directions_list = dict.get_item("directions")
                .map_err(|e| anyhow!("Failed to get directions: {}", e))?
                .ok_or_else(|| anyhow!("Missing directions"))?;
            let sizes_list = dict.get_item("sizes")
                .map_err(|e| anyhow!("Failed to get sizes: {}", e))?
                .ok_or_else(|| anyhow!("Missing sizes"))?;
            
            let directions: Vec<i32> = directions_list.extract()
                .map_err(|e| anyhow!("Failed to extract directions: {}", e))?;
            let sizes: Vec<f64> = sizes_list.extract()
                .map_err(|e| anyhow!("Failed to extract sizes: {}", e))?;
            let log_prob: f64 = dict.get_item("log_prob")
                .map_err(|e| anyhow!("Failed to get log_prob: {}", e))?
                .ok_or_else(|| anyhow!("Missing log_prob"))?
                .extract()
                .map_err(|e| anyhow!("Failed to extract log_prob: {}", e))?;
            let value: f64 = dict.get_item("value")
                .map_err(|e| anyhow!("Failed to get value: {}", e))?
                .ok_or_else(|| anyhow!("Missing value"))?
                .extract()
                .map_err(|e| anyhow!("Failed to extract value: {}", e))?;
            
            Ok(ActionInfo {
                directions,
                sizes,
                log_prob,
                value,
            })
        })
    }

    pub fn store_transition(
        &self,
        state: State,
        action: ActionInfo,
        reward: f64,
        done: i32,
    ) -> Result<()> {
        Python::with_gil(|py| {
            let py_symbol_seqs = PyList::new_bound(py,
                state.symbol_sequences.iter().map(|seq| {
                    PyList::new_bound(py, seq.iter().map(|features| {
                        PyList::new_bound(py, features)
                    }))
                })
            );
            
            let py_portfolio_feats = PyList::new_bound(py, &state.portfolio_features);
            let py_dirs = PyList::new_bound(py, &action.directions);
            let py_sizes = PyList::new_bound(py, &action.sizes);
            
            self.store_transition_func.bind(py).call1((
                py_symbol_seqs,
                py_portfolio_feats,
                py_dirs,
                py_sizes,
                reward,
                action.value,
                action.log_prob,
                done,
            )).map_err(|e| anyhow!("Python store_transition error: {}", e))?;
            
            Ok(())
        })
    }

    pub fn get_value(&self, state: &State) -> Result<f64> {
        // For now, just call get_action and extract value
        let action_info = self.get_action(state.clone())?;
        Ok(action_info.value)
    }

    pub fn train(&self, next_value: f64) -> Result<HashMap<String, f64>> {
        Python::with_gil(|py| {
            let result = self.train_func.bind(py).call1((next_value,))
                .map_err(|e| anyhow!("Python train error: {}", e))?;
            let dict = result.downcast::<PyDict>()
                .map_err(|e| anyhow!("Failed to downcast train result: {}", e))?;
            
            let mut stats = HashMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = key.extract()
                    .map_err(|e| anyhow!("Failed to extract key: {}", e))?;
                let val_f64: f64 = value.extract()
                    .map_err(|e| anyhow!("Failed to extract value: {}", e))?;
                stats.insert(key_str, val_f64);
            }
            
            Ok(stats)
        })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        Python::with_gil(|py| {
            self.save_func.bind(py).call1((path,))
                .map_err(|e| anyhow!("Python save error: {}", e))?;
            Ok(())
        })
    }

    pub fn load(&self, path: &str) -> Result<bool> {
        Python::with_gil(|py| {
            let result = self.load_func.bind(py).call1((path,))
                .map_err(|e| anyhow!("Python load error: {}", e))?;
            Ok(result.extract()
                .map_err(|e| anyhow!("Failed to extract load result: {}", e))?)
        })
    }
}
