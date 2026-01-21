use anyhow::{Result};
use std::collections::HashMap;
use chrono::NaiveDate;
use tracing::info;

use crate::config::Config;
use crate::data_loader::MultiSymbolMultiTFData;
use crate::portfolio::{PortfolioAction, Portfolio, SymbolAction};

use std::process::{Command, Stdio};
use std::io::Write;

pub fn get_action_from_python(state: &[f64], seed: u64) -> anyhow::Result<Vec<i32>> {
    let mut child = Command::new("python3")
        .arg("python/agent.py")
        .arg(seed.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn python agent");

    {
        let child_stdin = child.stdin.as_mut().unwrap();
        let state_json = serde_json::to_string(&state).unwrap();
        child_stdin.write_all(state_json.as_bytes()).unwrap();
    }

    let output = child.wait_with_output().unwrap();
    let action: Vec<i32> = serde_json::from_slice(&output.stdout).unwrap();
    Ok(action)
}

pub struct TrainingReport {
    pub train_steps: usize,
    pub train_profit: f64,
    pub test_steps: usize,
    pub test_profit: f64,
    pub test_log: Vec<(i64, f64)>, // (timestamp, equity)
}

/// RL & backtest orchestrator
pub struct TrainingEngine {
    config: Config,
    data: MultiSymbolMultiTFData,
    portfolio: Portfolio,
    pub report: TrainingReport,
    pub seed: Option<u64>, // <== Thêm field seed
}

impl TrainingEngine {
    pub fn new(config: Config, data: MultiSymbolMultiTFData) -> Self {
        let symbols = config.symbols.list.clone();
        let portfolio = Portfolio::new(config.backtest.initial_balance, symbols, &config);
        Self {
            config,
            data,
            portfolio,
            report: TrainingReport {
                train_steps: 0,
                train_profit: 0.0,
                test_steps: 0,
                test_profit: 0.0,
                test_log: Vec::new(),
            },
            seed: None,
        }
    }

    /// Cho phép khởi tạo với seed (dùng cho loop kiếm best seed)
    pub fn new_with_seed(config: Config, data: MultiSymbolMultiTFData, seed: u64) -> Self {
        let mut engine = Self::new(config, data);
        engine.seed = Some(seed);
        engine
    }

    pub fn get_split(&self) -> (usize, usize) {
        let timeline = self.data.timeline();
        let train_last_ts = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap()
            .and_hms_opt(23, 59, 59).unwrap()
            .timestamp();
        let test_first_idx = timeline.iter().position(|&ts| ts > train_last_ts).unwrap_or(timeline.len());
        let train_last_idx = test_first_idx.saturating_sub(1);
        (train_last_idx, test_first_idx)
    }

    pub fn train_agent(&mut self) -> Result<()> {
        let timeline = self.data.timeline();
        let (train_last_idx, _) = self.get_split();
        let warmup_steps = self.config.training.warmup_steps.min(train_last_idx);
        self.portfolio = Portfolio::new(self.config.backtest.initial_balance, self.config.symbols.list.clone(), &self.config);

        for step in warmup_steps..=train_last_idx {
            let mut multi_tf_sequences = HashMap::new();
            for symbol in self.data.symbols() {
                multi_tf_sequences.insert(
                    symbol.clone(),
                    self.data.get_multi_tf_sequence(
                        symbol,
                        step + 1 - self.config.model.sequence_length,
                        step,
                    )
                );
            }
            // Đúng: flatten state sang Vec<f64> (close)
            let mut flat_state = Vec::new();
            for symbol_seq in multi_tf_sequences.values() {
                for tf_seq in symbol_seq.values() {
                    for candle in tf_seq {
                        flat_state.push(candle.as_ref().map(|x| x.close).unwrap_or(0.0));
                    }
                }
            }
            // GỌI AGENT PYTHON LẤY ACTION
            let agent_seed = self.seed.unwrap_or(42);
            let acts = get_action_from_python(&flat_state, agent_seed)?;
            // Map acts (i32) về SymbolAction theo thứ tự symbol
            let mut actions = HashMap::new();
            for (symbol, &act) in self.data.symbols().iter().zip(acts.iter()) {
                actions.insert(symbol.clone(), SymbolAction::from_i32(act));
            }
            let action = PortfolioAction { actions };
            self.portfolio.execute_action(
                &action,
                &self.get_prices(step),
                timeline[step],
            )?;
        }

        let profit = self.portfolio.equity() - self.config.backtest.initial_balance;
        self.report.train_steps = train_last_idx + 1;
        self.report.train_profit = profit;
        info!("TRAIN done: {} bước, profit: {}", self.report.train_steps, profit);
        Ok(())
    }

    pub fn run_backtest(&mut self) -> Result<()> {
        let timeline = self.data.timeline();
        let (_, test_first_idx) = self.get_split();
        let mut portfolio = Portfolio::new(self.config.backtest.initial_balance, self.config.symbols.list.clone(), &self.config);
        let mut test_log = Vec::new();

        for step in test_first_idx..timeline.len() {
            let mut multi_tf_sequences = HashMap::new();
            for symbol in self.data.symbols() {
                multi_tf_sequences.insert(
                    symbol.clone(),
                    self.data.get_multi_tf_sequence(
                        symbol,
                        step + 1 - self.config.model.sequence_length,
                        step,
                    )
                );
            }
            let mut flat_state = Vec::new();
            for symbol_seq in multi_tf_sequences.values() {
                for tf_seq in symbol_seq.values() {
                    for candle in tf_seq {
                        flat_state.push(candle.as_ref().map(|x| x.close).unwrap_or(0.0));
                    }
                }
            }
            let agent_seed = self.seed.unwrap_or(42);
            let acts = get_action_from_python(&flat_state, agent_seed)?;
            let mut actions = HashMap::new();
            for (symbol, &act) in self.data.symbols().iter().zip(acts.iter()) {
                actions.insert(symbol.clone(), SymbolAction::from_i32(act));
            }
            let action = PortfolioAction { actions };
            portfolio.execute_action(
                &action,
                &self.get_prices(step),
                timeline[step],
            )?;

            test_log.push((timeline[step], portfolio.equity()));
        }

        let profit = portfolio.equity() - self.config.backtest.initial_balance;
        self.report.test_steps = timeline.len() - test_first_idx;
        self.report.test_profit = profit;
        self.report.test_log = test_log;
        info!("BACKTEST done: {} bước, profit: {}", self.report.test_steps, profit);
        Ok(())
    }

    /// Giá tại bước hiện tại cho từng symbol
    fn get_prices(&self, step: usize) -> HashMap<String, f64> {
        let mut prices = HashMap::new();
        for symbol in self.data.symbols() {
            let anchor_tf = &self.data.timeframes()[0];
            let seq = self.data.get_sequence(symbol, anchor_tf, step, step);
            let price = seq.first().and_then(|c| c.map(|x| x.close)).unwrap_or(0.0);
            prices.insert(symbol.clone(), price);
        }
        prices
    }
}