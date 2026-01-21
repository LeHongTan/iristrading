use anyhow::{anyhow, Result};
use std::collections::HashMap;
use chrono::{NaiveDate, NaiveDateTime};
use tracing::info;

use crate::config::Config;
use crate::data_loader::MultiSymbolMultiTFData;
use crate::portfolio::{PortfolioAction, Portfolio};

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
        }
    }

    /// Chia index theo timestamp (chia tập train/test cực chuẩn)
    pub fn get_split(&self) -> (usize, usize) {
        let timeline = self.data.timeline();
        // Chia tới hết năm 2024 => train, 2025+ => test
        let train_last_ts = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap()
            .and_hms_opt(23, 59, 59).unwrap()
            .timestamp();
        let test_first_idx = timeline.iter().position(|&ts| ts > train_last_ts).unwrap_or(timeline.len());
        let train_last_idx = test_first_idx.saturating_sub(1);
        (train_last_idx, test_first_idx)
    }

    /// Train AI agent: lặp qua tập train (vì đây là demo, chỉ giả lập random, bạn nối agent RL sau)
    pub fn train_agent(&mut self) -> Result<()> {
        let timeline = self.data.timeline();
        let (train_last_idx, test_first_idx) = self.get_split();
        let warmup_steps = self.config.training.warmup_steps.min(train_last_idx);
        self.portfolio = Portfolio::new(self.config.backtest.initial_balance, self.config.symbols.list.clone(), &self.config);

        for step in warmup_steps..=train_last_idx {
            // Build multi-symbol multi-tf state cho AI (có thể nối agent RL sau)
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
            // Gọi AI agent để ra action -- DEMO dùng random
            let mut actions = HashMap::new();
            for symbol in self.data.symbols() {
                actions.insert(symbol.clone(), 0); // 0 = Hold, 1 = Buy, -1 = Sell
            }
            let action = PortfolioAction { actions };
            self.portfolio.execute_action(action, self.get_prices(step));
        }

        let profit = self.portfolio.equity() - self.config.backtest.initial_balance;
        self.report.train_steps = train_last_idx + 1;
        self.report.train_profit = profit;
        info!("TRAIN done: {} bước, profit: {}", self.report.train_steps, profit);
        Ok(())
    }

    /// Chạy backtest, log toàn bộ trạng thái từng bước test (mô phỏng live trading như thật)
    pub fn run_backtest(&mut self) -> Result<()> {
        let timeline = self.data.timeline();
        let (train_last_idx, test_first_idx) = self.get_split();
        let mut portfolio = Portfolio::new(self.config.backtest.initial_balance, self.config.symbols.list.clone(), &self.config);
        let mut test_log = Vec::new();

        for step in test_first_idx..timeline.len() {
            // Lấy multi-tf sequence tại bước này (hoàn toàn không nhìn tương lai!)
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
            // AI agent sẽ dự đoán action dựa trên state hiện tại; ở đây demo random
            let mut actions = HashMap::new();
            for symbol in self.data.symbols() {
                actions.insert(symbol.clone(), 0);
            }
            let action = PortfolioAction { actions };
            portfolio.execute_action(action, self.get_prices(step));

            test_log.push((timeline[step], portfolio.equity()));
        }

        let profit = portfolio.equity() - self.config.backtest.initial_balance;
        self.report.test_steps = timeline.len() - test_first_idx;
        self.report.test_profit = profit;
        self.report.test_log = test_log;
        info!("BACKTEST done: {} bước, profit: {}", self.report.test_steps, profit);
        Ok(())
    }

    /// Giá tại bước hiện tại cho từng symbol, có thể truyền vào mỗi lần tick
    fn get_prices(&self, step: usize) -> HashMap<String, f64> {
        let mut prices = HashMap::new();
        for symbol in self.data.symbols() {
            // Lấy giá close symbol, tf anchor tại step này
            let anchor_tf = &self.data.timeframes()[0];
            let seq = self.data.get_sequence(symbol, anchor_tf, step, step);
            let price = seq.first().and_then(|c| c.map(|x| x.close)).unwrap_or(0.0);
            prices.insert(symbol.clone(), price);
        }
        prices
    }
}