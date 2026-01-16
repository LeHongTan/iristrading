use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::Deserialize;
use serde_json::Value;
use std::collections::VecDeque;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

mod database;
mod ict;
mod risk;
mod config;
mod data_loader;
mod portfolio;
mod training;

use database::{generate_sample_data, BacktestResult, Database};
use ict::{calculate_ict_features, find_fvg, find_order_block, Candle, FVGType, OrderBlockType};
use risk::{BacktestMetrics, RiskManager};
use config::Config;
use data_loader::MultiSymbolData;
use training::{PythonBrain as MultiSymbolBrain, TrainingEngine};

#[derive(Debug, Clone, ValueEnum)]
enum TradingMode {
    Live,
    Backtest,
    Train,
}

#[derive(Parser, Debug)]
#[command(name = "IrisTrading")]
#[command(about = "Hybrid Trading Bot for Bybit")]
struct Args {
    #[arg(short, long, value_enum, default_value = "backtest")]
    mode: TradingMode,

    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    #[arg(short, long, default_value = "1000.0")]
    balance: f64,

    #[arg(long, default_value = "iris_trading.db")]
    database: String,

    #[arg(long, default_value = "500")]
    candles: usize,
}

#[derive(Debug, Deserialize)]
struct BybitWsMessage {
    topic: Option<String>,
    data: Option<Value>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    success: Option<bool>,
    ret_msg: Option<String>,
}

struct PythonBrain {
    get_action_func: Py<PyAny>,
}

impl PythonBrain {
    fn new() -> Result<Self> {
        Python::with_gil(|py| {
            let sys = PyModule::import_bound(py, "sys")?;
            let path = sys.getattr("path")?;

            // 1) Always add local python/ folder first (brain.py lives here)
            path.call_method1("insert", (0, "./python"))?;

            // 2) If running inside a venv, add its site-packages so imports like torch/numpy work
            if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
                // NOTE: this project uses Python 3.12 venv
                let site_packages = format!("{}/lib/python3.12/site-packages", venv);
                let lib_dynload = format!("{}/lib/python3.12/site-packages/lib-dynload", venv);

                // Insert at front so venv packages win over system packages
                path.call_method1("insert", (0, site_packages))?;
                path.call_method1("insert", (0, lib_dynload))?;

                info!("Using VIRTUAL_ENV for Python imports: {}", venv);
            } else {
                warn!("VIRTUAL_ENV not set; embedded Python may not see venv packages (torch/numpy). Run: source .venv/bin/activate");
            }

            let brain_module = PyModule::import_bound(py, "brain")?;
            brain_module.getattr("initialize_agent")?.call0()?;

            let get_action_func = brain_module.getattr("get_action")?.into_py(py);

            info!("Python AI Brain initialized successfully");
            Ok(Self { get_action_func })
        })
    }

    fn get_action(&self, state: Vec<f64>) -> Result<i32> {
        Python::with_gil(|py| {
            let py_list = PyList::new_bound(py, state);
            let result = self.get_action_func.bind(py).call1((py_list,))?;
            Ok(result.extract::<i32>()?)
        })
    }
}

#[derive(Debug, Clone)]
struct Position {
    side: PositionSide,
    entry_price: f64,
    quantity: f64,
    entry_time: i64,
}

#[derive(Debug, Clone, PartialEq)]
enum PositionSide {
    Long,
    Short,
}

struct TradingEngine {
    candles: VecDeque<Candle>,
    risk_manager: RiskManager,
    position: Option<Position>,
    max_candles: usize,
    trade_returns: Vec<f64>,
    total_trades: usize,
}

impl TradingEngine {
    fn new(initial_balance: f64, max_candles: usize) -> Self {
        Self {
            candles: VecDeque::with_capacity(max_candles),
            risk_manager: RiskManager::new(initial_balance),
            position: None,
            max_candles,
            trade_returns: Vec::new(),
            total_trades: 0,
        }
    }

    fn add_candle(&mut self, candle: Candle) {
        self.candles.push_back(candle);
        while self.candles.len() > self.max_candles {
            self.candles.pop_front();
        }
    }

    fn get_candles_vec(&self) -> Vec<Candle> {
        self.candles.iter().cloned().collect()
    }

    fn get_state(&self) -> Vec<f64> {
        // 20 features in total: 15 ICT + 5 risk
        let mut state = calculate_ict_features(&self.get_candles_vec());
        let risk_features = self.risk_manager.get_risk_features();

        while state.len() < 15 {
            state.push(0.0);
        }
        state.truncate(15);
        state.extend(risk_features);

        while state.len() < 20 {
            state.push(0.0);
        }
        state.truncate(20);

        state
    }

    fn execute_action(&mut self, action: i32, current_price: f64, timestamp: i64) -> Option<f64> {
        let mut pnl = None;

        match action {
            1 => {
                // BUY: close short then open long
                if let Some(pos) = &self.position {
                    if pos.side == PositionSide::Short {
                        let trade_pnl = (pos.entry_price - current_price) * pos.quantity;
                        self.close_position(trade_pnl);
                        pnl = Some(trade_pnl);
                    }
                }

                if self.position.is_none() && self.risk_manager.is_safe_to_trade() {
                    self.open_position(PositionSide::Long, current_price, timestamp);
                }
            }
            2 => {
                // SELL: close long then open short
                if let Some(pos) = &self.position {
                    if pos.side == PositionSide::Long {
                        let trade_pnl = (current_price - pos.entry_price) * pos.quantity;
                        self.close_position(trade_pnl);
                        pnl = Some(trade_pnl);
                    }
                }

                if self.position.is_none() && self.risk_manager.is_safe_to_trade() {
                    self.open_position(PositionSide::Short, current_price, timestamp);
                }
            }
            _ => {}
        }

        pnl
    }

    fn open_position(&mut self, side: PositionSide, price: f64, timestamp: i64) {
        let risk_pct = self.risk_manager.check_risk(self.risk_manager.current_equity);
        let risk_amount = self.risk_manager.current_equity * risk_pct;

        // stop distance 2% (demo)
        let stop_distance = price * 0.02;
        let quantity = if stop_distance > 0.0 {
            risk_amount / stop_distance
        } else {
            0.0
        };

        self.position = Some(Position {
            side,
            entry_price: price,
            quantity,
            entry_time: timestamp,
        });

        self.total_trades += 1;
    }

    fn close_position(&mut self, pnl: f64) {
        let new_equity = self.risk_manager.current_equity + pnl;
        self.risk_manager.update_equity(new_equity, pnl);
        self.trade_returns.push(pnl);
        self.position = None;
    }

    fn force_close(&mut self, current_price: f64) -> Option<f64> {
        if let Some(pos) = &self.position {
            let pnl = match pos.side {
                PositionSide::Long => (current_price - pos.entry_price) * pos.quantity,
                PositionSide::Short => (pos.entry_price - current_price) * pos.quantity,
            };
            self.close_position(pnl);
            return Some(pnl);
        }
        None
    }
}

fn parse_bybit_kline(data: &Value) -> Result<Candle> {
    let arr = data
        .as_array()
        .ok_or_else(|| anyhow!("Expected array for kline data"))?;
    if arr.is_empty() {
        return Err(anyhow!("Empty kline array"));
    }

    let item = &arr[0];

    let timestamp = item
        .get("start")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("Missing start"))?;

    let open = item
        .get("open")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("Missing open"))?;

    let high = item
        .get("high")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("Missing high"))?;

    let low = item
        .get("low")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("Missing low"))?;

    let close = item
        .get("close")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("Missing close"))?;

    let volume = item
        .get("volume")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(Candle::new(timestamp, open, high, low, close, volume))
}

async fn run_live_mode(args: &Args, db: &Database, brain: &PythonBrain) -> Result<()> {
    info!("Starting LIVE mode for {}", args.symbol);

    let mut engine = TradingEngine::new(args.balance, 200);

    let url = "wss://stream.bybit.com/v5/public/linear";
    info!("Connecting to Bybit WebSocket: {}", url);

    let (mut ws_stream, _) = connect_async(url)
        .await
        .context("Failed to connect to Bybit WebSocket")?;

    info!("Connected to Bybit WebSocket");

    let topic = format!("kline.1.{}", args.symbol);
    let subscribe_msg = serde_json::json!({
        "op": "subscribe",
        "args": [topic]
    });

    ws_stream.send(Message::Text(subscribe_msg.to_string())).await?;
    info!("Subscribed to kline.1.{}", args.symbol);

    loop {
        match ws_stream.next().await {
            Some(Ok(Message::Text(text))) => match serde_json::from_str::<BybitWsMessage>(&text) {
                Ok(msg) => {
                    if let Some(success) = msg.success {
                        if success {
                            info!("Subscription confirmed");
                        } else {
                            warn!("Subscription failed: {:?}", msg.ret_msg);
                        }
                        continue;
                    }

                    if let (Some(topic), Some(data)) = (&msg.topic, &msg.data) {
                        if topic.contains("kline.1.") {
                            match parse_bybit_kline(data) {
                                Ok(candle) => {
                                    db.save_candle(&args.symbol, &candle)?;
                                    engine.add_candle(candle);

                                    if engine.candles.len() >= 20 {
                                        let state = engine.get_state();
                                        let action = brain.get_action(state)?;

                                        let action_str = match action {
                                            0 => "HOLD",
                                            1 => "BUY",
                                            2 => "SELL",
                                            _ => "UNKNOWN",
                                        };

                                        info!(
                                            "Price: ${:.2} | Action: {} | Equity: ${:.2} | Candles: {}",
                                            candle.close,
                                            action_str,
                                            engine.risk_manager.current_equity,
                                            engine.candles.len()
                                        );

                                        if let Some(fvg) = find_fvg(&engine.get_candles_vec()) {
                                            let fvg_type = match fvg.gap_type {
                                                FVGType::Bullish => "Bullish",
                                                FVGType::Bearish => "Bearish",
                                            };
                                            info!(
                                                "FVG detected: {} (strength: {:.2})",
                                                fvg_type, fvg.strength
                                            );
                                        }

                                        if let Some(ob) = find_order_block(&engine.get_candles_vec()) {
                                            let ob_type = match ob.ob_type {
                                                OrderBlockType::Bullish => "Bullish",
                                                OrderBlockType::Bearish => "Bearish",
                                            };
                                            info!(
                                                "Order Block detected: {} (strength: {:.2})",
                                                ob_type, ob.strength
                                            );
                                        }

                                        if let Some(pnl) = engine.execute_action(
                                            action,
                                            candle.close,
                                            candle.timestamp,
                                        ) {
                                            info!("Trade closed with PnL: ${:.2}", pnl);
                                        }
                                    }
                                }
                                Err(e) => debug!("Failed to parse kline: {}", e),
                            }
                        }
                    }
                }
                Err(e) => debug!("Failed to parse message: {}", e),
            },

            Some(Ok(Message::Ping(data))) => {
                ws_stream.send(Message::Pong(data)).await?;
            }

            Some(Ok(Message::Binary(_))) => {}
            Some(Ok(Message::Pong(_))) => {}
            Some(Ok(Message::Frame(_))) => {}

            Some(Ok(Message::Close(_))) => {
                warn!("WebSocket closed, reconnecting...");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

                let (new_ws, _) = connect_async(url).await?;
                ws_stream = new_ws;
                ws_stream.send(Message::Text(subscribe_msg.to_string())).await?;
                info!("Reconnected");
            }

            Some(Err(e)) => {
                error!("WebSocket error: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }

            None => {
                warn!("WebSocket stream ended");
                break;
            }
        }
    }

    Ok(())
}

fn run_backtest_mode(args: &Args, db: &Database, brain: &PythonBrain) -> Result<()> {
    info!("Starting BACKTEST mode for {}", args.symbol);
    info!("Initial balance: ${:.2}", args.balance);

    let candles = match db.load_history(&args.symbol, Some(args.candles)) {
        Ok(c) if !c.is_empty() => {
            info!("Loaded {} candles from database", c.len());
            c
        }
        _ => {
            info!("No data in database, generating {} sample candles", args.candles);
            let sample = generate_sample_data(args.candles, 42000.0);
            db.save_candles(&args.symbol, &sample)?;
            sample
        }
    };

    if candles.len() < 50 {
        return Err(anyhow!("Not enough candles for backtest (need at least 50)"));
    }

    let mut engine = TradingEngine::new(args.balance, 200);
    let warmup_period = 30;

    info!(
        "Running backtest with {} candles (warmup: {})",
        candles.len(),
        warmup_period
    );

    for (i, candle) in candles.iter().enumerate() {
        engine.add_candle(*candle);

        if i < warmup_period {
            continue;
        }

        let visible_candles = &candles[..=i];
        let mut full_state = calculate_ict_features(visible_candles);

        let risk_features = engine.risk_manager.get_risk_features();
        full_state.extend(risk_features);

        while full_state.len() < 20 {
            full_state.push(0.0);
        }
        full_state.truncate(20);

        let action = brain.get_action(full_state)?;

        if let Some(pnl) = engine.execute_action(action, candle.close, candle.timestamp) {
            let status = if pnl >= 0.0 { "WIN" } else { "LOSS" };
            debug!(
                "Trade #{}: {} | PnL: {:+.2} | Equity: {:.2}",
                engine.total_trades,
                status,
                pnl,
                engine.risk_manager.current_equity
            );
        }
    }

    if let Some(last_candle) = candles.last() {
        engine.force_close(last_candle.close);
    }

    let metrics = BacktestMetrics::calculate(
        &engine.trade_returns,
        args.balance,
        engine.risk_manager.current_equity,
    );

    metrics.print_report(args.balance, engine.risk_manager.current_equity);

    let result = BacktestResult {
        run_id: chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string(),
        symbol: args.symbol.clone(),
        start_time: candles.first().map(|c| c.timestamp).unwrap_or(0),
        end_time: candles.last().map(|c| c.timestamp).unwrap_or(0),
        initial_balance: args.balance,
        final_balance: engine.risk_manager.current_equity,
        total_trades: engine.total_trades as i64,
        winning_trades: metrics.winning_trades as i64,
        losing_trades: metrics.losing_trades as i64,
        max_drawdown: metrics.max_drawdown,
        sharpe_ratio: Some(metrics.sharpe_ratio),
    };

    db.save_backtest_result(&result)?;
    info!("Backtest results saved to database");

    Ok(())
}

fn run_train_mode(_args: &Args, db: &Database) -> Result<()> {
    info!("Starting TRAINING mode for multi-symbol portfolio");
    
    // Load configuration
    let config_path = std::env::var("CONFIG_PATH")
        .unwrap_or_else(|_| Config::default_path().to_string());
    
    let config = if std::path::Path::new(&config_path).exists() {
        Config::load(&config_path)?
    } else {
        info!("Config file not found, using defaults");
        Config::default()
    };
    
    info!("Configuration loaded:");
    info!("  Symbols: {:?}", config.symbols.list);
    info!("  Sequence length: {}", config.model.sequence_length);
    info!("  Initial balance: ${:.2}", config.backtest.initial_balance);
    info!("  Max episodes: {}", config.training.max_episodes);
    
    // Load or generate data for all symbols
    info!("Loading historical data for all symbols...");
    
    for symbol in &config.symbols.list {
        let candles = db.load_history(symbol, Some(config.backtest.max_candles_per_symbol))?;
        if candles.is_empty() {
            info!("No data for {}, generating sample data", symbol);
            
            // Generate sample data with different base prices
            let base_price = match symbol.as_str() {
                "BTCUSDT" => 42000.0,
                "ETHUSDT" => 2200.0,
                "SOLUSDT" => 100.0,
                "BNBUSDT" => 300.0,
                "XRPUSDT" => 0.6,
                _ => 1000.0,
            };
            
            let sample = generate_sample_data(config.backtest.max_candles_per_symbol, base_price);
            db.save_candles(symbol, &sample)?;
            info!("Generated {} sample candles for {}", sample.len(), symbol);
        } else {
            info!("Loaded {} candles for {}", candles.len(), symbol);
        }
    }
    
    // Load multi-symbol data with alignment
    let data = MultiSymbolData::load(
        db,
        &config.symbols.list,
        Some(config.backtest.max_candles_per_symbol),
    )?;
    
    info!("Multi-symbol data loaded: {} timestamps aligned", data.len());
    
    // Initialize Python brain
    info!("Initializing Python multi-symbol brain...");
    let python_brain = MultiSymbolBrain::new(&config)?;
    
    // Try to load existing checkpoint
    let checkpoint_path = "checkpoints/model_latest.pt";
    if std::path::Path::new(checkpoint_path).exists() {
        info!("Loading checkpoint from {}", checkpoint_path);
        python_brain.load(checkpoint_path)?;
    } else {
        info!("No checkpoint found, starting fresh");
    }
    
    // Create training engine
    let mut engine = TrainingEngine::new(config.clone(), data);
    
    info!("Starting training loop...");
    
    for episode in 0..config.training.max_episodes {
        let stats = engine.run_episode(&python_brain)?;
        
        let return_pct = ((stats.final_equity / stats.initial_equity) - 1.0) * 100.0;
        info!(
            "Episode {}/{}: Return={:.2}%, Final Equity=${:.2}, Trades={}",
            episode + 1,
            config.training.max_episodes,
            return_pct,
            stats.final_equity,
            stats.num_trades,
        );
        
        // Save checkpoint periodically
        if (episode + 1) % config.training.save_interval == 0 {
            std::fs::create_dir_all("checkpoints")?;
            let checkpoint_path = format!("checkpoints/model_episode_{}.pt", episode + 1);
            python_brain.save(&checkpoint_path)?;
            python_brain.save("checkpoints/model_latest.pt")?;
            info!("Checkpoint saved: {}", checkpoint_path);
        }
    }
    
    info!("Training complete!");
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    dotenv::dotenv().ok();

    let args = Args::parse();

    info!("Initializing database: {}", args.database);
    let db = Database::new(&args.database)?;

    info!("Initializing Python AI Brain...");
    let brain = PythonBrain::new()?;

    match args.mode {
        TradingMode::Live => run_live_mode(&args, &db, &brain).await?,
        TradingMode::Backtest => run_backtest_mode(&args, &db, &brain)?,
        TradingMode::Train => run_train_mode(&args, &db)?,
    }

    Ok(())
}