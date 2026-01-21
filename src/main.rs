use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

mod database;
mod ict;
mod risk;
mod config;
mod data_loader;
mod portfolio;
mod training;

use database::Database;
use config::Config;
use data_loader::MultiSymbolMultiTFData;
use training::TrainingEngine;

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
    #[arg(long, default_value = "5m")]
    timeframe: String,
    #[arg(short, long, default_value = "1000.0")]
    balance: f64,
    #[arg(long, default_value = "iris_trading.db")]
    database: String,
    #[arg(long, default_value = "500")]
    candles: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load config
    let config = Config::load("config.toml")?;
    let db = Database::new(&args.database)?;

    // Lấy tập symbol, timeframe từ config (ưu tiên args nếu bạn truyền vào)
    let symbols = if !config.symbols.list.is_empty() {
        config.symbols.list.clone()
    } else {
        vec![args.symbol.clone()]
    };
    let timeframes: Vec<String> = if !config.symbols.timeframes.is_empty() {
        config.symbols.timeframes.clone()
    } else {
        vec![args.timeframe.clone()]
    };
    let anchor_tf = &timeframes[0];

    // Dùng loader mới chuẩn multi-symbol, multi-tf
    let multi_data = MultiSymbolMultiTFData::load(
        &db,
        &symbols,
        &timeframes,
        Some(args.candles),
        anchor_tf,
    )?;

    println!(
        "Loaded data for {} symbols, {} timeframes, anchor timeline len {}",
        multi_data.symbols().len(),
        multi_data.timeframes().len(),
        multi_data.timeline().len()
    );

    // === ĐÂY: Khởi tạo training engine & chạy train/backtest pipeline ===
    let mut engine = TrainingEngine::new(config.clone(), multi_data);

    // TRAIN phase
    engine.train_agent()?;
    println!("TRAIN: steps {}, profit {}", engine.report.train_steps, engine.report.train_profit);

    // BACKTEST phase
    engine.run_backtest()?;
    println!("BACKTEST: steps {}, profit {}", engine.report.test_steps, engine.report.test_profit);

    // (Nếu muốn, bạn có thể vẽ equity curve từ engine.report.test_log)

    Ok(())
}