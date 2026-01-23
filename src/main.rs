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
        None,
        anchor_tf,
    )?;

    println!(
        "Loaded data for {} symbols, {} timeframes, anchor timeline len {}",
        multi_data.symbols().len(),
        multi_data.timeframes().len(),
        multi_data.timeline().len()
    );

    // ============ VÒNG LẶP CHẠY ĐA SEED =============
    let seeds = vec![42, 123, 77, 233, 1337, 2024, 7, 21];
    println!("Seed,TRAIN_Profit,TEST_Profit");

    for &seed in &seeds {
        let mut engine = TrainingEngine::new_with_seed(config.clone(), multi_data.clone(), seed);

        engine.train_agent()?;
        engine.run_backtest()?;
        println!("{},{},{}", seed, engine.report.train_profit, engine.report.test_profit);
    }

    Ok(())
}