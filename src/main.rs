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
    // Nếu config có timeframes array, lấy từ đó, nếu không thì lấy từ arg
    let timeframes: Vec<String> = if let Some(tfvec) = {
        // Dùng reflection/trick nếu config chưa có timeframes, fallback 1 tf
        #[allow(unused)]
        struct _Tmp { timeframes: Option<Vec<String>> }
        None
    } {
        // add custom code if you implement timeframes[] in config
        vec![]
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

    // Example: bạn có thể debug số nến
    println!(
        "Loaded data for {} symbols, {} timeframes, anchor timeline len {}",
        multi_data.symbols().len(),
        multi_data.timeframes().len(),
        multi_data.timeline().len()
    );

    // TODO: Tiếp tục pipeline của bạn (ví dụ training, backtest, truyền state cho AI agent...)

    Ok(())
}