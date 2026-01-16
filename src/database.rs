use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

use crate::ict::Candle;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open SQLite database")?;
        let db = Self { conn };
        db.init_tables()?;
        Ok(db)
    }

    fn init_tables(&self) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
             ON candles(symbol, timestamp DESC)",
            [],
        )?;

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                initial_balance REAL NOT NULL,
                final_balance REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(())
    }

    pub fn save_candle(&self, symbol: &str, candle: &Candle) -> Result<()> {
        // Fix: remove "? 4" spacing
        self.conn.execute(
            "INSERT OR REPLACE INTO candles
             (symbol, timestamp, open, high, low, close, volume)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                symbol,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            ],
        )?;
        Ok(())
    }

    pub fn save_candles(&self, symbol: &str, candles: &[Candle]) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let mut count = 0usize;

        for candle in candles {
            tx.execute(
                "INSERT OR REPLACE INTO candles
                 (symbol, timestamp, open, high, low, close, volume)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    symbol,
                    candle.timestamp,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume
                ],
            )?;
            count += 1;
        }

        tx.commit()?;
        Ok(count)
    }

    pub fn load_history(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Candle>> {
        let query = match limit {
            Some(n) => format!(
                "SELECT timestamp, open, high, low, close, volume
                 FROM candles
                 WHERE symbol = ?1
                 ORDER BY timestamp ASC
                 LIMIT {}",
                n
            ),
            None => String::from(
                "SELECT timestamp, open, high, low, close, volume
                 FROM candles
                 WHERE symbol = ?1
                 ORDER BY timestamp ASC",
            ),
        };

        let mut stmt = self.conn.prepare(&query)?;
        let candles = stmt
            .query_map(params![symbol], |row| {
                Ok(Candle {
                    timestamp: row.get(0)?,
                    open: row.get(1)?,
                    high: row.get(2)?,
                    low: row.get(3)?,
                    close: row.get(4)?,
                    volume: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(candles)
    }

    pub fn save_backtest_result(&self, result: &BacktestResult) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO backtest_results
             (run_id, symbol, start_time, end_time, initial_balance, final_balance,
              total_trades, winning_trades, losing_trades, max_drawdown, sharpe_ratio)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                result.run_id,
                result.symbol,
                result.start_time,
                result.end_time,
                result.initial_balance,
                result.final_balance,
                result.total_trades,
                result.winning_trades,
                result.losing_trades,
                result.max_drawdown,
                result.sharpe_ratio
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }
}

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub run_id: String,
    pub symbol: String,
    pub start_time: i64,
    pub end_time: i64,
    pub initial_balance: f64,
    pub final_balance: f64,
    pub total_trades: i64,
    pub winning_trades: i64,
    pub losing_trades: i64,
    pub max_drawdown: f64,
    pub sharpe_ratio: Option<f64>,
}

pub fn generate_sample_data(count: usize, start_price: f64) -> Vec<Candle> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut candles = Vec::with_capacity(count);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let mut price = start_price;
    let volatility = 0.002;

    for i in 0..count {
        let timestamp = now - ((count - i) as i64 * 60_000);

        let change = (rand_simple(i) - 0.5) * volatility * price;
        let open = price;
        price += change;
        let close = price;

        let high = open.max(close) * (1.0 + rand_simple(i + 1000) * 0.001);
        let low = open.min(close) * (1.0 - rand_simple(i + 2000) * 0.001);
        let volume = 100.0 + rand_simple(i + 3000) * 1000.0;

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

fn rand_simple(seed: usize) -> f64 {
    let x = ((seed as u64)
        .wrapping_mul(1103515245)
        .wrapping_add(12345))
        % (1 << 31);
    (x as f64) / (1u64 << 31) as f64
}