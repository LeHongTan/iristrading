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
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
             ON candles(symbol, timeframe, timestamp DESC)",
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

    pub fn save_candle(&self, symbol: &str, timeframe: &str, candle: &Candle) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO candles
             (symbol, timeframe, timestamp, open, high, low, close, volume)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                symbol,
                timeframe,
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

    pub fn save_candles(&self, symbol: &str, timeframe: &str, candles: &[Candle]) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let mut count = 0usize;

        for candle in candles {
            tx.execute(
                "INSERT OR REPLACE INTO candles
                 (symbol, timeframe, timestamp, open, high, low, close, volume)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    symbol,
                    timeframe,
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

        pub fn load_history_symbol_tf(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: Option<usize>,
    ) -> Result<Vec<Candle>> {
        let query = match limit {
            Some(n) => format!(
                "SELECT timestamp, open, high, low, close, volume
                 FROM candles
                 WHERE symbol = ?1 AND timeframe = ?2
                 ORDER BY timestamp ASC
                 LIMIT {}",
                n
            ),
            None => String::from(
                "SELECT timestamp, open, high, low, close, volume
                 FROM candles
                 WHERE symbol = ?1 AND timeframe = ?2
                 ORDER BY timestamp ASC",
            ),
        };
        let mut stmt = self.conn.prepare(&query)?;
        let candles = stmt
            .query_map(params![symbol, timeframe], |row| {
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
}