use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use crate::ict::Candle;
use crate::database::Database;

/// Data loader: multi-symbol & multi-timeframe (align theo anchor timeframe nhỏ nhất)
#[derive(Debug)]
pub struct MultiSymbolMultiTFData {
    /// data[symbol][timeframe] = Vec<Candle>
    data: HashMap<String, HashMap<String, Vec<Candle>>>,
    /// anchor_timeline: vector timestamp của khung nhỏ nhất (ví dụ: 1m)
    anchor_timeline: Vec<i64>,
    /// mapping[symbol][timeframe][timestamp] = idx in Vec<Candle>
    indices: HashMap<String, HashMap<String, HashMap<i64, usize>>>,
    /// Danh sách all symbol, tf
    symbols: Vec<String>,
    timeframes: Vec<String>,
}

impl MultiSymbolMultiTFData {
    /// Load tất cả symbol, tất cả timeframe về cùng anchor timeline (smallest tf)
    pub fn load(
        db: &Database,
        symbols: &[String],
        timeframes: &[String],
        max_candles: Option<usize>,
        anchor_tf: &str
    ) -> Result<Self> {
        let mut data: HashMap<String, HashMap<String, Vec<Candle>>> = HashMap::new();
        let mut all_timestamps: HashSet<i64> = HashSet::new();

        // Load all candles, build mapping cho từng symbol-tf
        for symbol in symbols {
            let mut tf_map = HashMap::new();
            for tf in timeframes {
                let sym_tf = format!("{}_{}", symbol, tf);
                let candles = db.load_history_symbol_tf(symbol, tf, max_candles)?;
                for candle in &candles {
                    all_timestamps.insert(candle.timestamp);
                }
                tf_map.insert(tf.clone(), candles);
            }
            data.insert(symbol.clone(), tf_map);
        }

        // anchor timeline = timeline của anchor_tf, symbol đầu tiên (giả định mọi symbol anchor đều đủ dài)
        let anchor_symbol = &symbols[0];
        let anchor_candles = data
            .get(anchor_symbol)
            .and_then(|m| m.get(anchor_tf))
            .ok_or_else(|| anyhow!("No anchor tf data for {}", anchor_symbol))?;

        let anchor_timeline: Vec<i64> = anchor_candles.iter().map(|c| c.timestamp).collect();

        // Build indices fast-lookup: [symbol][tf][timestamp] = idx
        let mut indices = HashMap::new();
        for symbol in symbols {
            let mut tf_indices = HashMap::new();
            for tf in timeframes {
                let col = data.get(symbol)
                    .and_then(|m| m.get(tf)).unwrap_or(&vec![]);
                let mut t_idx = HashMap::new();
                for (i, candle) in col.iter().enumerate() {
                    t_idx.insert(candle.timestamp, i);
                }
                tf_indices.insert(tf.clone(), t_idx);
            }
            indices.insert(symbol.clone(), tf_indices);
        }

        Ok(Self {
            data,
            anchor_timeline,
            indices,
            symbols: symbols.to_vec(),
            timeframes: timeframes.to_vec(),
        })
    }

    /// Lấy sequence của 1 symbol, 1 tf, align với anchor timeline
    pub fn get_sequence(
        &self,
        symbol: &str,
        timeframe: &str,
        start: usize,
        end: usize,
    ) -> Vec<Option<&Candle>> {
        let timeline = &self.anchor_timeline;
        let symbol_indices = self.indices.get(symbol)?.get(timeframe)?;
        let symbol_data = self.data.get(symbol)?.get(timeframe)?;

        (start..=end)
            .map(|i| {
                if i >= timeline.len() { return None; }
                let ts = timeline[i];
                symbol_indices
                    .get(&ts)
                    .and_then(|idx| symbol_data.get(*idx))
            })
            .collect()
    }

    /// Lấy sequence multi-tf cho 1 symbol trong range (align anchor)
    pub fn get_multi_tf_sequence(
        &self,
        symbol: &str,
        start: usize,
        end: usize,
    ) -> HashMap<String, Vec<Option<&Candle>>> {
        let mut out = HashMap::new();
        for tf in &self.timeframes {
            out.insert(tf.clone(), self.get_sequence(symbol, tf, start, end));
        }
        out
    }

    /// Trả về anchor timeline
    pub fn timeline(&self) -> &[i64] {
        &self.anchor_timeline
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
    pub fn timeframes(&self) -> &[String] {
        &self.timeframes
    }
}