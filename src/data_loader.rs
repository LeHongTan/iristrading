use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use crate::ict::Candle;
use crate::database::Database;

/// Multi-symbol, multi-timeframe data loader.
#[derive(Debug, Clone)] // <--- thêm Clone ở đây
pub struct MultiSymbolMultiTFData {
    /// data[symbol][timeframe] = Vec<Candle>
    data: HashMap<String, HashMap<String, Vec<Candle>>>,
    /// anchor_timeline = timeline theo tf nhỏ nhất (vector timestamp)
    anchor_timeline: Vec<i64>,
    /// mapping: [symbol][timeframe][timestamp] = idx in Vec<Candle>
    indices: HashMap<String, HashMap<String, HashMap<i64, usize>>>,
    /// List symbol, list timeframe được load
    symbols: Vec<String>,
    timeframes: Vec<String>,
} // <-- xoá chữ "s" ở đây!

impl MultiSymbolMultiTFData {
    pub fn load(
        db: &Database,
        symbols: &[String],
        timeframes: &[String],
        max_candles: Option<usize>,
        anchor_tf: &str
    ) -> Result<Self> {
        let mut data: HashMap<String, HashMap<String, Vec<Candle>>> = HashMap::new();
        let mut all_timestamps: HashSet<i64> = HashSet::new();

        for symbol in symbols {
            let mut tf_map = HashMap::new();
            for tf in timeframes {
                let candles = db.load_history_symbol_tf(symbol, tf, max_candles)?;
                for candle in &candles {
                    all_timestamps.insert(candle.timestamp);
                }
                tf_map.insert(tf.clone(), candles);
            }
            data.insert(symbol.clone(), tf_map);
        }

        let anchor_symbol = &symbols[0];
        let anchor_candles = data
            .get(anchor_symbol)
            .and_then(|m| m.get(anchor_tf))
            .ok_or_else(|| anyhow!("No anchor tf data for {}", anchor_symbol))?;
        let anchor_timeline: Vec<i64> = anchor_candles.iter().map(|c| c.timestamp).collect();

        let mut indices = HashMap::new();
        for symbol in symbols {
            let mut tf_indices = HashMap::new();
            for tf in timeframes {
                let col = if let Some(m) = data.get(symbol) { m.get(tf) } else { None };
                let tf_data = match col { Some(candles) => candles, None => &Vec::new() };
                let mut t_idx = HashMap::new();
                for (i, candle) in tf_data.iter().enumerate() {
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

    pub fn get_sequence(
        &self,
        symbol: &str,
        timeframe: &str,
        start: usize,
        end: usize,
    ) -> Vec<Option<&Candle>> {
        let timeline = &self.anchor_timeline;
        let tf_indices =
            match self.indices.get(symbol).and_then(|m| m.get(timeframe)) {
                Some(x) => x,
                None => return vec![None; end - start + 1],
            };
        let tf_data = match self.data.get(symbol).and_then(|m| m.get(timeframe)) {
            Some(x) => x,
            None => return vec![None; end - start + 1],
        };
        (start..=end)
            .map(|i| {
                if i >= timeline.len() {
                    None
                } else {
                    let ts = timeline[i];
                    tf_indices
                        .get(&ts)
                        .and_then(|idx| tf_data.get(*idx))
                }
            })
            .collect()
    }

    pub fn get_multi_tf_sequence(
        &self,
        symbol: &str,
        start: usize,
        end: usize,
    ) -> HashMap<String, Vec<Option<&Candle>>> {
        let mut out: HashMap<String, Vec<Option<&Candle>>> = HashMap::new(); // rõ type
        for tf in &self.timeframes {
            out.insert(tf.clone(), self.get_sequence(symbol, tf, start, end));
        }
        out
    }

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