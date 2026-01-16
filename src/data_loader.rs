use anyhow::{anyhow, Result};
use std::collections::HashMap;
use crate::ict::Candle;
use crate::database::Database;

/// Multi-symbol data loader with timestamp alignment
#[derive(Debug)]
pub struct MultiSymbolData {
    /// Map from symbol to sorted candles
    data: HashMap<String, Vec<Candle>>,
    /// Unified timeline of timestamps across all symbols
    timeline: Vec<i64>,
    /// Map from timestamp to index in each symbol's data
    indices: HashMap<String, HashMap<i64, usize>>,
}

impl MultiSymbolData {
    /// Load historical data for multiple symbols with timestamp alignment
    pub fn load(db: &Database, symbols: &[String], max_candles: Option<usize>) -> Result<Self> {
        let mut data = HashMap::new();
        let mut all_timestamps = std::collections::HashSet::new();

        // Load data for each symbol
        for symbol in symbols {
            let candles = db.load_history(symbol, max_candles)?;
            if candles.is_empty() {
                return Err(anyhow!("No data for symbol: {}", symbol));
            }

            // Collect timestamps
            for candle in &candles {
                all_timestamps.insert(candle.timestamp);
            }

            data.insert(symbol.clone(), candles);
        }

        // Create sorted unified timeline
        let mut timeline: Vec<i64> = all_timestamps.into_iter().collect();
        timeline.sort_unstable();

        // Build indices for fast lookup
        let mut indices = HashMap::new();
        for (symbol, candles) in &data {
            let mut symbol_indices = HashMap::new();
            for (idx, candle) in candles.iter().enumerate() {
                symbol_indices.insert(candle.timestamp, idx);
            }
            indices.insert(symbol.clone(), symbol_indices);
        }

        Ok(Self {
            data,
            timeline,
            indices,
        })
    }

    /// Get candle for a symbol at a specific timestamp
    pub fn get_candle(&self, symbol: &str, timestamp: i64) -> Option<&Candle> {
        let symbol_indices = self.indices.get(symbol)?;
        let idx = symbol_indices.get(&timestamp)?;
        self.data.get(symbol)?.get(*idx)
    }

    /// Get all available symbols
    pub fn symbols(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Get unified timeline
    pub fn timeline(&self) -> &[i64] {
        &self.timeline
    }

    /// Get number of timestamps in timeline
    pub fn len(&self) -> usize {
        self.timeline.len()
    }

    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.timeline.is_empty()
    }

    /// Get candles for a symbol in a time range (for building sequences)
    /// Returns candles from start_idx to end_idx (inclusive) on the unified timeline
    pub fn get_sequence(
        &self,
        symbol: &str,
        start_idx: usize,
        end_idx: usize,
    ) -> Vec<Option<&Candle>> {
        if start_idx > end_idx || end_idx >= self.timeline.len() {
            return Vec::new();
        }

        let symbol_indices = match self.indices.get(symbol) {
            Some(indices) => indices,
            None => return Vec::new(),
        };

        let symbol_data = match self.data.get(symbol) {
            Some(data) => data,
            None => return Vec::new(),
        };

        (start_idx..=end_idx)
            .map(|i| {
                let timestamp = self.timeline[i];
                symbol_indices
                    .get(&timestamp)
                    .and_then(|idx| symbol_data.get(*idx))
            })
            .collect()
    }

    /// Get the most recent valid candle for a symbol at or before a given timeline index
    /// This is useful for filling gaps when a symbol doesn't have data at exact timestamp
    pub fn get_last_valid_candle(
        &self,
        symbol: &str,
        timeline_idx: usize,
    ) -> Option<&Candle> {
        if timeline_idx >= self.timeline.len() {
            return None;
        }

        let symbol_indices = self.indices.get(symbol)?;
        let symbol_data = self.data.get(symbol)?;

        // Search backwards from timeline_idx to find the most recent valid candle
        for i in (0..=timeline_idx).rev() {
            let timestamp = self.timeline[i];
            if let Some(&idx) = symbol_indices.get(&timestamp) {
                return symbol_data.get(idx);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_alignment() {
        // This test would require setting up a test database
        // For now, we'll skip implementation
    }
}
