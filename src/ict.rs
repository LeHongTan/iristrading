use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct FairValueGap {
    pub index: usize,
    pub gap_high: f64,
    pub gap_low: f64,
    pub gap_type: FVGType,
    pub strength: f64,
    pub filled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FVGType {
    Bullish,
    Bearish,
}

#[derive(Debug, Clone)]
pub struct OrderBlock {
    pub index: usize,
    pub high: f64,
    pub low: f64,
    pub ob_type: OrderBlockType,
    pub strength: f64,
    pub mitigated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderBlockType {
    Bullish,
    Bearish,
}

impl Candle {
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[inline]
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    #[inline]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    #[inline]
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    #[inline]
    pub fn lower_wick(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    #[inline]
    pub fn body_ratio(&self) -> f64 {
        let r = self.range();
        if r > 0.0 {
            self.body_size() / r
        } else {
            0.0
        }
    }

    #[inline]
    pub fn mid_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }
}

pub fn find_fvg(candles: &[Candle]) -> Option<FairValueGap> {
    if candles.len() < 3 {
        return None;
    }

    for i in (0..candles.len().saturating_sub(2)).rev() {
        let c1 = &candles[i];
        let c2 = &candles[i + 1];
        let c3 = &candles[i + 2];

        let avg_range = (c1.range() + c2.range() + c3.range()) / 3.0;
        let min_gap = avg_range * 0.1;

        // Bullish FVG
        if c3.low > c1.high {
            let gap_size = c3.low - c1.high;
            if gap_size > min_gap {
                let strength = gap_size / avg_range.max(0.0001);
                return Some(FairValueGap {
                    index: i + 1,
                    gap_high: c3.low,
                    gap_low: c1.high,
                    gap_type: FVGType::Bullish,
                    strength: strength.min(3.0),
                    filled: false,
                });
            }
        }

        // Bearish FVG
        if c1.low > c3.high {
            let gap_size = c1.low - c3.high;
            if gap_size > min_gap {
                let strength = gap_size / avg_range.max(0.0001);
                return Some(FairValueGap {
                    index: i + 1,
                    gap_high: c1.low,
                    gap_low: c3.high,
                    gap_type: FVGType::Bearish,
                    strength: strength.min(3.0),
                    filled: false,
                });
            }
        }
    }

    None
}

pub fn find_all_fvgs(candles: &[Candle], lookback: usize) -> Vec<FairValueGap> {
    let mut fvgs = Vec::new();

    if candles.len() < 3 {
        return fvgs;
    }

    let start = if candles.len() > lookback {
        candles.len() - lookback
    } else {
        0
    };

    for i in start..(candles.len() - 2) {
        let c1 = &candles[i];
        let c2 = &candles[i + 1];
        let c3 = &candles[i + 2];

        let avg_range = (c1.range() + c2.range() + c3.range()) / 3.0;
        let min_gap = avg_range * 0.1;

        if c3.low > c1.high {
            let gap_size = c3.low - c1.high;
            if gap_size > min_gap {
                fvgs.push(FairValueGap {
                    index: i + 1,
                    gap_high: c3.low,
                    gap_low: c1.high,
                    gap_type: FVGType::Bullish,
                    strength: (gap_size / avg_range.max(0.0001)).min(3.0),
                    filled: false,
                });
            }
        }

        if c1.low > c3.high {
            let gap_size = c1.low - c3.high;
            if gap_size > min_gap {
                fvgs.push(FairValueGap {
                    index: i + 1,
                    gap_high: c1.low,
                    gap_low: c3.high,
                    gap_type: FVGType::Bearish,
                    strength: (gap_size / avg_range.max(0.0001)).min(3.0),
                    filled: false,
                });
            }
        }
    }

    fvgs
}

pub fn find_order_block(candles: &[Candle]) -> Option<OrderBlock> {
    if candles.len() < 5 {
        return None;
    }

    let len = candles.len();

    // FIX: 20.min(len) (không có khoảng trắng kiểu "20. min")
    let window = 20usize.min(len);
    let avg_volume: f64 = candles
        .iter()
        .rev()
        .take(window)
        .map(|c| c.volume)
        .sum::<f64>()
        / (window as f64).max(1.0);

    for i in (2..len.saturating_sub(2)).rev() {
        let candle = &candles[i];

        if candle.volume < avg_volume * 1.2 {
            continue;
        }

        // Bullish OB
        let prev_bearish = candles[i.saturating_sub(2)..i].iter().all(|c| c.is_bearish());
        let next_bullish = candles
            .get(i + 1)
            .map(|c| c.is_bullish())
            .unwrap_or(false);
        let breaks_high = candles
            .get(i + 2)
            .map(|c| c.high > candle.high)
            .unwrap_or(false);

        if prev_bearish && candle.is_bullish() && next_bullish && breaks_high {
            return Some(OrderBlock {
                index: i,
                high: candle.high,
                low: candle.low,
                ob_type: OrderBlockType::Bullish,
                strength: (candle.volume / avg_volume.max(0.0001)).min(3.0),
                mitigated: false,
            });
        }

        // Bearish OB
        let prev_bullish = candles[i.saturating_sub(2)..i].iter().all(|c| c.is_bullish());
        let next_bearish = candles
            .get(i + 1)
            .map(|c| c.is_bearish())
            .unwrap_or(false);
        let breaks_low = candles
            .get(i + 2)
            .map(|c| c.low < candle.low)
            .unwrap_or(false);

        if prev_bullish && candle.is_bearish() && next_bearish && breaks_low {
            return Some(OrderBlock {
                index: i,
                high: candle.high,
                low: candle.low,
                ob_type: OrderBlockType::Bearish,
                strength: (candle.volume / avg_volume.max(0.0001)).min(3.0),
                mitigated: false,
            });
        }
    }

    None
}

pub fn find_swing_high(candles: &[Candle], lookback: usize) -> Option<(usize, f64)> {
    if candles.len() < lookback * 2 + 1 {
        return None;
    }

    let len = candles.len();

    for i in (lookback..(len - lookback)).rev() {
        let current_high = candles[i].high;

        let is_swing = candles[(i - lookback)..i]
            .iter()
            .all(|c| c.high < current_high)
            && candles[(i + 1)..=(i + lookback)]
                .iter()
                .all(|c| c.high < current_high);

        if is_swing {
            return Some((i, current_high));
        }
    }

    None
}

pub fn find_swing_low(candles: &[Candle], lookback: usize) -> Option<(usize, f64)> {
    if candles.len() < lookback * 2 + 1 {
        return None;
    }

    let len = candles.len();

    for i in (lookback..(len - lookback)).rev() {
        let current_low = candles[i].low;

        let is_swing = candles[(i - lookback)..i]
            .iter()
            .all(|c| c.low > current_low)
            && candles[(i + 1)..=(i + lookback)]
                .iter()
                .all(|c| c.low > current_low);

        if is_swing {
            return Some((i, current_low));
        }
    }

    None
}

pub fn calculate_ict_features(candles: &[Candle]) -> Vec<f64> {
    let mut features = Vec::with_capacity(20);

    if candles.is_empty() {
        return vec![0.0; 20];
    }

    let len = candles.len();
    let latest = &candles[len - 1];

    // 1) Momentum
    if len >= 2 {
        let prev = &candles[len - 2];
        let momentum = (latest.close - prev.close) / prev.close.max(0.0001);
        features.push(momentum.clamp(-0.1, 0.1) * 10.0);
    } else {
        features.push(0.0);
    }

    // 2) Body ratio
    features.push(latest.body_ratio());

    // 3-4) Wicks
    if latest.range() > 0.0 {
        features.push(latest.upper_wick() / latest.range());
        features.push(latest.lower_wick() / latest.range());
    } else {
        features.push(0.0);
        features.push(0.0);
    }

    // 5) Volume strength
    let avg_volume = if len >= 10 {
        candles[len - 10..].iter().map(|c| c.volume).sum::<f64>() / 10.0
    } else {
        candles.iter().map(|c| c.volume).sum::<f64>() / len as f64
    };
    features.push((latest.volume / avg_volume.max(0.0001)).min(5.0) / 5.0);

    // 6-8) SMA deviations
    for period in [5, 10, 20] {
        if len >= period {
            let sma = candles[len - period..].iter().map(|c| c.close).sum::<f64>() / period as f64;
            let dev = (latest.close - sma) / sma.max(0.0001);
            features.push(dev.clamp(-0.1, 0.1) * 10.0);
        } else {
            features.push(0.0);
        }
    }

    // 9-10) FVG
    if let Some(fvg) = find_fvg(candles) {
        features.push(fvg.strength / 3.0);
        features.push(if matches!(fvg.gap_type, FVGType::Bullish) { 1.0 } else { -1.0 });
    } else {
        features.push(0.0);
        features.push(0.0);
    }

    // 11-12) OB
    if let Some(ob) = find_order_block(candles) {
        features.push(ob.strength / 3.0);
        features.push(if matches!(ob.ob_type, OrderBlockType::Bullish) { 1.0 } else { -1.0 });
    } else {
        features.push(0.0);
        features.push(0.0);
    }

    // 13-14) RSI-like
    for period in [7, 14] {
        if len >= period + 1 {
            let mut gains = 0.0;
            let mut losses = 0.0;

            for i in (len - period)..len {
                let change = candles[i].close - candles[i - 1].close;
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }

            let avg_gain = gains / period as f64;
            let avg_loss = losses / period as f64;

            if avg_loss > 0.0001 {
                let rs = avg_gain / avg_loss;
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                features.push((rsi - 50.0) / 50.0);
            } else {
                features.push(1.0);
            }
        } else {
            features.push(0.0);
        }
    }

    // 15) BB position
    if len >= 20 {
        let closes: Vec<f64> = candles[len - 20..].iter().map(|c| c.close).collect();
        let mean = closes.iter().sum::<f64>() / 20.0;
        let variance = closes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / 20.0;
        let std_dev = variance.sqrt();

        let bb_position = if std_dev > 0.0001 {
            (latest.close - mean) / (2.0 * std_dev)
        } else {
            0.0
        };
        features.push(bb_position.clamp(-1.0, 1.0));
    } else {
        features.push(0.0);
    }

    // 16) Position in recent range
    if len >= 20 {
        let recent_high = candles[len - 20..].iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let recent_low = candles[len - 20..].iter().map(|c| c.low).fold(f64::MAX, f64::min);
        let range = recent_high - recent_low;

        if range > 0.0001 {
            features.push((latest.close - recent_low) / range);
        } else {
            features.push(0.5);
        }
    } else {
        features.push(0.5);
    }

    // 17) Volatility
    if len >= 10 {
        let returns: Vec<f64> = candles[len - 10..]
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close.max(0.0001))
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        features.push((variance.sqrt() * 100.0).min(1.0));
    } else {
        features.push(0.0);
    }

    // 18) Candle direction
    features.push(if latest.is_bullish() { 1.0 } else { -1.0 });

    while features.len() < 20 {
        features.push(0.0);
    }
    features.truncate(20);

    features
}