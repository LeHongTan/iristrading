use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct RiskManager {
    pub initial_balance: f64,
    pub current_equity: f64,
    pub base_risk: f64,
    pub snowball_enabled: bool,
    pub trade_history: VecDeque<TradeResult>,
    pub max_history: usize,
    pub max_daily_loss: f64,
    pub daily_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct TradeResult {
    pub timestamp: i64,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub equity_after: f64,
}

#[derive(Debug, Clone)]
pub struct PositionSize {
    pub quantity: f64,
    pub risk_amount: f64,
    pub risk_percent: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
}

impl RiskManager {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            current_equity: initial_balance,
            base_risk: 0.02,
            snowball_enabled: true,
            trade_history: VecDeque::new(),
            max_history: 100,
            max_daily_loss: 0.05,
            daily_pnl: 0.0,
        }
    }

    pub fn check_risk(&self, equity: f64) -> f64 {
        if !self.snowball_enabled {
            return self.base_risk;
        }
        snowball_risk(equity, self.initial_balance, self.base_risk)
    }

    pub fn calculate_position_size(
        &self,
        entry_price: f64,
        stop_loss_price: f64,
        take_profit_price: Option<f64>,
    ) -> PositionSize {
        let risk_percent = self.check_risk(self.current_equity);
        let risk_amount = self.current_equity * risk_percent;

        let price_risk = (entry_price - stop_loss_price).abs();

        let quantity = if price_risk > 0.0 {
            risk_amount / price_risk
        } else {
            0.0
        };

        let tp = take_profit_price.unwrap_or_else(|| {
            if entry_price > stop_loss_price {
                entry_price + (entry_price - stop_loss_price) * 2.0
            } else {
                entry_price - (stop_loss_price - entry_price) * 2.0
            }
        });

        PositionSize {
            quantity,
            risk_amount,
            risk_percent,
            stop_loss: stop_loss_price,
            take_profit: tp,
        }
    }

    pub fn update_equity(&mut self, new_equity: f64, pnl: f64) {
        let pnl_percent = if self.current_equity.abs() > 0.0001 {
            pnl / self.current_equity
        } else {
            0.0
        };

        let result = TradeResult {
            timestamp: chrono::Utc::now().timestamp(),
            pnl,
            pnl_percent,
            equity_after: new_equity,
        };

        self.trade_history.push_back(result);
        while self.trade_history.len() > self.max_history {
            self.trade_history.pop_front();
        }

        self.daily_pnl += pnl;
        self.current_equity = new_equity;
    }

    pub fn reset_daily_pnl(&mut self) {
        self.daily_pnl = 0.0;
    }

    pub fn is_safe_to_trade(&self) -> bool {
        if self.current_equity < 50.0 {
            return false;
        }

        if self.daily_pnl < -(self.initial_balance * self.max_daily_loss) {
            return false;
        }

        if self.trade_history.len() >= 3 {
            let last_3: Vec<&TradeResult> = self.trade_history.iter().rev().take(3).collect();
            if last_3.iter().all(|t| t.pnl < 0.0) {
                return false;
            }
        }

        true
    }

    pub fn get_win_rate(&self, lookback: usize) -> f64 {
        if self.trade_history.is_empty() {
            return 0.5;
        }

        let trades: Vec<&TradeResult> = self.trade_history.iter().rev().take(lookback).collect();
        let wins = trades.iter().filter(|t| t.pnl > 0.0).count();

        wins as f64 / trades.len() as f64
    }

    pub fn get_profit_factor(&self, lookback: usize) -> f64 {
        if self.trade_history.is_empty() {
            return 1.0;
        }

        let trades: Vec<&TradeResult> = self.trade_history.iter().rev().take(lookback).collect();

        let gross_profit: f64 = trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();

        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| -t.pnl)
            .sum();

        if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            10.0
        } else {
            1.0
        }
    }

    pub fn get_max_drawdown(&self) -> f64 {
        if self.trade_history.is_empty() {
            return 0.0;
        }

        let mut peak: f64 = self.initial_balance;
        let mut max_dd: f64 = 0.0;

        for trade in &self.trade_history {
            peak = peak.max(trade.equity_after);
            let dd = (peak - trade.equity_after) / peak;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }

    pub fn get_risk_features(&self) -> Vec<f64> {
        vec![
            self.current_equity / self.initial_balance,
            self.check_risk(self.current_equity) / self.base_risk,
            self.get_win_rate(20),
            self.get_profit_factor(20) / 2.0,
            if self.is_safe_to_trade() { 1.0 } else { 0.0 },
        ]
    }
}

pub fn snowball_risk(equity: f64, initial_balance: f64, base_risk: f64) -> f64 {
    if equity < 200.0 {
        let recovery_factor = (200.0 - equity) / 200.0;
        let high_risk = base_risk * (1.0 + recovery_factor * 3.0);
        high_risk.min(0.08)
    } else {
        let growth_factor = equity / initial_balance;

        if growth_factor <= 1.0 {
            base_risk
        } else {
            let risk_reduction = (growth_factor.ln() / 2.0).min(0.5);
            let reduced_risk = base_risk * (1.0 - risk_reduction);
            reduced_risk.max(0.005)
        }
    }
}

pub fn check_risk(equity: f64) -> f64 {
    snowball_risk(equity, 1000.0, 0.02)
}

pub fn kelly_criterion(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    if avg_loss <= 0.0 || win_rate <= 0.0 || win_rate >= 1.0 {
        return 0.02;
    }

    let b = avg_win / avg_loss;
    let p = win_rate;
    let q = 1.0 - win_rate;

    let kelly = (p * b - q) / b;

    (kelly / 4.0).clamp(0.005, 0.10)
}

#[derive(Debug)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub avg_trade_return: f64,
}

impl BacktestMetrics {
    pub fn calculate(returns: &[f64], initial_balance: f64, final_balance: f64) -> Self {
        let total_trades = returns.len();
        let winning_trades = returns.iter().filter(|&&r| r > 0.0).count();
        let losing_trades = returns.iter().filter(|&&r| r < 0.0).count();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            10.0
        } else {
            1.0
        };

        let mut peak: f64 = initial_balance;
        let mut max_dd: f64 = 0.0;
        let mut equity: f64 = initial_balance;

        for &ret in returns {
            equity += ret;
            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }

        let avg_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance = if returns.len() > 1 {
            returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();

        let sharpe_ratio = if std_dev > 0.0 {
            (avg_return / std_dev) * (252_f64).sqrt()
        } else {
            0.0
        };

        Self {
            total_return: final_balance - initial_balance,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            max_drawdown: max_dd,
            sharpe_ratio,
            avg_trade_return: avg_return,
        }
    }

    pub fn print_report(&self, initial_balance: f64, final_balance: f64) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BACKTEST RESULTS                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Initial Balance:     ${:>12.2}                           ║",
            initial_balance
        );
        println!(
            "║ Final Balance:       ${:>12.2}                           ║",
            final_balance
        );
        println!(
            "║ Total Return:        ${:>12.2} ({:>+6.2}%)                ║",
            self.total_return,
            (self.total_return / initial_balance) * 100.0
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Total Trades:        {:>6}                                   ║",
            self.total_trades
        );
        println!(
            "║ Winning Trades:      {:>6} ({:>5.1}%)                         ║",
            self.winning_trades,
            self.win_rate * 100.0
        );
        println!(
            "║ Losing Trades:       {:>6}                                   ║",
            self.losing_trades
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Profit Factor:       {:>8.2}                                 ║",
            self.profit_factor
        );
        println!(
            "║ Max Drawdown:        {:>7.2}%                                 ║",
            self.max_drawdown * 100.0
        );
        println!(
            "║ Sharpe Ratio:        {:>8.2}                                 ║",
            self.sharpe_ratio
        );
        println!(
            "║ Avg Trade Return:    ${:>8.2}                                ║",
            self.avg_trade_return
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}