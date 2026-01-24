import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.data_loader import load_multisymbol_multitf, get_state_window

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_dir="data",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        timeframes=("1h",),  # edit as needed
        sequence_length=256,
        initial_balance=20.0,
        fee=0.00055,
        slippage=0.0001,
        reward_mode="equity"
    ):
        super().__init__()
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)
        self.seq_len = sequence_length
        self.data_dir = data_dir
        self.fee = fee
        self.slip = slippage
        self.init_balance = initial_balance
        self.reward_mode = reward_mode

        # Load all data
        self.symbol_tf_to_df, self.anchor_timeline = load_multisymbol_multitf(
            self.data_dir, self.symbols, self.timeframes, self.seq_len
        )
        self.n_steps = len(self.anchor_timeline)
        self.n_symbols = len(self.symbols)
        self.n_tfs = len(self.timeframes)

        feature_per_candle = 5 # OHLCV
        ict_feats_per_tf = 7 # (number of ICT features in .py)
        self.state_dim = self.n_symbols * self.n_tfs * self.seq_len * (feature_per_candle) + self.n_symbols * self.n_tfs * (self.seq_len * ict_feats_per_tf)

        self.action_space = spaces.MultiDiscrete([3] * self.n_symbols) # direction -1/0/1 for each symbol
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim,),np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.account_equity = self.init_balance
        self.balances = {sym: 0.0 for sym in self.symbols}
        self.position = {sym: 0 for sym in self.symbols}  # -1 short, 0 flat, 1 long
        self.position_size = {sym: 0.0 for sym in self.symbols}
        self.entry_price = {sym: 0.0 for sym in self.symbols}
        self.equity_curve = [self.account_equity]
        self.last_step_pnl = 0.0
        self._step = self.seq_len
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Nếu vượt range thì pad zeros để không lỗi tiếp
        if self._step >= len(self.anchor_timeline):
            return np.zeros(self.state_dim, dtype=np.float32)
        statevec = get_state_window(
            self.symbol_tf_to_df,
            self.anchor_timeline,
            self._step,
            self.symbols,
            self.timeframes,
            self.seq_len,
            use_ict=True
        )
        return np.array(statevec, dtype=np.float32)
    
    def _get_prices(self):
        prices = {}
        if self._step >= len(self.anchor_timeline):
            # Trả lại giá cuối cùng hoặc entry_price để không lỗi
            for sym in self.symbols:
                prices[sym] = self.entry_price.get(sym, 1.0)
            return prices

        for sym in self.symbols:
            tf = self.timeframes[0]
            idx = self.anchor_timeline[self._step]
            df = self.symbol_tf_to_df[sym][tf]
            price = df.loc[df.index == idx]['close']
            px = price.values[0] if len(price) > 0 else self.entry_price.get(sym, 1.0)
            prices[sym] = px
        return prices

    def step(self, action):
        info = {}

        # ======== FIX: Nếu đã done hoặc vượt chỉ số timeline, trả obs zeros và done ngay =======
        if self.done or self._step >= len(self.anchor_timeline):
            self.done = True
            obs = np.zeros(self.state_dim, dtype=np.float32)
            return obs, 0.0, self.done, False, info
        # ===================================================
        prices = self._get_prices()
        realized_pnl = 0.0
        cost = 0.0

        # Execute action for each symbol
        for i, sym in enumerate(self.symbols):
            direction = action[i] - 1  # 0=short, 1=hold, 2=long
            last_dir = self.position[sym]

            if direction != last_dir:
                # close previous
                if last_dir != 0:
                    entry_px = self.entry_price[sym]
                    qty = self.position_size[sym]
                    close_px = prices[sym]
                    close_notional = qty * close_px
                    old_notional = qty * entry_px
                    raw_pnl = (close_px - entry_px) * qty * last_dir
                    realized_pnl += raw_pnl
                    trx_fee = close_notional * self.fee
                    cost += trx_fee
                    self.position[sym] = 0
                    self.position_size[sym] = 0
                    self.entry_price[sym] = 0

            # open new
            if direction != 0 and direction != last_dir:
                avail = self.account_equity / self.n_symbols  # simple position sizing
                price = prices[sym]
                qty = avail / price if price > 0 else 0
                open_notional = qty * price
                trx_fee = open_notional * self.fee
                cost += trx_fee
                self.position[sym] = direction
                self.position_size[sym] = qty
                self.entry_price[sym] = price

        # Update unrealized PnL
        unreal = 0.0
        for i, sym in enumerate(self.symbols):
            dir = self.position[sym]
            if dir != 0:
                qty = self.position_size[sym]
                price = prices[sym]
                entry_px = self.entry_price[sym]
                pnl = (price - entry_px) * qty * dir
                unreal += pnl

        # Update equity and reward
        self.account_equity += realized_pnl - cost
        self.equity_curve.append(self.account_equity + unreal)
        reward = (realized_pnl + unreal - cost) / self.init_balance  # normalized

        self.last_step_pnl = reward
        self._step += 1

        # ==== dừng env khi index tới limit hoặc equity cháy
        if self._step >= len(self.anchor_timeline) or self.account_equity <= 0.1:
            self.done = True

        return self._get_obs(), reward, self.done, False, {
            "equity": self.account_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unreal,
            "cost": cost,
            "prices": prices
        }

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        plt.plot(self.equity_curve)
        plt.title("Equity Curve")
        plt.show()