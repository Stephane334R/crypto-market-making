"""Synthetic book + trade data so the whole pipeline runs before you have recorded data.

The generator deliberately contains the effects a market maker must deal with:

* volatility regimes (calm / stressed), so a fixed spread is wrong half the time;
* informed flow: a share of market orders correctly predicts the next few
  seconds of price movement, which creates adverse selection on our fills;
* occasional sweeps that walk the book several ticks deep.

It is NOT a substitute for real data: conclusions about real profitability must
come from recorded Hyperliquid data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    hours: float = 2.0
    step_ms: int = 100
    start_price: float = 100_000.0
    tick: float = 1.0
    annual_vol_calm: float = 0.45
    annual_vol_stressed: float = 1.2
    regime_switch_prob: float = 1 / 9_000  # per step: mean regime length ~15 min
    trades_per_second: float = 6.0
    informed_share: float = 0.30
    informed_horizon_s: float = 3.0
    sweep_prob: float = 0.04
    sweep_mean_bps: float = 3.0
    sweep_informed_share: float = 0.75
    mean_trade_size: float = 0.05  # in coin
    seed: int = 7


def generate(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.hours * 3600 * 1000 / cfg.step_ms)
    steps_per_year = 365 * 24 * 3600 * 1000 / cfg.step_ms

    # --- volatility regimes and efficient price -------------------------------
    stressed = np.zeros(n, dtype=bool)
    state = False
    switches = rng.random(n) < cfg.regime_switch_prob
    for i in range(n):
        if switches[i]:
            state = not state
        stressed[i] = state
    vol = np.where(stressed, cfg.annual_vol_stressed, cfg.annual_vol_calm) / np.sqrt(steps_per_year)
    log_ret = rng.standard_normal(n) * vol
    price = cfg.start_price * np.exp(np.cumsum(log_ret))

    # --- book: 1-tick spread most of the time, wider in stress ----------------
    spread_ticks = 1 + rng.geometric(np.where(stressed, 0.35, 0.8)) - 1
    bid = np.floor((price - spread_ticks * cfg.tick / 2) / cfg.tick) * cfg.tick
    ask = bid + spread_ticks * cfg.tick
    ts = np.arange(n, dtype=np.int64) * cfg.step_ms + 1_700_000_000_000
    book = pd.DataFrame({
        "ts": ts,
        "recv_ts": ts + 5,
        "bid_px": bid,
        "bid_sz": rng.lognormal(0, 0.8, n),
        "ask_px": ask,
        "ask_sz": rng.lognormal(0, 0.8, n),
        "bid_depth5": rng.lognormal(1.5, 0.6, n),
        "ask_depth5": rng.lognormal(1.5, 0.6, n),
    })
    book["mid"] = (book["bid_px"] + book["ask_px"]) / 2

    # --- trades -----------------------------------------------------------------
    lam = cfg.trades_per_second * cfg.step_ms / 1000 * np.where(stressed, 2.5, 1.0)
    counts = rng.poisson(lam)
    step_idx = np.repeat(np.arange(n), counts)
    m = len(step_idx)
    h = int(cfg.informed_horizon_s * 1000 / cfg.step_ms)
    future = price[np.minimum(step_idx + h, n - 1)] - price[step_idx]
    # large aggressive orders (sweeps) are more often informed than small ones
    sweep = rng.random(m) < cfg.sweep_prob
    informed = rng.random(m) < np.where(sweep, cfg.sweep_informed_share, cfg.informed_share)
    side = np.where(informed, np.sign(future), rng.choice([-1, 1], m)).astype(int)
    side[side == 0] = 1

    depth = np.where(sweep, rng.exponential(cfg.sweep_mean_bps, m) * 1e-4 * price[step_idx], 0.0)
    depth = np.round(depth / cfg.tick) * cfg.tick
    touch = np.where(side == 1, ask[step_idx], bid[step_idx])
    px = touch + side * depth
    sz = rng.exponential(cfg.mean_trade_size, m) * np.where(sweep, 6.0, 1.0)

    offs = rng.integers(1, cfg.step_ms, m)
    trades = pd.DataFrame({
        "ts": ts[step_idx] + offs,
        "recv_ts": ts[step_idx] + offs + 5,
        "px": px,
        "sz": sz,
        "side": side,
    }).sort_values("ts", kind="stable").reset_index(drop=True)
    return book, trades
