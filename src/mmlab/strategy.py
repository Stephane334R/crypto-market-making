"""Quoting logic: volatility-scaled spread with inventory skew.

This is a practical simplification of Avellaneda & Stoikov (2008):

    reservation price   r = mid - q * gamma * sigma^2 * (T - t)
    optimal spread      d = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

The closed form needs parameters (gamma, k, T) that are awkward to calibrate on
a 24/7 market with no session end. Desks typically use the same *structure*
with directly interpretable parameters, which is what we do:

    half_spread_bps = max(min_half_spread_ticks * tick_bps, vol_mult * sigma_bps(horizon))
    skew_bps        = skew_ticks_at_max * tick_bps * (q / max_inventory)
    r               = mid * (1 - skew_bps / 1e4)
    bid, ask        = r * (1 -/+ half_spread_bps / 1e4), rounded away from mid

`sigma_bps(horizon)` is the expected mid move over `vol_horizon_s` seconds,
estimated with a time-aware EWMA of squared mid returns.

Why the floors are in TICKS, not bps
------------------------------------
A bps-denominated floor is meaningless until you know the price. One tick is
0.125 bps on BTC near $80k but 4.3 bps on ETH near $2.3k with a $0.1 tick, so
a single `min_half_spread_bps=1.0` is 8 ticks on one coin and a quarter of a
tick on the other. Since the venue's minimum spread is one tick, the only
portable unit for "how far behind the touch do I rest" is the tick itself.

`vol_horizon_s` defaults to the time a quote is actually exposed --- the
requote interval plus one latency hop (~0.3 s with the default SimConfig) ---
not an arbitrary 5 s. Pricing 5 s of variance into a quote you can cancel in
300 ms is what pushes quotes a full 8 ticks off BTC's mid and earns zero fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QuoterParams:
    order_size: float = 0.02             # coin per quote
    max_inventory: float = 0.10          # coin, absolute
    min_half_spread_ticks: float = 1.0   # floor on the half-spread, in ticks
    min_half_spread_bps: float = 0.0     # optional extra floor in bps (0 = unused)
    vol_mult: float = 0.5
    vol_horizon_s: float = 0.3           # quote exposure: requote interval + latency
    skew_ticks_at_max: float = 2.0       # reservation shift, ticks, when |q| = max_inventory
    vol_halflife_s: float = 60.0
    tick: float = 1.0


class EwmaVol:
    """Time-aware EWMA of variance per millisecond of log mid returns."""

    def __init__(self, halflife_s: float, init_bps_per_sqrt_s: float = 0.5):
        self.halflife_ms = halflife_s * 1000
        self.var_per_ms = (init_bps_per_sqrt_s * 1e-4) ** 2 / 1000
        self.last_ts: int | None = None
        self.last_mid: float | None = None

    def update(self, ts: int, mid: float) -> None:
        if self.last_ts is not None and ts > self.last_ts:
            dt = ts - self.last_ts
            r = math.log(mid / self.last_mid)
            w = 1 - 0.5 ** (dt / self.halflife_ms)
            self.var_per_ms = (1 - w) * self.var_per_ms + w * (r * r / dt)
        self.last_ts, self.last_mid = ts, mid

    def sigma_bps(self, horizon_s: float) -> float:
        return math.sqrt(self.var_per_ms * horizon_s * 1000) * 1e4


@dataclass
class Quote:
    bid_px: float | None
    ask_px: float | None
    bid_sz: float
    ask_sz: float
    half_spread_bps: float
    skew_bps: float


class InventorySkewQuoter:
    def __init__(self, params: QuoterParams = QuoterParams()):
        self.p = params
        self.vol = EwmaVol(params.vol_halflife_s)

    def on_mid(self, ts: int, mid: float) -> None:
        self.vol.update(ts, mid)

    def quote(self, mid: float, best_bid: float, best_ask: float, inventory: float) -> Quote:
        p = self.p
        tick_bps = p.tick / mid * 1e4
        half = max(p.min_half_spread_ticks * tick_bps,
                   p.min_half_spread_bps,
                   p.vol_mult * self.vol.sigma_bps(p.vol_horizon_s))
        skew = p.skew_ticks_at_max * tick_bps * inventory / p.max_inventory
        r = mid * (1 - skew / 1e4)
        bid = math.floor(r * (1 - half / 1e4) / p.tick) * p.tick
        ask = math.ceil(r * (1 + half / 1e4) / p.tick) * p.tick
        # post-only: never cross the current book
        bid = min(bid, best_ask - p.tick)
        ask = max(ask, best_bid + p.tick)

        bid_sz = max(0.0, min(p.order_size, p.max_inventory - inventory))
        ask_sz = max(0.0, min(p.order_size, p.max_inventory + inventory))
        return Quote(
            bid_px=bid if bid_sz > 0 else None,
            ask_px=ask if ask_sz > 0 else None,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
            half_spread_bps=half,
            skew_bps=skew,
        )
