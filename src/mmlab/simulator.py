"""Event-driven market-making simulator on recorded book + trade data.

Fill model (the most important assumption in any MM backtest, state it in
your write-up):

* A seller-initiated print strictly BELOW our bid means the aggressor walked
  through our price level, so our resting size there is fully filled.
* A print exactly AT our bid means we were somewhere in the queue: we fill
  `touch_fill_ratio * print size`, capped at our remaining size.
* Symmetric for asks.
* New quotes go live `latency_ms` after the decision. A quote that would cross
  the book at activation time is rejected (post-only / ALO behaviour).
* Our orders have no market impact: the recorded tape does not react to us.
  This flatters results, especially with large sizes.
* Inventory can exceed `max_inventory` by up to one `order_size`. A quote is
  sized against the inventory known at DECISION time, but it goes live one
  latency hop later, and a resting order fills whether or not we have since
  breached our own limit. The overshoot is exactly zero at `latency_ms=0` and
  exactly one `order_size` otherwise, so read `max_abs_inventory` as
  `max_inventory + order_size`, and size `order_size` accordingly.

Accounting identity (checked in tests), everything marked to mid:

    equity = cash + q * mid = spread_capture + inventory_pnl - fees

    spread_capture = sum over fills of side * qty * (mid_at_fill - fill_px)
    inventory_pnl  = sum over book updates of q * (mid_t - mid_{t-1})
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import InventorySkewQuoter


@dataclass
class SimConfig:
    latency_ms: int = 50
    requote_ms: int = 250
    maker_fee: float = 0.00015   # fraction of notional; negative = rebate. Check the venue's current fee tier.
    touch_fill_ratio: float = 0.25


@dataclass
class SimResult:
    fills: pd.DataFrame
    states: pd.DataFrame
    quotes: pd.DataFrame
    config: SimConfig
    meta: dict = field(default_factory=dict)


def run(book: pd.DataFrame, trades: pd.DataFrame, strategy: InventorySkewQuoter, cfg: SimConfig = SimConfig()) -> SimResult:
    b_ts = book["ts"].to_numpy(np.int64)
    b_bid = book["bid_px"].to_numpy(float)
    b_ask = book["ask_px"].to_numpy(float)
    b_mid = book["mid"].to_numpy(float)
    t_ts = trades["ts"].to_numpy(np.int64)
    t_px = trades["px"].to_numpy(float)
    t_sz = trades["sz"].to_numpy(float)
    t_side = trades["side"].to_numpy(np.int64)

    # merged event stream: book updates (kind 0) before trades (kind 1) at equal ts
    ev_ts = np.concatenate([b_ts, t_ts])
    ev_kind = np.concatenate([np.zeros(len(b_ts), np.int8), np.ones(len(t_ts), np.int8)])
    ev_idx = np.concatenate([np.arange(len(b_ts)), np.arange(len(t_ts))])
    order = np.lexsort((ev_kind, ev_ts))

    q = cash = edge = inv_pnl = fees = 0.0
    mid = best_bid = best_ask = np.nan
    bid_px = ask_px = None
    bid_rem = ask_rem = 0.0
    pending = None  # (activation_ts, Quote)
    last_quote_ts = -(10**18)

    fills, quotes = [], []
    st_ts, st_mid, st_q, st_edge, st_inv, st_fee, st_cash = [], [], [], [], [], [], []

    for k in order:
        ts = int(ev_ts[k])
        i = int(ev_idx[k])

        # activate pending quote once latency has elapsed
        if pending is not None and pending[0] <= ts and not np.isnan(mid):
            qt = pending[1]
            bid_px = qt.bid_px if (qt.bid_px is not None and qt.bid_px < best_ask) else None
            ask_px = qt.ask_px if (qt.ask_px is not None and qt.ask_px > best_bid) else None
            bid_rem = qt.bid_sz if bid_px is not None else 0.0
            ask_rem = qt.ask_sz if ask_px is not None else 0.0
            pending = None

        if ev_kind[k] == 0:  # ---- book update
            new_mid = b_mid[i]
            if not np.isnan(mid):
                inv_pnl += q * (new_mid - mid)
            mid, best_bid, best_ask = new_mid, b_bid[i], b_ask[i]
            strategy.on_mid(ts, mid)

            if ts - last_quote_ts >= cfg.requote_ms:
                qt = strategy.quote(mid, best_bid, best_ask, q)
                pending = (ts + cfg.latency_ms, qt)
                last_quote_ts = ts
                quotes.append((ts, mid, qt.bid_px, qt.ask_px, qt.half_spread_bps, qt.skew_bps, q))

            st_ts.append(ts); st_mid.append(mid); st_q.append(q)
            st_edge.append(edge); st_inv.append(inv_pnl); st_fee.append(fees); st_cash.append(cash)

        else:  # ---- trade print
            if np.isnan(mid):
                continue
            px, sz, side = t_px[i], t_sz[i], t_side[i]
            fill_side, fill_px, qty = 0, 0.0, 0.0
            if side == -1 and bid_px is not None and bid_rem > 0 and px <= bid_px:
                fill_side, fill_px = 1, bid_px
                qty = bid_rem if px < bid_px else min(bid_rem, sz * cfg.touch_fill_ratio)
                bid_rem -= qty
            elif side == 1 and ask_px is not None and ask_rem > 0 and px >= ask_px:
                fill_side, fill_px = -1, ask_px
                qty = ask_rem if px > ask_px else min(ask_rem, sz * cfg.touch_fill_ratio)
                ask_rem -= qty
            if qty > 0:
                fee = cfg.maker_fee * fill_px * qty
                edge += fill_side * qty * (mid - fill_px)
                cash -= fill_side * fill_px * qty + fee
                fees += fee
                q += fill_side * qty
                fills.append((ts, fill_side, fill_px, qty, mid, fee, q, px < bid_px if fill_side == 1 else px > ask_px))

    if len(order) and not np.isnan(mid):  # final state, includes fills after the last book update
        st_ts.append(int(ev_ts[order[-1]])); st_mid.append(mid); st_q.append(q)
        st_edge.append(edge); st_inv.append(inv_pnl); st_fee.append(fees); st_cash.append(cash)

    fills_df = pd.DataFrame(fills, columns=["ts", "side", "px", "qty", "mid", "fee", "inventory", "through"])
    states = pd.DataFrame({
        "ts": st_ts, "mid": st_mid, "inventory": st_q, "cash": st_cash,
        "spread_capture": st_edge, "inventory_pnl": st_inv, "fees": st_fee,
    })
    states["equity"] = states["cash"] + states["inventory"] * states["mid"]
    quotes_df = pd.DataFrame(quotes, columns=["ts", "mid", "bid_px", "ask_px", "half_spread_bps", "skew_bps", "inventory"])
    return SimResult(fills=fills_df, states=states, quotes=quotes_df, config=cfg)
