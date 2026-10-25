import numpy as np
import pandas as pd
import pytest

from mmlab import analytics
from mmlab.simulator import SimConfig, run
from mmlab.strategy import InventorySkewQuoter, QuoterParams
from mmlab.synthetic import SyntheticConfig, generate


def _book(ts, bid, ask):
    b = pd.DataFrame({"ts": ts, "recv_ts": ts, "bid_px": bid, "bid_sz": 1.0, "ask_px": ask, "ask_sz": 1.0,
                      "bid_depth5": 1.0, "ask_depth5": 1.0})
    b["mid"] = (b["bid_px"] + b["ask_px"]) / 2
    return b


class FixedQuoter(InventorySkewQuoter):
    """Always quotes a fixed bid/ask, for hand-checkable tests."""

    def __init__(self, bid, ask, size=1.0):
        super().__init__(QuoterParams(order_size=size, max_inventory=10.0))
        self.b, self.a = bid, ask

    def quote(self, mid, best_bid, best_ask, inventory):
        q = super().quote(mid, best_bid, best_ask, inventory)
        q.bid_px, q.ask_px = self.b, self.a
        return q


def test_accounting_identity_on_synthetic_data():
    book, trades = generate(SyntheticConfig(hours=0.5, seed=1))
    res = run(book, trades, InventorySkewQuoter(QuoterParams()), SimConfig())
    s = res.states
    assert len(res.fills) > 0
    lhs = s["cash"] + s["inventory"] * s["mid"]
    rhs = s["spread_capture"] + s["inventory_pnl"] - s["fees"]
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_print_through_our_bid_fills_full_size_at_our_price():
    book = _book([0, 1000, 2000], [99, 99, 99], [101, 101, 101])
    trades = pd.DataFrame({"ts": [1500], "recv_ts": [1500], "px": [97.0], "sz": [0.1], "side": [-1]})
    res = run(book, trades, FixedQuoter(98.0, 102.0, size=1.0), SimConfig(latency_ms=10, requote_ms=10_000, maker_fee=0))
    f = res.fills.iloc[0]
    assert (f["side"], f["px"], f["qty"]) == (1, 98.0, 1.0)  # walked through our level: full size
    assert res.fills["through"].iloc[0]


def test_print_at_our_bid_fills_queue_fraction():
    book = _book([0, 1000], [98, 98], [101, 101])
    trades = pd.DataFrame({"ts": [500], "recv_ts": [500], "px": [98.0], "sz": [2.0], "side": [-1]})
    res = run(book, trades, FixedQuoter(98.0, 102.0, size=1.0),
              SimConfig(latency_ms=10, requote_ms=10_000, maker_fee=0, touch_fill_ratio=0.25))
    assert res.fills["qty"].iloc[0] == pytest.approx(0.5)


def test_quote_not_live_before_latency_elapsed():
    book = _book([0, 1000], [99, 99], [101, 101])
    trades = pd.DataFrame({"ts": [20], "recv_ts": [20], "px": [97.0], "sz": [1.0], "side": [-1]})
    res = run(book, trades, FixedQuoter(98.0, 102.0), SimConfig(latency_ms=50, requote_ms=10_000))
    assert res.fills.empty


def test_crossing_quote_rejected_post_only():
    # quote decided at ts=0 with bid 98; by activation the ask has dropped to 97.5
    book = _book([0, 30, 1000], [99, 97, 97], [101, 97.5, 97.5])
    trades = pd.DataFrame({"ts": [60], "recv_ts": [60], "px": [96.0], "sz": [1.0], "side": [-1]})
    res = run(book, trades, FixedQuoter(98.0, 102.0), SimConfig(latency_ms=50, requote_ms=10_000))
    assert res.fills.empty


def test_markout_sign():
    book = _book([0, 1000, 2000, 3000], [99, 99, 103, 103], [101, 101, 105, 105])
    trades = pd.DataFrame({"ts": [500], "recv_ts": [500], "px": [97.0], "sz": [1.0], "side": [-1]})
    res = run(book, trades, FixedQuoter(98.0, 110.0), SimConfig(latency_ms=10, requote_ms=10_000, maker_fee=0))
    mo = analytics.markouts(res, book, horizons_s=(2,))
    # bought at 98, mid was 100 at fill (+2 => ~204 bps) and 104 two seconds later (+6)
    assert mo["h_0s"].iloc[0] == pytest.approx(2 / 98 * 1e4)
    assert mo["h_2s"].iloc[0] == pytest.approx(6 / 98 * 1e4)


def test_inventory_limit_respected_exactly_without_latency():
    book, trades = generate(SyntheticConfig(hours=0.5, seed=3))
    p = QuoterParams(max_inventory=0.05, order_size=0.02)
    res = run(book, trades, InventorySkewQuoter(p), SimConfig(latency_ms=0))
    assert res.states["inventory"].abs().max() <= p.max_inventory + 1e-9


def test_inventory_overshoot_is_bounded_by_one_order_size():
    """With latency, a quote is sized on stale inventory and still fills.

    The quoter caps size using the inventory it knew when it decided; by the
    time the order is live, an earlier quote may already have filled. A resting
    order is not cancelled because we breached our own limit, so the breach is
    real -- but it is bounded by the one order that was in flight.
    """
    book, trades = generate(SyntheticConfig(hours=0.5, seed=3))
    p = QuoterParams(max_inventory=0.05, order_size=0.02)
    res = run(book, trades, InventorySkewQuoter(p), SimConfig(latency_ms=50))
    peak = res.states["inventory"].abs().max()
    assert peak > p.max_inventory                                  # the breach is real
    assert peak <= p.max_inventory + p.order_size + 1e-9           # and bounded
