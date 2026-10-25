"""Offline tests for the Tardis adapter: no network, fixtures only."""
import datetime as dt
import gzip

import pandas as pd
import pytest

from mmlab import tardis
from mmlab.backtest import ensure_day, run_by_day
from mmlab.simulator import SimConfig
from mmlab.strategy import QuoterParams

BOOK_CSV = """exchange,symbol,timestamp,local_timestamp,asks[0].price,asks[0].amount,bids[0].price,bids[0].amount,asks[1].price,asks[1].amount,bids[1].price,bids[1].amount,asks[2].price,asks[2].amount,bids[2].price,bids[2].amount,asks[3].price,asks[3].amount,bids[3].price,bids[3].amount,asks[4].price,asks[4].amount,bids[4].price,bids[4].amount
hyperliquid,BTC,1748736000147000,1748736005245132,104585,2.0,104584,6.0,104586,0.1,104583,0.4,104589,0.3,104582,0.5,104590,1.0,104581,0.7,104591,0.2,104580,0.4
hyperliquid,BTC,1748736001147000,1748736006245132,104586,1.0,104585,2.0,104587,0.1,104584,0.4,104588,0.3,104583,0.5,104589,1.0,104582,0.7,104590,0.2,104581,0.4
"""

TRADES_CSV = """exchange,symbol,timestamp,local_timestamp,id,side,price,amount
hyperliquid,BTC,1748736000500000,1748736005944854,1,buy,104585,1.5
hyperliquid,BTC,1748736001500000,1748736006944854,2,sell,104584,0.25
"""


@pytest.fixture
def book_file(tmp_path):
    p = tmp_path / "book.csv.gz"
    p.write_bytes(gzip.compress(BOOK_CSV.encode()))
    return p


@pytest.fixture
def trades_file(tmp_path):
    p = tmp_path / "trades.csv.gz"
    p.write_bytes(gzip.compress(TRADES_CSV.encode()))
    return p


def test_free_days_is_first_of_each_month_inside_the_window():
    days = tardis.free_days(dt.date(2024, 10, 29), dt.date(2025, 2, 15))
    # 2024-10-01 precedes availableSince, 2025-03-01 follows availableTo
    assert days == [dt.date(2024, 11, 1), dt.date(2024, 12, 1),
                    dt.date(2025, 1, 1), dt.date(2025, 2, 1)]


def test_free_days_excludes_a_first_before_available_since():
    assert tardis.free_days(dt.date(2025, 1, 2), dt.date(2025, 2, 1)) == [dt.date(2025, 2, 1)]


def test_parse_book_maps_schema_and_sums_five_levels(book_file):
    b = tardis.parse_book(book_file)
    assert list(b.columns) == ["ts", "recv_ts", "bid_px", "bid_sz", "ask_px", "ask_sz",
                               "bid_depth5", "ask_depth5"]
    assert b["ts"].iloc[0] == 1748736000147          # microseconds -> milliseconds
    assert b["recv_ts"].iloc[0] == 1748736005245
    assert b["bid_px"].iloc[0] == 104584
    assert b["bid_depth5"].iloc[0] == pytest.approx(6.0 + 0.4 + 0.5 + 0.7 + 0.4)
    assert b["ask_depth5"].iloc[0] == pytest.approx(2.0 + 0.1 + 0.3 + 1.0 + 0.2)


def test_parse_trades_maps_aggressor_side(trades_file):
    t = tardis.parse_trades(trades_file)
    assert t["side"].tolist() == [1, -1]             # buy -> +1, sell -> -1
    assert t["ts"].tolist() == [1748736000500, 1748736001500]


def test_run_by_day_keeps_days_independent():
    """Two sessions a month and 300 points apart must not share state.

    Running them together must give exactly what running each alone gives; a
    single run over the concatenation would book the overnight gap as inventory
    PnL instead.
    """
    books, trades = [], []
    for day, base_ts, px in ((20250601, 1_700_000_000_000, 100.0),
                             (20250701, 1_702_592_000_000, 400.0)):
        b = pd.DataFrame({"ts": [base_ts + i * 1000 for i in range(40)]})
        b = b.assign(recv_ts=b["ts"], bid_px=px, bid_sz=1.0, ask_px=px + 1, ask_sz=1.0,
                     bid_depth5=5.0, ask_depth5=5.0, day=day,
                     mid=px + 0.5)
        books.append(b)
        trades.append(pd.DataFrame({"ts": [base_ts + 5000, base_ts + 15000],
                                    "recv_ts": [base_ts + 5000, base_ts + 15000],
                                    "px": [px - 2, px + 3], "sz": [1.0, 1.0],
                                    "side": [-1, 1], "day": [day, day]}))
    book, trade = pd.concat(books, ignore_index=True), pd.concat(trades, ignore_index=True)

    params, cfg = QuoterParams(tick=1.0), SimConfig()
    together = run_by_day(book, trade, params, cfg)
    assert together["day"].tolist() == [20250601, 20250701]
    assert together["fills"].sum() > 0, "fixture produced no fills, test proves nothing"

    for _, row in together.iterrows():
        d = row["day"]
        alone = run_by_day(book[book["day"] == d].reset_index(drop=True),
                           trade[trade["day"] == d].reset_index(drop=True), params, cfg)
        assert alone["pnl_total"].iloc[0] == pytest.approx(row["pnl_total"])


def test_run_by_day_keeps_columns_when_no_day_is_usable():
    book = pd.DataFrame({"ts": [1], "recv_ts": [1], "bid_px": [1.0], "bid_sz": [1.0],
                         "ask_px": [2.0], "ask_sz": [1.0], "bid_depth5": [1.0],
                         "ask_depth5": [1.0], "mid": [1.5], "day": [20250601]})
    empty = pd.DataFrame(columns=["ts", "recv_ts", "px", "sz", "side", "day"])
    out = run_by_day(book, empty, QuoterParams(tick=1.0), SimConfig())
    assert out.empty and "day" in out.columns


def test_ensure_day_derives_day_from_ts_when_absent():
    book = pd.DataFrame({"ts": [1748736000147, 1748822400147]})
    trades = pd.DataFrame({"ts": [1748736000500]})
    b, t = ensure_day(book, trades)
    assert b["day"].tolist() == [20250601, 20250602]
    assert t["day"].tolist() == [20250601]
