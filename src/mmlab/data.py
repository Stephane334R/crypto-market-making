"""Parse raw recorder files into clean book / trade tables.

Output schemas (all timestamps are integer milliseconds, exchange time):

book   : ts, recv_ts, bid_px, bid_sz, ask_px, ask_sz, bid_depth5, ask_depth5, mid
trades : ts, recv_ts, px, sz, side      (side = +1 buyer-initiated, -1 seller-initiated)

    python -m mmlab.data --raw data/raw/BTC --out data/processed/BTC
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("data")

# Hyperliquid trade `side`: "B" = bid side / buyer aggressor, "A" = ask side /
# seller aggressor. `check_trade_side_convention` verifies this on your data.
SIDE_MAP = {"B": 1, "A": -1}


def parse_files(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    book_rows, trade_rows = [], []
    for path in sorted(files):
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # truncated last line after a crash
                    continue
                msg, recv_ts = rec["msg"], rec["recv_ts"]
                if msg["channel"] == "l2Book":
                    d = msg["data"]
                    bids, asks = d["levels"]
                    if not bids or not asks:
                        continue
                    book_rows.append((
                        d["time"], recv_ts,
                        float(bids[0]["px"]), float(bids[0]["sz"]),
                        float(asks[0]["px"]), float(asks[0]["sz"]),
                        sum(float(l["sz"]) for l in bids[:5]),
                        sum(float(l["sz"]) for l in asks[:5]),
                    ))
                elif msg["channel"] == "trades":
                    for t in msg["data"]:
                        trade_rows.append((t["time"], recv_ts, float(t["px"]), float(t["sz"]), SIDE_MAP[t["side"]]))

    book = pd.DataFrame(book_rows, columns=["ts", "recv_ts", "bid_px", "bid_sz", "ask_px", "ask_sz", "bid_depth5", "ask_depth5"])
    trades = pd.DataFrame(trade_rows, columns=["ts", "recv_ts", "px", "sz", "side"])
    return clean_book(book), clean_trades(trades)


def clean_book(book: pd.DataFrame) -> pd.DataFrame:
    book = book.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    book = book[book["ask_px"] > book["bid_px"]]  # drop crossed / locked snapshots
    book = book.assign(mid=(book["bid_px"] + book["ask_px"]) / 2)
    return book.reset_index(drop=True)


def clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
    return trades.sort_values("ts", kind="stable").reset_index(drop=True)


def infer_tick(book: pd.DataFrame) -> float:
    """Smallest price increment actually observed in the book.

    Hyperliquid quotes to at most 5 significant figures, so the tick depends on
    the coin's price level ($1 for BTC near $80k, $0.1 for ETH near $2.3k).
    Guessing it wrong rescales every spread parameter, so infer it from the data
    rather than passing a default. Floats coming out of JSON carry noise
    (0.1 arrives as 0.09999999999990905), hence the rounding.
    """
    px = np.unique(np.concatenate([book["bid_px"].to_numpy(), book["ask_px"].to_numpy()]))
    diffs = np.diff(np.sort(px))
    diffs = diffs[diffs > 1e-12]
    if not len(diffs):
        raise ValueError("cannot infer tick: fewer than two distinct prices in the book")
    return float(f"{diffs.min():.6g}")


def check_trade_side_convention(book: pd.DataFrame, trades: pd.DataFrame) -> float:
    """Share of buyer-initiated trades printing at or above the prevailing mid.

    Should be close to 1.0. If it is close to 0.0, the side mapping is inverted.
    """
    idx = np.searchsorted(book["ts"].to_numpy(), trades["ts"].to_numpy(), side="right") - 1
    ok = idx >= 0
    mid = book["mid"].to_numpy()[idx[ok]]
    t = trades[ok]
    buys = t["side"].to_numpy() == 1
    return float(np.mean(t["px"].to_numpy()[buys] >= mid[buys])) if buys.any() else float("nan")


def load_processed(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(folder / "book.parquet"), pd.read_parquet(folder / "trades.parquet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True, help="folder with *.jsonl.gz for one coin")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files = sorted(args.raw.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no raw files in {args.raw}")
    book, trades = parse_files(files)
    args.out.mkdir(parents=True, exist_ok=True)
    book.to_parquet(args.out / "book.parquet", index=False)
    trades.to_parquet(args.out / "trades.parquet", index=False)
    log.info("book rows: %d  trade rows: %d", len(book), len(trades))
    log.info("inferred tick: %g  (median spread: %g ticks)", infer_tick(book),
             float((book["ask_px"] - book["bid_px"]).median()) / infer_tick(book))
    log.info("side convention check (should be ~1.0): %.3f", check_trade_side_convention(book, trades))


if __name__ == "__main__":
    main()
