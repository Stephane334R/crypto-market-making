"""Download free historical Hyperliquid data from Tardis.dev.

Tardis publishes the **first day of every month** without an API key, which is
more useful than it sounds: ~24 days are available today (2024-11-01 onwards),
and 24 non-contiguous days sample 24 independent volatility regimes, whereas a
couple of consecutively recorded days sample one. That is exactly what the
parameter sweep needs. Use `mmlab.recorder` for ground truth at your own
latency, and this for breadth.

    python -m mmlab.tardis --coin BTC --out data/processed/BTC

`book_snapshot_5` carries 5 levels per side, so it maps onto the same schema
`mmlab.data` produces, plus a `day` column used for leave-days-out validation:

    timestamp (us)          -> ts        (// 1000)
    local_timestamp (us)    -> recv_ts   (Tardis's collector, NOT your latency)
    bids[0].price/amount    -> bid_px / bid_sz
    sum(bids[0..4].amount)  -> bid_depth5
    trades side buy/sell    -> +1 / -1   (aggressor, same as mmlab.data)

Set TARDIS_API_KEY to reach the paid days; without it, non-free days 401.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .data import clean_book, clean_trades, infer_tick

log = logging.getLogger("tardis")

API_URL = "https://api.tardis.dev/v1/exchanges/{exchange}"
DATASETS_URL = "https://datasets.tardis.dev/v1/{exchange}/{data_type}/{y:04d}/{m:02d}/{d:02d}/{symbol}.csv.gz"
EXCHANGE = "hyperliquid"
DATA_TYPES = ("book_snapshot_5", "trades")


def _request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    key = os.environ.get("TARDIS_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return req


def symbol_availability(symbol: str, exchange: str = EXCHANGE) -> tuple[dt.date, dt.date]:
    """(available_since, available_to) for one symbol, from the Tardis API."""
    with urllib.request.urlopen(_request(API_URL.format(exchange=exchange))) as r:
        info = json.load(r)
    for s in info.get("datasets", {}).get("symbols", []):
        if s.get("id", "").upper() == symbol.upper():
            missing = set(DATA_TYPES) - set(s.get("dataTypes", []))
            if missing:
                raise SystemExit(f"{exchange}:{symbol} has no {sorted(missing)} datasets")
            return (dt.date.fromisoformat(s["availableSince"][:10]),
                    dt.date.fromisoformat(s["availableTo"][:10]))
    raise SystemExit(f"{symbol} not available as a dataset on {exchange}")


def free_days(since: dt.date, to: dt.date) -> list[dt.date]:
    """First of each month inside [since, to] -- the days Tardis serves for free."""
    days, y, m = [], since.year, since.month
    while (y, m) <= (to.year, to.month):
        d = dt.date(y, m, 1)
        if since <= d <= to:
            days.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return days


def download(symbol: str, data_type: str, day: dt.date, cache: Path,
             exchange: str = EXCHANGE) -> Path | None:
    """Fetch one coin-day-datatype into `cache`, skipping files already there."""
    path = cache / symbol / data_type / f"{day:%Y%m%d}.csv.gz"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = DATASETS_URL.format(exchange=exchange, data_type=data_type,
                              y=day.year, m=day.month, d=day.day, symbol=symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(_request(url)) as r:
            blob = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            log.warning("%s %s %s: not free (HTTP %d); set TARDIS_API_KEY to fetch it",
                        symbol, data_type, day, exc.code)
            return None
        raise
    path.write_bytes(blob)
    log.info("downloaded %s (%.1f MB)", path, len(blob) / 1e6)
    return path


def parse_book(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bid_cols = [f"bids[{i}].amount" for i in range(5)]
    ask_cols = [f"asks[{i}].amount" for i in range(5)]
    return pd.DataFrame({
        "ts": df["timestamp"] // 1000,
        "recv_ts": df["local_timestamp"] // 1000,
        "bid_px": df["bids[0].price"].astype(float),
        "bid_sz": df["bids[0].amount"].astype(float),
        "ask_px": df["asks[0].price"].astype(float),
        "ask_sz": df["asks[0].amount"].astype(float),
        "bid_depth5": df[bid_cols].astype(float).sum(axis=1),
        "ask_depth5": df[ask_cols].astype(float).sum(axis=1),
    })


def parse_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "ts": df["timestamp"] // 1000,
        "recv_ts": df["local_timestamp"] // 1000,
        "px": df["price"].astype(float),
        "sz": df["amount"].astype(float),
        # Tardis `side` is the aggressor, matching mmlab.data's SIDE_MAP.
        "side": df["side"].map({"buy": 1, "sell": -1}).astype("int64"),
    })


def load_days(symbol: str, days: list[dt.date], cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    books, trades = [], []
    for day in days:
        b = download(symbol, "book_snapshot_5", day, cache)
        t = download(symbol, "trades", day, cache)
        if b is None or t is None:
            continue
        stamp = int(f"{day:%Y%m%d}")
        books.append(parse_book(b).assign(day=stamp))
        trades.append(parse_trades(t).assign(day=stamp))
    if not books:
        raise SystemExit("no days downloaded")
    # Per-day cleaning: each day is a separate session, so a dedup or sort must
    # not reach across the overnight gap.
    book = pd.concat([clean_book(b) for b in books], ignore_index=True)
    trade = pd.concat([clean_trades(t) for t in trades], ignore_index=True)
    return book, trade


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coin", default="BTC")
    p.add_argument("--out", type=Path, help="where book/trades parquet go (not needed with --list)")
    p.add_argument("--cache", type=Path, default=Path("data/tardis"), help="raw csv.gz cache")
    p.add_argument("--max-days", type=int, default=None, help="only the most recent N free days")
    p.add_argument("--list", action="store_true", help="list the free days and exit")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    since, to = symbol_availability(args.coin)
    days = free_days(since, to)
    log.info("%s: datasets %s..%s -> %d free days", args.coin, since, to, len(days))
    if args.max_days:
        days = days[-args.max_days:]
    if args.list:
        print("\n".join(str(d) for d in days))
        return
    if args.out is None:
        p.error("--out is required unless --list is given")

    book, trades = load_days(args.coin, days, args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    book.to_parquet(args.out / "book.parquet", index=False)
    trades.to_parquet(args.out / "trades.parquet", index=False)
    log.info("days: %d  book rows: %d  trade rows: %d",
             book["day"].nunique(), len(book), len(trades))
    log.info("inferred tick: %g", infer_tick(book))


if __name__ == "__main__":
    main()
