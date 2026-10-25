"""Command-line entry point: run one backtest, or a parameter sweep with a
train/test split.

    # offline demo on synthetic data
    python -m mmlab.backtest --synthetic

    # real recorded data (after `python -m mmlab.data ...`); the tick is
    # inferred from the book unless you pass --tick
    python -m mmlab.backtest --data data/processed/BTC

    # grid search on the first half, evaluate the chosen parameters on the second half
    # (only meaningful on one continuous recording session)
    python -m mmlab.backtest --data data/processed/BTC --sweep

    # leave-days-out: select on the earlier days, test on the held-out later
    # ones, and report the per-day PnL distribution rather than one number.
    # This is the right mode for Tardis data (see `mmlab.tardis`).
    python -m mmlab.backtest --data data/processed/BTC_tardis --cv
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, plots
from .data import infer_tick, load_processed
from .simulator import SimConfig, run
from .strategy import InventorySkewQuoter, QuoterParams
from .synthetic import SyntheticConfig, generate


def split_by_time(book: pd.DataFrame, trades: pd.DataFrame, frac: float):
    cut = book["ts"].iloc[int(len(book) * frac)]
    return (
        (book[book["ts"] < cut].reset_index(drop=True), trades[trades["ts"] < cut].reset_index(drop=True)),
        (book[book["ts"] >= cut].reset_index(drop=True), trades[trades["ts"] >= cut].reset_index(drop=True)),
    )


def ensure_day(book: pd.DataFrame, trades: pd.DataFrame):
    """Guarantee a `day` column (YYYYMMDD int). `mmlab.tardis` sets it; data
    coming from `mmlab.data` does not, so derive it from the timestamp."""
    out = []
    for df in (book, trades):
        if "day" in df.columns:
            out.append(df)
        else:
            stamp = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y%m%d").astype(int)
            out.append(df.assign(day=stamp))
    return out[0], out[1]


def run_by_day(book: pd.DataFrame, trades: pd.DataFrame, params: QuoterParams, cfg: SimConfig) -> pd.DataFrame:
    """Run one simulation per day, with state reset between days.

    Days from Tardis are non-contiguous (the 1st of each month), so inventory
    must not carry across the gap and a single run over the concatenation would
    invent an overnight mid jump worth more than a day of spread capture.
    """
    rows = []
    for day, b in book.groupby("day", sort=True):
        t = trades[trades["day"] == day]
        if len(b) < 2 or t.empty:
            continue
        _, _, st = run_once(b.reset_index(drop=True), t.reset_index(drop=True), params, cfg)
        rows.append({"day": int(day), "pnl_total": st["pnl_total"], "fills": st["fills"],
                     "pnl_per_volume_bps": st["pnl_per_volume_bps"],
                     "max_drawdown": st["max_drawdown"],
                     "max_abs_inventory": st["max_abs_inventory"]})
    cols = ["day", "pnl_total", "fills", "pnl_per_volume_bps", "max_drawdown", "max_abs_inventory"]
    return pd.DataFrame(rows, columns=cols)  # keep the columns when every day is skipped


def _dist(per_day: pd.DataFrame) -> dict:
    """Mean per-day PnL with its standard error -- the honest headline number."""
    v = per_day["pnl_total"].to_numpy(float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"days": n, "pnl_per_day_mean": round(float(v.mean()), 2),
            "pnl_per_day_std": round(sd, 2) if n > 1 else None,
            "pnl_per_day_stderr": round(se, 2) if n > 1 else None,
            "t_stat": round(float(v.mean() / se), 2) if n > 1 and se > 0 else None,
            "profitable_days": f"{int((v > 0).sum())}/{n}"}


def run_once(book, trades, params: QuoterParams, cfg: SimConfig):
    result = run(book, trades, InventorySkewQuoter(params), cfg)
    mo = analytics.markouts(result, book)
    stats = analytics.summary(result, mo)
    analytics.add_inventory_limit_share(stats, result, params.max_inventory)
    return result, mo, stats


def write_report(out: Path, result, mo, stats, params: QuoterParams, cfg: SimConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stats = {**stats, "params": asdict(params), "sim_config": asdict(cfg)}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    result.fills.to_csv(out / "fills.csv", index=False)
    plots.pnl_attribution(result.states, out / "pnl_attribution.png")
    plots.inventory(result.states, params.max_inventory, out / "inventory.png")
    if not mo.empty:
        curve = analytics.markout_curve(mo)
        curve.to_csv(out / "markout_curve.csv", index=False)
        plots.markout_curve(curve, out / "markout_curve.png")


# In ticks, so the same grid is meaningful on BTC ($1 tick) and ETH ($0.1).
SWEEP_GRID = {"min_half_spread_ticks": [1.0, 2.0, 4.0, 8.0], "skew_ticks_at_max": [0.0, 1.0, 2.0, 4.0]}


def _cv(book, trades, params: QuoterParams, cfg: SimConfig, args) -> None:
    """Grid search on the earlier days, evaluate on the held-out later ones.

    Reports the per-day PnL distribution on the test days, not a single total.
    One number from one split cannot tell you whether a parameter choice
    survived or whether one day's trend carried it; the mean, its standard
    error and the profitable-day count can.
    """
    book, trades = ensure_day(book, trades)
    days = sorted(book["day"].unique())
    n_test = args.test_days if args.test_days is not None else max(2, round(0.25 * len(days)))
    if len(days) - n_test < 2:
        raise SystemExit(f"need at least 2 train + {n_test} test days, have {len(days)}")
    train_days, test_days = days[:-n_test], days[-n_test:]
    print(f"{len(days)} days: train on {len(train_days)} ({train_days[0]}..{train_days[-1]}), "
          f"test on {len(test_days)} ({test_days[0]}..{test_days[-1]})")

    tr_b, tr_t = book[book["day"].isin(train_days)], trades[trades["day"].isin(train_days)]
    rows = []
    for hs, sk in itertools.product(*SWEEP_GRID.values()):
        ps = replace(params, min_half_spread_ticks=hs, skew_ticks_at_max=sk)
        per_day = run_by_day(tr_b, tr_t, ps, cfg)
        d = _dist(per_day)
        rows.append({"min_half_spread_ticks": hs, "skew_ticks_at_max": sk,
                     "pnl_total": float(per_day["pnl_total"].sum()), "fills": int(per_day["fills"].sum()),
                     **d})
        print(f"train  hs={hs:<4} skew={sk:<4} pnl/day={d['pnl_per_day_mean']:>9.2f} "
              f"+/-{d['pnl_per_day_stderr'] or float('nan'):<8.2f} good_days={d['profitable_days']}")

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "cv_train.csv", index=False)
    plots.sweep_heatmap(table, "pnl_per_day_mean", args.out / "sweep_train_pnl.png")

    best = table.sort_values("pnl_per_day_mean", ascending=False).iloc[0]
    chosen = replace(params, min_half_spread_ticks=float(best["min_half_spread_ticks"]),
                     skew_ticks_at_max=float(best["skew_ticks_at_max"]))
    te_b, te_t = book[book["day"].isin(test_days)], trades[trades["day"].isin(test_days)]
    per_day = run_by_day(te_b, te_t, chosen, cfg)
    per_day.to_csv(args.out / "cv_test_per_day.csv", index=False)

    print("\nheld-out days, parameters chosen on the train days:")
    print(per_day.to_string(index=False))
    out = {"selected_on_train": {"min_half_spread_ticks": chosen.min_half_spread_ticks,
                                 "skew_ticks_at_max": chosen.skew_ticks_at_max,
                                 "train_pnl_per_day": float(best["pnl_per_day_mean"])},
           "test": _dist(per_day), "sim_config": asdict(cfg), "params": asdict(chosen)}
    (args.out / "cv_summary.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out["test"], indent=2))
    if out["test"]["t_stat"] is not None and abs(out["test"]["t_stat"]) < 2:
        print("\nNOTE: |t| < 2 across held-out days -- this result is not distinguishable "
              "from zero. More days, not more grid points.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/latest"))
    p.add_argument("--tick", type=float, default=None, help="default: inferred from the book")
    p.add_argument("--order-size", type=float, default=0.02)
    p.add_argument("--max-inventory", type=float, default=0.10)
    p.add_argument("--min-half-spread-ticks", type=float, default=1.0)
    p.add_argument("--vol-mult", type=float, default=0.5)
    p.add_argument("--vol-horizon-s", type=float, default=None,
                   help="default: (requote_ms + latency_ms) / 1000, i.e. the quote's exposure")
    p.add_argument("--skew-ticks", type=float, default=2.0)
    p.add_argument("--latency-ms", type=int, default=50)
    p.add_argument("--requote-ms", type=int, default=250)
    p.add_argument("--maker-fee", type=float, default=0.00015)
    p.add_argument("--touch-fill-ratio", type=float, default=0.25)
    p.add_argument("--sweep", action="store_true", help="grid search on first half, test on second half")
    p.add_argument("--cv", action="store_true",
                   help="leave-days-out: select on earlier days, test on held-out later days")
    p.add_argument("--test-days", type=int, default=None,
                   help="days held out by --cv (default: 25%% of days, at least 2)")
    args = p.parse_args()

    if args.synthetic:
        book, trades = generate(SyntheticConfig(hours=4.0))
    else:
        book, trades = load_processed(args.data)

    tick = args.tick if args.tick is not None else infer_tick(book)
    cfg = SimConfig(latency_ms=args.latency_ms, requote_ms=args.requote_ms,
                    maker_fee=args.maker_fee, touch_fill_ratio=args.touch_fill_ratio)
    # A quote is exposed for one requote interval plus one latency hop; that is
    # the horizon whose variance it has to cover.
    vol_horizon_s = (args.vol_horizon_s if args.vol_horizon_s is not None
                     else (cfg.requote_ms + cfg.latency_ms) / 1000)
    params = QuoterParams(
        order_size=args.order_size, max_inventory=args.max_inventory,
        min_half_spread_ticks=args.min_half_spread_ticks, vol_mult=args.vol_mult,
        vol_horizon_s=vol_horizon_s, skew_ticks_at_max=args.skew_ticks, tick=tick,
    )
    print(f"tick={tick:g}  median spread={float((book['ask_px'] - book['bid_px']).median()) / tick:g} ticks  "
          f"vol_horizon={vol_horizon_s:g}s")

    if args.cv:
        _cv(book, trades, params, cfg, args)
        return

    if not args.sweep:
        result, mo, stats = run_once(book, trades, params, cfg)
        write_report(args.out, result, mo, stats, params, cfg)
        print(json.dumps(stats, indent=2))
        return

    (train_b, train_t), (test_b, test_t) = split_by_time(book, trades, 0.5)
    rows = []
    for hs, sk in itertools.product(*SWEEP_GRID.values()):
        ps = replace(params, min_half_spread_ticks=hs, skew_ticks_at_max=sk)
        _, _, st = run_once(train_b, train_t, ps, cfg)
        rows.append({"min_half_spread_ticks": hs, "skew_ticks_at_max": sk, "pnl_total": st["pnl_total"],
                     "fills": st["fills"], "max_drawdown": st["max_drawdown"],
                     "pnl_per_volume_bps": st["pnl_per_volume_bps"]})
        print(f"train  hs={hs:<4} skew={sk:<4} pnl={st['pnl_total']:>10.2f} fills={st['fills']}")  # ticks
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "sweep_train.csv", index=False)
    plots.sweep_heatmap(table, "pnl_total", args.out / "sweep_train_pnl.png")

    best = table.sort_values("pnl_total", ascending=False).iloc[0]
    chosen = replace(params, min_half_spread_ticks=float(best["min_half_spread_ticks"]),
                     skew_ticks_at_max=float(best["skew_ticks_at_max"]))
    result, mo, stats = run_once(test_b, test_t, chosen, cfg)
    stats["selected_on_train"] = {"min_half_spread_ticks": chosen.min_half_spread_ticks,
                                  "skew_ticks_at_max": chosen.skew_ticks_at_max,
                                  "train_pnl": float(best["pnl_total"])}
    write_report(args.out / "test", result, mo, stats, chosen, cfg)
    print("\nout-of-sample (second half) with parameters chosen on the first half:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
