"""Command-line entry point: run one backtest, or a parameter sweep with a
train/test split.

    # offline demo on synthetic data
    python -m mmlab.backtest --synthetic

    # real recorded data (after `python -m mmlab.data ...`)
    python -m mmlab.backtest --data data/processed/BTC --tick 1

    # grid search on the first half, evaluate the chosen parameters on the second half
    python -m mmlab.backtest --data data/processed/BTC --sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from . import analytics, plots
from .data import load_processed
from .simulator import SimConfig, run
from .strategy import InventorySkewQuoter, QuoterParams
from .synthetic import SyntheticConfig, generate


def split_by_time(book: pd.DataFrame, trades: pd.DataFrame, frac: float):
    cut = book["ts"].iloc[int(len(book) * frac)]
    return (
        (book[book["ts"] < cut].reset_index(drop=True), trades[trades["ts"] < cut].reset_index(drop=True)),
        (book[book["ts"] >= cut].reset_index(drop=True), trades[trades["ts"] >= cut].reset_index(drop=True)),
    )


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/latest"))
    p.add_argument("--tick", type=float, default=1.0)
    p.add_argument("--order-size", type=float, default=0.02)
    p.add_argument("--max-inventory", type=float, default=0.10)
    p.add_argument("--min-half-spread-bps", type=float, default=1.0)
    p.add_argument("--vol-mult", type=float, default=1.0)
    p.add_argument("--skew-bps", type=float, default=3.0)
    p.add_argument("--latency-ms", type=int, default=50)
    p.add_argument("--requote-ms", type=int, default=250)
    p.add_argument("--maker-fee", type=float, default=0.00015)
    p.add_argument("--touch-fill-ratio", type=float, default=0.25)
    p.add_argument("--sweep", action="store_true", help="grid search on first half, test on second half")
    args = p.parse_args()

    if args.synthetic:
        book, trades = generate(SyntheticConfig(hours=4.0))
    else:
        book, trades = load_processed(args.data)

    params = QuoterParams(
        order_size=args.order_size, max_inventory=args.max_inventory,
        min_half_spread_bps=args.min_half_spread_bps, vol_mult=args.vol_mult,
        skew_bps_at_max=args.skew_bps, tick=args.tick,
    )
    cfg = SimConfig(latency_ms=args.latency_ms, requote_ms=args.requote_ms,
                    maker_fee=args.maker_fee, touch_fill_ratio=args.touch_fill_ratio)

    if not args.sweep:
        result, mo, stats = run_once(book, trades, params, cfg)
        write_report(args.out, result, mo, stats, params, cfg)
        print(json.dumps(stats, indent=2))
        return

    (train_b, train_t), (test_b, test_t) = split_by_time(book, trades, 0.5)
    grid = {"min_half_spread_bps": [1.0, 2.0, 4.0, 6.0], "skew_bps_at_max": [0.0, 2.0, 5.0, 10.0]}
    rows = []
    for hs, sk in itertools.product(*grid.values()):
        ps = replace(params, min_half_spread_bps=hs, skew_bps_at_max=sk)
        _, _, st = run_once(train_b, train_t, ps, cfg)
        rows.append({"min_half_spread_bps": hs, "skew_bps_at_max": sk, "pnl_total": st["pnl_total"],
                     "fills": st["fills"], "max_drawdown": st["max_drawdown"],
                     "pnl_per_volume_bps": st["pnl_per_volume_bps"]})
        print(f"train  hs={hs:<4} skew={sk:<4} pnl={st['pnl_total']:>10.2f} fills={st['fills']}")
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "sweep_train.csv", index=False)
    plots.sweep_heatmap(table, "pnl_total", args.out / "sweep_train_pnl.png")

    best = table.sort_values("pnl_total", ascending=False).iloc[0]
    chosen = replace(params, min_half_spread_bps=float(best["min_half_spread_bps"]),
                     skew_bps_at_max=float(best["skew_bps_at_max"]))
    result, mo, stats = run_once(test_b, test_t, chosen, cfg)
    stats["selected_on_train"] = {"min_half_spread_bps": chosen.min_half_spread_bps,
                                  "skew_bps_at_max": chosen.skew_bps_at_max,
                                  "train_pnl": float(best["pnl_total"])}
    write_report(args.out / "test", result, mo, stats, chosen, cfg)
    print("\nout-of-sample (second half) with parameters chosen on the first half:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
