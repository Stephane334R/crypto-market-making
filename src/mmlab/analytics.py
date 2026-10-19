"""PnL attribution, markouts and summary statistics for a simulation run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SimResult

MARKOUT_HORIZONS_S = (0.1, 0.5, 1, 2, 5, 10, 30, 60)


def markouts(result: SimResult, book: pd.DataFrame, horizons_s=MARKOUT_HORIZONS_S) -> pd.DataFrame:
    """Per-fill markout in bps: side * (mid(t + h) - fill_px) / fill_px.

    Positive = the fill made money h seconds later. The markout at h=0
    equals the spread captured; the decay as h grows is adverse selection.
    """
    f = result.fills
    if f.empty:
        return pd.DataFrame()
    b_ts = book["ts"].to_numpy(np.int64)
    b_mid = book["mid"].to_numpy(float)
    out = {"ts": f["ts"].to_numpy(), "side": f["side"].to_numpy(), "qty": f["qty"].to_numpy()}
    out["h_0s"] = f["side"] * (f["mid"] - f["px"]) / f["px"] * 1e4
    for h in horizons_s:
        target = f["ts"].to_numpy(np.int64) + int(h * 1000)
        idx = np.clip(np.searchsorted(b_ts, target, side="right") - 1, 0, len(b_ts) - 1)
        valid = target <= b_ts[-1]
        mo = f["side"].to_numpy() * (b_mid[idx] - f["px"].to_numpy()) / f["px"].to_numpy() * 1e4
        out[f"h_{h:g}s"] = np.where(valid, mo, np.nan)
    return pd.DataFrame(out)


def markout_curve(mo: pd.DataFrame) -> pd.DataFrame:
    """Size-weighted average markout per horizon, all fills and by side."""
    cols = [c for c in mo.columns if c.startswith("h_")]
    rows = []
    for c in cols:
        h = float(c[2:-1])
        row = {"horizon_s": h}
        for name, sel in (("all", mo), ("buys", mo[mo["side"] == 1]), ("sells", mo[mo["side"] == -1])):
            v = sel[[c, "qty"]].dropna()
            row[name] = float(np.average(v[c], weights=v["qty"])) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("horizon_s").reset_index(drop=True)


def summary(result: SimResult, mo: pd.DataFrame | None = None) -> dict:
    s, f = result.states, result.fills
    last = s.iloc[-1]
    hours = (s["ts"].iloc[-1] - s["ts"].iloc[0]) / 3.6e6
    notional = float((f["px"] * f["qty"]).sum()) if len(f) else 0.0

    # hourly PnL distribution (marked to mid)
    eq = s.set_index(pd.to_datetime(s["ts"], unit="ms"))["equity"]
    hourly = eq.resample("1h").last().diff().dropna()

    out = {
        "hours": round(hours, 2),
        "fills": int(len(f)),
        "fills_through_pct": round(100 * float(f["through"].mean()), 1) if len(f) else None,
        "volume_notional": round(notional, 2),
        "pnl_total": round(float(last["equity"]), 2),
        "pnl_spread_capture": round(float(last["spread_capture"]), 2),
        "pnl_inventory": round(float(last["inventory_pnl"]), 2),
        "pnl_fees": round(-float(last["fees"]), 2),
        "pnl_per_volume_bps": round(float(last["equity"]) / notional * 1e4, 3) if notional else None,
        "max_abs_inventory": round(float(s["inventory"].abs().max()), 4),
        "time_at_inventory_limit_pct": None,
        "hourly_pnl_mean": round(float(hourly.mean()), 2) if len(hourly) else None,
        "hourly_pnl_std": round(float(hourly.std()), 2) if len(hourly) > 1 else None,
        "max_drawdown": round(float((eq - eq.cummax()).min()), 2),
    }
    if mo is not None and not mo.empty:
        curve = markout_curve(mo).set_index("horizon_s")["all"]
        out["markout_bps"] = {f"{h:g}s": round(float(v), 3) for h, v in curve.items()}
    return out


def add_inventory_limit_share(stats: dict, result: SimResult, max_inventory: float) -> dict:
    inv = result.states["inventory"].abs()
    stats["time_at_inventory_limit_pct"] = round(100 * float((inv >= max_inventory - 1e-12).mean()), 2)
    return stats
