# Crypto market making: where does the PnL actually come from?

A research framework for passive market making on a crypto perpetual
(Hyperliquid BTC by default): record the live order book and tape, replay it
through an event-driven simulator with explicit latency and queue assumptions,
and **decompose the PnL** into spread capture, inventory PnL and fees, with
**fill markouts** to measure adverse selection.

The question this project answers is not "can I find a profitable bot?" but
**"why does a simple quoting strategy make or lose money, and which levers
(spread, inventory skew, volatility scaling, latency, fee tier) change that?"**

> Status: framework complete and tested. The figures below come from the
> synthetic demo dataset; [first results on real
> data](#first-results-on-real-hyperliquid-data) use 8 free Tardis.dev days of
> Hyperliquid BTC. Write-up goes in
> [`notes/research_note.md`](notes/research_note.md).

## Pipeline

```
Hyperliquid websocket ──► recorder ──► raw .jsonl.gz (hourly) ──► data.py
                                                                     │
Tardis.dev free datasets ──► tardis.py: book_snapshot_5 + trades ────┤
                                                                     ▼
                                       book + trades tables (parquet)
                                                     │
              strategy.py (quotes) ──► simulator.py (event-driven replay)
                                                     │
                                                     ▼
              analytics.py: PnL attribution · markouts · summary ──► reports/
```

Two data sources, deliberately: the recorder gives ground truth at **your own**
latency, Tardis gives **regime breadth** for free. See [Getting data](#getting-data).

| Module | What it does |
|---|---|
| `recorder.py` | Async websocket client: L2 snapshots + trades, reconnects with backoff, hourly gzip files, raw messages kept untouched. Clean shutdown on SIGINT **and SIGTERM**, with periodic flushes |
| `tardis.py` | Downloads the free Tardis.dev history (first of each month, no API key) into the same schema, tagged by `day` |
| `data.py` | Parses raw files into clean `book` / `trades` tables, drops crossed books, infers the tick size, checks the trade-side convention |
| `strategy.py` | Volatility-scaled spread + inventory skew, a practical form of Avellaneda–Stoikov (mapping in the docstring) |
| `simulator.py` | Event-driven replay: latency, post-only rejection, queue-aware fills, maker fees, exact PnL accounting |
| `analytics.py` | Markouts at 0.1s–60s, PnL decomposition, drawdown, time at inventory limit |
| `backtest.py` | CLI: single run, `--sweep` (one continuous session), or `--cv` leave-days-out selection reporting the per-day PnL distribution |
| `synthetic.py` | Synthetic data with vol regimes, informed flow and sweeps, so everything runs offline |

## Key modelling choices (and their biases)

The fill model drives every MM backtest result, so it is explicit:

- **Print through our price** (a sell below our bid): the aggressor consumed
  our level, so we fill our **full** resting size.
- **Print at our price**: we are somewhere in the queue, so we fill
  `touch_fill_ratio × print size` (default 25%).
- **Latency**: a quote goes live `latency_ms` after the decision (default 50ms);
  if it would cross the book at that moment it is **rejected** (post-only).
- **No market impact**: the recorded tape doesn't react to our orders. This
  flatters results, more so with larger sizes.

Accounting identity, enforced by the test suite:

```
equity (cash + q·mid) = spread capture − fees + inventory PnL
spread capture = Σ side · qty · (mid at fill − fill price)
inventory PnL  = Σ q · Δmid
```

## Demo results (synthetic data, 4 hours)

![PnL attribution](docs/img/pnl_attribution.png)

The default parameters **capture spread** (about 2.9 bps per fill) but **lose
money**: inventory PnL and fees together exceed the spread captured.

![Markouts](docs/img/markout_curve.png)

The markout curve shows why. Edge at the moment of the fill is about 2.9 bps,
but it decays to about 1.4 bps within 10 seconds as informed flow moves the
price against the position. That residual sits **below the 1.5 bps maker
fee**, so every fill loses on average. Three ways out: a better fee tier (or
rebates), wider quotes when flow is toxic, or a short-term signal that avoids
being picked off.

![Parameter sweep](docs/img/sweep_train_pnl.png)

Parameter sweep, selected on the first 2 hours and tested on the last 2:
wider spreads win in-sample (best train PnL **+221**) and stay profitable
out-of-sample (**+160**) with fewer fills. On this synthetic data, inventory
skew *hurts*: it makes the bot quote more aggressively on the side that
reduces inventory, which is often the side informed flow is about to hit.
Whether that holds on real data is one of the questions for the research note.

## Getting data

### Tardis.dev free datasets — available immediately

Tardis publishes Hyperliquid datasets from **2024-10-29**, and serves the
**first day of every month without an API key**. That is ~24 days today, per
coin, and `book_snapshot_5` carries 5 levels per side, which maps exactly onto
this project's `bid_depth5` / `ask_depth5`.

```bash
python -m mmlab.tardis --coin BTC --list                        # which days are free
python -m mmlab.tardis --coin BTC --out data/processed/BTC_tardis
```

Files are cached under `data/tardis/` and never re-downloaded. Set
`TARDIS_API_KEY` to reach the paid days; without it they return 401 and are
skipped with a warning.

24 non-contiguous days is **better** for parameter selection than a couple of
consecutively recorded days: it samples 24 independent volatility regimes
instead of one. It also measures the book far more finely than a single
websocket client typically manages:

| | own websocket recording | Tardis `book_snapshot_5` |
|---|---|---|
| book cadence (median) | ~5,500 ms (observed) | **545 ms** (p90 627 ms) |
| book rows / day | ~15,700 | **~150,000** |
| trades / day | ~310,000 | ~197,000 |
| cost | a VM | free, 24 days |

That 545 ms is tight enough to suggest Hyperliquid itself publishes `l2Book`
about twice a second — so ~0.5 s is the ceiling, and a recorder seeing 5.5 s
is losing ~10× somewhere local. Worth checking from the VM before committing
to weeks of recording.

**Because the free days are non-contiguous, every run over them must treat each
day as a separate session.** `--cv` does; a single run over the concatenation
does not, and will invent an overnight mid jump worth more than a day of spread
capture (it reports `"hours": 5160` for 8 days, which is the tell).

Note also that Tardis's `local_timestamp` is *their* collector's receive time
(median 219 ms behind the exchange), not yours. It is kept as `recv_ts` for
schema compatibility but is useless for modelling your own latency — use
`--latency-ms` for that.

### Your own recorder — ground truth at your latency

```bash
python -m mmlab.recorder --coins BTC ETH --out data/raw
```

Run it on a small always-on VM. It handles SIGTERM, so `systemctl stop` and
container stops shut it down cleanly instead of discarding the compressor
buffer; gzip buffers are flushed every 5 s, so even a SIGKILL costs seconds.

### How much data is enough?

- **One or two days**: enough to verify the pipeline, the accounting identity
  and the *shape* of the markout curve (adverse selection resolves within
  seconds, and the horizons stop at 60 s).
- **Not enough for `--sweep`**: a 50/50 split of two days is one regime each.
  48 hourly PnL observations give a ~10% standard error on the volatility
  alone, and taking a max over 16 configs on a single day selects whichever one
  fit that day's drift.
- **For parameter selection**: 2–4 weeks recorded, or the Tardis free days with
  `--cv`, which reports a mean, a standard error and a profitable-day count
  instead of one total.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                   # 16 tests, including the accounting identity

# offline demo
python -m mmlab.backtest --synthetic --out reports/demo
python -m mmlab.backtest --synthetic --sweep --out reports/demo_sweep

# free history, no API key: 24 days (first of each month) from Tardis.dev
python -m mmlab.tardis --coin BTC --out data/processed/BTC_tardis
python -m mmlab.backtest --data data/processed/BTC_tardis --cv --out reports/btc_cv

# your own recording (always-on VM), for ground truth at your latency
python -m mmlab.recorder --coins BTC ETH --out data/raw
python -m mmlab.data --raw data/raw/BTC --out data/processed/BTC
python -m mmlab.backtest --data data/processed/BTC --out reports/btc
python -m mmlab.backtest --data data/processed/BTC --sweep --out reports/btc_sweep
```

`--cv` writes `cv_train.csv`, `cv_test_per_day.csv` and `cv_summary.json`; it
holds out the chronologically last 25% of days (override with `--test-days`),
picks parameters by mean per-day PnL on the rest, and warns when the held-out
result has `|t| < 2`.

Each run writes `summary.json`, `fills.csv`, `markout_curve.csv` and PNG
figures to the output folder.

Before trusting results on real data:

- The **tick size** is inferred from the recorded book and printed by both
  `mmlab.data` and `mmlab.backtest`; override with `--tick` only if you have a
  reason to. Hyperliquid prices use at most 5 significant figures, so the tick
  depends on the price level: $1 for BTC near $80k, $0.1 for ETH near $2.3k.
  Spread and skew parameters are therefore specified in **ticks**
  (`--min-half-spread-ticks`, `--skew-ticks`), which keeps one grid meaningful
  across coins; a bps-denominated floor is 8 ticks on BTC and a quarter tick
  on ETH for the same number.
- `--vol-horizon-s` defaults to the quote's actual exposure
  (`requote_ms + latency_ms`, 0.3s with the defaults). Pricing several seconds
  of variance into a quote you can cancel in 300ms puts it ~8 ticks off BTC's
  mid and earns **zero fills**.
- Expect `max_abs_inventory` to reach `max_inventory + order_size`: a quote is
  sized on the inventory known when it was decided and still fills one latency
  hop later. Size `order_size` with that in mind.
- Check the current **maker fee** for your volume tier and pass it with
  `--maker-fee` (negative for a rebate).
- Run `python -m mmlab.data`, which prints a trade-side sanity check; it
  should be close to 1.0. On the Tardis BTC day checked, buy prints land at or
  above mid 95.6% of the time and sell prints at or below 95.1%, confirming
  that `side` is the aggressor on both sources.
- If the sweep is **monotonic to the edge of the grid**, the optimum is outside
  it and the reported "best" is just the boundary. Widen the grid rather than
  reading the corner as a result.

## First results on real Hyperliquid data

8 free Tardis days of BTC (2026-03-01 .. 2026-10-01; 1.25M book rows, 3.5M
trades), `--cv` with 6 train / 2 held-out days, default fill model:

```
train  hs=1 tick   pnl/day  -4764 ± 960    profitable days 0/6
train  hs=8 ticks  pnl/day  -1285 ± 334    profitable days 0/6
held out (hs=8, skew=0)      -1405 ± 171    profitable days 0/2
```

Markouts over those days: **+0.17 bps at the fill**, −0.54 bps at 0.5 s,
−1.14 bps at 60 s. Edge at the fill is an order of magnitude below the 1.5 bps
maker fee, and PnL per volume stays negative (−1.1 to −2.1 bps) even with
`--maker-fee 0`, so **adverse selection alone sinks it** — the fee is not the
whole story. PnL improves monotonically as quotes widen, i.e. toward not
trading at all; the grid never finds an interior optimum.

Read as: a naive vol-scaled quoter resting at or near the touch on BTC perps
has **no edge**. That is the expected answer, and it is what makes the
toxicity filter below the next real experiment rather than a refinement.

## Next steps

- [x] Free historical data via `mmlab.tardis`, with leave-days-out validation
      (`--cv`) replacing the single time split
- [ ] Extend the sweep grid past 8 ticks, since the current grid is monotonic
      to its boundary on real data
- [ ] Record 2–4 weeks of BTC and ETH at your own latency and compare the
      markout curve with the Tardis days (the recorder's coarser cadence
      flatters queue position)
- [ ] Toxicity filter: widen or pull quotes when order book imbalance or recent trade flow is one-sided
- [ ] Sensitivity of PnL to latency (10ms → 500ms) and to `touch_fill_ratio`
- [ ] Hedging inventory on another venue instead of skewing
- [ ] Small live deployment on Hyperliquid testnet via the official Python SDK, comparing live fills with simulated fills (fill-model validation)

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem*.
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Hyperliquid API documentation (websocket subscriptions, fees, tick sizes).
