# Market making BTC on Hyperliquid: PnL attribution and adverse selection

*Research note. Fill this in once you have 2+ weeks of recorded data. Aim
for 3–5 pages: an interviewer should get the conclusion from page 1 alone.*

## 1. Summary (5 lines max)

- Question:
- Data: venue, coin, period, number of book updates and trades
- Main finding (one number):
- What drives it:
- What I would change next:

## 2. Data

- Recording setup (VM, uptime, gaps and how they were handled)
- Sanity checks: side-convention check result, share of crossed books dropped, median spread in ticks and bps
- Basic market stats: realized vol by hour of day, trades per second, typical print size

## 3. Strategy and simulator assumptions

- Quoting rule (spread, skew, vol scaling) and baseline parameters
- Fill model, latency, fee tier: why these values, and which direction each biases results

## 4. Results

- PnL attribution table: spread capture / inventory / fees / total, per hour and per $1M traded
- Markout curve: edge at fill vs after 1s, 10s, 60s; buys vs sells
- Where the losses come from: high-vol regimes? hours of day? sweeps?

## 5. Sensitivity

- Parameter sweep (train) and out-of-sample result (test)
- Latency: 10 / 50 / 200 / 500 ms
- Queue assumption (`touch_fill_ratio`): 0.1 / 0.25 / 0.5
- Fee tier: base vs top-tier maker rebate

## 6. What a real desk would do differently

- Hedging inventory on another venue
- Toxicity signals (book imbalance, trade flow)
- Risk limits and kill switches

## 7. Limitations

- No market impact, queue position approximated, single venue, short sample
