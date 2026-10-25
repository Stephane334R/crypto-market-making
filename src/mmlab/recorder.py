"""Live market data recorder for Hyperliquid (L2 order book + trades).

Writes raw websocket messages, stamped with the local receive time, into
hourly gzipped JSON-lines files:

    data/raw/<COIN>/<YYYYMMDD_HH>.jsonl.gz

Each line: {"recv_ts": <ms>, "msg": <raw message>}

Run it on a small always-on VM (a cheap GCP e2-micro is enough) for a few
weeks before running the backtests on real data:

    python -m mmlab.recorder --coins BTC ETH --out data/raw

Notes
-----
* Hyperliquid pushes a full L2 snapshot (top levels per side) on each
  `l2Book` update, so no incremental book reconstruction is needed.
* The server drops idle connections, so we send an application-level ping.
* Raw messages are stored untouched; parsing happens in `mmlab.data`, so a
  parsing bug never costs you recorded data.
* SIGINT *and* SIGTERM shut down cleanly, and the gzip buffers are flushed
  every `FLUSH_EVERY_S`. Without this a `systemctl stop` / container stop
  (both SIGTERM) discards whatever is still sitting in the compressor.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

WS_URL = "wss://api.hyperliquid.xyz/ws"
PING_EVERY_S = 30
FLUSH_EVERY_S = 5.0

log = logging.getLogger("recorder")


class HourlyWriter:
    """Append lines to one gzip file per coin per UTC hour."""

    def __init__(self, root: Path, flush_every_s: float = FLUSH_EVERY_S):
        self.root = root
        self.flush_every_s = flush_every_s
        self._files: dict[str, tuple[str, gzip.GzipFile]] = {}
        self._last_flush = time.monotonic()

    def write(self, coin: str, record: dict) -> None:
        hour = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
        current = self._files.get(coin)
        if current is None or current[0] != hour:
            if current is not None:
                current[1].close()
            path = self.root / coin / f"{hour}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[coin] = (hour, gzip.open(path, "at"))
            log.info("writing %s", path)
        self._files[coin][1].write(json.dumps(record, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if now - self._last_flush >= self.flush_every_s:
            self.flush()
            self._last_flush = now

    def flush(self) -> None:
        """Flush the compressors so an unclean kill costs seconds, not minutes."""
        for _, f in self._files.values():
            f.flush()

    def close(self) -> None:
        for _, f in self._files.values():
            f.close()
        self._files.clear()


def _coin_of(msg: dict) -> str | None:
    channel = msg.get("channel")
    data = msg.get("data")
    if channel == "l2Book" and isinstance(data, dict):
        return data.get("coin")
    if channel == "trades" and isinstance(data, list) and data:
        return data[0].get("coin")
    return None


async def _pinger(ws) -> None:
    while True:
        await asyncio.sleep(PING_EVERY_S)
        await ws.send(json.dumps({"method": "ping"}))


async def _consume(ws, writer: HourlyWriter) -> None:
    async for raw in ws:
        recv_ts = int(time.time() * 1000)
        msg = json.loads(raw)
        coin = _coin_of(msg)
        if coin is None:  # pong, subscription acks, etc.
            continue
        writer.write(coin, {"recv_ts": recv_ts, "msg": msg})


async def record(coins: list[str], out: Path) -> None:
    writer = HourlyWriter(out)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # non-POSIX
            pass

    backoff = 1.0
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, max_size=2**23) as ws:
                    for coin in coins:
                        for sub_type in ("l2Book", "trades"):
                            await ws.send(json.dumps({
                                "method": "subscribe",
                                "subscription": {"type": sub_type, "coin": coin},
                            }))
                    log.info("subscribed to %s", coins)
                    backoff = 1.0
                    tasks = [asyncio.create_task(_consume(ws, writer)),
                             asyncio.create_task(_pinger(ws)),
                             asyncio.create_task(stop.wait())]
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for t in tasks:
                            t.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    for t in done:  # re-raise a real failure so the retry logic sees it
                        if not t.cancelled() and t.exception() is not None:
                            raise t.exception()
            except (websockets.WebSocketException, OSError) as exc:
                if stop.is_set():
                    break
                log.warning("connection lost (%s); reconnecting in %.0fs", exc, backoff)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                    break  # asked to stop while waiting to reconnect
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)
    finally:
        writer.close()
        log.info("stopped; files closed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", nargs="+", default=["BTC"])
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(record(args.coins, args.out))


if __name__ == "__main__":
    main()
