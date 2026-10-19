import gzip
import json

from mmlab.data import check_trade_side_convention, parse_files


def test_parse_hyperliquid_messages(tmp_path):
    msgs = [
        {"channel": "l2Book", "data": {"coin": "BTC", "time": 1000, "levels": [
            [{"px": "99", "sz": "1.5", "n": 3}, {"px": "98", "sz": "2", "n": 1}],
            [{"px": "101", "sz": "0.5", "n": 2}],
        ]}},
        {"channel": "trades", "data": [
            {"coin": "BTC", "side": "B", "px": "101", "sz": "0.1", "time": 1100, "hash": "x", "tid": 1},
            {"coin": "BTC", "side": "A", "px": "99", "sz": "0.2", "time": 1200, "hash": "y", "tid": 2},
        ]},
    ]
    path = tmp_path / "20260101_00.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for i, m in enumerate(msgs):
            f.write(json.dumps({"recv_ts": 2000 + i, "msg": m}) + "\n")
        f.write('{"recv_ts": 3000, "msg": {"chan')  # truncated line, as after a crash

    book, trades = parse_files([path])
    assert book.loc[0, ["bid_px", "ask_px", "mid", "bid_depth5"]].tolist() == [99.0, 101.0, 100.0, 3.5]
    assert trades["side"].tolist() == [1, -1]
    assert check_trade_side_convention(book, trades) == 1.0
