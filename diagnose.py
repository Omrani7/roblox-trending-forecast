"""Where does the log loss come from? (dev tool)

For every cutoff, place and scored read: the loss -ln max(P, 1e-4) of the game actually there, grouped by what that
game was at the cutoff's last read (same place, moved 1-2 places, moved 3+, from 11-20, from 21+, not on the chart),
and by hours since the last read.
usage: python diagnose.py history.ndjson [--set k=v ...]
"""
import argparse
import json
import math
from collections import defaultdict

import backtest as bt
import forecast_model as fm
import scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("--set", action="append")
    ap.add_argument("--model", default="sim")
    a = ap.parse_args()
    raw = sorted((json.loads(l) for l in open(a.history, encoding="utf-8") if l.strip()), key=lambda s: s["at"])
    reads = fm.clean_snapshots(raw)
    fn = bt.model_fn(a.model, bt.parse_sets(a.set))
    cat_loss, cat_n = defaultdict(float), defaultdict(int)
    hz_loss, hz_n = defaultdict(float), defaultdict(int)
    place_loss = defaultdict(float)
    nreads = 0
    cal = [[0.0, 0.0, 0] for _ in range(10)]
    for c in bt.cutoffs_of(reads, 1.0, -1e18, 36.0, 0.0):
        hist = [s for s in raw if fm.parse_time(s["at"]) < c]
        ans = bt.answer_charts(reads, c)
        if not ans:
            continue
        last = fm.clean_snapshots(hist)[-1]
        pos = {u: i + 1 for i, u in enumerate(last[1])}
        f, _ = scoring.validate_forecast(json.loads(json.dumps(fn(hist, bt.make_task(c, len(hist))))))
        for t, chart in ans:
            nreads += 1
            for k in range(1, 11):
                for u, q in f.named[k - 1].items():
                    b = min(9, int(q * 10))
                    cal[b][0] += q
                    cal[b][1] += 1.0 if chart[k - 1] == u else 0.0
                    cal[b][2] += 1
            for k in range(1, 11):
                u = chart[k - 1]
                loss = -math.log(max(scoring.place_probability(f, k, u, chart), 1e-4))
                r = pos.get(u)
                cat = ("not on chart" if r is None else "same place" if r == k else "moved 1-2" if abs(r - k) <= 2
                       else "moved 3+ (was top10)" if r <= 10 else "from 11-20" if r <= 20 else "from 21+")
                cat_loss[cat] += loss
                cat_n[cat] += 1
                hb = int((t - last[0]) // 6) * 6
                hz_loss[hb] += loss
                hz_n[hb] += 1
                place_loss[k] += loss
    print(f"reads scored: {nreads}; mean loss per read {sum(cat_loss.values()) / nreads:.3f}")
    print("by outcome            n   share-of-loss  mean-loss  (loss per read)")
    for c in sorted(cat_loss, key=lambda c: -cat_loss[c]):
        print(f"  {c:22s} {cat_n[c]:4d}  {cat_loss[c] / sum(cat_loss.values()):6.1%}  {cat_loss[c] / cat_n[c]:6.2f}"
              f"  {cat_loss[c] / nreads:6.2f}")
    print("by hours after last read:", {h: round(hz_loss[h] / hz_n[h] * 10, 2) for h in sorted(hz_loss)})
    print("by place (loss per read):", {k: round(place_loss[k] / nreads, 2) for k in sorted(place_loss)})
    print("calibration  bucket  n  mean-p  observed")
    for b, (sp, hit, n) in enumerate(cal):
        if n:
            print(f"  {b / 10:.1f}-{(b + 1) / 10:.1f} {n:6d}  {sp / n:.3f}  {hit / n:.3f}")


if __name__ == "__main__":
    main()
