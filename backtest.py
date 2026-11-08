"""Walk-forward backtest of forecast models on a history file (dev tool; not run by main.py).

For each cutoff T (a read's time + gap hours, on or after --start), a model sees only reads that started before T and
forecasts the 24 hourly slots of [T, T + 24h). It is scored, exactly as the competition scores, against the first
read in each slot that holds one. Windows overlap and the sample is small: read the mean with its standard error.

usage: python backtest.py history.ndjson [--model sim|baseline|...] [--gap 1] [--start ISO] [--set k=v ...]
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

import forecast_model as fm
import main as entry
import scoring


def iso(h):
    return datetime.fromtimestamp(h * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def make_task(cutoff_h, n_snaps):
    return {"history_until": iso(cutoff_h), "window": {"from": iso(cutoff_h), "until": iso(cutoff_h + 24),
            "slot_minutes": 60, "slots": 24}, "places": 10, "output": "forecast.json", "snapshots": n_snaps}


def answer_charts(reads, cutoff_h):
    slots = {}
    for t, ids, _ in reads:
        if cutoff_h <= t < cutoff_h + 24:
            s = int((t - cutoff_h) // 1)
            slots.setdefault(s, (t, ids))
    return [slots[s] for s in sorted(slots)]


def blend(docs, weights):
    """Linear pool of forecasts, place by place (each place's sum stays <= 1); the first doc's top10."""
    acc = {}
    for d, w in zip(docs, weights):
        for r in d["probabilities"]:
            key = (r["place"], r["universe_id"])
            acc[key] = acc.get(key, 0.0) + w * r["p"]
    rows = [{"universe_id": u, "place": k, "p": int(q * 1e6) / 1e6} for (k, u), q in sorted(acc.items())]
    by_place = {}
    for r in rows:
        by_place.setdefault(r["place"], []).append(r)
    rows = [r for k in sorted(by_place) for r in sorted(by_place[k], key=lambda r: -r["p"])[:100]]
    return {"probabilities": rows, "top10": docs[0]["top10"]}


def model_fn(name, overrides):
    if name == "baseline":
        return lambda snaps, task: entry.persistence_forecast(snaps, 10)
    if name == "sim":
        return lambda snaps, task: fm.forecast(snaps, task, overrides)
    if name == "kernel":
        import kernel_model

        return lambda snaps, task: kernel_model.forecast(snaps, task, overrides)
    if name == "ensemble":
        import kernel_model

        w = overrides.get("w_sim", 0.7)
        sim_over = {k: v for k, v in overrides.items() if k != "w_sim"}
        return lambda snaps, task: blend([fm.forecast(snaps, task, sim_over),
                                          kernel_model.forecast(snaps, task, {})], [w, 1 - w])
    if name == "entry":
        return lambda snaps, task: entry.build_forecast(snaps, task)
    raise SystemExit(f"unknown model {name}")


def cutoffs_of(reads, gap, start_h, min_hist_h, step_h):
    out, last = [], -1e9
    t0 = reads[0][0]
    for t, _, _ in reads:
        c = t + gap
        if c < start_h or c - t0 < min_hist_h or c - last < step_h:
            continue
        out.append(c)
        last = c
    return out


def run(raw, reads, fn, cutoffs, late_from=0, verbose=False):
    res = []
    for c in cutoffs:
        hist = [s for s in raw if fm.parse_time(s["at"]) < c]
        ans = [a for a in answer_charts(reads, c) if a[0] - c >= late_from]
        if not ans:
            continue
        doc = fn(hist, make_task(c, len(hist)))
        f, why = scoring.validate_forecast(json.loads(json.dumps(doc)))
        if f is None:
            raise SystemExit(f"invalid forecast at {iso(c)}: {why}")
        sc = scoring.score_all(f, [ids for _, ids in ans])
        res.append((c, len(ans), sc))
        if verbose:
            print(f"  {iso(c)[:16]} reads={len(ans):2d} log={sc['log_score']:8.3f} hits={sc['top10_hits']:.2f}")
    return res


def summary(res):
    v = [r[2]["log_score"] for r in res]
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, n - 1))
    hits = sum(r[2]["top10_hits"] for r in res) / n
    agr = sum(r[2]["agreement"] for r in res) / n
    return {"cutoffs": n, "mean_log": round(m, 4), "se": round(sd / math.sqrt(n), 3), "hits": round(hits, 3),
            "agreement": round(agr, 2)}


def parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        out[k] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("--model", default="sim")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--min-hist", type=float, default=36.0)
    ap.add_argument("--step", type=float, default=0.0)
    ap.add_argument("--late-from", type=float, default=0.0)
    ap.add_argument("--set", action="append")
    ap.add_argument("--exclude", default=None, help="FROM,UNTIL: drop cutoffs whose window overlaps it")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    raw = [json.loads(l) for l in open(a.history, encoding="utf-8") if l.strip()]
    raw.sort(key=lambda s: s["at"])
    reads = fm.clean_snapshots(raw)
    start = fm.parse_time(a.start) if a.start else -1e18
    cuts = cutoffs_of(reads, a.gap, start, a.min_hist, a.step)
    if a.exclude:
        lo, hi = (fm.parse_time(x) for x in a.exclude.split(","))
        cuts = [c for c in cuts if c + 24 <= lo or c >= hi]
    t0 = time.time()
    res = run(raw, reads, model_fn(a.model, parse_sets(a.set)), cuts, a.late_from, a.v)
    s = summary(res)
    s["secs"] = round(time.time() - t0, 1)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
