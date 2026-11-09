"""Coordinate-descent tuning of the simulation model's parameters on walk-forward cutoffs (dev tool).

Each candidate value is scored on every cutoff with the same seed (common random numbers), and the halves of the
period are reported apart so a value that only helps one half is visible.
usage: python tune.py history.ndjson [--sims 400] [--rounds 2] [--params a,b,c] [--set k=v ...]
"""
import argparse
import json

import backtest as bt
import forecast_model as fm

GRID = {
    "sigma_top": [0.04, 0.05, 0.06, 0.07, 0.08],
    "sigma_low": [0.04, 0.048, 0.055, 0.065],
    "alpha": [0.3, 0.4, 0.5, 0.6, 0.75],
    "c_mom": [0.0, 0.1, 0.25, 0.4, 0.6],
    "c_grow": [0.5, 0.75, 1.0, 1.3],
    "rel_growth": [0, 1],
    "vol_w": [0.0, 0.5, 0.75, 1.0],
    "game_vol_w": [0.5, 1.0, 1.5, 2.0, 3.0],
    "c_mom_s": [-0.2, 0.0, 0.15, 0.3],
    "exit_rate": [0.0, 0.002, 0.004, 0.007, 0.01],
    "exit_top": [0.001, 0.002, 0.004, 0.007, 0.01],
    "exit_grow": [0.0, 2.0, 4.0, 8.0],
    "exit_cap": [5.0, 20.0, 50.0],
    "jump_rate": [0.0, 0.001, 0.002, 0.004, 0.007],
    "jump_thresh": [0.1, 0.15, 0.25],
    "jump_land": [10, 15, 25],
    "jump_from": [5, 10, 15],
    "entry_rate": [0.0, 0.015, 0.03, 0.05, 0.08],
    "smooth": [0.02, 0.04, 0.06, 0.1, 0.15],
    "prior_width": [0.5, 0.7, 0.9, 1.2],
    "unnamed": [0.001, 0.0025, 0.005, 0.01],
}


def evaluate(raw, reads, cuts, params):
    res = bt.run(raw, reads, lambda s, t: fm.forecast(s, t, params), cuts)
    half = len(res) // 2
    a = bt.summary(res)
    return a["mean_log"], bt.summary(res[:half])["mean_log"], bt.summary(res[half:])["mean_log"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--params", default=",".join(GRID))
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--min-hist", type=float, default=36.0)
    ap.add_argument("--set", action="append")
    a = ap.parse_args()
    raw = [json.loads(l) for l in open(a.history, encoding="utf-8") if l.strip()]
    raw.sort(key=lambda s: s["at"])
    reads = fm.clean_snapshots(raw)
    cuts = bt.cutoffs_of(reads, a.gap, -1e18, a.min_hist, 0.0)
    cur = dict(fm.DEFAULT_PARAMS)
    cur.update(bt.parse_sets(a.set))
    cur["sims"] = a.sims
    best, h1, h2 = evaluate(raw, reads, cuts, cur)
    print(f"start {best:.4f} ({h1:.3f} / {h2:.3f})", flush=True)
    for rnd in range(a.rounds):
        changed = False
        for name in a.params.split(","):
            line = []
            for v in GRID[name]:
                trial = dict(cur, **{name: v})
                s, s1, s2 = evaluate(raw, reads, cuts, trial)
                line.append(f"{v}:{s:.3f}({s1:.2f}/{s2:.2f})")
                if s > best + 0.01:
                    best, cur, changed = s, trial, True
            print(f"r{rnd} {name:12s} -> {cur[name]}  | " + "  ".join(line), flush=True)
        if not changed:
            break
    print("best", round(best, 4), json.dumps({k: cur[k] for k in GRID}))


if __name__ == "__main__":
    main()
