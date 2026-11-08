"""Shared helpers for the tests: synthetic histories and running main.py the way the grader does."""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_INPUT = os.path.join(os.path.dirname(ROOT), "roblox-trending-forecast-01-kit", "input")


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def synthetic_history(n_reads=30, step_h=6, n_games=97, seed=1, start=None):
    """A history whose chart drifts: game i moves a little every read, one game leaves, one enters."""
    import random

    rng = random.Random(seed)
    start = start or datetime(2026, 11, 1, tzinfo=timezone.utc)
    ids = [1000 + i for i in range(n_games)]
    score = {u: -i + rng.uniform(-0.5, 0.5) for i, u in enumerate(ids)}
    snaps = []
    nxt = 5000
    for r in range(n_reads):
        for u in list(score):
            score[u] += rng.gauss(0, 0.8)
        if r % 7 == 3:
            score.pop(max(score, key=score.get))  # the leader leaves
            score[nxt] = -3.0  # a newcomer
            nxt += 1
        order = sorted(score, key=lambda u: -score[u])
        rows = [{"rank": k + 1, "universe_id": u, "root_place_id": u * 10, "name": f"g{u}",
                 "playing": 1000 + 50 * int(score[u] + 100), "up_votes": None, "down_votes": None, "genre": None}
                for k, u in enumerate(order)]
        snaps.append({"at": iso(start + timedelta(hours=r * step_h)), "source": "test", "filters": None, "ok": True,
                      "top_trending": rows, "top_playing_now": None, "details": None})
    until = start + timedelta(hours=(n_reads - 1) * step_h, minutes=30)
    return snaps, task_for(until, len(snaps))


def task_for(until, n):
    return {"schema": "makefun/roblox-trending-forecast/task/v2", "competition": "roblox-trending-forecast-01",
            "history_until": iso(until), "window": {"from": iso(until), "until": iso(until + timedelta(hours=24)),
                                                     "slot_minutes": 60, "slots": 24},
            "places": 10, "chart": {"sort_id": "top-trending", "device": "computer", "country": "us"},
            "output": "forecast.json", "snapshots": n}


def write_input(snaps, task, raw_lines=None):
    d = tempfile.mkdtemp(prefix="rtf-in-")
    with open(os.path.join(d, "history.ndjson"), "w", encoding="utf-8") as f:
        for s in snaps:
            f.write(json.dumps(s) + "\n")
        for line in raw_lines or []:
            f.write(line + "\n")
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task, f)
    return d


def run_main(input_dir):
    out = tempfile.mkdtemp(prefix="rtf-out-")
    env = dict(os.environ, MAKEFUN_INPUT_DIR=input_dir, MAKEFUN_OUTPUT_DIR=out)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")], env=env, capture_output=True, timeout=300)
    path = os.path.join(out, "forecast.json")
    data = open(path, "rb").read() if os.path.exists(path) else None
    return p, data
