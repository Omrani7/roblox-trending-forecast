"""Roblox Top Trending forecast entry (Python 3.13, standard library only).

Reads $MAKEFUN_INPUT_DIR (else /input): history.ndjson and task.json.
Writes $MAKEFUN_OUTPUT_DIR (else /output): the forecast file task.json names.
"""
import json
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"


def load_history(path):
    snaps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snaps.append(json.loads(line))
            except ValueError:
                continue
    return snaps


def persistence_forecast(snaps, places):
    """Iteration 1: the house baseline's persistence rule, ported."""
    chart = []
    for s in reversed(snaps):
        rows = s.get("top_trending") if isinstance(s, dict) else None
        if isinstance(rows, list):
            good = [r for r in rows if isinstance(r, dict) and isinstance(r.get("universe_id"), int)]
            good.sort(key=lambda r: r["rank"] if isinstance(r.get("rank"), (int, float)) else 1e9)
            chart = list(dict.fromkeys(r["universe_id"] for r in good))
            if len(chart) >= places:
                break
    chart = (chart + [i for i in range(1, places + 1) if i not in chart])[: max(places, len(chart))]
    rows = []
    for place in range(1, min(places, len(chart)) + 1):
        rows.append({"universe_id": chart[place - 1], "place": place, "p": 0.5})
        if place == 1:
            if len(chart) > 1:
                rows.append({"universe_id": chart[1], "place": place, "p": 0.3})
            continue
        rows.append({"universe_id": chart[place - 2], "place": place, "p": 0.15})
        if place < len(chart):
            rows.append({"universe_id": chart[place], "place": place, "p": 0.15})
    return {"probabilities": rows, "top10": chart[:places]}


def valid_doc(doc, places):
    """A light check of the forecast's shape (the scorer's rules), so a broken model falls back, never fails."""
    try:
        rows, top = doc["probabilities"], doc["top10"]
        sums, seen = [0.0] * places, set()
        for r in rows:
            k, u, p = r["place"], r["universe_id"], r["p"]
            if not (isinstance(u, int) and 1 <= k <= places and 0 <= p <= 1) or (u, k) in seen:
                return False
            seen.add((u, k))
            sums[k - 1] += p
        per_place = [sum(1 for r in rows if r["place"] == k) for k in range(1, places + 1)]
        return (all(s <= 1 + 1e-12 for s in sums) and max(per_place, default=0) <= 100
                and len(top) == places and len(set(top)) == places and all(isinstance(u, int) for u in top))
    except (KeyError, TypeError, ValueError):
        return False


def build_forecast(snaps, task):
    places = int(task.get("places") or 10)
    try:
        import forecast_model

        doc = forecast_model.forecast(snaps, task)
        if valid_doc(doc, places):
            return doc
        sys.stderr.write("model forecast failed its shape check; falling back to persistence\n")
    except Exception as e:  # noqa: BLE001 - never leave the run without a forecast
        sys.stderr.write(f"model failed ({type(e).__name__}: {e}); falling back to persistence\n")
    return persistence_forecast(snaps, places)


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    snaps = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    snaps.sort(key=lambda s: str(s.get("at", "")) if isinstance(s, dict) else "")
    doc = build_forecast(snaps, task)
    out = os.path.join(OUTPUT_DIR, task.get("output") or "forecast.json")
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
