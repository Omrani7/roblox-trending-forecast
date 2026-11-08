"""Python port of the competition's ranked-forecast scorer (grader/forecast/ranked.ts in the kit's harness source).

Used only by the backtest and the tests; main.py does not need it. It mirrors:
  parseRanked        -> validate_forecast
  placeProbability   -> place_probability
  orderLogScore      -> order_log_score
  meanLogScore, hitsOf, agreementOf
"""
import json
import math

P_FLOOR = 1e-4
SUM_TOLERANCE = 1e-9
MAX_ROWS_PER_PLACE = 100
MAX_BYTES = 1_048_576


class Forecast:
    def __init__(self, places, named, sums, top):
        self.places = places
        self.named = named  # list (index place-1) of {id: p}
        self.sums = sums
        self.top = top


def validate_forecast(raw, places=10):
    """Returns (Forecast, None) or (None, reason), following parseRanked's rules."""
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_BYTES:
            return None, "over max bytes"
        try:
            doc = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return None, f"not UTF-8 JSON: {e}"
    else:
        doc = raw
    if not isinstance(doc, dict) or sorted(doc.keys()) != ["probabilities", "top10"]:
        return None, "keys are not exactly probabilities and top10"
    rows = doc["probabilities"]
    if not isinstance(rows, list):
        return None, "probabilities is not a list"
    named = [dict() for _ in range(places)]
    sums = [0.0] * places
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or sorted(row.keys()) != ["p", "place", "universe_id"]:
            return None, f"probabilities[{i}] is not exactly {{universe_id, place, p}}"
        uid, place, p = row["universe_id"], row["place"], row["p"]
        if not (isinstance(uid, int) and not isinstance(uid, bool) and 1 <= uid <= 2**53 - 1):
            return None, f"probabilities[{i}].universe_id is not a positive integer"
        if not (isinstance(place, int) and not isinstance(place, bool) and 1 <= place <= places):
            return None, f"probabilities[{i}].place out of range"
        if not (isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1):
            return None, f"probabilities[{i}].p is not a number from 0 to 1"
        at = named[place - 1]
        if uid in at:
            return None, f"{uid} named twice at place {place}"
        if len(at) == MAX_ROWS_PER_PLACE:
            return None, f"place {place} has more than {MAX_ROWS_PER_PLACE} rows"
        at[uid] = float(p)
        sums[place - 1] += float(p)
    for k in range(places):
        if sums[k] > 1 + SUM_TOLERANCE:
            return None, f"place {k + 1} sums to {sums[k]}"
    top = doc["top10"]
    if (not isinstance(top, list) or len(top) != places or len(set(top)) != places
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in top)):
        return None, "top10 is not 10 distinct ids"
    return Forecast(places, named, sums, list(top)), None


def place_probability(f, place, uid, chart_ids):
    at = f.named[place - 1]
    if uid in at:
        return at[uid]
    unnamed = sum(1 for c in set(chart_ids) if c not in at)
    left = max(0.0, 1.0 - f.sums[place - 1])
    return 0.0 if unnamed == 0 else left / unnamed


def order_log_score(f, chart, floor=P_FLOOR):
    n = min(f.places, len(chart))
    return sum(math.log(max(place_probability(f, k, chart[k - 1], chart), floor)) for k in range(1, n + 1))


def mean_log_score(f, charts, floor=P_FLOOR):
    return sum(order_log_score(f, c, floor) for c in charts) / len(charts)


def hits_of(top, actual):
    head = actual[: len(top)]
    return sum(1 for u in top if u in head)


def agreement_of(top, actual):
    width = len(top)
    head = actual[:width]
    pts = 0
    for i, u in enumerate(top):
        if u in head:
            pts += width - abs(i - head.index(u))
    return pts


def score_all(f, charts):
    """Log score, typical probability per place, mean hits, mean agreement over the reads."""
    ls = mean_log_score(f, charts)
    return {
        "log_score": round(ls, 6),
        "p_place": round(math.exp(ls / 10), 6),
        "top10_hits": round(sum(hits_of(f.top, c) for c in charts) / len(charts), 6),
        "agreement": round(sum(agreement_of(f.top, c) for c in charts) / len(charts), 6),
    }
