"""Approach B: an empirical rank-transition kernel (comparison model; the entry uses forecast_model.py).

From every pair of history reads 1 to 30 hours apart, each game in the top 40 of the earlier read gives one sample:
its log-rank displacement (or 'gone'). A game now at rank i is forecast at horizon h by the samples weighted by how
close their gap is to h and their starting rank to i, each sample moved to i * exp(displacement). Places are then
normalised so each sums to 1 - unnamed.
"""
import math

import forecast_model as fm

DEFAULT = {"bw_h": 6.0, "bw_rank": 0.4, "unnamed": 0.03, "smooth": 0.05, "max_rank": 40}


def samples(reads, max_rank):
    out = []
    for a in range(len(reads)):
        ta, ida, _ = reads[a]
        for b in range(a + 1, len(reads)):
            tb, idb, _ = reads[b]
            dt = tb - ta
            if dt < 1:
                continue
            if dt > 30:
                break
            pos = {u: k + 1 for k, u in enumerate(idb)}
            for k, u in enumerate(ida[:max_rank]):
                out.append((dt, math.log(k + 1), None if u not in pos else math.log(pos[u]) - math.log(k + 1)))
    return out


def forecast(snaps, task, params=None):
    p = dict(DEFAULT)
    p.update(params or {})
    try:
        until = fm.parse_time(task.get("history_until"))
    except (TypeError, ValueError, AttributeError):
        until = None
    reads = fm.clean_snapshots(snaps, until)
    places = int(task.get("places") or 10)
    t_last, ids, _ = reads[-1]
    hours = fm.window_hours(task, t_last)
    groups = [sum(hours[i:i + 6]) / len(hours[i:i + 6]) for i in range(0, len(hours), 6)]
    S = samples(reads, int(p["max_rank"]))
    occ = [dict() for _ in range(places)]
    for i, u in enumerate(ids[:40]):
        li = math.log(i + 1)
        for h in groups:
            tot = 0.0
            acc = [0.0] * places
            for dt, l0, d in S:
                w = math.exp(-0.5 * ((dt - h) / p["bw_h"]) ** 2 - 0.5 * ((l0 - li) / p["bw_rank"]) ** 2)
                if w < 1e-4:
                    continue
                tot += w
                if d is None:
                    continue
                j = int(round(math.exp(li + d)))
                if 1 <= j <= places:
                    acc[j - 1] += w
            if tot > 0:
                for k in range(places):
                    if acc[k]:
                        occ[k][u] = occ[k].get(u, 0.0) + acc[k] / tot / len(groups)
    if not S:  # one read: persistence
        for k in range(min(places, len(ids))):
            occ[k][ids[k]] = 1.0
    sim = []
    for k in range(places):
        s = sum(occ[k].values()) or 1.0
        sim.append({u: q / s for u, q in occ[k].items()})
    bp = {"smooth": p["smooth"], "unnamed": p["unnamed"], "prior_width": 0.9}
    return fm.build_doc(ids, sim, bp, places)
