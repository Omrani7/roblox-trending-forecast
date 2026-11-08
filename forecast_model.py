"""Probabilistic forecast of Roblox's Top Trending top 10 over a 24-hour window (standard library only).

Model (a seeded Monte-Carlo over a latent log-rank):
  * Each game on the last chart starts at y = log(rank).
  * Drift per hour from features of the history (no future data): the game's 24h log-rank trend (momentum) and its
    24h growth in concurrent players (the chart ranks relative growth in daily players, which concurrent players lead).
  * Diffusion: y(h) = y0 + drift*h + sigma(rank) * h**alpha * Z, Z a standard normal fixed per path (the marginal at
    each horizon is what is scored, so a per-path scale is enough).
  * Exits: each game leaves the chart at a constant hazard per hour; the games below move up.
  * Entrants: unseen games arrive in the top places at a constant rate; they are unnamed, they only push others down.
  * At every scored hour of the window the simulated chart is re-ranked; the place occupancy is averaged over the
    window's hours and the paths, then smoothed toward a broad prior so no plausible game gets a near-zero share.
"""
import math
import random
from datetime import datetime, timezone

DEFAULT_PARAMS = {
    "sims": 3000,
    "seed": 20261110,
    "sigma_top": 0.10,      # sd of log-rank per hour**alpha at rank 1
    "sigma_low": 0.07,      # ... at rank >= 20
    "alpha": 0.5,
    "c_mom": 0.25,          # weight of the 24h log-rank trend on the 24h drift
    "c_grow": 0.35,         # weight of the 24h log player growth on the 24h drift (minus: growth lifts the game)
    "rel_growth": 1,        # growth taken relative to the chart's median growth
    "c_mom_s": 0.0,         # weight of the 6h log-rank trend (scaled to 6h) on the 24h drift
    "drift_cap": 1.0,       # max |drift| over 24h in log-rank
    "vol_w": 0.0,           # exponent on (recent volatility / vol_ref) scaling sigma; 0 = off
    "vol_ref": 0.049,       # the archive's volatility statistic (recent_vol over 2026-10-30..11-08)
    "exit_rate": 0.004,     # per hour per game
    "entry_rate": 0.03,     # unseen games reaching the top 12 per hour
    "smooth": 0.06,         # mass mixed in from the broad prior
    "prior_width": 0.9,     # sd in log-rank of the broad prior kernel
    "unnamed": 0.02,        # mass of each place left unnamed (for unseen games)
}

LAG_H = 24.0


def parse_time(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 3600.0


def clean_snapshots(snaps, until_h=None):
    """Valid reads, oldest first, one per distinct time, each as (t_hours, [ids in order], {id: playing})."""
    out = {}
    for s in snaps:
        if not isinstance(s, dict) or s.get("ok") is False:
            continue
        rows = s.get("top_trending")
        at = s.get("at")
        if not isinstance(rows, list) or not rows or not isinstance(at, str):
            continue
        try:
            t = parse_time(at)
        except ValueError:
            continue
        if until_h is not None and t >= until_h:
            continue
        good = []
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            uid = r.get("universe_id")
            if isinstance(uid, str) and uid.isdigit():
                uid = int(uid)
            if not isinstance(uid, int) or isinstance(uid, bool) or uid < 1:
                continue
            rank = r.get("rank")
            rank = rank if isinstance(rank, (int, float)) and not isinstance(rank, bool) else 10_000 + i
            good.append((rank, i, uid, r.get("playing")))
        good.sort()
        ids, playing, seen = [], {}, set()
        for _, _, uid, pl in good:
            if uid in seen:
                continue
            seen.add(uid)
            ids.append(uid)
            if isinstance(pl, (int, float)) and not isinstance(pl, bool) and pl > 0:
                playing[uid] = float(pl)
        if len(ids) >= 10:
            out[t] = (t, ids, playing)
    return [out[t] for t in sorted(out)]


def read_near(reads, t_target, tol):
    best = None
    for r in reads:
        d = abs(r[0] - t_target)
        if d <= tol and (best is None or d < abs(best[0] - t_target)):
            best = r
    return best


def lagged(reads, uid_list, y_now, lag, tol):
    """Per game: (log-rank change, log player growth) over about `lag` hours, scaled to `lag`, or None."""
    t, ids, playing = reads[-1]
    past = read_near(reads[:-1], t - lag, tol)
    if past is None:
        return None
    scale = lag / max(0.5, t - past[0])
    pos = {u: i for i, u in enumerate(past[1])}
    out = {}
    for uid in uid_list:
        prev_rank = pos[uid] + 1 if uid in pos else len(past[1]) + 10  # off the chart then: it has been rising
        mom = (y_now[uid] - math.log(prev_rank)) * scale
        grow = None
        if uid in playing and uid in past[2]:
            grow = (math.log(playing[uid]) - math.log(past[2][uid])) * scale
        out[uid] = (mom, grow)
    return out


def features(reads, lag=LAG_H, tol=6.0, rel_growth=True, short_lag=6.0):
    """Per game on the last chart: (log rank now, 24h log-rank trend, 24h log player growth, short log-rank trend).

    Growth is relative to the chart's median growth when rel_growth: the chart ranks games against each other, and a
    weekday or seasonal swing that lifts every game's players moves no game's place."""
    t, ids, playing = reads[-1]
    y_now = {u: math.log(i + 1) for i, u in enumerate(ids)}
    long = lagged(reads, ids, y_now, lag, tol)
    short = lagged(reads, ids, y_now, short_lag, short_lag / 2)
    med = 0.0
    if long is not None and rel_growth:
        g = sorted(v[1] for u, v in long.items() if v[1] is not None and y_now[u] < math.log(61))
        if g:
            med = g[len(g) // 2]
    feats = {}
    for uid in ids:
        mom, grow, mom_s = 0.0, 0.0, 0.0
        if long is not None:
            mom = long[uid][0]
            if long[uid][1] is not None:
                grow = long[uid][1] - med
        if short is not None:
            mom_s = short[uid][0]
        feats[uid] = (y_now[uid], mom, grow, mom_s)
    return feats


def recent_vol(reads, alpha, span=72.0, target=6.0, max_rank=30):
    """Mean |change in log rank| / h**alpha over pairs of reads about `target` hours apart in the last `span` hours,
    for games in the top `max_rank` that stayed on the chart; (value, number of game-pairs)."""
    t_end = reads[-1][0]
    rs = [r for r in reads if r[0] >= t_end - span]
    tot, n = 0.0, 0
    j = 0
    for i, (ta, ida, _) in enumerate(rs):
        # the later read closest to target hours after this one, within 4..10 h
        best = None
        for tb, idb, _ in rs[i + 1:]:
            dt = tb - ta
            if dt > 10:
                break
            if dt >= 4 and (best is None or abs(dt - target) < abs(best[0] - ta - target)):
                best = (tb, idb)
        if best is None:
            continue
        posb = {u: k for k, u in enumerate(best[1])}
        hp = (best[0] - ta) ** alpha
        for k, u in enumerate(ida[:max_rank]):
            if u in posb:
                tot += abs(math.log(posb[u] + 1) - math.log(k + 1)) / hp
                n += 1
        j += 1
    return (tot / n if n else None), n


def window_hours(task, t_last):
    """Hours from the last read to each scored slot (the slot's start: the recorder reads on the hour)."""
    w = task.get("window") or {}
    try:
        start = parse_time(w.get("from") or task.get("history_until"))
    except (TypeError, ValueError, AttributeError):
        start = t_last
    slots = int(w.get("slots") or 24)
    minutes = float(w.get("slot_minutes") or 60)
    return [max(0.05, start + s * minutes / 60.0 - t_last) for s in range(slots)]


def sigma_of(rank, p):
    if rank >= 20:
        return p["sigma_low"]
    f = math.log(rank) / math.log(20)
    return p["sigma_top"] + (p["sigma_low"] - p["sigma_top"]) * f


def simulate(reads, task, params):
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    places = int(task.get("places") or 10)
    t_last, ids, _ = reads[-1]
    feats = features(reads, rel_growth=bool(p["rel_growth"]))
    hours = window_hours(task, t_last)
    H = len(hours)
    rng = random.Random(p["seed"])
    n = len(ids)
    y0 = [feats[u][0] for u in ids]
    drift = []
    for u in ids:
        _, mom, grow, mom_s = feats[u]
        d = p["c_mom"] * mom - p["c_grow"] * grow + p["c_mom_s"] * mom_s
        d = max(-p["drift_cap"], min(p["drift_cap"], d))
        drift.append(d / 24.0)
    vol_mult = 1.0
    if p["vol_w"] > 0:
        v, npairs = recent_vol(reads, p["alpha"])
        if v is not None and npairs >= 50:
            vol_mult = max(0.5, min(2.0, (v / p["vol_ref"]) ** p["vol_w"]))
    sig = [sigma_of(i + 1, p) * vol_mult for i in range(n)]
    hpow = [h ** p["alpha"] for h in hours]
    hmax = hours[-1]
    counts = [dict() for _ in range(places)]
    gauss = rng.gauss
    expo = rng.expovariate
    exit_rate, entry_rate = p["exit_rate"], p["entry_rate"]
    log12 = math.log(12)
    sims = int(p["sims"])
    for _ in range(sims):
        z = [gauss(0.0, 1.0) for _ in range(n)]
        # exit time per game (hours from the last read); inf when it stays the whole window
        ex = [expo(exit_rate) if exit_rate > 0 else math.inf for _ in range(n)]
        # entrants: (arrival hour, latent y)
        ent = []
        if entry_rate > 0:
            tt = expo(entry_rate)
            while tt < hmax:
                ent.append((tt, rng.uniform(-0.3, log12)))
                tt += expo(entry_rate)
        for hi in range(H):
            h = hours[hi]
            hp = hpow[hi]
            ys = [(y0[i] + drift[i] * h + sig[i] * hp * z[i], i) for i in range(n) if ex[i] > h]
            for (ta, ye) in ent:
                if ta <= h:
                    ys.append((ye, -1))
            ys.sort()
            for k in range(min(places, len(ys))):
                i = ys[k][1]
                if i >= 0:
                    c = counts[k]
                    c[i] = c.get(i, 0) + 1
    total = float(sims * H)
    return ids, [{ids[i]: c / total for i, c in ck.items()} for ck in counts], p


def prior_row(k, ids, width):
    """A broad kernel around place k over the chart's games, in log-rank."""
    lk = math.log(k)
    w = {}
    for i, u in enumerate(ids[:60]):
        d = (math.log(i + 1) - lk) / width
        w[u] = math.exp(-0.5 * d * d)
    s = sum(w.values())
    return {u: v / s for u, v in w.items()}


def build_doc(ids, sim, p, places):
    rows = []
    smooth, unnamed = p["smooth"], p["unnamed"]
    for k in range(1, places + 1):
        prior = prior_row(k, ids, p["prior_width"])
        mass = 1.0 - unnamed
        cand = set(sim[k - 1]) | set(prior)
        probs = {}
        for u in cand:
            probs[u] = mass * ((1 - smooth) * sim[k - 1].get(u, 0.0) + smooth * prior.get(u, 0.0))
        # keep the 100 largest; round down to 6 decimals so the place never sums over 1
        best = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
        for u, q in best:
            q = math.floor(q * 1e6) / 1e6
            if q > 0:
                rows.append({"universe_id": u, "place": k, "p": q})
    # point guess: by expected place over the window, then chart order
    exp_place = {}
    for k in range(1, places + 1):
        for u, q in sim[k - 1].items():
            exp_place[u] = exp_place.get(u, 0.0) + q * k
    occ = {u: sum(sim[k].get(u, 0.0) for k in range(places)) for u in ids}

    def key(u):
        o = occ.get(u, 0.0)
        # expected place given in the top 10, with absence counted as place places+3
        e = (exp_place.get(u, 0.0) + (1 - o) * (places + 3))
        return (e, ids.index(u))

    top = sorted(ids, key=key)[:places]
    return {"probabilities": rows, "top10": top}


def forecast(snaps, task, params=None):
    try:
        until = parse_time(task.get("history_until"))
    except (TypeError, ValueError, AttributeError):
        until = None
    reads = clean_snapshots(snaps, until)
    if not reads:
        reads = clean_snapshots(snaps, None)
    places = int(task.get("places") or 10)
    ids, sim, p = simulate(reads, task, params)
    return build_doc(ids, sim, p, places)
