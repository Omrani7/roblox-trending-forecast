# Roblox Top Trending forecaster (make.fun `roblox-trending-forecast-01`)

A Python 3.13, standard-library-only program that reads the history of Roblox's Top Trending chart
(device=computer, country=us) and writes one probabilistic forecast of its top 10 for every hour of the
next 24 hours: for each place 1–10, a probability for each named game, plus a point-guess top 10.

```
MAKEFUN_INPUT_DIR=<dir with history.ndjson, task.json> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Without the variables it reads `/input` and writes `/output/<task.output>`. The scored window, the places and the
output name all come from `task.json`; nothing about the practice or final dates is in the code.

## The model (`forecast_model.py`)

A seeded Monte-Carlo simulation of a latent log-rank, re-ranked at every scored hour:

* **State.** Each game on the last read starts at `y = log(rank)`.
* **Drift** (per game, from the history only): its 24-hour growth in concurrent players *relative to the chart's
  median growth*. The chart ranks the "largest relative increase in daily active users", and concurrent players lead
  it, so a game whose players are growing faster than the chart's climbs. Growth is compared at the same time of day
  (24 h, optionally 48 h) because concurrent players swing several-fold through the day. Rank momentum was tested
  and dropped: once player growth is in, it adds nothing.
* **Diffusion.** `sigma(rank) * h**alpha * Z` with `alpha = 0.5`, larger near the top in log-rank terms, and scaled
  per game by its own jumpiness over the last 48 hours (shrunk toward 1 by how many read pairs it rests on): a game
  that has been bouncing around keeps a wider distribution, a settled one a tighter one.
* **Exits and entrants.** Each game leaves the chart at a constant hazard (the top games do leave: DONT SPLASH went
  from #1 to off the chart); unseen games arrive in the top places at a constant rate. Both shift everyone else.
* **Window integration.** The occupancy of each place is counted at each of the window's 24 hourly slots (the hour's
  distance from the last read) across all paths and averaged: one forecast for the whole window, as it is scored.
* **Calibration.** The simulated marginals are mixed with a broad log-rank kernel prior (`smooth`), a small mass is
  left unnamed for games not yet on the chart, probabilities are rounded down so no place sums over 1.

`main.py` validates the model's output shape and falls back to the house persistence rule if anything goes wrong,
so a run always writes a valid forecast. The simulation uses a fixed seed: two runs write byte-identical files.

## Validation (`backtest.py`, `tune.py`)

Walk-forward: for each cutoff (a read's time + 1 h, as the final's last read is 1 h before its window), the model
sees only reads that started before the cutoff and is scored with an exact port of the competition's scorer
(`scoring.py`, checked against the harness's `ranked.ts`) on the reads in the following 24 hours, one per hour slot.

Development data: the kit's 26 practice reads plus the public archive's later reads (36 reads, 2026-10-30 to
2026-11-08, converted the kit's way; the conversion reproduces 25 of the kit's 26 reads exactly, the 26th is in
the kit). 28 cutoffs, each forecasting a 24-hour window holding 1 to 4 archive reads. The windows overlap and
the sample is small, so the ± is a standard error of the mean over correlated cutoffs: read it as optimistic.

| model | mean log score, 28 cutoffs | 21 cutoffs whose window misses the practice window |
|---|---|---|
| house persistence baseline (`starter/main.ts`, ported) | -23.87 ± 1.51 | -23.49 ± 1.78 |
| empirical rank-transition kernel (`kernel_model.py`) | -19.27 ± 1.00 | -18.64 ± 1.02 |
| simulation, untuned first version | -18.93 ± 0.84 | |
| **simulation, final parameters** | **-17.91 ± 1.03** | **-17.06 ± 1.09** |

The final model beats the baseline on 25 of the 28 cutoffs (mean paired gain +5.9 nats a read) and in the window's
later hours (12 to 24 h after the cutoff: -20.1 against -27.1). Its probabilities are calibrated: games given
0.2-0.3 at a place held it 24.5% of the time, 0.3-0.4 34.4%, 0.4-0.5 46.5%.

What was tried and rejected on the evidence: rank momentum (24 h and 6 h; no gain once player growth is in), a
volatility scaled by the whole chart's recent volatility (worse: a calm stretch did not predict a calm next day),
a linear pool with the kernel model (worse at every weight), a separate exit hazard for the top 3, and "rocket"
jumps for games shooting up from far down the chart. The rockets improved the full backtest only through the
practice window's own event (Ragdoll Physics #42 to #5); on the cutoffs away from it they lost, so they are off.
Each kept change was checked on the 21 cutoffs away from the practice window as well.

The practice window itself is an unusual day: the #1 of a week, DONT SPLASH, left the chart and two games rose from
#35 and #42 into the top 7. The program scores -31.94 on it (baseline -32.68), by the harness's own `ranked.ts` on
the archive's 3 reads in that window; those reads were among the backtest's answers, so this is not an
out-of-sample number.

## Tests

```
python -m unittest discover -s tests -t .
```

Schema and size, byte-for-byte determinism, runtime, sparse/malformed histories, reads after the cutoff ignored,
unusual ranks, late entrants, mixed archive + hourly cadence.

## Data

The repository ships no data. The archive used for development is the public MIT-licensed
[Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots) (as converted in the competition kit).

MIT licensed.
