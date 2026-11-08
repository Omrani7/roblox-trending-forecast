"""Robustness: sparse or malformed histories, unusual ranks, reads past history_until, late entrants."""
import copy
import json
import os
import sys
import unittest
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forecast_model as fm  # noqa: E402
import main as entry  # noqa: E402
import scoring  # noqa: E402
from tests.helpers import run_main, synthetic_history, write_input  # noqa: E402


def ok(doc):
    f, why = scoring.validate_forecast(json.loads(json.dumps(doc)))
    return f is not None, why


class Robustness(unittest.TestCase):
    def test_single_snapshot(self):
        snaps, task = synthetic_history(n_reads=1)
        doc = entry.build_forecast(snaps, task)
        self.assertTrue(ok(doc)[0])
        self.assertEqual(doc["top10"][0], snaps[0]["top_trending"][0]["universe_id"])

    def test_two_snapshots_no_24h_lag(self):
        snaps, task = synthetic_history(n_reads=2, step_h=1)
        self.assertTrue(ok(fm.forecast(snaps, task))[0])

    def test_missing_optional_fields(self):
        snaps, task = synthetic_history()
        for s in snaps:
            for k in ("source", "filters", "details", "top_playing_now"):
                s.pop(k, None)
            for r in s["top_trending"]:
                r["playing"] = None
                r.pop("name", None)
        self.assertTrue(ok(fm.forecast(snaps, task))[0])

    def test_malformed_lines_and_rows(self):
        snaps, task = synthetic_history()
        snaps[3]["top_trending"] = None
        snaps[4]["ok"] = False
        snaps[5]["top_trending"][2] = "garbage"
        snaps[6]["top_trending"][0]["universe_id"] = str(snaps[6]["top_trending"][0]["universe_id"])
        snaps[7]["top_trending"][1]["rank"] = None
        p, data = run_main(write_input(snaps, task, raw_lines=["{not json", "", "[]"]))
        self.assertEqual(p.returncode, 0, p.stderr)
        f, why = scoring.validate_forecast(data)
        self.assertIsNotNone(f, why)

    def test_reads_after_cutoff_are_ignored(self):
        snaps, task = synthetic_history()
        future = copy.deepcopy(snaps[-1])
        future["at"] = "2030-01-01T00:00:00.000Z"
        future["top_trending"].reverse()
        a = fm.forecast(snaps, task)
        b = fm.forecast(snaps + [future], task)
        self.assertEqual(a, b)

    def test_unusual_ranks_and_short_chart(self):
        snaps, task = synthetic_history()
        last = snaps[-1]
        last["top_trending"] = last["top_trending"][:12]
        for i, r in enumerate(last["top_trending"]):
            r["rank"] = 100 + 3 * i  # gaps and offsets: order still decides the place
        doc = fm.forecast(snaps, task)
        self.assertTrue(ok(doc)[0])
        self.assertEqual(doc["top10"][0], last["top_trending"][0]["universe_id"])

    def test_late_entrant_gets_probability(self):
        snaps, task = synthetic_history()
        newcomer = 999999
        snaps[-1]["top_trending"].insert(2, dict(snaps[-1]["top_trending"][2], universe_id=newcomer))
        for i, r in enumerate(snaps[-1]["top_trending"]):
            r["rank"] = i + 1
        doc = fm.forecast(snaps, task)
        p3 = {r["universe_id"]: r["p"] for r in doc["probabilities"] if r["place"] == 3}
        self.assertGreater(p3.get(newcomer, 0), 0.1)

    def test_window_read_from_task(self):
        snaps, task = synthetic_history()
        hours = fm.window_hours(task, fm.clean_snapshots(snaps)[-1][0])
        self.assertEqual(len(hours), 24)
        self.assertAlmostEqual(hours[0], 0.5, places=6)
        self.assertAlmostEqual(hours[-1], 23.5, places=6)

    def test_mixed_cadence_history(self):
        # an archive-like sparse part followed by an hourly part, as the final history is
        sparse, _ = synthetic_history(n_reads=20, step_h=6, seed=3)
        last = fm.parse_time(sparse[-1]["at"])
        dense, task = synthetic_history(n_reads=60, step_h=1, seed=4)
        shift = timedelta(hours=last - fm.parse_time(dense[0]["at"]) + 1)
        from datetime import datetime
        for s in dense:
            dt = datetime.fromisoformat(s["at"].replace("Z", "+00:00")) + shift
            s["at"] = dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        until = datetime.fromisoformat(dense[-1]["at"].replace("Z", "+00:00")) + timedelta(hours=1)
        from tests.helpers import task_for
        doc = fm.forecast(sparse + dense, task_for(until, 80))
        self.assertTrue(ok(doc)[0])


if __name__ == "__main__":
    unittest.main()
