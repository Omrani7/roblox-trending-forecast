import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring  # noqa: E402
from tests.helpers import KIT_INPUT, run_main, synthetic_history, write_input  # noqa: E402


class OutputSchema(unittest.TestCase):
    def check(self, data):
        self.assertIsNotNone(data, "no forecast file")
        self.assertLessEqual(len(data), 1_048_576)
        f, why = scoring.validate_forecast(data)
        self.assertIsNotNone(f, why)
        doc = json.loads(data)
        self.assertEqual(sorted(doc), ["probabilities", "top10"])
        for k in range(1, 11):
            rows = [r for r in doc["probabilities"] if r["place"] == k]
            self.assertLessEqual(len(rows), 100)
            self.assertLessEqual(sum(r["p"] for r in rows), 1.0)
            self.assertEqual(len({r["universe_id"] for r in rows}), len(rows))
            self.assertGreater(len(rows), 0, f"place {k} names no game")
        self.assertEqual(len(set(doc["top10"])), 10)
        return doc

    def test_synthetic(self):
        snaps, task = synthetic_history()
        p, data = run_main(write_input(snaps, task))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.check(data)
        self.assertNotIn(b"falling back", p.stderr)

    @unittest.skipUnless(os.path.exists(os.path.join(KIT_INPUT, "history.ndjson")), "kit input not present")
    def test_kit_practice_input(self):
        p, data = run_main(KIT_INPUT)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = self.check(data)
        # the forecast's favourite at place 1 is a game from the last read's top 3
        last = [json.loads(l) for l in open(os.path.join(KIT_INPUT, "history.ndjson"), encoding="utf-8")][-1]
        top3 = [r["universe_id"] for r in last["top_trending"][:3]]
        p1 = max((r for r in doc["probabilities"] if r["place"] == 1), key=lambda r: r["p"])
        self.assertIn(p1["universe_id"], top3)


if __name__ == "__main__":
    unittest.main()
