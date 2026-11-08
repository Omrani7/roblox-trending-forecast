import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import KIT_INPUT, run_main, synthetic_history, write_input  # noqa: E402


class Determinism(unittest.TestCase):
    def test_two_runs_byte_identical(self):
        snaps, task = synthetic_history(seed=7)
        d = write_input(snaps, task)
        t0 = time.time()
        _, a = run_main(d)
        elapsed = time.time() - t0
        _, b = run_main(d)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        self.assertLess(elapsed, 120, "one run should take well under the 300 s limit")

    @unittest.skipUnless(os.path.exists(os.path.join(KIT_INPUT, "history.ndjson")), "kit input not present")
    def test_kit_input_byte_identical(self):
        _, a = run_main(KIT_INPUT)
        _, b = run_main(KIT_INPUT)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
