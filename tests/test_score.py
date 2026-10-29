import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from edge_escalate.__main__ import main
from edge_escalate.score import score_path

ROOT = Path(__file__).resolve().parents[1]
TURNS = ROOT / "fixtures" / "turns.jsonl"


class ScoreTests(unittest.TestCase):
    def test_holdout_score(self) -> None:
        report = score_path(TURNS)
        self.assertEqual(report.threshold, 4)
        self.assertEqual(report.holdout, 8)
        self.assertEqual(report.precision, 0.75)
        self.assertEqual(report.recall, 0.75)
        self.assertEqual(report.escalations, 4)
        self.assertEqual(report.cloud_milli, 8)
        self.assertEqual(report.saved_milli, 8)
        self.assertEqual(report.false_escalate, ("h-closed-novel-risk",))
        self.assertEqual(report.false_stay, ("h-long-unresolved",))

    def test_cli_prints_the_holdout(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            code = main([str(TURNS), "--verbose"])
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn("0.75", text)
        self.assertIn("$0.008", text)
        self.assertIn("Threshold 4", text)
        self.assertIn("h-closed-novel-risk", text)
        self.assertIn("h-long-unresolved", text)
        self.assertIn("illustrative", text)


if __name__ == "__main__":
    unittest.main()
