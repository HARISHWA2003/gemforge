"""Real OpenAI API run on one batch. Costs money; skipped unless explicitly enabled:

    GEMSTONE_INTEGRATION=1 GEMSTONE_INTEGRATION_BATCH=input/Ruby_001 python -m unittest tests.test_integration

The batch is copied to a temp dir; outputs are kept in output/_integration/ for visual review.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import processor as P  # noqa: E402


@unittest.skipUnless(os.getenv("GEMSTONE_INTEGRATION") == "1", "set GEMSTONE_INTEGRATION=1 to call the real API")
class TestRealAPI(unittest.TestCase):
    def test_one_batch_end_to_end(self):
        src = (ROOT / os.environ["GEMSTONE_INTEGRATION_BATCH"]).resolve()
        cfg = P.load_config()
        tmp = Path(tempfile.mkdtemp())
        cfg.input_dir, cfg.failed_dir = tmp / "input", tmp / "failed"
        cfg.output_dir = ROOT / "output" / "_integration"
        shutil.copytree(src, cfg.input_dir / src.name)
        P.setup_logging(cfg.log_dir)
        from openai import OpenAI
        client = OpenAI(max_retries=0, timeout=cfg.timeout)

        code = P.run(P.parse_args(["--force", src.name]), cfg, lambda: client)

        self.assertEqual(code, 0)
        out = cfg.output_dir / P.sanitize(src.name)
        pngs = sorted(out.glob("*.png"))
        self.assertEqual(len(pngs), 3)
        self.assertTrue(all(P.is_valid_output(p) for p in pngs))
        print("\nReview these images by eye:", *pngs, sep="\n  ")
        shutil.rmtree(tmp)
