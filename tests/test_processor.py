"""Unit tests with a fake OpenAI client. No network. Run: python -m unittest discover tests"""
import base64
import contextlib
import hashlib
import io
import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import processor as P  # noqa: E402

P.BACKOFF_BASE = 0
logging.getLogger("gemstone").addHandler(logging.NullHandler())
logging.getLogger("gemstone").propagate = False


def png_b64(size=(1088, 1088), color=(255, 255, 255)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


GOOD = png_b64()
NET_ERR = openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com"))


def classification(names, background="white", stone="other", **override):
    data = {"reasoning": "test", "front_source": names[1], "back_source": names[0], "side_source": names[2],
            "background": background, "stone_type": stone,
            "confidence": {"front": .9, "back": .9, "side": .9, "background": .9}}
    data.update(override)
    return json.dumps(data)


class FakeClient:
    """classify: list of output_text strings (or exceptions), consumed in order; last one repeats.
    edit(call_no, source_name, prompt): return None for a good image, 'bad' for garbage, or an exception."""

    def __init__(self, classify=None, edit=None):
        self._classify = classify
        self._edit = edit or (lambda *a: None)
        self.classify_calls, self.edit_calls = 0, []
        self.responses = SimpleNamespace(create=self.create)
        self.images = SimpleNamespace(edit=self.edit)

    def create(self, **kw):
        self.classify_calls += 1
        names = [c["text"].split(": ", 1)[1] for c in kw["input"][0]["content"][1::2]]
        outs = self._classify or [classification(names)]
        out = outs[min(self.classify_calls, len(outs)) - 1]
        out = out(names) if callable(out) else out
        if isinstance(out, BaseException):
            raise out
        return SimpleNamespace(output_text=out)

    def edit(self, **kw):
        name = kw["image"][0]
        self.edit_calls.append(name)
        r = self._edit(len(self.edit_calls), name, kw["prompt"])
        if isinstance(r, BaseException):
            raise r
        b64 = {None: GOOD, "bad": "not-an-image", "rect": png_b64((1536, 1024))}[r]
        return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = P.Config(*(self.tmp / d for d in ("input", "output", "failed", "logs")))
        self.cfg.input_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make_batch(self, name, files=("a.jpg", "b.jpg", "c.jpg")):
        d = self.cfg.input_dir / name
        d.mkdir()
        for i, f in enumerate(files):
            p = d / f
            if p.suffix.lower() in P.IMAGE_EXTS:
                Image.new("RGB", (64, 48), (i * 60, 10, 10)).save(p, P.Image.registered_extensions()[p.suffix.lower()])
            else:
                p.write_text("not an image")
        return d

    def run_cli(self, client, *argv):
        self.stdout = io.StringIO()
        with contextlib.redirect_stdout(self.stdout):
            return P.run(P.parse_args(list(argv)), self.cfg, lambda: client)

    def snapshot(self, d):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}

    def outputs(self, name):
        return sorted(p.name for p in (self.cfg.output_dir / name).glob("*.png"))


class TestValidation(Base):
    def test_1_three_valid_images(self):
        d = self.make_batch("Ruby_001")
        before = self.snapshot(d)
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 0)
        self.assertEqual(self.outputs("Ruby_001"), ["Ruby_001_Back_White_BG.png", "Ruby_001_Front_White_BG.png",
                                                    "Ruby_001_Side_Angle_White_BG.png"])
        for f in self.outputs("Ruby_001"):
            self.assertTrue(P.is_valid_output(self.cfg.output_dir / "Ruby_001" / f))
        m = json.loads((self.cfg.output_dir / "Ruby_001" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["mapping"], {"front": "b.jpg", "back": "a.jpg", "side": "c.jpg"})
        # each edit got only its own source
        self.assertEqual(c.classify_calls, 1)
        self.assertEqual(c.edit_calls, ["b.jpg", "a.jpg", "c.jpg"])
        self.assertEqual(self.snapshot(d), before, "input must be untouched")

    def test_black_background_naming_and_prompt(self):
        self.make_batch("Pearl_1")
        prompts = []
        c = FakeClient(classify=[lambda n: classification(n, "black", "white_or_warm_white_opal")],
                       edit=lambda i, n, p: prompts.append(p))
        self.assertEqual(self.run_cli(c), 0)
        self.assertIn("Pearl_1_Side_Angle_Black_BG.png", self.outputs("Pearl_1"))
        self.assertIn("#080808", prompts[0])
        self.assertIn("This is the FRONT output.", prompts[0])

    def assert_invalid(self, name, files):
        self.make_batch(name, files)
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 1)
        self.assertEqual((c.classify_calls, c.edit_calls), (0, []))
        self.assertFalse((self.cfg.output_dir / name).exists())

    def test_2_two_images(self):
        self.assert_invalid("Two", ("a.jpg", "b.jpg"))

    def test_3_four_images(self):
        self.assert_invalid("Four", ("a.jpg", "b.jpg", "c.jpg", "d.png"))

    def test_4_unsupported_files_ignored(self):
        self.make_batch("Ok", ("a.jpg", "b.JPEG", "c.webp", "notes.txt", ".hidden.jpg", "x.gif"))
        self.assertIsNone(P.validate_batch(self.cfg.input_dir / "Ok")[1])
        self.assertIn("found 2", P.validate_batch(self.make_batch("Gif", ("a.jpg", "b.jpg", "c.gif")))[1])

    def test_4b_corrupt_image(self):
        d = self.make_batch("Corrupt")
        (d / "c.jpg").write_bytes(b"garbage")
        self.assertIn("unreadable", P.validate_batch(d)[1])

    def test_5_empty_folder(self):
        self.assert_invalid("Empty", ())


class TestResume(Base):
    def test_6_completed_batch_skipped(self):
        self.make_batch("Ruby_001")
        self.run_cli(FakeClient())
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 0)
        self.assertEqual((c.classify_calls, c.edit_calls), (0, []))

    def test_7_partial_batch_regenerates_only_missing(self):
        self.make_batch("Ruby_001")
        self.run_cli(FakeClient())
        out = self.cfg.output_dir / "Ruby_001"
        (out / "Ruby_001_Side_Angle_White_BG.png").unlink()
        (out / "Ruby_001_Back_White_BG.png").write_bytes(b"")  # corrupt -> invalid -> regenerate
        front_mtime = (out / "Ruby_001_Front_White_BG.png").stat().st_mtime_ns
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 0)
        self.assertEqual(c.classify_calls, 0, "classification must be reused")
        self.assertEqual(c.edit_calls, ["a.jpg", "c.jpg"])
        self.assertEqual((out / "Ruby_001_Front_White_BG.png").stat().st_mtime_ns, front_mtime)

    def test_changed_sources_reclassify(self):
        d = self.make_batch("Ruby_001")
        self.run_cli(FakeClient())
        Image.new("RGB", (64, 48), "blue").save(d / "a.jpg")
        c = FakeClient()
        self.run_cli(c)
        self.assertEqual((c.classify_calls, len(c.edit_calls)), (1, 3))

    def test_11_interrupted_run_resumes(self):
        self.make_batch("Ruby_001")
        c = FakeClient(edit=lambda i, n, p: KeyboardInterrupt() if i == 2 else None)
        self.assertEqual(self.run_cli(c), 130)
        out = self.cfg.output_dir / "Ruby_001"
        self.assertEqual(self.outputs("Ruby_001"), ["Ruby_001_Front_White_BG.png"])
        self.assertEqual(list(out.glob("*.tmp")), [])
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 0)
        self.assertEqual((c.classify_calls, c.edit_calls), (0, ["a.jpg", "c.jpg"]))

    def test_dry_run_and_status_make_no_calls(self):
        self.make_batch("Ruby_001")
        self.make_batch("Bad", ("a.jpg",))
        for flag in ("--dry-run", "--status"):
            self.assertEqual(self.run_cli(None, flag), 0)
        self.assertIn("WOULD PROCESS", self.stdout.getvalue()) if flag == "--dry-run" else None
        self.assertFalse(self.cfg.output_dir.exists())


class TestFailures(Base):
    def test_8_failed_api_call_records_failure_and_continues(self):
        self.make_batch("A_Ruby")
        self.make_batch("B_Emerald")
        c = FakeClient(edit=lambda i, n, p: NET_ERR if "This is the BACK output." in p and len(c.edit_calls) <= 5 else None)
        self.assertEqual(self.run_cli(c), 1)
        err = json.loads((self.cfg.failed_dir / "A_Ruby" / "error.json").read_text(encoding="utf-8"))
        self.assertEqual((err["batch"], err["stage"], err["attempts"], err["error_type"]),
                         ("A_Ruby", "back_edit", 3, "network_error"))
        self.assertTrue((self.cfg.failed_dir / "A_Ruby" / "manifest.json").exists())
        self.assertEqual(len(self.outputs("A_Ruby")), 2, "front and side still produced")
        self.assertEqual(len(self.outputs("B_Emerald")), 3, "next batch continues")

    def test_non_retryable_error_fails_fast(self):
        self.make_batch("A")
        resp = httpx2.Response(400, request=httpx2.Request("POST", "https://api.openai.com"))
        bad = openai.BadRequestError("moderation", response=resp, body=None)
        c = FakeClient(edit=lambda *a: bad)
        self.run_cli(c)
        err = json.loads((self.cfg.failed_dir / "A" / "error.json").read_text(encoding="utf-8"))
        self.assertEqual((err["attempts"], err["error_type"]), (1, "api_error"))
        self.assertEqual(len(c.edit_calls), 3)  # one attempt per role

    def test_9_malformed_classification_retried(self):
        self.make_batch("Ruby_001")
        dup = lambda n: classification(n, side_source=n[1])  # front and side same file
        clash = lambda n: classification(n, "black", "other")  # background contradicts stone type
        c = FakeClient(classify=["not json", dup, clash, lambda n: classification(n)])  # 4th never reached
        self.assertEqual(self.run_cli(c), 1)
        self.assertEqual((c.classify_calls, c.edit_calls), (3, []))
        err = json.loads((self.cfg.failed_dir / "Ruby_001" / "error.json").read_text(encoding="utf-8"))
        self.assertEqual((err["stage"], err["error_type"]), ("classification", "malformed_output"))

        c = FakeClient(classify=["{}", lambda n: classification(n)])
        self.assertEqual(self.run_cli(c, "--retry-failed"), 0)
        self.assertEqual(c.classify_calls, 2)

    def test_10_invalid_generated_image(self):
        self.make_batch("Ruby_001")
        c = FakeClient(edit=lambda i, n, p: {1: "bad", 2: "rect"}.get(i))
        self.assertEqual(self.run_cli(c), 0)
        self.assertEqual(len(c.edit_calls), 5)
        self.assertEqual(len(self.outputs("Ruby_001")), 3)

        self.make_batch("Always_Bad")
        c = FakeClient(edit=lambda *a: "bad")
        self.assertEqual(self.run_cli(c, "--batch", "Always_Bad"), 1)
        self.assertEqual(self.outputs("Always_Bad"), [])

    def test_12_retry_failed_only_touches_failed(self):
        self.make_batch("A_Ruby")
        self.make_batch("B_Emerald")
        self.run_cli(FakeClient(edit=lambda i, n, p: NET_ERR if i <= 3 else None))  # A front fails 3x
        self.assertTrue((self.cfg.failed_dir / "A_Ruby").exists())

        c = FakeClient()
        self.assertEqual(self.run_cli(c), 0, "default run skips previously failed batches")
        self.assertEqual((c.classify_calls, c.edit_calls), (0, []))

        b_before = self.snapshot(self.cfg.output_dir / "B_Emerald")
        c = FakeClient()
        self.assertEqual(self.run_cli(c, "--retry-failed"), 0)
        self.assertEqual((c.classify_calls, c.edit_calls), (0, ["b.jpg"]))
        self.assertFalse((self.cfg.failed_dir / "A_Ruby").exists())
        self.assertEqual(self.snapshot(self.cfg.output_dir / "B_Emerald"), b_before)


class TestNaming(Base):
    def test_13_sanitized_names(self):
        self.assertEqual(P.sanitize("Ruby_001"), "Ruby_001")
        self.assertEqual(P.sanitize("Blue Sapphire #7"), "Blue_Sapphire_7")
        self.assertEqual(P.sanitize("a/b:c*?"), "a_b_c")
        self.assertEqual(P.sanitize("..."), "batch")
        self.assertEqual(P.sanitize("con"), "_con")
        self.assertEqual(P.sanitize("Com1.x"), "_Com1.x")
        self.make_batch("Blue Sapphire #7")
        self.assertEqual(self.run_cli(FakeClient(), "--batch", "Blue Sapphire #7"), 0)
        self.assertIn("Blue_Sapphire_7_Front_White_BG.png", self.outputs("Blue_Sapphire_7"))
        m = json.loads((self.cfg.output_dir / "Blue_Sapphire_7" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((m["batch"], m["output_name"]), ("Blue Sapphire #7", "Blue_Sapphire_7"))

    def test_14_duplicate_output_prevention(self):
        self.make_batch("Ruby_001")
        self.run_cli(FakeClient())
        # forced reclassification flips the background: old-background files must not linger
        c = FakeClient(classify=[lambda n: classification(n, "black", "white_or_warm_white")])
        self.assertEqual(self.run_cli(c, "--force", "Ruby_001"), 0)
        self.assertEqual(self.outputs("Ruby_001"), ["Ruby_001_Back_Black_BG.png", "Ruby_001_Front_Black_BG.png",
                                                    "Ruby_001_Side_Angle_Black_BG.png"])
        # two folders mapping to the same output name are both refused
        self.make_batch("Ruby 002")
        self.make_batch("Ruby_002")
        self.make_batch("RUBY#002")  # -> RUBY_002: same folder as Ruby_002 on Windows
        c = FakeClient()
        self.assertEqual(self.run_cli(c), 1)
        self.assertEqual(c.edit_calls, [])
        self.assertFalse((self.cfg.output_dir / "Ruby_002").exists())
        self.assertFalse((self.cfg.output_dir / "RUBY_002").exists())


if __name__ == "__main__":
    unittest.main()
