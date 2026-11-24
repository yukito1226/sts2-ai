"""解説（tools/hint/explain.py）のテスト。Claude API は呼ばない（偽のクライアントを差し込む）。

  ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HINT = ROOT / "tools" / "hint"
sys.path.insert(0, str(HINT))
sys.path.insert(0, str(ROOT / "tests"))

import explain  # noqa: E402


class FakeStream:
    def __init__(self, chunks, stop_reason, content, model):
        self.chunks, self.stop_reason, self.content, self.model = chunks, stop_reason, content, model

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.chunks

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop_reason, content=self.content, model=self.model,
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class FakeClient:
    """client.beta.messages.stream(**kw) だけを真似る。呼ばれた引数を calls に残す。"""

    def __init__(self, replies=None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, ensure_ascii=False)))      # 後から messages が変わっても残るように写す
        r = self.replies.pop(0) if self.replies else ("ok", "end_turn")
        text, stop = r[0], r[1]
        extra = r[2] if len(r) > 2 else []
        content = extra + ([SimpleNamespace(type="text", text=text)] if text else [])
        return FakeStream([text[: len(text) // 2], text[len(text) // 2:]] if text else [], stop, content, r[3] if len(r) > 3 else explain.MODEL)


def sample_rec() -> dict:
    strike = {"action": "カードを使う", "action_kind": 2, "text": "ストライク → ニビット#0", "a_key": "CARD.STRIKE_IRONCLAD",
              "facts": {"damage": {"per_hit": 6, "hits": 1, "total": 6, "kills": 0}, "hp_loss_if_end_turn_after": 6.0,
                        "hp_after_end_turn": 69.0, "raw": [0.0] * 20}}
    return {
        "step": 3,
        "decision": {"kind": 1, "screen": "戦闘", "phase": "combat", "num_candidates": 4},
        "state": {"act": 1, "floor": 2, "hp": 75, "max_hp": 80, "in_combat": True, "energy": 3,
                  "enemies": [{"name": "ニビット#0", "hp": 30, "max_hp": 44, "block": 0, "attack": {"per_hit": 6, "hits": 1, "total": 6}}]},
        "net": {"value": 0.0213, "value_scalar": 0.0213, "head": "combat", "combat_win_prob": 0.48, "combat_hp_loss_frac": 0.22},
        "candidates": [
            {"index": 3, "rank": 1, "action": "カードを使う", "action_kind": 2, "text": "強打 → ニビット#0", "prob": 0.4,
             "facts": {"damage": {"per_hit": 8, "hits": 1, "total": 8, "kills": 0}, "raw": [0.0] * 20},
             "readout": {"advantage": 0.5},
             "after": {"value_mean": 0.02, "value_delta": -0.001, "value_se": 0.002, "n": 4, "hp_mean": 69.0, "enemy_hp_mean": 13.0,
                       "died_frac": 0.0, "horizon": "end_of_turn", "best_by_lookahead": True, "steps_mean": 2.0}},
            {"index": 0, "rank": 2, **strike, "prob": 0.2, "group_prob": 0.4, "same_as": 0},
            {"index": 1, "rank": 3, **strike, "prob": 0.2, "group_prob": 0.4, "same_as": 0},
            {"index": 2, "rank": 4, "action": "カードを使う", "action_kind": 2, "text": "防御", "prob": 0.2,
             "facts": {"block": 5, "hp_loss_if_end_turn_after": 1.0, "raw": [0.0] * 20}},
        ],
        "greedy": 3,
        "glossary": {"強打": "{Damage}ダメージを与える。弱体を2付与。"},
    }


class TestPrompt(unittest.TestCase):
    def test_required_statements(self):
        self.assertIn("渡した数値と事実に基づいて説明すること。数値と矛盾することは言わないこと。", explain.SYSTEM)
        self.assertIn("この説明は後付けの推測であり、ネットが実際に考えた理由ではない", explain.SYSTEM)
        self.assertIn("勝率ではない", explain.SYSTEM)

    def test_compact(self):
        c = explain.compact(sample_rec())
        self.assertEqual([x["index"] for x in c["candidates"]], [3, 0, 2])        # 同じストライク2枚は1つに
        self.assertEqual(c["candidates"][1]["copies"], 2)
        self.assertEqual(c["candidates"][1]["group_prob"], 0.4)
        for x in c["candidates"]:
            self.assertNotIn("raw", x["facts"])
            self.assertNotIn("readout", x)
        self.assertNotIn("steps_mean", c["candidates"][0]["after"])
        self.assertEqual(c["candidates"][0]["after"]["value_delta"], -0.001)
        self.assertEqual(c["glossary"]["強打"], "{Damage}ダメージを与える。弱体を2付与。")
        c2 = explain.compact(sample_rec(), top=2)
        self.assertEqual(len(c2["candidates"]), 2)
        self.assertEqual(c2["omitted_candidates"], 1)


class TestExplainer(unittest.TestCase):
    def make(self, replies=None, **kw):
        out = io.StringIO()
        fc = FakeClient(replies)
        return explain.Explainer(client=fc, out=out, **kw), fc, out

    def test_start_and_follow_up(self):
        ex, fc, out = self.make([("強打で弱体を付けるのが1位です。", "end_turn"), ("防御は被ダメージ 1 まで減らせますが…", "end_turn")])
        self.assertEqual(ex.start(sample_rec()), "強打で弱体を付けるのが1位です。")
        k = fc.calls[0]
        self.assertEqual(k["model"], "claude-opus-5-5")
        self.assertEqual(k["system"], explain.SYSTEM)
        self.assertEqual(k["thinking"], {"type": "adaptive"})
        self.assertEqual(k["output_config"], {"effort": "medium"})
        self.assertEqual(k["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(k["fallbacks"], "default")
        self.assertEqual(k["messages"][-1]["role"], "user")                     # 先頭埋め（prefill）はしない
        self.assertIn("強打 → ニビット#0", k["messages"][0]["content"])
        self.assertIn('"value_delta":-0.001', k["messages"][0]["content"])
        self.assertIn(explain.DISCLAIMER, out.getvalue())
        self.assertIn("強打で弱体を付けるのが1位です。", out.getvalue())

        ex.ask("なぜ防御じゃないの？")
        m = fc.calls[1]["messages"]
        self.assertEqual([x["role"] for x in m], ["user", "assistant", "user"])
        self.assertEqual(m[1]["content"], "強打で弱体を付けるのが1位です。")
        self.assertEqual(m[2]["content"], "なぜ防御じゃないの？")
        self.assertEqual(len(ex.messages), 4)

        ex.start(sample_rec())                                                  # 新しい局面では会話をやり直す
        self.assertEqual(len(fc.calls[2]["messages"]), 1)

    def test_options(self):
        ex, fc, _ = self.make(effort="low", fallbacks=False, model="claude-sonnet-5-5")
        ex.start(sample_rec(), "防御について教えて")
        k = fc.calls[0]
        self.assertNotIn("fallbacks", k)
        self.assertNotIn("betas", k)
        self.assertEqual(k["output_config"], {"effort": "low"})
        self.assertEqual(k["model"], "claude-sonnet-5-5")
        self.assertTrue(k["messages"][0]["content"].endswith("防御について教えて"))

    def test_refusal_is_dropped(self):
        ex, fc, out = self.make([("ok", "end_turn"), ("途中まで", "refusal")])
        ex.start(sample_rec())
        self.assertIsNone(ex.ask("変な質問"))
        self.assertEqual(len(ex.messages), 2)                                   # 断られたやりとりは履歴に残さない
        self.assertIn("答えられない", out.getvalue())
        self.assertIn("途中までの文は使わない", out.getvalue())

    def test_fallback_and_truncation_notes(self):
        fb = [SimpleNamespace(type="fallback")]
        ex, fc, out = self.make([("答え", "max_tokens", fb, "claude-opus-5")])
        self.assertEqual(ex.start(sample_rec()), "答え")
        self.assertIn("claude-opus-5 が代わりに答えました", out.getvalue())
        self.assertIn("途中で切れました", out.getvalue())

    def test_missing_key_message(self):
        class NoKey:
            beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: (_ for _ in ()).throw(
                TypeError("Could not resolve authentication method."))))
        out = io.StringIO()
        ex = explain.Explainer(client=NoKey(), out=out)
        self.assertIsNone(ex.start(sample_rec()))
        self.assertEqual(ex.messages, [])
        self.assertIn("ANTHROPIC_API_KEY", out.getvalue())

    def test_ask_before_start(self):
        ex, _, _ = self.make()
        with self.assertRaises(explain.ExplainError):
            ex.ask("?")


@contextlib.contextmanager
def fake_client_factory(fc):
    old = explain.make_client
    explain.make_client = lambda: fc
    try:
        yield
    finally:
        explain.make_client = old


class TestWithMockGame(unittest.TestCase):
    """モックのゲームで対話モードの e / ? と、JSONL からの解説を通す。"""

    @classmethod
    def setUpClass(cls):
        import adrepo
        cls.repo = adrepo.setup()
        import make_mock_ckpt
        cls.tmp = Path(tempfile.mkdtemp(prefix="explain_test_"))
        cls.ckpt = cls.tmp / "mock.pt"
        make_mock_ckpt.make(str(cls.ckpt), milestone_from=0)
        loc = cls.tmp / "loc" / "jpn"
        loc.mkdir(parents=True)
        (loc / "cards.json").write_text(json.dumps({"BASH.title": "強打", "BASH.description": "[gold]{Damage}[/gold]ダメージを与える。"},
                                                   ensure_ascii=False), encoding="utf-8")

    def args(self, *extra):
        import hint
        return hint.build_parser().parse_args(["--ad-repo", str(self.repo), "interactive", "--mock", "--allow-unlisted",
                                               "--ckpt", str(self.ckpt), "--loc-dir", str(self.tmp / "loc"), *extra])

    def run_interactive(self, stdin: str, *extra) -> tuple[str, FakeClient]:
        import hint
        fc = FakeClient([("解説その1", "end_turn"), ("聞き返しの答え", "end_turn"), ("次の局面の解説", "end_turn")])
        out = io.StringIO()
        old_in = sys.stdin
        sys.stdin = io.StringIO(stdin)
        try:
            with fake_client_factory(fc), contextlib.redirect_stdout(out):
                hint.interactive(self.args(*extra))
        finally:
            sys.stdin = old_in
        return out.getvalue(), fc

    def test_interactive_explain(self):
        # 1つ目（イベント）は Enter、2つ目（戦闘）で l 2 → e → ? → Enter、3つ目で ? だけ（前の会話は持ち越さない）→ q
        text, fc = self.run_interactive("\nl 2\ne\n? なぜ防御じゃないの？\n\n? まだある？\nq\n")
        self.assertIn(explain.DISCLAIMER, text)
        self.assertIn("解説その1", text)
        self.assertIn("聞き返しの答え", text)
        self.assertIn("先に e で解説を始めてください", text)
        self.assertEqual(len(fc.calls), 2)
        first = fc.calls[0]["messages"][0]["content"]
        data = json.loads(first.split("```json\n", 1)[1].split("\n```", 1)[0])
        self.assertEqual(data["decision"]["phase"], "combat")
        self.assertTrue(all("after" in c for c in data["candidates"]))         # l の見込みも渡る
        self.assertEqual(data["glossary"].get("強打"), "{Damage}ダメージを与える。")    # 書式タグは外す
        self.assertEqual(fc.calls[1]["messages"][2]["content"], "なぜ防御じゃないの？")

    def test_explain_from_jsonl(self):
        import subprocess
        out = self.tmp / "run.jsonl"
        r = subprocess.run([sys.executable, "-I", str(HINT / "hint.py"), "autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt),
                            "--out", str(out), "--quiet"], capture_output=True, text=True, encoding="utf-8",
                           env=dict(os.environ, ALPHADEFECT_REPO=str(self.repo)), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        steps = [json.loads(l).get("step") for l in out.read_text(encoding="utf-8").splitlines() if '"candidates"' in l]
        fc = FakeClient([("この局面は…", "end_turn"), ("それは…", "end_turn")])
        buf = io.StringIO()
        old_in = sys.stdin
        sys.stdin = io.StringIO("なぜ？\n\n")
        try:
            with fake_client_factory(fc), contextlib.redirect_stdout(buf):
                rc = explain.main([str(out), "--step", str(steps[1])])
        finally:
            sys.stdin = old_in
        self.assertEqual(rc, 0)
        self.assertEqual(len(fc.calls), 2)
        self.assertIn("それは…", buf.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(explain.main([str(out), "--step", "99999"]), 1)
        self.assertIn("step 99999", err.getvalue())


if __name__ == "__main__":
    unittest.main()
