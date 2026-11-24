"""ヒントツール（tools/hint/）のモックテスト。ゲーム本体と学習済みの重みは要らない。

要るもの: AlphaDefect-Public のチェックアウト（環境変数 ALPHADEFECT_REPO）、Python 3.12 以上、numpy、torch（CPU 版でよい）。
重みは乱数で作った小さなネット（tests/make_mock_ckpt.py）、エンジンは tools/hint/mock_engine.py。

  ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HINT = ROOT / "tools" / "hint"
sys.path.insert(0, str(HINT))
sys.path.insert(0, str(ROOT / "tests"))

import adrepo  # noqa: E402

REPO = adrepo.setup()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import make_mock_ckpt  # noqa: E402


class _Evil:
    """weights_only=True なら読み込みで弾かれるはずのオブジェクト。"""
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


def run_hint(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, ALPHADEFECT_REPO=str(REPO))
    return subprocess.run([sys.executable, "-I", str(HINT / "hint.py"), *args], input=stdin, capture_output=True, text=True, encoding="utf-8",
                          env=env, timeout=300)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hint_test_"))
        cls.ckpt = cls.tmp / "mock.pt"
        make_mock_ckpt.make(str(cls.ckpt), milestone_from=0)


class TestWeights(Base):
    def test_unlisted_refused(self):
        import weights
        with self.assertRaises(SystemExit):
            weights.verify(self.ckpt, REPO, allow_unlisted=False)
        self.assertEqual(weights.verify(self.ckpt, REPO, allow_unlisted=True)["status"], "unlisted")

    def test_checksum_list(self):
        import weights
        fake = self.tmp / "fakerepo"
        (fake / "weights").mkdir(parents=True, exist_ok=True)
        good = weights.sha256_of(self.ckpt)
        (fake / "weights" / "SHA256SUMS_t.txt").write_text(f"{good}  mock.pt\n", encoding="utf-8")
        self.assertEqual(weights.verify(self.ckpt, fake)["status"], "ok")
        (fake / "weights" / "SHA256SUMS_t.txt").write_text(f"{'0' * 64}  mock.pt\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            weights.verify(self.ckpt, fake, allow_unlisted=True)      # 一覧にあって値が違うときは、--allow-unlisted でも読まない

    def test_real_sums_are_parsed(self):
        import weights
        sums = weights.known_sums(REPO)
        self.assertIn("ironclad_ema_v24782.pt", sums)
        self.assertEqual(sums["ironclad_ema_v24782.pt"][0][0], "78ba58312335b1a49823fdaf6aa263ee46f9dae46b5d4753bc609eb671588f20")

    def test_weights_only_blocks_code(self):
        import weights
        evil = self.tmp / "evil.pt"
        with open(evil, "wb") as f:
            pickle.dump({"x": _Evil()}, f)
        with self.assertRaises(Exception):
            weights.load(evil, REPO, allow_unlisted=True)

    def test_load(self):
        import weights
        L = weights.load(self.ckpt, REPO, allow_unlisted=True)
        self.assertEqual(L.version, 1000)
        self.assertEqual(L.obs_format, 28)


class TestInfer(Base):
    def _frames(self):
        """モックエンジンから、戦闘と戦闘外のフレームを1つずつ取る。"""
        from engine import Engine, mock_cmd
        from sts2env.protocol import parse
        cmd, env = mock_cmd()
        env["ALPHADEFECT_REPO"] = str(REPO)
        with Engine(cmd, env) as e:
            ev = parse(e.request({"op": "reset", "mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": "T", "labels": True}))
            cb = parse(e.request({"op": "step", "action": 0}))
        return ev, cb

    def test_matches_existing_eval_path(self):
        """方策と価値が、評価コードと同じ経路（collate_ragged → net(batch)、特権ゼロ）の結果と一致する。"""
        import weights
        from infer import evaluate
        from sts2rl.features import Ragged, collate_ragged
        L = weights.load(self.ckpt, REPO, allow_unlisted=True)
        for o in self._frames():
            out = evaluate(L.net, o)
            r = Ragged.from_obs([o], np.array([1]))
            r.tok_priv[:] = 0
            with torch.no_grad():
                lg, v = L.net(collate_ragged(r).to("cpu"))
                lg2, v2 = L.net.infer_ready().forward_infer(collate_ragged(r).to("cpu"))      # 探索で使う推論専用の経路
            np.testing.assert_allclose(lg2[0, :o.num_actions].numpy(), lg[0, :o.num_actions].numpy(), atol=1e-5)
            p = torch.softmax(lg[0, :o.num_actions].double(), -1).numpy()
            np.testing.assert_allclose(out.probs, p, rtol=0, atol=1e-12)
            self.assertAlmostEqual(out.value, float(v[0]), places=6)
            self.assertAlmostEqual(float(out.probs.sum()), 1.0, places=9)

    def test_privileged_inputs_ignored(self):
        """フレームに特権入力が入っていても、出力は特権ゼロのときと同じ。"""
        import copy
        import weights
        from infer import evaluate
        L = weights.load(self.ckpt, REPO, allow_unlisted=True)
        _, o = self._frames()
        o2 = copy.copy(o)
        o2.tok_priv = np.random.default_rng(0).random(o.tok_priv.shape).astype(np.float32)
        a, b = evaluate(L.net, o), evaluate(L.net, o2)
        np.testing.assert_array_equal(a.probs, b.probs)
        self.assertEqual(a.value, b.value)

    def test_heads(self):
        import weights
        from infer import evaluate
        L = weights.load(self.ckpt, REPO, allow_unlisted=True)
        ev, cb = self._frames()
        a, b = evaluate(L.net, ev), evaluate(L.net, cb)
        self.assertTrue(a.is_macro)
        self.assertFalse(b.is_macro)
        self.assertIsNotNone(b.combat_win_logit)
        self.assertEqual(b.readout.shape, (cb.num_actions, 5))


class TestAutoplay(Base):
    def test_autoplay_mock(self):
        out = self.tmp / "run.jsonl"
        r = run_hint(["autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(out), "--quiet"])
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        rows = read_jsonl(out)
        self.assertIn("meta", rows[0])
        self.assertEqual(rows[0]["meta"]["privileged_inputs"], "zeroed")
        self.assertEqual(rows[0]["meta"]["reset_extra"]["obs_format"], 28)
        self.assertIn("result", rows[-1])
        self.assertTrue(rows[-1]["result"]["win"])
        recs = rows[1:-1]
        screens = {x["decision"]["screen"] for x in recs}
        self.assertTrue({"イベント", "戦闘", "カード報酬"} <= screens, screens)
        for x in recs:
            ps = [c["prob"] for c in x["candidates"]]
            self.assertAlmostEqual(sum(ps), 1.0, places=3)
            self.assertEqual([c["rank"] for c in x["candidates"]], list(range(1, len(ps) + 1)))
            self.assertEqual(ps, sorted(ps, reverse=True))
            self.assertEqual(x["chosen"], x["greedy"])
            self.assertEqual(x["candidates"][0]["index"], x["greedy"])
        combat = [x for x in recs if x["decision"]["phase"] == "combat"]
        c0 = combat[0]
        self.assertIn("combat_win_prob", c0["net"])
        end = [c for c in c0["candidates"] if c["action_kind"] == 1][0]
        self.assertEqual(end["facts"]["hp_loss_if_end_turn_after"], 6.0)      # モックの敵は 6 ダメージ、ブロック 0
        strike = [c for c in c0["candidates"] if c.get("a_key") == "CARD.STRIKE_IRONCLAD"][0]
        self.assertEqual(strike["facts"]["damage"]["total"], 6)
        self.assertIn("group_prob", strike)                                     # 打撃が2枚ある
        self.assertEqual(c0["state"]["enemies"][0]["attack"]["total"], 6)

    def test_deterministic(self):
        outs = []
        for k in range(2):
            out = self.tmp / f"det{k}.jsonl"
            r = run_hint(["autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(out), "--quiet"])
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append([x for x in read_jsonl(out) if "meta" not in x])
        self.assertEqual(outs[0], outs[1])

    def test_screens_filter(self):
        out = self.tmp / "cr.jsonl"
        r = run_hint(["autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(out), "--quiet",
                      "--screens", "CARD_REWARD"])
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = [x for x in read_jsonl(out) if "net" in x]
        self.assertEqual({x["decision"]["kind"] for x in recs}, {3})

    def test_unlisted_ckpt_refused_by_cli(self):
        r = run_hint(["autoplay", "--mock", "--ckpt", str(self.ckpt), "--out", str(self.tmp / "x.jsonl")])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SHA256SUMS", r.stderr + r.stdout)

    def test_seeds_and_calibrate(self):
        seeds = self.tmp / "seeds.json"
        seeds.write_text(json.dumps({"seeds": ["A1", "A2", "A3"]}), encoding="utf-8")
        d = self.tmp / "many"
        r = run_hint(["autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(d), "--quiet",
                      "--seeds-file", str(seeds), "--runs", "3"])
        self.assertEqual(r.returncode, 0, r.stderr)
        logs = sorted(str(p) for p in d.glob("*.jsonl"))
        self.assertEqual(len(logs), 3)
        cal = self.tmp / "calib.json"
        r = subprocess.run([sys.executable, "-I", str(HINT / "calibrate.py"), *logs, "--out", str(cal)], capture_output=True, text=True,
                           encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        c = json.loads(cal.read_text(encoding="utf-8"))
        self.assertEqual(c["runs"], 3)
        self.assertEqual(c["tables"]["all"]["y"][-1], 1.0)      # モックでは全部勝つ
        out = self.tmp / "with_calib.jsonl"
        r = run_hint(["autoplay", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(out), "--quiet", "--calib", str(cal)])
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = [x for x in read_jsonl(out) if "net" in x]
        self.assertTrue(all(x["net"]["win_prob_method"] == "calibrated" for x in recs))

    def test_interactive(self):
        out = self.tmp / "inter.jsonl"
        # 1つ目（イベント）は Enter、2つ目（戦闘）は index 2 を選ぶ、3つ目で j を見てから q
        r = run_hint(["interactive", "--mock", "--allow-unlisted", "--ckpt", str(self.ckpt), "--out", str(out)], stdin="\n2\nj\nq\n")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("画面: イベント", r.stdout)
        self.assertIn("画面: 戦闘", r.stdout)
        recs = [x for x in read_jsonl(out) if "net" in x]
        self.assertEqual(recs[1]["chosen"], 2)


class TestNamesAndWinprob(unittest.TestCase):
    def test_localized_names(self):
        from names import Namer
        from sts2rl.catalog import Catalog
        vocab = Catalog().vocab
        d = Path(tempfile.mkdtemp()) / "jpn"
        d.mkdir()
        (d / "cards.json").write_text(json.dumps({"STRIKE_IRONCLAD.title": "[gold]ストライク[/gold]"}, ensure_ascii=False), encoding="utf-8")
        (d / "monsters.json").write_text(json.dumps({"NIBBIT.name": "ニビット"}, ensure_ascii=False), encoding="utf-8")
        n = Namer(vocab, d.parent, ["jpn", "eng"])
        self.assertEqual(n.name(vocab.index("CARD.STRIKE_IRONCLAD")), "ストライク")
        self.assertEqual(n.name(vocab.index("MONSTER.NIBBIT")), "ニビット")
        self.assertEqual(n.name(vocab.index("CARD.BASH")), "CARD.BASH")           # 翻訳がなければキー
        self.assertEqual(n.name(vocab.index("MAPNODE:Elite")), "エリート")

    def test_winprob_linear(self):
        from winprob import WinProb
        cfg = {"floor_reward": 0.02, "act1_boss_reward": 0.5, "act2_boss_reward": 0.5, "victory_reward": 1.0, "milestone_from": 0,
               "milestone_ramp": 500, "run_hp_shaping": 0.0}
        w = WinProb(cfg, version=1000)
        self.assertEqual(w.ms_scale, 0.0)
        self.assertAlmostEqual(w.r_win, 1.0 + 0.02 * 49 + 1.0)
        self.assertAlmostEqual(w.estimate(w.r_win / 2, 0, 50, 80, 80)["p"], 0.5)

    def test_winprob_refuses_with_milestones(self):
        from winprob import WinProb
        w = WinProb({"floor_reward": 0.02, "victory_reward": 1.0}, version=1000)       # milestone_from なし → 里程碑あり
        e = w.estimate(1.0, 0, 50, 80, 80)
        self.assertIsNone(e["p"])
        self.assertIn("較正", e["reason"])

    def test_pav_monotone(self):
        from winprob import pav
        y = pav([0.1, 0.5, 0.3, 0.4, 0.9], [1, 1, 1, 1, 1])
        self.assertEqual(y, sorted(y))
        self.assertAlmostEqual(sum(y), 0.1 + 0.5 + 0.3 + 0.4 + 0.9)


if __name__ == "__main__":
    unittest.main()
