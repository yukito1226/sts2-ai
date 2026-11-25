"""採集の高速化（tools/finetune/）のテスト。ゲーム本体・学習済みの重み・GPU は要らない（CPU 版の torch でよい）。

  ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest tests.test_finetune -v

- 指定の読み取り、既定（指定なし）では本体と同じ動作であること
- bf16（CPU の autocast）でも候補の順位がほぼ変わらないこと（乱数初期化の小さなネットとモックの観測で）
- train_fast の差し替えで、spawn で起動した子の中でパッチが効くこと
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "tools" / "finetune"
HINT = ROOT / "tools" / "hint"
for p in (str(FT), str(HINT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import adrepo  # noqa: E402

REPO = adrepo.setup()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fastinfer  # noqa: E402
import make_mock_ckpt  # noqa: E402


def mock_batches(n_batches: int = 6, bs: int = 8):
    """モックエンジンのフレームから、採集進程と同じ組批を作る。"""
    import check_equiv
    from sts2rl.features import Ragged, collate_ragged
    data = check_equiv.collect_obs(n_batches * bs, "IRONCLAD", seed=0, mock=True)
    out = []
    for i in range(0, len(data), bs):
        grp = data[i:i + bs]
        out.append((collate_ragged(Ragged.from_obs([o for o, _ in grp], np.array([m for _, m in grp]))), [o.num_actions for o, _ in grp]))
    return out


class TestModes(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(fastinfer.parse_modes(""), frozenset())
        self.assertEqual(fastinfer.parse_modes(None), frozenset())
        self.assertEqual(fastinfer.parse_modes(" BF16, fast "), frozenset({"bf16", "fast"}))
        with self.assertRaises(ValueError):
            fastinfer.parse_modes("bf16,fp8")


class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import weights as W
        cls.tmp = tempfile.TemporaryDirectory()
        path = make_mock_ckpt.make(os.path.join(cls.tmp.name, "mock.pt"))
        cls.net = W.load(path, REPO, allow_unlisted=True).net.eval()
        cls.batches = mock_batches()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, modes: str):
        from sts2rl.model import PolicyValueNet
        f = fastinfer.make_forward(PolicyValueNet.forward_phi, fastinfer.parse_modes(modes))
        outs = []
        with torch.no_grad():
            for b, _ in self.batches:
                outs.append(f(self.net, fastinfer.to_device(b, fastinfer.parse_modes(modes), "cpu")))
        return outs

    def test_default_is_identical(self):
        """指定なしの make_forward は本体の forward_phi と同じ値（float32 に揃えるだけ）。"""
        with torch.no_grad():
            ref = [self.net.forward_phi(b.to("cpu")) for b, _ in self.batches]
        got = self.run_mode("")
        for r, g in zip(ref, got):
            for x, y in zip(r, g):
                self.assertEqual(y.dtype, torch.float32)
                self.assertTrue(torch.equal(x.float(), y))

    def test_bf16_close(self):
        """bf16 でも候補の top1 はほぼ一致し、出力は float32 に戻っている。"""
        ref, got = self.run_mode(""), self.run_mode("bf16,packed,fast")     # CPU では packed・fast は効かず bf16 だけ
        # 乱数初期化のネットは候補の得点がほぼ同点になりやすいので、fp32 で 1 位と 2 位の差が 0.05 以上ある行だけ top1 を比べる
        agree = n = 0
        for (lr, vr, _), (lg, vg, _), (_, na) in zip(ref, got, self.batches):
            self.assertEqual(lg.dtype, torch.float32)
            for i, k in enumerate(na):
                pr, pg = lr[i, :k].softmax(-1), lg[i, :k].softmax(-1)
                self.assertLess(float((pr - pg).abs().max()), 0.05)
                top2 = lr[i, :k].topk(min(2, k)).values
                if k == 1 or float(top2[0] - top2[-1]) >= 0.05:
                    agree += int(torch.argmax(pr) == torch.argmax(pg)); n += 1
            self.assertLess(float((vr - vg).abs().max()), 0.1)
        self.assertGreater(n, 0)
        self.assertEqual(agree, n, f"top1 一致 {agree}/{n}")

    def test_grad_path_untouched(self):
        """勾配ありの呼び出し（学習側）は本体のまま。"""
        from sts2rl.model import PolicyValueNet
        f = fastinfer.make_forward(PolicyValueNet.forward_phi, fastinfer.parse_modes("bf16"))
        b = self.batches[0][0].to("cpu")
        lg, _, _ = f(self.net, b)
        self.assertTrue(lg.requires_grad)
        with torch.no_grad():
            ref = self.net.forward_phi(b)[0]
        # 勾配ありと無しでは内容先験の表の計算（キャッシュの有無）だけが違うので、丸め誤差の範囲で一致する
        self.assertTrue(torch.allclose(lg.detach(), ref, atol=1e-5, rtol=1e-5))


def _spawn_probe(modes: str) -> list:
    """train_fast.install のあと、本体と同じく spawn で sts2rl.train.run_actor を target にして子を起動する。"""
    import queue as _q
    import train_fast
    os.environ["STS2FT_DRYRUN"] = "1"
    train_fast.install(modes)
    from sts2rl import train
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=train.run_actor, args=(r, None, "", q, None)) for r in range(2)]
    for p in ps:
        p.start()
    got = []
    try:
        for _ in ps:
            got.append(q.get(timeout=120))
    except _q.Empty:
        pass
    for p in ps:
        p.join(30)
    return got


class TestSpawn(unittest.TestCase):
    def run_probe(self, modes: str) -> list:
        code = textwrap.dedent(f"""
            import sys, json
            sys.path[:0] = [{str(FT)!r}, {str(HINT)!r}, {str(ROOT / 'tests')!r}]
            import test_finetune
            print("RESULT", json.dumps(test_finetune._spawn_probe({modes!r})))
        """)
        env = dict(os.environ, ALPHADEFECT_REPO=str(REPO))
        env.pop("STS2FT_ACTOR", None)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
        import json
        return json.loads(line[len("RESULT "):]), p.stderr

    def test_child_gets_patch(self):
        got, err = self.run_probe("bf16,fast")
        self.assertEqual(len(got), 2)
        for tag, rank, modes, fname, pid in got:
            self.assertEqual(tag, "fastinfer")
            self.assertEqual(modes, ["bf16", "fast"])
            self.assertEqual(fname, "forward_phi_fast")
            self.assertNotEqual(pid, os.getpid())
        self.assertIn("[fastinfer] 採集進程 rank=0 パッチ: bf16,fast", err)

    def test_child_default_untouched(self):
        got, _ = self.run_probe("")
        for _tag, _rank, modes, fname, _pid in got:
            self.assertEqual(modes, [])
            self.assertEqual(fname, "forward_phi")


if __name__ == "__main__":
    unittest.main()
