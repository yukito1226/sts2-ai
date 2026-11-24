"""ゲーム内 Mod 用のヒントサーバー（tools/hint/server.py）のテスト。ゲーム本体と学習済みの重みは要らない。

Mod の代わりにこのテストがソケットでつなぎ、モックエンジンが作ったフレームを Mod と同じ形式で送る。

  ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest tests.test_server -v
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hint"))
sys.path.insert(0, str(ROOT / "tests"))

import adrepo  # noqa: E402

REPO = adrepo.setup()
import make_mock_ckpt  # noqa: E402
import server  # noqa: E402


def mock_frames() -> list[bytes]:
    """モックエンジンの生のフレーム（戦闘外と戦闘）。"""
    import os
    from engine import Engine, mock_cmd
    cmd, env = mock_cmd()
    env["ALPHADEFECT_REPO"] = str(REPO)
    with Engine(cmd, env) as e:
        a = e.request({"op": "reset", "mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": "T", "labels": True})
        b = e.request({"op": "step", "action": 0})
    return [a, b]


def send(sock, kind: int, body: bytes) -> None:
    sock.sendall(struct.pack("<I", len(body) + 1) + bytes([kind]) + body)


def recv(sock) -> dict:
    (n,) = struct.unpack("<I", server.read_exact(sock, 4))
    return json.loads(server.read_exact(sock, n).decode("utf-8"))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="hint_server_"))
        ckpt = tmp / "ironclad_mock.pt"
        make_mock_ckpt.make(str(ckpt), milestone_from=0)
        a = server.build_parser().parse_args(["--ckpt", str(ckpt), "--allow-unlisted", "--threads", "1", "--port", "0"])
        cls.svc = server.HintService(a)
        ready, bound = threading.Event(), []
        threading.Thread(target=server.serve, args=(cls.svc, "127.0.0.1", 0, None, ready, bound), daemon=True).start()
        ready.wait(10)
        cls.port = bound[0]
        cls.frames = mock_frames()

    def connect(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        send(s, 0, json.dumps({"type": "hello"}).encode())
        return s, recv(s)

    def test_hello(self):
        s, h = self.connect()
        with s:
            self.assertEqual(h["type"], "hello")
            self.assertEqual(h["ckpt"], "ironclad_mock.pt")
            self.assertEqual(h["character"], "CHARACTER.IRONCLAD")      # ファイル名から
            self.assertEqual(h["search"], "none")
            self.assertEqual(h["privileged_inputs"], "zeroed")

    def test_hint_matches_service(self):
        """ソケット越しの返事が、同じフレームを直接 HintService に通した結果と同じ。通し番号も返る。"""
        s, _ = self.connect()
        phases = []
        with s:
            for i, f in enumerate(self.frames, 1):
                send(s, 1, struct.pack("<I", i) + f)
                r = recv(s)
                self.assertEqual(r["type"], "hint")
                self.assertEqual(r["seq"], i)
                want = json.loads(json.dumps(self.svc.hint(f), ensure_ascii=False))
                self.assertEqual(r["hint"], want)
                probs = [c["prob"] for c in r["hint"]["candidates"]]
                self.assertAlmostEqual(sum(probs), 1.0, places=3)
                self.assertNotIn("win_prob", r["hint"]["net"])       # 局全体の勝率は出さない
                phases.append(r["hint"]["decision"]["phase"])
        self.assertEqual(sorted(phases), ["combat", "out_of_combat"])

    def test_bad_frame_keeps_connection(self):
        """壊れたフレームはエラーを返すだけで、接続はそのまま使える。"""
        s, _ = self.connect()
        with s:
            send(s, 1, struct.pack("<I", 7) + b"\x01\x02\x03")
            r = recv(s)
            self.assertEqual(r["type"], "error")
            self.assertEqual(r["seq"], 7)
            send(s, 1, struct.pack("<I", 8) + self.frames[0])
            self.assertEqual(recv(s)["type"], "hint")

    def test_oversized_length_closes(self):
        s, _ = self.connect()
        with s:
            s.sendall(struct.pack("<I", server.MAX_MSG + 1))
            self.assertEqual(s.recv(1), b"")       # サーバーが切る

    def test_loopback_only(self):
        with self.assertRaises(SystemExit):
            server.serve(self.svc, "0.0.0.0", 0)

    def test_dump(self):
        tmp = Path(tempfile.mkdtemp(prefix="hint_dump_"))
        self.svc.dump, self.svc._dump_n = tmp, 0
        try:
            for f in self.frames:
                self.svc.hint(f)
        finally:
            self.svc.dump = None
        data = (tmp / "frames.bin").read_bytes()
        (n0,) = struct.unpack_from("<I", data)
        self.assertEqual(data[4:4 + n0], self.frames[0])
        lines = (tmp / "hints.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(l)["n"] for l in lines], [1, 2])

    def test_character_of(self):
        self.assertEqual(server.character_of("weights/files_v6/ironclad_ema_v24782.pt"), "CHARACTER.IRONCLAD")
        self.assertEqual(server.character_of("fool_ema_v24181.pt"), "CHARACTER.FOOL")
        self.assertIsNone(server.character_of("mock.pt"))


if __name__ == "__main__":
    unittest.main()
