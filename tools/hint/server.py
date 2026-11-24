"""ゲーム内 Mod（tools/hintmod, Sts2Hint）のためのヒントサーバー。

Mod はゲームの今の状態を AlphaDefect と同じエンコーダで観測フレームにして送ってくる。このサーバーはそれをネットに通し、
Step 1 と同じ形のヒント（record.py）を返す。ゲームの状態は変えない。探索・先読み・解説はまだしない（探索なし、CPU）。

  python -I tools/hint/server.py --ckpt weights/files_v6/ironclad_ema_v24782.pt [--port 8765]

通信は 127.0.0.1 だけで待ち受ける（外からはつなげない）。
  Mod → サーバー: [u32 長さ][u8 種類][本体]  種類 0 = JSON（あいさつなど）、1 = フレーム（[u32 通し番号] + エンジンと同じバイト列）
  サーバー → Mod: [u32 長さ][UTF-8 JSON]
    あいさつへの返事 {"type":"hello", "ckpt", "obs_format", "character", ...}
    フレームへの返事 {"type":"hint", "seq", "ms", "hint": {...record.py の形...}}、失敗したら {"type":"error", "seq", "message"}
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrepo  # noqa: E402

MAX_MSG = 16 << 20       # 1フレームは数十KB。壊れた長さで大きな確保をしないための上限
KIND_JSON, KIND_FRAME = 0, 1


def character_of(ckpt: str) -> str | None:
    """重みのファイル名（ironclad_ema_v24782.pt など）からキャラの語彙キー（CHARACTER.IRONCLAD）を推す。"""
    head = Path(ckpt).name.split("_")[0].upper()
    return f"CHARACTER.{head}" if head in {"IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL"} else None


class HintService:
    """フレーム（bytes）→ ヒント（dict）。ネットワークとは切り離してあるので、テストから直接呼べる。"""

    def __init__(self, a):
        self.repo = adrepo.setup(a.ad_repo)
        import torch
        torch.set_num_threads(max(1, a.threads))
        import weights
        from names import Namer
        from record import Builder
        from sts2env.gamever import GAME_VERSION, localization_dir
        from sts2rl.catalog import Catalog

        self.a = a
        self.L = weights.load(a.ckpt, self.repo, allow_unlisted=a.allow_unlisted, device=a.device)
        loc_root = Path(a.loc_dir).expanduser() if a.loc_dir else localization_dir("eng").parent
        self.namer = Namer(Catalog().vocab, loc_root, [x for x in a.lang.split(",") if x])
        self.builder = Builder(self.namer)
        self.game_version = GAME_VERSION
        self.character = a.character or character_of(a.ckpt)
        self.lock = threading.Lock()      # ネットは1つなので、同時に来ても順番に通す
        self.dump = Path(a.dump) if getattr(a, "dump", None) else None
        if self.dump:
            self.dump.mkdir(parents=True, exist_ok=True)
            self._dump_n = 0

    def hello(self) -> dict:
        return {"type": "hello", "tool": "alphadefect-hint-server/1", "ckpt": Path(self.L.path).name,
                "sha256_status": self.L.check["status"], "version": self.L.version, "obs_format": self.L.obs_format,
                "character": self.character, "game_version": self.game_version, "search": "none",
                "privileged_inputs": "zeroed"}

    def hint(self, frame: bytes) -> dict:
        from infer import evaluate
        from sts2env.protocol import parse
        from sts2rl.greedy import equivalence_classes
        o = parse(frame)
        with self.lock:
            out = evaluate(self.L.net, o, self.a.device)
        classes = equivalence_classes(o) if o.num_actions else None
        # 局全体の勝率は出さない（価値は較正されていない。ユーザーの決定）。戦闘の勝率（補助ヘッド）は record.py が出す
        rec = self.builder.build(o, out, None, merge_classes=classes)
        if self.dump:
            self._save(frame, rec)
        return rec

    def _save(self, frame: bytes, rec: dict) -> None:
        """受け取ったフレームとヒントを残す（実機の確認用）。frames.bin は「u32 長さ + フレーム」の連続（live_fidelity.py と同じ形）。"""
        with self.lock:
            self._dump_n += 1
            with open(self.dump / "frames.bin", "ab") as f:
                f.write(struct.pack("<I", len(frame)) + frame)
            with open(self.dump / "hints.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"n": self._dump_n, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, ensure_ascii=False) + "\n")


def read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return bytes(buf)


def read_msg(sock: socket.socket) -> tuple[int, bytes]:
    (n,) = struct.unpack("<I", read_exact(sock, 4))
    if n < 1 or n > MAX_MSG:
        raise ValueError(f"不正な長さ {n}")
    body = read_exact(sock, n)
    return body[0], body[1:]


def send_json(sock: socket.socket, obj: dict) -> None:
    b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("<I", len(b)) + b)


def serve_client(svc: HintService, sock: socket.socket, log) -> None:
    with sock:
        while True:
            try:
                kind, body = read_msg(sock)
            except (ConnectionError, OSError, ValueError, struct.error):
                return
            if kind == KIND_JSON:
                req = json.loads(body.decode("utf-8"))
                if req.get("type") == "hello":
                    send_json(sock, svc.hello())
                else:
                    send_json(sock, {"type": "error", "message": f"知らない要求 {req.get('type')}"})
            elif kind == KIND_FRAME:
                (seq,) = struct.unpack_from("<I", body)
                t0 = time.perf_counter()
                try:
                    rec = svc.hint(body[4:])
                    ms = round((time.perf_counter() - t0) * 1000, 1)
                    send_json(sock, {"type": "hint", "seq": seq, "ms": ms, "hint": rec})
                    if log:
                        top = rec["candidates"][0]["text"] if rec["candidates"] else "-"
                        print(f"[{seq}] {rec['decision']['screen']} 候補 {rec['decision']['num_candidates']} 1位 {top} ({ms} ms)", file=log, flush=True)
                except Exception as e:      # 1つのフレームで失敗しても接続は切らない
                    if log:
                        traceback.print_exc(file=log)
                    send_json(sock, {"type": "error", "seq": seq, "message": str(e)})
            else:
                send_json(sock, {"type": "error", "message": f"知らない種類 {kind}"})


def serve(svc: HintService, host: str, port: int, log=sys.stderr, ready: threading.Event | None = None,
          bound: list | None = None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("このサーバーは自分の PC の中（127.0.0.1）だけで使う前提です")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(4)
    if bound is not None:
        bound.append(srv.getsockname()[1])
    if log:
        print(f"ヒントサーバー 127.0.0.1:{srv.getsockname()[1]}  重み {svc.hello()['ckpt']}（{svc.character}, 観測格式 {svc.L.obs_format}）",
              file=log, flush=True)
    if ready:
        ready.set()
    while True:
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if log:
            print(f"Mod が接続しました {addr}", file=log, flush=True)
        threading.Thread(target=serve_client, args=(svc, c, log), daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ad-repo", default=None, help="AlphaDefect-Public のチェックアウト（環境変数 ALPHADEFECT_REPO でも可）")
    ap.add_argument("--ckpt", required=True, help="重みファイル（.pt）。SHA-256 を weights/SHA256SUMS_*.txt と照合してから読む")
    ap.add_argument("--character", default=None, help="重みのキャラ（CHARACTER.IRONCLAD など）。既定はファイル名から推す")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lang", default="jpn,eng")
    ap.add_argument("--loc-dir", default=None, help="localization フォルダ（既定 ~/.local/share/sls2/game/<版>/res/localization）")
    ap.add_argument("--allow-unlisted", action="store_true", help="チェックサム一覧にない重みも読む（テスト用）")
    ap.add_argument("--quiet", action="store_true", help="判断ごとのログを出さない")
    ap.add_argument("--dump", default=None, help="受け取ったフレーム（frames.bin）とヒント（hints.jsonl）をこのフォルダに残す")
    return ap


def main(argv: list[str] | None = None) -> None:
    from hint import utf8_stdio
    utf8_stdio()
    a = build_parser().parse_args(argv)
    svc = HintService(a)
    serve(svc, "127.0.0.1", a.port, None if a.quiet else sys.stderr)


if __name__ == "__main__":
    main()
