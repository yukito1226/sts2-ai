"""ヘッドレスエンジンとのやり取り（python/sts2env/env.py の _Process と同じ通信手順）。

要求は JSON を1行ずつ標準入力へ、返事は「4バイトの長さ（リトルエンディアン）＋本体」。本体の先頭4バイトが -1 ならエラー文。
env.py との違いは、起動するコマンドを差し替えられることだけ（モックエンジン tools/hint/mock_engine.py でテストするため）。
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


class EngineError(RuntimeError):
    pass


def default_cmd(repo: Path) -> tuple[list[str], dict]:
    """実機のエンジン（Sts2Headless.dll serve）を起動するコマンドと環境変数。env.py と同じ探し方をする。"""
    dll = repo / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
    if not dll.is_file():
        raise SystemExit(f"エンジンがビルドされていません: {dll}\n"
                         "  (cd engine/Sts2Headless && dotnet build -c Release -p:Sts2GameVersion=v0.113.1) を先に実行してください。")
    home_dotnet = Path.home() / ".dotnet" / "dotnet"
    dotnet = str(home_dotnet) if home_dotnet.exists() else (shutil.which("dotnet") or "dotnet")
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_TieredPGO="1")
    if home_dotnet.exists():
        env["DOTNET_ROOT"] = str(home_dotnet.parent)
    return [dotnet, str(dll), "serve"], env


def mock_cmd() -> tuple[list[str], dict]:
    return [sys.executable, "-I", str(Path(__file__).with_name("mock_engine.py"))], dict(os.environ)


class Engine:
    def __init__(self, cmd: list[str], env: dict | None = None, log: Any = subprocess.DEVNULL):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)

    def send(self, req: dict) -> None:
        self.p.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def recv(self) -> bytes:
        (n,) = struct.unpack("<i", self._read(4))
        body = self._read(n)
        if n >= 4 and struct.unpack_from("<i", body)[0] == -1:
            raise EngineError(body[4:].decode("utf-8", "replace"))
        return body

    def request(self, req: dict) -> bytes:
        self.send(req)
        return self.recv()

    def _read(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            c = self.p.stdout.read(n - got)
            if not c:
                raise EngineError(f"エンジンのプロセスが終了しました（終了コード {self.p.poll()}）")
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

    def close(self) -> None:
        try:
            self.send({"op": "close"})
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
