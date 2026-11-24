"""重みファイルの確認と読み込み。

- 読み込む前に SHA-256 を計算し、AlphaDefect-Public/weights/SHA256SUMS_*.txt に載っている値と一致するか確かめる。
  一致しなければ読み込まない（--allow-unlisted を付けたときだけ、一覧にないファイルを警告つきで読む。モックテスト用）。
- torch.load は必ず weights_only=True で呼ぶ（任意のコードが実行される pickle を受け付けない）。
- ネットの組み立ては tools/ability_test.py の load_net と同じ手順（load_state で内容先験の列を名前で合わせ、
  重みより後に増えた語彙の埋め込み行をゼロにする）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VOCAB_N_LEGACY = 1873      # ability_test.py と同じ：vocab_n を記録する前のチェックポイントは 1,873 語


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def known_sums(repo: Path) -> dict[str, list[tuple[str, str]]]:
    """weights/SHA256SUMS_*.txt を読み、{ファイル名: [(sha256, 一覧ファイル名), ...]} を返す。"""
    out: dict[str, list[tuple[str, str]]] = {}
    for f in sorted((repo / "weights").glob("SHA256SUMS_*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                out.setdefault(parts[-1].lstrip("*"), []).append((parts[0].lower(), f.name))
    return out


def verify(path: str | Path, repo: Path, allow_unlisted: bool = False) -> dict:
    """SHA-256 を確かめる。返り値は {"sha256", "status", "list"}。合わなければ SystemExit。"""
    p = Path(path)
    digest = sha256_of(p)
    entries = known_sums(repo).get(p.name, [])
    for sha, lst in entries:
        if sha == digest:
            return {"sha256": digest, "status": "ok", "list": lst}
    if entries:
        raise SystemExit(f"SHA-256 が一致しません: {p.name}\n  計算値 {digest}\n  一覧の値 {[e[0] for e in entries]}\n"
                         "ダウンロードし直してください。")
    if not allow_unlisted:
        raise SystemExit(f"{p.name} は weights/SHA256SUMS_*.txt に載っていません（SHA-256 {digest}）。"
                         "配布された重みだけを使ってください（テスト用のファイルなら --allow-unlisted）。")
    print(f"警告: {p.name} はチェックサム一覧にないファイルです（--allow-unlisted で読み込み、SHA-256 {digest}）", flush=True)
    return {"sha256": digest, "status": "unlisted", "list": None}


@dataclass
class Loaded:
    net: Any                 # sts2rl.model.PolicyValueNet（eval モード、CPU）
    blob: dict               # チェックポイントの中身（state 以外も含む）
    path: str
    check: dict              # verify の結果

    @property
    def version(self) -> int:
        return int(self.blob.get("version") or 0)

    @property
    def config(self) -> dict:
        return dict(self.blob.get("config") or {})

    @property
    def obs_format(self) -> int | None:
        v = self.blob.get("obs_format")
        return int(v) if v is not None else None


def load(path: str | Path, repo: Path, allow_unlisted: bool = False, device: str = "cpu") -> Loaded:
    import torch
    from sts2rl.catalog import Catalog
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, zero_new_vocab

    check = verify(path, repo, allow_unlisted)
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    old_n, vocab_n = blob.get("vocab_n") or VOCAB_N_LEGACY, Catalog().vocab_size
    if old_n < vocab_n:
        with torch.no_grad():
            zero_new_vocab(net, old_n, vocab_n)
    net = net.to(device).eval()
    return Loaded(net, blob, str(path), check)
