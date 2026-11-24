"""AlphaDefect-Public のチェックアウトを探し、その python/ を import できるようにする。

AlphaDefect 本体のコードは書き換えずに読み込むだけにする。場所は次の順で決める。
  1. 引数 --ad-repo
  2. 環境変数 ALPHADEFECT_REPO
  3. このファイルが AlphaDefect-Public/tools/hint/ に置かれている場合は、その2つ上
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _looks_like_repo(p: Path) -> bool:
    return (p / "python" / "sts2rl" / "model.py").is_file() and (p / "engine" / "catalog" / "vocab.txt").is_file()


def find_repo(explicit: str | os.PathLike | None = None) -> Path:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    env = os.environ.get("ALPHADEFECT_REPO")
    if env:
        cands.append(Path(env).expanduser())
    cands.append(_HERE.parents[1])
    for c in cands:
        if _looks_like_repo(c):
            return c.resolve()
    tried = ", ".join(str(c) for c in cands)
    raise SystemExit(
        "AlphaDefect-Public のチェックアウトが見つかりません。--ad-repo か環境変数 ALPHADEFECT_REPO で場所を指定してください"
        f"（探した場所: {tried}）"
    )


def setup(explicit: str | os.PathLike | None = None) -> Path:
    """リポジトリを探して sys.path に python/ を足し、その場所を返す。何度呼んでもよい。"""
    repo = find_repo(explicit)
    py = str(repo / "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    return repo
