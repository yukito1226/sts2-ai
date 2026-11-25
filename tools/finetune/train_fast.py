"""AlphaDefect の追加学習（PPO の再開）を、採集進程の推論を速くした状態で起動する。AlphaDefect 本体のファイルは書き換えない。

  python -I tools/finetune/train_fast.py [--ad-repo ~/AlphaDefect-Public] [--actor-infer bf16,fast,packed] --config cfg.json [本体の引数 ...]

- --actor-infer は環境変数 STS2FT_ACTOR と同じ（fastinfer.py の説明を参照）。どちらも無ければ本体と同じ動作。
- --ad-repo か環境変数 ALPHADEFECT_REPO で AlphaDefect-Public のチェックアウトを指定する。
- 残りの引数は、そのまま sts2rl.train.main（python/train_defect.py と同じもの）に渡す。

仕組み：sts2rl.train の中の名前 run_actor を fastinfer.run_actor_patched に差し替えてから main を呼ぶ。
main は ctx.Process(target=run_actor, ...) で採集進程を spawn するので、子はこの関数から始まり、
子の中でパッチを当ててから本体の run_actor を呼ぶ（親でのパッチは spawn の子に届かないため）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def setup_paths(ad_repo: str | None) -> Path:
    for p in (str(HERE), str(ROOT / "tools" / "hint")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import adrepo
    return adrepo.setup(ad_repo)


def install(modes: str | None) -> None:
    """sts2rl.train の採集進程の target を差し替える。modes が None なら環境変数をそのまま使う。"""
    import fastinfer
    if modes is not None:
        os.environ[fastinfer.ENV] = modes      # spawn の子は環境変数を受け継ぐ
    fastinfer.parse_modes(os.environ.get(fastinfer.ENV))   # 書き間違いはここで止める
    from sts2rl import train
    train.run_actor = fastinfer.run_actor_patched


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ad-repo", default=None)
    ap.add_argument("--actor-infer", default=None)
    a, rest = ap.parse_known_args(argv)
    repo = setup_paths(a.ad_repo)
    install(a.actor_infer)
    import fastinfer
    print(f"[train_fast] AlphaDefect-Public: {repo}、採集の推論: {os.environ.get(fastinfer.ENV) or '（本体のまま）'}", flush=True)
    from sts2rl.train import main as train_main
    train_main(rest)


if __name__ == "__main__":
    main()
