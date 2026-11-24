"""テスト用の小さな重みファイルを作る（乱数で初期化した PolicyValueNet。学習済みではないので、確率や価値の値そのものに意味はない）。

形式は配布されている重みと同じ（format, version, obs_format, vocab_n, config{model,...}, state, content_names）。
モデルのフラグはアイアンクラッドの weights-v6（MANIFEST_v6.json の model_flags）に合わせ、幅と層数だけ小さくする。

  python3 -I tests/make_mock_ckpt.py --ad-repo ~/AlphaDefect-Public --out /tmp/mock.pt [--milestone-from 0]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "hint"))
import adrepo  # noqa: E402

FLAGS = ("content_enchant", "content_upgrade", "obs_v16", "obs_v17", "obs_v18", "obs_v19", "obs_v26", "obs_v27", "future_content",
         "value_hl", "combat_aux", "future_head", "recycle")


def make(out: str, milestone_from: int | None = 0, seed: int = 0) -> str:
    import torch
    from sts2rl.catalog import Catalog
    from sts2rl.model import ModelConfig, PolicyValueNet
    torch.manual_seed(seed)
    cfg = ModelConfig(d=32, layers=1, heads=2, ff_mult=2, **{k: True for k in FLAGS})
    net = PolicyValueNet(cfg).eval()
    blob = {"format": "sts2-defect-rl/ckpt-1", "version": 1000, "obs_format": 28, "vocab_n": Catalog().vocab_size, "total_steps": None,
            "config": {"model": dataclasses.asdict(cfg), "floor_reward": 0.02, "act1_boss_reward": 0.5, "act2_boss_reward": 0.5,
                       "victory_reward": 1.0, "act3_boss_reward": 0.0, "act3_boss2_reward": 0.0, "milestone_from": milestone_from,
                       "milestone_ramp": 500, "run_hp_shaping": 0.3, "hp_concavity": 0.5},
            "state": {k: v.detach().clone() for k, v in net.state_dict().items()},
            "content_names": tuple(net.content_names) if getattr(net, "content_names", None) is not None else None,
            "export": {"name": "mock", "notes": "test only, random init"}}
    torch.save(blob, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad-repo", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--milestone-from", type=int, default=0)
    a = ap.parse_args()
    adrepo.setup(a.ad_repo)
    print(make(a.out, a.milestone_from))
