"""採集進程の推論を速くした版（fastinfer）が、本体の推論と同じ答えを出すかを確かめ、1 回の前向き計算の時間も測る。

  python -I tools/finetune/check_equiv.py --weights ~/AlphaDefect-Public/weights/files_v6/ironclad_ema_v24782.pt \
      [--ad-repo ~/AlphaDefect-Public] [--n-obs 2048] [--batch 16,32] [--modes "bf16;bf16,fast;bf16,fast,packed"] [--out out.json]

- 観測はヘッドレスエンジンでランダムに打って集める（キャラは重みの config の character。整局と戦闘カリキュラムを半々）。
  エンジンが無い環境では --mock で tools/hint/mock_engine.py のフレームを使う。
- 組批は採集進程と同じ（collate_ragged(Ragged.from_obs(obs, modes))）。基準は本体の採集と同じ設定（fp32、TF32 あり、
  打包注意力なし、コンパイルなし）。
- 比べるもの：候補の確率の最大差、top1（いちばん確率の高い候補）の一致率、KL(基準‖高速版)、
  基準の分布から引いた行動の対数確率の差（PPO の比率に入るずれ）、価値の差。
- 重みは SHA-256 を照合し、weights_only=True で読む（tools/hint/weights.py）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "hint"))

import adrepo  # noqa: E402

DEFAULT_MODES = "bf16;fast;bf16,fast;bf16,fast,packed;bf16,fast,packed,compile"


def collect_obs(n: int, character: str, seed: int = 0, mock: bool = False):
    """ランダムに打って観測を n 個集める。返り値は [(Obs, mode)]（mode は採集進程と同じ MODE_RUN / MODE_COMBAT）。"""
    import numpy as np
    from sts2env import VecEnv
    from sts2rl.actor import MODE_COMBAT, MODE_RUN
    rng = np.random.default_rng(seed)
    out = []
    if mock:
        import mock_engine
        from sts2env.protocol import parse
        g = mock_engine.Game({"character": character, "seed": str(seed)})
        while len(out) < n:
            o = parse(g.frame())
            if o.done or o.num_actions == 0:
                g = mock_engine.Game({"character": character, "seed": str(len(out) + seed)})
                continue
            out.append((o, MODE_COMBAT if o.in_combat else MODE_RUN))
            g.step(int(rng.integers(o.num_actions)))
        return out
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character(character), seed=seed)
    k = 8
    with VecEnv(k) as env:
        mode = [MODE_RUN if i % 2 == 0 else MODE_COMBAT for i in range(k)]
        cnt = [0]

        def reset(i):
            cnt[0] += 1
            if mode[i] == MODE_RUN:
                return env.envs[i].reset_run(character, 10, f"EQV{seed}_{cnt[0]}")
            return env.envs[i].reset_combat(cur.sample(0.8))
        obs = [reset(i) for i in range(k)]
        while len(out) < n:
            acts = []
            for i, o in enumerate(obs):
                if o.done or o.stuck or o.num_actions == 0:
                    acts.append(None)
                else:
                    out.append((o, mode[i])); acts.append(int(rng.integers(o.num_actions)))
            nxt = env.step(acts)
            obs = [reset(i) if a is None else x for i, (a, x) in enumerate(zip(acts, nxt))]
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad-repo", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--allow-unlisted", action="store_true", help="チェックサム一覧にない重み（テスト用）も読む")
    ap.add_argument("--n-obs", type=int, default=2048)
    ap.add_argument("--batch", default="16,32")
    ap.add_argument("--modes", default=DEFAULT_MODES, help="比べる高速版。; で区切り、各々は fastinfer の指定（カンマ区切り）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeat", type=int, default=3, help="時間を測るときに全バッチを回す回数（1 回目は捨てる）")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = adrepo.setup(a.ad_repo)
    import numpy as np
    import torch
    import fastinfer
    import weights as W
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import PolicyValueNet, use_tf32

    use_tf32()      # 採集進程と同じ（actor.py の _actor_loop の最初）
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    L = W.load(a.weights, repo, allow_unlisted=a.allow_unlisted, device=str(dev))
    net = L.net.eval()
    character = (L.blob.get("config") or {}).get("character") or L.blob.get("character") or "IRONCLAD"
    t0 = time.time()
    data = collect_obs(a.n_obs, character, a.seed, a.mock)
    print(f"観測 {len(data)} 個（{character}、整局 {sum(m == 1 for _, m in data)} / 戦闘 {sum(m == 0 for _, m in data)}、{time.time() - t0:.0f} 秒）", flush=True)
    orig = PolicyValueNet.forward_phi
    report = {"weights": L.path, "sha256": L.check["sha256"], "device": str(dev), "torch": torch.__version__, "n_obs": len(data),
              "character": character, "results": {}}
    mode_list = [""] + [m for m in a.modes.split(";") if m.strip()]

    for bs_s in a.batch.split(","):
        # "r16" は採集進程に近い形：1〜16 行のばらばらな大きさの批
        if bs_s.startswith("r"):
            bs = int(bs_s[1:]); rr = np.random.default_rng(7); batches, i = [], 0
            while i < len(data):
                k = int(rr.integers(1, bs + 1)); batches.append(data[i:i + k]); i += k
        else:
            bs = int(bs_s)
            batches = [data[i:i + bs] for i in range(0, len(data) - bs + 1, bs)]
        cb = [collate_ragged(Ragged.from_obs([o for o, _ in grp], np.array([m for _, m in grp]))) for grp in batches]
        nact = [np.array([o.num_actions for o, _ in grp]) for grp in batches]
        base = None
        for ms in mode_list:
            modes = fastinfer.parse_modes(ms)
            f = fastinfer.make_forward(orig, modes)
            outs, times = [], []
            with torch.no_grad():
                for rep in range(max(1, a.repeat)):
                    for j, b in enumerate(cb):
                        import copy
                        bj = copy.copy(b)      # packed は Batch に偏移を書き足すので、元を汚さない
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        t = time.perf_counter()
                        bd = fastinfer.to_device(bj, modes, dev)
                        lg, v, _z = f(net, bd)
                        lg, v = lg.cpu().numpy(), v.cpu().numpy()
                        times.append(time.perf_counter() - t) if rep > 0 or a.repeat == 1 else None
                        if rep == 0:
                            outs.append((lg, v))
            ms_med = 1000 * statistics.median(times)
            name = ms or "base(fp32)"
            mean_rows = statistics.fmean(len(g) for g in batches)
            r = {"batch": bs_s, "mean_rows": round(mean_rows, 2), "ms_median": round(ms_med, 3), "ms_mean": round(1000 * statistics.fmean(times), 3),
                 "rows_per_s": round(mean_rows / (statistics.fmean(times)))}
            if base is None:
                base = outs
            else:
                dmax, agree, kls, dlogp, dval, n = 0.0, 0, [], [], [], 0
                rng = np.random.default_rng(1)
                for (lb, vb), (lm, vm), na in zip(base, outs, nact):
                    for i in range(len(na)):
                        k = int(na[i])
                        pb = np.exp(lb[i, :k] - np.logaddexp.reduce(lb[i, :k]))
                        pm = np.exp(lm[i, :k] - np.logaddexp.reduce(lm[i, :k]))
                        dmax = max(dmax, float(np.abs(pb - pm).max()))
                        agree += int(np.argmax(pb) == np.argmax(pm)); n += 1
                        kls.append(float(np.sum(pb * (np.log(pb + 1e-30) - np.log(pm + 1e-30)))))
                        act = int(rng.choice(k, p=pb / pb.sum()))
                        dlogp.append(abs(float(np.log(pb[act] + 1e-30) - np.log(pm[act] + 1e-30))))
                        dval.append(abs(float(vb[i] - vm[i])))
                r.update({"rows": n, "prob_max_abs_diff": round(dmax, 5), "top1_agree": round(agree / n, 5),
                          "kl_mean": float(f"{statistics.fmean(kls):.3g}"), "kl_max": float(f"{max(kls):.3g}"),
                          "dlogp_mean": float(f"{statistics.fmean(dlogp):.3g}"), "dlogp_p99": float(f"{np.quantile(dlogp, 0.99):.3g}"),
                          "value_max_abs_diff": round(max(dval), 5), "value_mean_abs_diff": float(f"{statistics.fmean(dval):.3g}")})
            report["results"].setdefault(f"batch{bs_s}", {})[name] = r
            print(f"批 {bs_s:>4s} {name:32s} {json.dumps(r, ensure_ascii=False)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("書き出し:", a.out)


if __name__ == "__main__":
    main()
