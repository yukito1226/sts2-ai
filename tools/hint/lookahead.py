"""候補ごとの「その手を選んだあとの見込み」（短い先読み）。

やり方は tools/future/boss_pimc.py のクローン世界と同じ:
  - 実際の対局とは別に、先読み用のエンジンをもう1つ起動する（実際の対局の乱数や状態には一切触らない）。
  - 先読み用のエンジンを、同じ reset と、これまでに選んだ手の再生で、今の局面に合わせてスロット 0 に保存する
    （2回目以降は、前回からの差分だけ再生する）。
  - 候補 c と世界 w ごとに、restore(スロット 0, reseed=世界の種, first=c)。reseed で「見えない未来」（山札の順番、
    この先の乱数、戦闘後の報酬）を引き直すので、本当の未来は覗かない（AlphaDefect の決定記録 214 の「公開情報の探索」と同じ扱い）。
    どの候補も同じ世界の種を使う（共通乱数）ので、候補どうしの差が比べやすい。
  - そのあとはネットの手（貪欲）で進める:
      戦闘中の判断  … ターンが変わる（次の自分のターンの最初の判断）か、戦闘が終わるまで
      戦闘外の判断  … 1手だけ（次の判断の局面）
    行き着いた局面の価値 ＋ 途中でもらった報酬（階層・幕・クリア）を「その手のあとの価値」とする。
    死んだら 0（整局モードでは、死んだあとは報酬がない）。
これは探索ではなく、1手ごとの見込みの比較。行動の決定（greedy）は変えない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_ROLL_STEPS = 60


@dataclass
class LookaheadConfig:
    worlds: int = 4                # 候補ごとの世界の数（reseed の種の数）
    top: int = 0                   # 確率の高い順にいくつの候補を調べるか（0 は全部）
    seed: int = 12345              # 世界の種の元


@dataclass
class _Root:
    key: tuple | None = None       # (シード, reset の設定)
    n_hist: int = 0                # スロット 0 に入っている局面までの手数


class Lookahead:
    def __init__(self, engine, net, wp, base_req: dict, cfg: LookaheadConfig, device: str = "cpu"):
        self.eng = engine            # 先読み専用のエンジン（engine.Engine）
        self.net = net
        self.wp = wp                 # winprob.WinProb（報酬の設定と ms_scale を使う）
        self.base = dict(base_req)   # 実際の対局の reset 要求（op, mode, character, ascension, seed, ...）
        self.cfg = cfg
        self.device = device
        self.root = _Root()

    # ------------------------------------------------------------ 先読みエンジンを今の局面に合わせる

    def _sync(self, hist: list[int]) -> None:
        from sts2env.protocol import parse
        key = (self.base.get("seed"), tuple(sorted((k, str(v)) for k, v in self.base.items())))
        if self.root.key != key or self.root.n_hist > len(hist):
            req = {**self.base, "labels": True, "replay": list(hist), "snapshot": 0}
        elif self.root.n_hist < len(hist):
            req = {"op": "restore", "slot": 0, "replay": list(hist[self.root.n_hist:]), "snapshot": 0}
        else:
            return
        parse(self.eng.request(req))
        self.root = _Root(key, len(hist))

    # ------------------------------------------------------------ 報酬（run_search.RunTreeRunner._path_reward の簡略版）

    def _path_reward(self, o_prev, o) -> float:
        rw, ms = self.wp.rw, self.wp.ms_scale
        r = 0.0
        if o.floor > o_prev.floor:
            r += ms * rw["floor_reward"] * (int(o.floor) - int(o_prev.floor))
        for act in range(int(o_prev.act) + 1, int(o.act) + 1):
            r += ms * {1: rw["act1_boss_reward"], 2: rw["act2_boss_reward"]}.get(act, 0.0)
        if o.done and o.victory and not o.stuck:
            r += rw["victory_reward"] + rw["act3_boss2_reward"]
        # 第3幕1体目のボスの報酬（act3_boss_reward）は、どの戦闘がそれかを追う必要があるので数えない（第3幕ボス戦の先読みでは過小になる）
        return r

    # ------------------------------------------------------------ 1つの世界

    def _rollout(self, root, c: int, world_seed: int) -> dict:
        from infer import evaluate
        from sts2env.protocol import parse
        o = parse(self.eng.request({"op": "restore", "slot": 0, "reseed": int(world_seed) & 0xFFFFFFFF, "first": int(c),
                                    "rewards": True, "auto_single": True}))
        acc = self._path_reward(root, o)
        prev = o
        steps = 1
        in_combat0, turn0, floor0 = bool(root.in_combat), _turn(root), int(root.floor)
        while not (o.done or o.stuck or o.num_actions == 0):
            if not in_combat0:
                break                                  # 戦闘外の判断：1手だけ
            if not o.in_combat or int(o.floor) != floor0 or _turn(o) != turn0:
                break                                  # 戦闘が終わった / 次のターンになった
            if steps >= MAX_ROLL_STEPS:
                break
            out = evaluate(self.net, o, self.device)
            o = parse(self.eng.request({"op": "step", "action": int(np.argmax(out.probs)), "auto_single": True}))
            acc += self._path_reward(prev, o)
            prev = o
            steps += 1
        if o.done or o.stuck or o.num_actions == 0:
            v_end = 0.0
        else:
            v_end = evaluate(self.net, o, self.device).value
        died = bool(o.done and not o.victory)
        enemy_hp = None
        if o.in_combat:
            ti, tf = o.tok_ints, o.tok_feats
            m = (ti[:, 0] == 2) & (ti[:, 2] == 2) & (tf[:, 3] > 0)
            enemy_hp = float(tf[m, 0].sum())
        return {"value": acc + v_end, "hp": int(o.hp) if not died else 0, "died": died, "steps": steps,
                "combat_over": bool(in_combat0 and (not o.in_combat or int(o.floor) != floor0)), "enemy_hp": enemy_hp}

    # ------------------------------------------------------------ まとめ

    def run(self, root, probs: np.ndarray, root_value: float, hist: list[int], decision_key: int,
            classes: list[int] | None = None) -> dict[int, dict]:
        """候補 index → {"value_mean", "value_delta", "value_sd", "n", "hp_mean", "died_frac", ...}。
        classes（greedy.equivalence_classes）を渡すと、見分けのつかない同じ手は代表の1つだけ計算して結果を写す。"""
        n = int(root.num_actions)
        if n <= 1:
            return {}
        self._sync(hist)
        order = list(np.argsort(-probs, kind="stable"))
        cands = order if not self.cfg.top else order[: self.cfg.top]
        rng = np.random.default_rng([self.cfg.seed, decision_key])
        seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=max(1, self.cfg.worlds))]
        out: dict[int, dict] = {}
        for c in cands:
            rep = int(classes[int(c)]) if classes is not None else int(c)
            if rep != int(c) and rep in out:
                out[int(c)] = dict(out[rep], same_as=rep)
                continue
            rs = [self._rollout(root, int(c), s) for s in seeds]
            vals = [r["value"] for r in rs]
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            e = {"value_mean": round(mean, 4), "value_delta": round(mean - root_value, 4), "value_sd": round(sd, 4),
                 "value_se": round(sd / math.sqrt(len(vals)), 4), "n": len(vals),
                 "hp_mean": round(float(np.mean([r["hp"] for r in rs])), 1),
                 "died_frac": round(float(np.mean([r["died"] for r in rs])), 3),
                 "steps_mean": round(float(np.mean([r["steps"] for r in rs])), 1),
                 "horizon": "end_of_turn" if root.in_combat else "one_step"}
            if root.in_combat:
                e["combat_over_frac"] = round(float(np.mean([r["combat_over"] for r in rs])), 3)
                eh = [r["enemy_hp"] for r in rs if r["enemy_hp"] is not None]
                if eh:
                    e["enemy_hp_mean"] = round(float(np.mean(eh)), 1)
            out[int(c)] = e
        best = max(out, key=lambda k: out[k]["value_mean"])
        best_cls = int(classes[best]) if classes is not None else best
        for k in out:
            out[k]["best_by_lookahead"] = (int(classes[k]) if classes is not None else k) == best_cls
        return out


def _turn(o) -> int:
    gi = np.flatnonzero(o.tok_ints[:, 0] == 1)
    return int(round(float(o.tok_feats[gi[0], 9]))) if len(gi) else 0
