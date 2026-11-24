"""価値ヘッドの値 → 勝率の推定。

価値ヘッドが出すのは勝率ではなく「この局面から先にもらえる報酬の合計の予測」（割引なし）。報酬の中身はチェックポイントの
config に入っている（floor_reward × 階層、各幕ボスの報酬、クリア報酬など。tools/future/run_search.py の RunTreeRunner と同じ読み方）。

推定のしかた（method）:
  - "calibrated": tools/hint/calibrate.py で作った較正表（自動プレイのログの「価値 → 実際に勝ったか」）があれば、それを使う。いちばん信頼できる。
  - "linear": 較正表がないときの概算。里程碑報酬（階層・ボス）を撤去した重み（milestone_from 以降、ms_scale = 0）では、報酬はほぼ
    「クリアしたときにまとめて R_win」だけになるので、勝率 ≈ 価値 / R_win。HP の形づけ（run_hp_shaping）が入っていれば、
    その分を足し戻す（ポテンシャル型の形づけでは 価値 = 本来の価値 − Φ(今) + E[Φ(終局)] になるため）。
    E[Φ(終局)] は分からないので「クリア時のHP割合は今と同じ」と仮定する。仮定が入るので、結果は「未較正の概算」として扱う。
  - それ以外（里程碑報酬が残っている重み、他の形づけが入っている重み）は推定しない（None）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

F_END = 49      # クリア時の階層の目安（第3幕の2体目のボス）。較正表があれば使わない

REWARD_KEYS = ("floor_reward", "act1_boss_reward", "act2_boss_reward", "act3_boss_reward", "victory_reward", "act3_boss2_reward")
# 0 でないと「線形の概算」の前提が崩れる形づけ（run_search.py が 0 であることを確かめているもの）
OTHER_SHAPING = ("run_potion_coef", "combat_dmg_coef", "wp_shaping", "bv_shaping", "relic_coef", "potion_discard_penalty")


def _f(c: dict, k: str) -> float:
    try:
        return float(c.get(k) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class WinProb:
    def __init__(self, config: dict, version: int, calib_path: str | None = None):
        c = config or {}
        self.rw = {k: _f(c, k) for k in REWARD_KEYS}
        mf, mr = c.get("milestone_from"), int(c.get("milestone_ramp") or 500)
        self.ms_scale = 1.0 if mf is None or version < mf else (0.0 if mr <= 0 else 1.0 - min(1.0, (version - mf) / mr))
        self.hp_shaping = _f(c, "run_hp_shaping")
        self.hp_concavity = _f(c, "hp_concavity") if c.get("hp_concavity") is not None else 0.5
        self.other = {k: _f(c, k) for k in OTHER_SHAPING if _f(c, k) != 0.0}
        ms_total = self.rw["floor_reward"] * F_END + self.rw["act1_boss_reward"] + self.rw["act2_boss_reward"] + self.rw["act3_boss_reward"]
        # クリアした時の報酬合計：クリア報酬 + 第3幕2体目 + 撤去した分の里程碑の払い戻し（run_search の _path_reward と同じ）
        self.r_win = self.rw["victory_reward"] + self.rw["act3_boss2_reward"] + (1.0 - self.ms_scale) * ms_total
        self.calib = None
        if calib_path:
            self.calib = json.loads(Path(calib_path).read_text(encoding="utf-8"))

    def describe(self) -> dict:
        return {"rewards": self.rw, "ms_scale": self.ms_scale, "r_win": self.r_win, "run_hp_shaping": self.hp_shaping,
                "other_shaping": self.other, "calibration": bool(self.calib)}

    def hp_potential(self, hp: float, max_hp: float, max_hp0: float) -> float:
        """sts2rl.actor.hp_potential と同じ式。"""
        x = min(1.0, max(0.0, hp / max(1.0, max_hp)))
        return self.hp_shaping * max_hp / max(1.0, max_hp0) * (x + self.hp_concavity * x * (1.0 - x))

    def estimate(self, value: float, act: int, hp: float, max_hp: float, max_hp0: float) -> dict:
        if self.calib:
            p = apply_calibration(self.calib, value, act)
            if p is not None:
                return {"p": p, "method": "calibrated"}
        if self.ms_scale > 0.0:
            return {"p": None, "method": None, "reason": f"里程碑報酬が残っている重み（ms_scale={self.ms_scale:.2f}）なので、価値から勝率を直接は出せません。較正表を作ってください"}
        if self.other:
            return {"p": None, "method": None, "reason": f"ほかの形づけが入っている重み（{self.other}）なので概算を出しません。較正表を作ってください"}
        if self.r_win <= 0:
            return {"p": None, "method": None, "reason": "クリア報酬が 0 の設定です"}
        phi = self.hp_potential(hp, max_hp, max_hp0) if self.hp_shaping else 0.0
        # 価値 ≈ p × (R_win + Φ_win) − Φ(今)、Φ_win ≈ Φ(今) と仮定
        p = (value + phi) / (self.r_win + phi)
        return {"p": min(1.0, max(0.0, p)), "method": "linear", "raw": p,
                "note": "未較正の概算（価値 / クリア時の報酬。HPの形づけはクリア時のHP割合が今と同じと仮定）"}


# ---------------------------------------------------------------- 較正表（calibrate.py が作る）

def apply_calibration(calib: dict, value: float, act: int) -> float | None:
    """幕ごとの単調な区分線形の表（"tables": {"0": {"x": [...], "y": [...]}, ...}）で補間する。表がなければ全幕共通の "all"。"""
    t = calib.get("tables", {}).get(str(act)) or calib.get("tables", {}).get("all")
    if not t or not t.get("x"):
        return None
    xs, ys = t["x"], t["y"]
    if value <= xs[0]:
        return float(ys[0])
    if value >= xs[-1]:
        return float(ys[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= value:
            lo = mid
        else:
            hi = mid
    w = (value - xs[lo]) / max(1e-12, xs[hi] - xs[lo])
    return float(ys[lo] + w * (ys[hi] - ys[lo]))


def pav(y: list[float], w: list[float]) -> list[float]:
    """重み付きの単調回帰（Pool Adjacent Violators）。y は x の昇順に並べたもの。"""
    blocks: list[list[float]] = []      # [平均, 重み, 個数]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            ww = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / ww, ww, n1 + n2])
    out: list[float] = []
    for m, _, n in blocks:
        out.extend([m] * int(n))
    return out


def sigmoid(x: float | None) -> float | None:
    if x is None:
        return None
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
