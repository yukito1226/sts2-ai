"""1つの局面（Obs）をネットに通し、方策・価値・補助ヘッドの出力を取り出す。

組み立ては AlphaDefect の探索コード（tools/sim_pimc.py の ModelEvaluator、strip_priv=True）と同じ:
  Ragged.from_obs([obs], mode=[1]) → 特権入力 tok_priv をゼロ → collate_ragged → net.forward_all
forward_all は評価・実戦で使われる net(batch) と同じ計算で、方策と価値は net(batch) と一致する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODE_COMBAT, MODE_RUN = 0, 1


@dataclass
class NetOut:
    probs: np.ndarray                 # [候補数] 方策の確率（softmax）
    logits: np.ndarray                # [候補数]
    value: float                      # 価値ヘッド（整局モード。戦闘中は戦闘用、戦闘外は戦闘外用のヘッド）
    v_scalar: float | None            # 数値の価値ヘッド（分類ヘッドと混ぜる前）
    combat_win_logit: float | None    # 戦闘補助ヘッド：この戦闘に勝つ（ロジット）
    combat_hp_loss: float | None      # 戦闘補助ヘッド：この戦闘の終わりまでに失うHP（最大HPに対する割合）
    readout: np.ndarray | None        # [候補数, 5] 候補ごとの先読み予測（future_head があるとき）
    is_macro: bool                    # 戦闘外の判断か（戦闘外用ヘッドが使われたか）


def evaluate(net, obs, device: str = "cpu", mode: int = MODE_RUN) -> NetOut:
    import torch
    from sts2rl.features import Ragged, collate_ragged, macro_rows

    r = Ragged.from_obs([obs], np.array([mode]))
    r.tok_priv[:] = 0       # 特権入力（山札の順番など）は使わない。from_obs が作るのはコピーなので元の obs は変わらない
    b = collate_ragged(r)
    is_macro = bool(macro_rows(r.header, r.modes())[0])
    with torch.no_grad():
        logits, value, _phi, extra = net.forward_all(b.to(device))
    n = int(obs.num_actions)
    lg = logits[0, :n].double().cpu()
    probs = torch.softmax(lg, -1).numpy() if n else np.zeros(0)
    aux = extra.get("aux")
    fh = extra.get("fh")
    readout = None
    if fh is not None and n:
        # fh の行は cand_step / cand_slot の順。1局面だけなので候補の順そのまま
        readout = fh.float().cpu().numpy()[:n]
    return NetOut(
        probs=probs,
        logits=lg.numpy(),
        value=float(value[0]),
        v_scalar=float(extra["v_scalar"][0]) if "v_scalar" in extra else None,
        combat_win_logit=float(aux[0, 0]) if aux is not None else None,
        combat_hp_loss=float(aux[0, 1]) if aux is not None else None,
        readout=readout,
        is_macro=is_macro,
    )
