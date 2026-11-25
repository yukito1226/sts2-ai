"""1つの判断のヒント（JSON にできる dict）を組み立てる。

中身:
  decision  判断の種類（戦闘か戦闘外か、画面）
  state     盤面の要約（HP、エネルギー、敵、意図、手札、遺物など。エンジンが計算した値）
  net       価値ヘッドの値、勝率の推定、戦闘補助ヘッド（この戦闘に勝つ確率、戦闘終了までに失うHP）
  candidates 候補ごとに名前、方策の確率、順位、同じ手をまとめた確率、エンジンが計算した事実、先読み予測

エンジンが出す数値の意味は engine/Sts2Headless/Engine/Obs/Candidates.cs、Creatures.cs、EndTurn.cs、Observation.cs の AddGlobal と
zh/観測格式.md に書かれているものをそのまま使う（格式 19〜28 で確認。格式 29 は列が増えているが、ここで使う列の意味は同じとして扱う。未確認）。
"""
from __future__ import annotations

import numpy as np

from names import ACTION_JA, DECISION_JA, Namer

# sts2env.protocol の定数（import を軽くするためここに写す。値は protocol.py と同じ）
TK_GLOBAL, TK_CREATURE, TK_POWER, TK_CARD, TK_RELIC, TK_POTION, TK_ORB, TK_MAP_NODE, TK_OPTION, TK_MOVE = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
Z_ALLY, Z_ENEMY, Z_HAND, Z_DRAW, Z_DISCARD, Z_EXHAUST, Z_DECK = 1, 2, 3, 4, 5, 6, 7
A_END_TURN, A_PLAY_CARD, A_USE_POTION = 1, 2, 3
A_EVENT_OPTION = 12
A_MAP_MOVE = 15
ROOM_JA = {1: "通常戦闘", 2: "エリート", 3: "ボス"}
READOUT_KEYS = ("advantage", "hp_change_frac", "enemy_left_change", "win_within", "lose_within")


def _i(x) -> int:
    return int(round(float(x)))


def _r(x, nd: int = 4):
    return None if x is None else round(float(x), nd)


class Builder:
    def __init__(self, namer: Namer):
        self.N = namer

    # ------------------------------------------------------------ 対象の名前

    def token_name(self, o, j: int) -> str | None:
        if j is None or j < 0 or j >= len(o.tok_ints):
            return None
        ti, tf = o.tok_ints[j], o.tok_feats[j]
        kind, vid = int(ti[0]), int(ti[1])
        if kind == TK_CREATURE and tf[5]:
            return "自分"
        name = self.N.name(vid) if vid else None
        if kind in (TK_CARD, 13) and tf[3] > 0:       # 13 = 選択済みのカード。特徴 3 は強化回数
            name = f"{name}+" if tf[3] == 1 else f"{name}+{_i(tf[3])}"
        if kind == TK_CREATURE and int(o.tok_ints[j, 0]) == TK_CREATURE and int(ti[2]) == Z_ENEMY:
            name = f"{name}#{int(ti[4])}"             # 同じ敵が並ぶので位置（補助ID）を付ける
        return name

    def option_name(self, o, c: int, a: int) -> str | None:
        """語彙にないイベントの選択肢（古の存在の遺物の選択肢などは、エンコーダが意図的に <unk> にしている）の表示名。

        ネットには <unk> のまま渡っている（AlphaDefect の Events.cs、F113 の注記）。表示だけ、エンジンのラベル（選択肢の TextKey）から
        翻訳を引き、なければ選択肢に付いた遺物（記録の補助ID）の名前を使う。
        """
        label = o.labels[c] if o.labels and c < len(o.labels) else None
        if label:
            nm = self.N.name_of_key("EVENTOPT:" + label)
            if nm:
                return nm
        aux = int(o.tok_ints[a, 4])
        if aux > 1:
            return self.N.name(aux)
        if label:
            # 古の存在の選択肢は翻訳に選択肢のキーがなく、末尾が遺物の ID（TEZCATARA.pages.INITIAL.options.NUTRITIOUS_SOUP → 遺物「栄養満点のスープ」）
            nm = self.N.name_of_key("RELIC." + label.split(".")[-1])
            if nm:
                return nm
        if label:
            return label.split(".")[-1]
        return None

    def cand_name(self, o, c: int) -> dict:
        ci = o.cand_ints[c]
        kind, a, b, vid = int(ci[0]), int(ci[1]), int(ci[2]), int(ci[3])
        an, bn = self.token_name(o, a), self.token_name(o, b)
        key_name = self.N.name(vid) if vid else None
        if kind == A_EVENT_OPTION and 0 <= a < len(o.tok_ints) and int(o.tok_ints[a, 1]) <= 1:
            an = self.option_name(o, c, a)
        action = ACTION_JA.get(kind, str(kind))
        if kind == A_END_TURN:
            text = "ターン終了"
        elif kind in (A_PLAY_CARD, A_USE_POTION):
            text = f"{an}" + (f" → {bn}" if bn else "")
        else:
            parts = [p for p in (key_name, an) if p and p != "-"]
            # 同じ名前が2回並ばないように
            uniq: list[str] = []
            for p in parts:
                if p not in uniq:
                    uniq.append(p)
            text = " / ".join(uniq) if uniq else action
            if bn and kind != A_MAP_MOVE:            # 15 = マップ移動（対象Bは経路の要約で、名前にならない）
                text += f" → {bn}"
        out = {"action": action, "action_kind": kind, "text": text}
        if a >= 0 and a < len(o.tok_ints):
            out["a_key"] = self.N.key(int(o.tok_ints[a, 1]))
        if vid:
            out["key"] = self.N.key(vid)
        if o.labels and c < len(o.labels):
            out["engine_label"] = o.labels[c]
        return out

    # ------------------------------------------------------------ 盤面

    def state(self, o) -> dict:
        ti, tf = o.tok_ints, o.tok_feats
        kind, vid, zone, owner = ti[:, 0], ti[:, 1], ti[:, 2], ti[:, 3]
        gi = np.flatnonzero(kind == TK_GLOBAL)
        g = tf[gi[0]] if len(gi) else np.zeros(30, np.float32)
        in_combat = bool(o.in_combat)
        s: dict = {"act": int(o.act) + 1, "floor": int(o.floor), "hp": int(o.hp), "max_hp": int(o.max_hp), "gold": int(o.gold),
                   "in_combat": in_combat, "room": ROOM_JA.get(_i(g[14]), None),
                   "character": self.N.name(int(vid[gi[0]])) if len(gi) else None}

        def powers_of(j: int) -> list:
            out = []
            for k in np.flatnonzero((kind == TK_POWER) & (owner == j)):
                out.append({"name": self.N.name(int(vid[k])), "amount": _i(tf[k, 2] if tf[k, 2] else tf[k, 0]), "debuff": bool(tf[k, 1])})
            return out

        if in_combat:
            s.update(energy=_i(g[6]), max_energy=_i(g[7]), turn=_i(g[9]))
            enemies = []
            for j in np.flatnonzero(kind == TK_CREATURE):
                f = tf[j]
                if f[5]:
                    s["player"] = {"hp": _i(f[0]), "max_hp": _i(f[1]), "block": _i(f[2]), "powers": powers_of(int(j)),
                                   # 生物の特徴 16–19（EndTurn.cs）：プレイヤーは [冷気の受動ブロック, 今ターンを終えたときに受ける見込みのHP減少, 自傷, その他]
                                   "end_turn_preview": {"frost_block": _r(f[16], 1), "incoming_hp_loss": _r(f[17], 1), "self_damage": _r(f[18], 1)}}
                elif int(zone[j]) == Z_ENEMY and f[3] > 0:
                    intents = []
                    for k in np.flatnonzero((kind == TK_MOVE) & (owner == j)):
                        m = tf[k]
                        if _i(m[0]) != 0:       # 0 = 次に使う手。1 以降はその先の予告
                            continue
                        intents.append({"move": self.N.name(int(vid[k])), "prob": _r(m[1], 3), "damage_per_hit": _i(m[5]),
                                        "hits": max(1, _i(m[4])), "total": _i(m[6])})
                    enemies.append({"name": self.token_name(o, int(j)), "hp": _i(f[0]), "max_hp": _i(f[1]), "block": _i(f[2]),
                                    # 生物の特徴 6–8（Creatures.cs）：今の意図の1発あたり（修正値込み）、回数、合計
                                    "attack": {"per_hit": _i(f[6]), "hits": _i(f[7]), "total": _i(f[8])} if f[8] else None,
                                    "intents": intents, "powers": powers_of(int(j))})
            s["enemies"] = enemies
            cards = kind == TK_CARD
            s["hand"] = [{"name": self.token_name(o, int(j)), "cost": (-1 if tf[j, 1] else _i(tf[j, 0]))}
                         for j in np.flatnonzero(cards & (zone == Z_HAND))]
            s["draw_pile"] = int((cards & ((zone == Z_DRAW) | ((zone >= 16) & (zone <= 18)))).sum())
            s["discard_pile"] = int((cards & (zone == Z_DISCARD)).sum())
            s["exhaust_pile"] = int((cards & (zone == Z_EXHAUST)).sum())
            s["orbs"] = [self.N.name(int(vid[j])) for j in np.flatnonzero(kind == TK_ORB)]
        else:
            deck = [self.token_name(o, int(j)) for j in np.flatnonzero((kind == TK_CARD) & (zone == Z_DECK))]
            s["deck_size"] = len(deck)
            s["deck"] = deck
        s["relics"] = [self.N.name(int(vid[j])) for j in np.flatnonzero((kind == TK_RELIC) & (zone == Z_ALLY))]
        s["potions"] = [self.N.name(int(vid[j])) for j in np.flatnonzero((kind == TK_POTION) & (zone == Z_ALLY))]
        bosses = [self.N.name(b) for b in o.bosses if b]
        if bosses:
            s["act_bosses"] = bosses
        return s

    def glossary(self, o) -> dict:
        """盤面にあるカード・遺物・ポーション・パワーの説明文（ゲームの翻訳ファイルから。数値の差し込み記号はそのまま）。

        解説（explain.py）に渡して、カードの効果を推測させないためのもの。翻訳が見つからないものは入れない。
        """
        from names import clean
        out: dict = {}
        for k in np.flatnonzero(np.isin(o.tok_ints[:, 0], (TK_CARD, 13, TK_RELIC, TK_POTION, TK_POWER))):
            vid = int(o.tok_ints[k, 1])
            if not vid:
                continue
            name = self.N.name(vid)
            if name in out:
                continue
            d = clean(self.N.description(vid))
            if d:
                out[name] = d
        return out

    # ------------------------------------------------------------ 候補ごとの事実

    @staticmethod
    def cand_facts(o, c: int) -> dict:
        """Candidates.cs の候補特徴。戦闘中だけ意味がある列（4–13）は戦闘中だけ出す。"""
        cf = o.cand_feats[c]
        kind = int(o.cand_ints[c, 0])
        f: dict = {}
        if o.in_combat:
            if kind in (A_PLAY_CARD, A_USE_POTION) and (cf[4] or cf[6]):
                # 4 1発のダメージ（画面の表示と同じ、筋力・脱力・弱体込み）、5 回数、6 合計、7 倒せるか（全体攻撃は倒せる数）
                f["damage"] = {"per_hit": _i(cf[4]), "hits": _i(cf[5]), "total": _i(cf[6]), "kills": _i(cf[7])}
            if cf[8]:
                f["block"] = _i(cf[8])                       # この手でその場で得るブロック
            if cf[9]:
                f["frost_block_change"] = _i(cf[9])          # ターン終了時の冷気の受動ブロックの増減
            # 10 この手を打ってすぐターンを終えたら失う見込みのHP（ターン終了なら「今終えたら」）、11 その後に残るHP
            f["hp_loss_if_end_turn_after"] = _r(cf[10], 1)
            f["hp_after_end_turn"] = _r(cf[11], 1)
            if cf[12]:
                f["energy_gain"] = _i(cf[12])
            if cf[13]:
                f["cards_to_hand"] = _i(cf[13])
        f["raw"] = [_r(x, 3) for x in cf]
        return f

    @staticmethod
    def map_positions(o, n: int) -> dict[int, str]:
        """地図の行き先が2つ以上あるとき、画面の左から見た位置（左・中央・右、4つ以上は「左から k 番目」）。

        候補はエンジン（RunSession.MapDecision）も Mod（LiveDecision.Map）も列の小さい順に並べているので、その順が左からの順になる。
        行き先が同じ種類（通常戦闘が3つなど）だと名前だけでは見分けがつかないため。
        """
        ms = [c for c in range(n) if int(o.cand_ints[c, 0]) == A_MAP_MOVE]
        k = len(ms)
        if k < 2:
            return {}
        names = {2: ["左", "右"], 3: ["左", "中央", "右"]}.get(k) or [f"左から{i + 1}番目" for i in range(k)]
        return dict(zip(ms, names))

    # ------------------------------------------------------------ 組み立て

    def build(self, o, out, wp_est: dict | None, chosen: int | None = None, merge_classes: list[int] | None = None,
              cand_values: dict | None = None) -> dict:
        from winprob import sigmoid
        n = int(o.num_actions)
        p = out.probs
        order = list(np.argsort(-p, kind="stable"))
        rank = {int(c): r + 1 for r, c in enumerate(order)}
        group_p, size = None, None
        if merge_classes is not None and n:
            tot = np.zeros(n)
            np.add.at(tot, np.asarray(merge_classes), p)
            group_p = [float(tot[merge_classes[c]]) for c in range(n)]
            size = np.bincount(np.asarray(merge_classes), minlength=n)
        where = self.map_positions(o, n)
        cands = []
        for c in range(n):
            e = {"index": c, **self.cand_name(o, c), "prob": _r(p[c], 5), "rank": rank[c]}
            if c in where:
                e["text"] += f"（{where[c]}）"
                e["position"] = where[c]
            if group_p is not None and size[merge_classes[c]] > 1:
                e["group_prob"] = _r(group_p[c], 5)        # 見分けのつかない同じ手（同じカード2枚など）をまとめた確率
                e["same_as"] = int(merge_classes[c])
            e["facts"] = self.cand_facts(o, c)
            if out.readout is not None and o.in_combat:
                e["readout"] = {k: _r(v) for k, v in zip(READOUT_KEYS, out.readout[c])}
            if cand_values and c in cand_values:
                e["after"] = cand_values[c]
            cands.append(e)
        cands.sort(key=lambda x: x["rank"])
        net = {"value": _r(out.value), "value_scalar": _r(out.v_scalar), "head": "out_of_combat" if out.is_macro else "combat"}
        if wp_est and wp_est.get("p") is not None:      # 勝率は較正表か概算が使えるときだけ出す（既定では出さない）
            net["win_prob"] = _r(wp_est["p"])
            net["win_prob_method"] = wp_est.get("method")
            if wp_est.get("note"):
                net["win_prob_note"] = wp_est["note"]
        if o.in_combat and out.combat_win_logit is not None:
            net["combat_win_prob"] = _r(sigmoid(out.combat_win_logit))
            net["combat_hp_loss_frac"] = _r(out.combat_hp_loss)
        top = order[0] if n else None
        rec = {"decision": {"kind": int(o.decision), "screen": DECISION_JA.get(int(o.decision), str(int(o.decision))),
                            "phase": "combat" if o.in_combat else "out_of_combat", "num_candidates": n},
               "state": self.state(o), "net": net, "candidates": cands,
               "greedy": int(top) if top is not None else None}
        if chosen is not None:
            rec["chosen"] = int(chosen)
        return rec
