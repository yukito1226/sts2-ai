"""モックのエンジン（テスト用）。ゲーム本体なしで、実機エンジン（Sts2Headless.dll serve）と同じ通信手順・同じ形式（格式 28）のフレームを返す。

中身はとても小さな決め打ちの1局:
  1. イベント（選択肢2つ）
  2. 戦闘（アイアンクラッド対ニビット1体。打撃・防御・強打、ターン終了。敵は毎ターン 6 ダメージ）
  3. カード報酬（3枚＋スキップ）
  4. マップ（1か所）→ クリア（victory）
ゲームのルールを再現するものではない。フレームの形と通信の流れ、ツール側の処理を確かめるためだけに使う。
語彙ID は AlphaDefect-Public の engine/catalog/vocab.txt から引く（環境変数 ALPHADEFECT_REPO）。

  python -I tools/hint/mock_engine.py   # 標準入力に {"op": "reset", ...} などを1行ずつ
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

FORMAT = 28
HEADER_INTS, TOK_INTS, TOK_FEATS, TOK_PRIV, CAND_INTS, CAND_FEATS = 34, 5, 22, 2, 5, 14

repo = Path(os.environ.get("ALPHADEFECT_REPO", Path(__file__).resolve().parents[2]))
VOCAB = {k: i for i, k in enumerate((repo / "engine" / "catalog" / "vocab.txt").read_text(encoding="utf-8").split("\n"))}


def V(key: str) -> int:
    return VOCAB.get(key, 1)       # 1 = <unk>


class Frame:
    def __init__(self):
        self.tok_i: list[list[int]] = []
        self.tok_f: list[list[float]] = []
        self.cand_i: list[list[int]] = []
        self.cand_f: list[list[float]] = []
        self.labels: list[str] = []

    def tok(self, kind, vid, zone, owner=-1, aux=0, **feats) -> int:
        f = [0.0] * TOK_FEATS
        for k, v in feats.items():
            f[int(k[1:])] = float(v)
        self.tok_i.append([kind, vid, zone, owner, aux])
        self.tok_f.append(f)
        return len(self.tok_i) - 1

    def cand(self, kind, a=-1, b=-1, vid=0, label="", **feats) -> None:
        f = [0.0] * CAND_FEATS
        for k, v in feats.items():
            f[int(k[1:])] = float(v)
        self.cand_i.append([kind, a, b, vid, 0])
        self.cand_f.append(f)
        self.labels.append(label)

    def pack(self, header: dict, labels: bool, extra: dict | None) -> bytes:
        h = [0] * HEADER_INTS
        h[0] = FORMAT
        for k, v in header.items():
            h[k] = int(v)
        h[2], h[3] = len(self.tok_i), len(self.cand_i)
        out = struct.pack(f"<{HEADER_INTS}i", *h)
        for row in self.tok_i:
            out += struct.pack("<5i", *row)
        for row in self.tok_f:
            out += struct.pack(f"<{TOK_FEATS}f", *row)
        out += struct.pack(f"<{len(self.tok_i) * TOK_PRIV}f", *([0.0] * (len(self.tok_i) * TOK_PRIV)))
        for row in self.cand_i:
            out += struct.pack("<5i", *row)
        for row in self.cand_f:
            out += struct.pack(f"<{CAND_FEATS}f", *row)
        lab = json.dumps(self.labels, ensure_ascii=False).encode("utf-8") if labels else b""
        out += struct.pack("<i", len(lab)) + lab
        if extra is not None:
            ex = json.dumps(extra, ensure_ascii=False).encode("utf-8")
            out += struct.pack("<i", len(ex)) + ex
        return out


class Game:
    """決め打ちの1局。phase: event → combat → reward → map → over"""

    def __init__(self, req: dict):
        self.labels = bool(req.get("labels"))
        self.seed = str(req.get("seed", ""))
        self.phase = "event"
        self.hp, self.max_hp, self.gold, self.floor = 80, 80, 99, 1
        self.deck = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
        self.steps = 0
        self.fresh_combat = False

    # 共通の記録（全体・プレイヤー・遺物）
    def _common(self, fr: Frame, in_combat: bool):
        fr.tok(1, V("CHARACTER.IRONCLAD"), 0, -1, V("ACT.OVERGROWTH"), f0=0, f1=self.floor, f2=10, f3=self.gold, f4=self.hp, f5=self.max_hp,
               f6=self.energy if in_combat else 0, f7=3 if in_combat else 0, f9=self.turn if in_combat else 0, f11=3, f12=len(self.deck),
               f13=1 if in_combat else 0, f14=1 if in_combat else 0)
        fr.tok(5, V("RELIC.BURNING_BLOOD"), 1)

    def frame(self) -> bytes:
        fr = Frame()
        header = {6: 0, 7: self.floor, 8: self.hp, 9: self.max_hp, 10: self.gold, 13: self.steps}
        extra = None
        if self.phase == "event":
            header[1] = 6
            self._common(fr, False)
            for c in self.deck:
                fr.tok(4, V("CARD." + c), 7)
            o1 = fr.tok(9, V("EVENTOPT:ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN"), 0)
            o2 = fr.tok(9, V("EVENTOPT:ABYSSAL_BATHS.pages.ALL.options.LINGER"), 0)
            fr.cand(12, o1, -1, V("EVENTOPT:ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN"), "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN")
            fr.cand(12, o2, -1, V("EVENTOPT:ABYSSAL_BATHS.pages.ALL.options.LINGER"), "ABYSSAL_BATHS.pages.ALL.options.LINGER")
        elif self.phase == "combat":
            header.update({1: 1, 11: 1})
            if self.fresh_combat:
                extra = {"combat": {"room": "Monster", "act": 0, "floor": self.floor, "encounter": "NIBBIT_WEAK"}}
                self.fresh_combat = False
            self._common(fr, True)
            me = fr.tok(2, V("CHARACTER.IRONCLAD"), 1, -1, 0, f0=self.hp, f1=self.max_hp, f2=self.block, f3=1, f4=1, f5=1,
                        f17=max(0, 6 - self.block))
            if self.vuln:
                fr.tok(3, V("POWER.VULNERABLE_POWER"), 2, 0, 0)   # 所属は後で敵の下標に置き換える
            en = fr.tok(2, V("MONSTER.NIBBIT"), 2, -1, 0, f0=self.en_hp, f1=44, f2=0, f3=1, f4=1, f6=6, f7=1, f8=6)
            for k, (row, f) in enumerate(zip(fr.tok_i, fr.tok_f)):
                if row[0] == 3:
                    row[3] = en
            fr.tok(10, V("MOVE:NIBBIT.SLICE_MOVE"), 0, en, 0, f0=0, f1=1.0, f3=6, f4=1, f5=6, f6=6)
            hand_toks = []
            for c in self.hand:
                cost = {"STRIKE_IRONCLAD": 1, "DEFEND_IRONCLAD": 1, "BASH": 2}[c]
                hand_toks.append((c, cost, fr.tok(4, V("CARD." + c), 3, -1, 0, f0=cost)))
            mult = 1.5 if self.vuln else 1.0
            for c, cost, t in hand_toks:
                if cost > self.energy:
                    continue
                if c == "DEFEND_IRONCLAD":
                    fr.cand(2, t, -1, 0, f"出 {c}", f8=5, f10=max(0, 6 - self.block - 5), f11=self.hp - max(0, 6 - self.block - 5))
                else:
                    d = int((6 if c == "STRIKE_IRONCLAD" else 8) * mult)
                    fr.cand(2, t, en, 0, f"出 {c} → NIBBIT", f4=d, f5=1, f6=d, f7=1 if d >= self.en_hp else 0,
                            f10=0 if d >= self.en_hp else max(0, 6 - self.block), f11=self.hp - (0 if d >= self.en_hp else max(0, 6 - self.block)))
            loss = max(0, 6 - self.block)
            fr.cand(1, -1, -1, 0, "结束回合", f10=loss, f11=self.hp - loss)
        elif self.phase == "reward":
            header[1] = 3
            self._common(fr, False)
            for c in self.deck:
                fr.tok(4, V("CARD." + c), 7)
            for c in ("ANGER", "POMMEL_STRIKE", "TWIN_STRIKE"):
                t = fr.tok(4, V("CARD." + c), 8)
                fr.cand(7, t, -1, 0, f"选 {c}")
            fr.cand(8, -1, -1, V("ALT:SKIP"), "SKIP")
        elif self.phase == "map":
            header[1] = 11
            self._common(fr, False)
            n1 = fr.tok(8, V("MAPNODE:Monster"), 0, -1, 0, f0=0, f1=self.floor + 1)
            fr.cand(15, n1, -1, V("MAPNODE:Monster"), "→ Monster")
        else:   # over
            header.update({1: 12, 4: 1, 5: 1 if self.hp > 0 else 0})
            self._common(fr, False)
        return fr.pack(header, self.labels, extra)

    def start_combat(self):
        self.phase, self.fresh_combat = "combat", True
        self.en_hp, self.vuln, self.turn = 30, False, 0
        self.new_turn()

    def new_turn(self):
        self.turn += 1
        self.energy, self.block = 3, 0
        self.hand = ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "DEFEND_IRONCLAD"]

    def step(self, a: int) -> None:
        self.steps += 1
        if self.phase == "event":
            if a == 1:
                self.hp -= 5
                self.gold += 50
            self.floor += 1
            self.start_combat()
        elif self.phase == "combat":
            playable = [c for c in self.hand if {"STRIKE_IRONCLAD": 1, "DEFEND_IRONCLAD": 1, "BASH": 2}[c] <= self.energy]
            if a >= len(playable):      # ターン終了
                self.hp -= max(0, 6 - self.block)
                if self.hp <= 0:
                    self.phase = "over"
                    return
                self.new_turn()
                return
            c = playable[a]
            self.hand.remove(c)
            self.energy -= {"STRIKE_IRONCLAD": 1, "DEFEND_IRONCLAD": 1, "BASH": 2}[c]
            if c == "DEFEND_IRONCLAD":
                self.block += 5
            else:
                self.en_hp -= int((6 if c == "STRIKE_IRONCLAD" else 8) * (1.5 if self.vuln else 1.0))
                if c == "BASH":
                    self.vuln = True
            if self.en_hp <= 0:
                self.phase = "reward"
                self.floor += 1
        elif self.phase == "reward":
            if a < 3:
                self.deck.append(("ANGER", "POMMEL_STRIKE", "TWIN_STRIKE")[a])
            self.phase = "map"
        elif self.phase == "map":
            self.floor += 1
            self.phase = "over"


def reply(body: bytes) -> None:
    sys.stdout.buffer.write(struct.pack("<i", len(body)) + body)
    sys.stdout.buffer.flush()


def error(msg: str) -> None:
    reply(struct.pack("<i", -1) + msg.encode("utf-8"))


def n_cands(frame: bytes) -> int:
    return struct.unpack_from("<i", frame, 12)[0]      # ヘッダ 3 = 候補の数


def done(frame: bytes) -> bool:
    return struct.unpack_from("<i", frame, 16)[0] != 0  # ヘッダ 4 = 終局


def main() -> None:
    """実機エンジンの要求のうち、ツールが使うものだけを受け付ける:
    reset（replay・snapshot つき）、step、snapshot、restore（replay・reseed・first・snapshot）、drop、auto_single。
    reseed は受け付けるが、モックには乱数がないので何も変えない。"""
    import copy
    game: Game | None = None
    slots: dict[int, Game] = {}
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        req = json.loads(line)
        op = req.get("op")
        if op == "close":
            return
        if op == "snapshot":
            if game is None:
                error("局面がありません")
                continue
            slots[int(req.get("slot", 0))] = copy.deepcopy(game)
            reply(json.dumps({"slot": int(req.get("slot", 0)), "objects": 1, "ms": 0}).encode("utf-8"))
            continue
        if op == "drop":
            if "slot" in req:
                slots.pop(int(req["slot"]), None)
            else:
                slots.clear()
            reply(json.dumps({"dropped": 1, "slots": len(slots)}).encode("utf-8"))
            continue
        if op == "reset":
            if req.get("mode") != "run":
                error("モックは整局モードだけ")
                continue
            game = Game(req)
        elif op == "restore":
            sl = int(req.get("slot", 0))
            if sl not in slots:
                error(f"スロット {sl} は空です")
                continue
            game = copy.deepcopy(slots[sl])
            game.labels = game.labels or bool(req.get("labels"))
        elif op == "step":
            if game is None:
                error("reset の前に step が来ました")
                continue
        else:
            error(f"モックが知らない要求: {op}")
            continue
        if op in ("reset", "restore"):
            for x in req.get("replay") or []:
                game.step(int(x))
            if op == "restore" and "first" in req:
                game.step(int(req["first"]))
        if op == "step":
            game.step(int(req["action"]))
        fr = game.frame()
        if req.get("auto_single"):
            cap = 10000 if req["auto_single"] is True else int(req["auto_single"])
            k = 0
            while k < cap and n_cands(fr) == 1 and not done(fr):
                game.step(0)
                fr = game.frame()
                k += 1
        if "snapshot" in req and isinstance(req["snapshot"], int) and not isinstance(req["snapshot"], bool):
            slots[int(req["snapshot"])] = copy.deepcopy(game)
        reply(fr)


if __name__ == "__main__":
    main()
