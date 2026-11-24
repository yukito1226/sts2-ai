"""語彙ID → 人が読める名前。

ゲームに付いている翻訳ファイル（~/.local/share/sls2/game/<版>/res/localization/<言語>/*.json、tools/extract_pck.py で取り出す）を
指定した言語の順に探し、見つからなければ語彙のキー（CARD.STRIKE_IRONCLAD など）をそのまま返す。
キーの対応は AlphaDefect-Public の tools/replay_viewer.py の names()（MIT）と同じ。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ACTION_JA = {
    0: "なし", 1: "ターン終了", 2: "カードを使う", 3: "ポーションを使う", 4: "ポーションを捨てる", 5: "カードを選ぶ",
    6: "選択を確定", 7: "カード報酬を取る", 8: "カード報酬の別の選択肢", 9: "パックを選ぶ", 10: "報酬を受け取る", 11: "進む・やめる",
    12: "イベントの選択肢", 13: "休憩所の行動", 14: "買う", 15: "マップを進む", 16: "遺物を選ぶ", 17: "遺物を取らない", 18: "マス目を掘る",
}
DECISION_JA = {
    0: "停止", 1: "戦闘", 2: "カード選択", 3: "カード報酬", 4: "パック選択", 5: "報酬画面", 6: "イベント", 7: "休憩所", 8: "ショップ",
    9: "宝箱", 10: "ミニゲーム", 11: "マップ", 12: "ゲーム終了",
}
MAPNODE_JA = {"Monster": "通常戦闘", "Elite": "エリート", "Boss": "ボス", "RestSite": "休憩所", "Shop": "ショップ", "Treasure": "宝箱",
              "Unknown": "？部屋", "Ancient": "古の存在", "Unassigned": "未定"}
REWARD_JA = {"GoldReward": "ゴールド", "CardReward": "カード報酬", "PotionReward": "ポーション", "RelicReward": "遺物",
             "SpecialCardReward": "特別なカード報酬", "CardRemovalReward": "カード削除", "LinkedRewardSet": "まとまった報酬"}
TABLE_OF = {"CARD": "cards", "RELIC": "relics", "POTION": "potions", "POWER": "powers", "ORB": "orbs", "ENCHANTMENT": "enchantments",
            "AFFLICTION": "afflictions", "ENCOUNTER": "encounters", "CHARACTER": "characters"}


def clean(text: str | None) -> str:
    """[gold]…[/gold] などの書式タグを外す。"""
    if not text:
        return ""
    return re.sub(r"\[/?[a-zA-Z_]+[^\]]*\]", "", text).strip()


class Namer:
    def __init__(self, vocab: list[str], loc_root: Path | None, langs: list[str]):
        self.vocab = vocab
        self.tables: list[dict[str, dict]] = []
        self.langs_found: list[str] = []
        for lang in langs:
            d = loc_root / lang if loc_root else None
            if d is not None and d.is_dir():
                t = {}
                for p in d.glob("*.json"):
                    try:
                        t[p.stem] = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                if t:
                    self.tables.append(t)
                    self.langs_found.append(lang)
        self._cache: dict[int, str] = {}

    def _get(self, table: str, key: str) -> str | None:
        for t in self.tables:
            v = t.get(table, {}).get(key)
            if isinstance(v, str) and v:
                return clean(v)
        return None

    def key(self, vid: int) -> str:
        return self.vocab[vid] if 0 < vid < len(self.vocab) else "-"

    def name(self, vid: int) -> str:
        """語彙ID の表示名。翻訳が見つからなければキーを返す。"""
        if vid in self._cache:
            return self._cache[vid]
        k = self.key(vid)
        n = self._lookup(k) or k
        self._cache[vid] = n
        return n

    def name_of_key(self, k: str) -> str | None:
        """語彙にないキー（古の存在の選択肢など）でも、翻訳から名前を引く。見つからなければ None。"""
        return self._lookup(k)

    def _lookup(self, k: str) -> str | None:
        if k == "-":
            return None
        if "." in k and ":" not in k:
            cat, entry = k.split(".", 1)
            if cat == "MONSTER":
                return self._get("monsters", f"{entry}.name") or self._get(
                    "monsters", f"{re.sub(r'_(FRONT|BACK|MIDDLE|LEFT|RIGHT)$', '', entry)}.name")
            if cat == "EVENT":
                return self._get("events", f"{entry}.title") or self._get("ancients", f"{entry}.title")
            table = TABLE_OF.get(cat)
            if table:
                return self._get(table, f"{entry}.title") or self._get(table, f"{entry}.name")
            return None
        if k.startswith("EVENTOPT:"):
            key = k[9:]
            nm = self._get("events", f"{key}.title") or self._get("ancients", f"{key}.title")
            if nm is None and ".options." in key:
                # 古の存在（ネオー、テスカタラなど）の選択肢は翻訳にキーがなく、末尾が遺物の ID（NEOW.pages.INITIAL.options.LEAFY_POULTICE
                # → 遺物「薬草の湿布」）。2026-11-24 に実機の翻訳ファイルで確認。遺物名だけを返す（古の存在の名前は付けない）
                tail = key.rsplit(".", 1)[-1]
                rel = self._get("relics", f"{tail}.title")
                if rel:
                    return rel
            ev = key.split(".")[0]
            evname = self._get("events", f"{ev}.title") or self._get("ancients", f"{ev}.title")
            return f"{evname}・{nm}" if (evname and nm) else nm
        if k.startswith("REST:"):
            return self._get("rest_site_ui", f"OPTION_{k[5:]}.name")
        if k.startswith("ALT:"):
            return self._get("card_reward_ui", f"OPTION_{k[4:]}.name") or {"SKIP": "スキップ", "REROLL": "引き直し",
                                                                           "SACRIFICE": "生贄"}.get(k[4:])
        if k.startswith("REWARD:"):
            return REWARD_JA.get(k[7:], k[7:])
        if k.startswith("MAPNODE:"):
            return MAPNODE_JA.get(k[8:], k[8:])
        return None

    def description(self, vid: int) -> str | None:
        """カード・遺物・ポーション・パワーの説明文（数値の差し込み記号はそのまま）。"""
        k = self.key(vid)
        if "." not in k or ":" in k:
            return None
        cat, entry = k.split(".", 1)
        table = TABLE_OF.get(cat)
        if not table:
            return None
        if cat == "POWER":
            return self._get(table, f"{entry}.smartDescription") or self._get(table, f"{entry}.description")
        return self._get(table, f"{entry}.description")
