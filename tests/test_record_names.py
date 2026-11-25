"""イベントの選択肢の表示名（record.py の option_name）のテスト。ゲームと重みは要らない。

古の存在（ネオー以外）の選択肢は、エンコーダが意図的に語彙 <unk> にしている。表示ではエンジンのラベル（選択肢の TextKey）から翻訳を引き、
なければ選択肢に付いた遺物の名前を出す。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hint"))

from names import Namer  # noqa: E402
from record import Builder, TK_OPTION, TK_RELIC  # noqa: E402

VOCAB = ["<pad>", "<unk>", "RELIC.GOLDEN_COMPASS", "CARD.SOOT", "EVENTOPT:NEOW.pages.INITIAL.options.LEAFY_POULTICE"]


def frame(labels, aux):
    ti = np.zeros((3, 5), np.int64)
    ti[0] = [TK_OPTION, 1, 8, -1, aux]          # 語彙 <unk> の選択肢、補助IDに遺物
    ti[1] = [TK_RELIC, 2, 8, -1, 0]
    ti[2] = [4, 3, 8, -1, 0]                     # カード（すす）
    ci = np.array([[12, 0, 2, 0, 0]], np.int64)  # イベントの選択肢、A = 選択肢、B = すす
    return SimpleNamespace(tok_ints=ti, tok_feats=np.zeros((3, 30), np.float32), cand_ints=ci, labels=labels)


class TestOptionName(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix="loc_"))
        (root / "jpn").mkdir()
        (root / "jpn" / "ancients.json").write_text(json.dumps({
            "TEZCATARA.title": "テスカタラ",
            "TEZCATARA.pages.INITIAL.options.NUTRITIOUS_SOUP.title": "栄養満点のスープ"}, ensure_ascii=False), encoding="utf-8")
        (root / "jpn" / "relics.json").write_text(json.dumps({"GOLDEN_COMPASS.title": "黄金の羅針盤", "BIIIG_HUG.title": "アツアツの愛情",
                                                          "LEAFY_POULTICE.title": "薬草の湿布"},
                                                         ensure_ascii=False), encoding="utf-8")
        (root / "jpn" / "cards.json").write_text(json.dumps({"SOOT.title": "すす"}, ensure_ascii=False), encoding="utf-8")
        self.B = Builder(Namer(VOCAB, root, ["jpn"]))

    def test_from_label(self):
        o = frame(["TEZCATARA.pages.INITIAL.options.NUTRITIOUS_SOUP"], 0)
        t = self.B.cand_name(o, 0)["text"]
        self.assertIn("栄養満点のスープ", t)
        self.assertNotIn("<unk>", t)
        self.assertTrue(t.endswith("→ すす"))

    def test_from_relic(self):
        o = frame(["TEZCATARA.pages.INITIAL.options.GOLDEN_COMPASS"], 2)
        self.assertTrue(self.B.cand_name(o, 0)["text"].startswith("黄金の羅針盤"))

    def test_ancient_option_tail_is_relic(self):
        """実機の翻訳ファイルでは、古の存在の選択肢のキーはなく、末尾が遺物の ID になっている（2026-11-24 確認）。"""
        o = frame(["TEZCATARA.pages.INITIAL.options.BIIIG_HUG"], 0)
        self.assertTrue(self.B.cand_name(o, 0)["text"].startswith("アツアツの愛情"))

    def test_neow_option_in_vocab(self):
        """ネオーの選択肢は語彙にある（<unk> ではない）が、翻訳にキーがない。末尾の遺物名で出す。"""
        self.assertEqual(self.B.N.name(4), "薬草の湿布")

    def test_last_resort_key(self):
        o = frame(["SOMEWHERE.pages.INITIAL.options.ODD_THING"], 0)
        self.assertTrue(self.B.cand_name(o, 0)["text"].startswith("ODD_THING"))


if __name__ == "__main__":
    unittest.main()


class TestMapPositions(unittest.TestCase):
    """地図の行き先は列の小さい順に並んでいるので、その順に左・中央・右を付ける（同じ種類が並ぶと名前だけでは見分けがつかない）。"""

    def positions(self, kinds):
        ci = np.array([[k, -1, -1, 0, 0] for k in kinds], np.int64)
        return Builder.map_positions(SimpleNamespace(cand_ints=ci), len(kinds))

    def test_three(self):
        self.assertEqual(self.positions([15, 15, 15]), {0: "左", 1: "中央", 2: "右"})

    def test_two_with_potion(self):
        self.assertEqual(self.positions([15, 15, 3]), {0: "左", 1: "右"})

    def test_one_has_none(self):
        self.assertEqual(self.positions([15]), {})

    def test_four(self):
        self.assertEqual(self.positions([15] * 4)[3], "左から4番目")


class TestRankByGroup(unittest.TestCase):
    """同じカード2枚（27.6% ずつ、合わせて 55.3%）は、1枚だけの防御（35.1%）より上に並ぶ（2026-11-25 の実機の表示）。"""

    def test_merged_first(self):
        from types import SimpleNamespace as NS
        B = Builder(Namer(VOCAB, None, ["jpn"]))
        n = 4
        o = NS(num_actions=n, decision=1, in_combat=False, cand_ints=np.array([[2, -1, -1, 0, 0]] * 3 + [[1, -1, -1, 0, 0]], np.int64),
               tok_ints=np.zeros((1, 5), np.int64), tok_feats=np.zeros((1, 30), np.float32), labels=None,
               act=0, floor=1, hp=1, max_hp=1, gold=0)
        B.state = lambda o: {}
        B.cand_facts = lambda o, c: {}
        out = NS(probs=np.array([0.351, 0.2765, 0.2765, 0.096]), readout=None, value=0.0, v_scalar=0.0, is_macro=True, combat_win_logit=None)
        rec = B.build(o, out, None, merge_classes=[0, 1, 1, 3])
        self.assertEqual([c["index"] for c in rec["candidates"]], [1, 2, 0, 3])
        self.assertAlmostEqual(rec["candidates"][0]["group_prob"], 0.553, places=3)
        self.assertEqual(rec["greedy"], 0)
