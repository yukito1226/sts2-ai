"""ヒント JSON を Claude API に渡して、日本語の解説を作る（Step 2）。聞き返しのできるチャット。

hint.py の対話モードで `e`（解説）と `? 質問`（聞き返し）として使う。autoplay の JSONL の1局面を後から聞くこともできる:

  python3 -I tools/hint/explain.py out/hint/run.jsonl --step 40
  （そのあと「なぜ防御じゃないの？」などと打つ。空行か q で終わる）

API キーは環境変数 ANTHROPIC_API_KEY から読む（`ant auth login` の資格情報でもよい）。コードやファイルには書かない。

渡すのは hint.py が作った JSON だけ（盤面、候補ごとの方策の確率、価値、戦闘補助ヘッド、エンジンが計算した事実、先読みの見込み、
カードなどのゲーム内の説明文）。解説は「後付けの推測」であって、ネットが実際に考えた理由ではない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

MODEL = "claude-opus-5-5"
EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"     # 安全分類器に断られたとき、サーバー側で別のモデルが答え直す

DISCLAIMER = "※ この解説は、ネットの出した数値をもとに Claude が後から付けた推測です。ネットが実際に考えた理由ではありません。"

SYSTEM = """あなたは Slay the Spire 2 の対戦 AI「AlphaDefect」のヒントを、プレイヤーに日本語で解説する係です。

AlphaDefect は方策・価値ネット（探索なし）で手を選びます。あなたには、ある局面でのネットの出力と、ゲームエンジンが計算した事実を JSON で渡します。

# 守ること
1. 渡した数値と事実に基づいて説明すること。数値と矛盾することは言わないこと。
2. この説明は後付けの推測であり、ネットが実際に考えた理由ではない。最初の答えの冒頭で、そのことを一文で断ること。ネットの「意図」や「考え」を事実のように書かないこと（「数値からは〜と読める」「〜を重く見ているようだ」のように書く）。
3. 確定した事実（facts、state）と、ネットの予測（prob、value、combat_win_prob、after）を区別して書くこと。
4. 価値（value）を勝率と呼ばないこと。局全体の勝率は、net.win_prob が渡されたときだけ触れること。
5. カードや遺物の効果は glossary（ゲーム内の説明文。{Damage} のような記号は差し込まれる数値）を使うこと。glossary にない知識を使うときは「一般には」と断ること。
6. 確率の差が小さい（数ポイント以内）ときは、ネットにとってほぼ互角だと言うこと。after の value_delta の差が value_se と同じくらいなら、差ははっきりしないと言うこと。
7. 渡していない数値（ダメージ、HP、確率など）を作らないこと。分からないことは分からないと言うこと。

# 数値の意味
- candidates[].prob: ネットの方策がその手を選ぶ確率。group_prob は見分けのつかない同じ手（同じカード2枚など）を合計した確率、copies はその枚数。rank は順位（1 がネットの手）。
- net.value: 価値ヘッドの値。この先にもらえる報酬（階を進む、ボスを倒す、クリア）の見込みで、勝率ではない。局面どうし・手どうしの比較にだけ使える。
- net.combat_win_prob: 戦闘補助ヘッドが見積もった「この戦闘に勝つ確率」。net.combat_hp_loss_frac: この戦闘が終わるまでに失うHPの見込み（最大HPに対する割合）。どちらも予測。
- facts（エンジンが計算した確定値。筋力・脱力などの修正値込み）: damage（per_hit 1発、hits 回数、total 合計、kills 倒せる敵の数）、block 得るブロック、hp_loss_if_end_turn_after この手を打ってすぐターンを終えたら失うHP、hp_after_end_turn そのあとのHP、energy_gain、cards_to_hand。
- state: 盤面。enemies[].attack は敵の今の攻撃の意図（修正値込み）、player.end_turn_preview.incoming_hp_loss は今ターンを終えたら受けるHP減少の見込み。
- after（あるときだけ。先読み）: その手を打ってから、戦闘ならそのターンの終わりまで、戦闘外なら1手だけ、ネットの手で進めた n 通りの未来（山札の順番など見えない部分を引き直したもの）の平均。value_delta は今の価値からの差、value_se はその標準誤差、hp_mean はその時点のHP、enemy_hp_mean は敵のHPの合計、died_frac は死んだ割合、best_by_lookahead は見込みが最大の手。
- greedy: ネットが選ぶ手の index。chosen があればプレイヤーが実際に選んだ手。

# 答え方
- 最初の解説は、ネットの手を一言で言い、主な理由を2〜4点、比べる価値のある他の候補との違いを短く。全体で 300 字前後まで。
- 聞き返しには、その質問にだけ答える。数値を引くときは「確率 41%」「ダメージ 6×2=12」のように具体的に。
- 見出しや表は使わず、短い段落か箇条書きで。"""

AFTER_KEYS = ("value_delta", "value_se", "n", "hp_mean", "enemy_hp_mean", "died_frac", "combat_over_frac", "horizon", "best_by_lookahead")


class ExplainError(RuntimeError):
    pass


def compact(rec: dict, top: int = 8) -> dict:
    """プロンプトに入れる分だけにした局面のデータ。同じ手（same_as）は1つにまとめ、生の特徴量は外す。"""
    cands, seen = [], {}
    for c in rec["candidates"]:
        g = c.get("same_as")
        if g is not None and g in seen:
            seen[g]["copies"] = seen[g].get("copies", 1) + 1
            continue
        e = {k: c[k] for k in ("index", "rank", "action", "text", "prob", "group_prob") if k in c}
        e["facts"] = {k: v for k, v in c.get("facts", {}).items() if k != "raw"}
        if "after" in c:
            e["after"] = {k: c["after"][k] for k in AFTER_KEYS if k in c["after"]}
        if g is not None:
            seen[g] = e
        cands.append(e)
    out = {"decision": rec["decision"], "state": rec["state"], "net": rec["net"], "greedy": rec.get("greedy"),
           "candidates": cands[:top]}
    if len(cands) > top:
        out["omitted_candidates"] = len(cands) - top
    if "chosen" in rec:
        out["chosen"] = rec["chosen"]
    if rec.get("glossary"):
        out["glossary"] = rec["glossary"]
    return out


def first_question(rec: dict) -> str:
    g = rec.get("greedy")
    top = next((c for c in rec["candidates"] if c["index"] == g), rec["candidates"][0] if rec["candidates"] else None)
    name = top["text"] if top else "（候補なし）"
    return f"ネットがこの局面で「{name}」（index {g}）を1位にした理由を、ほかの有力な候補と比べて説明してください。"


def make_client():
    try:
        import anthropic
    except ImportError as e:
        raise ExplainError("解説には anthropic パッケージが必要です（pip install anthropic）") from e
    return anthropic.Anthropic()


def _api_errors() -> tuple:
    try:
        import anthropic
        return (anthropic.APIError,)
    except ImportError:
        return ()


class Explainer:
    """1局面ぶんの会話。start で新しい局面を渡し、ask で聞き返す。"""

    def __init__(self, client=None, model: str = MODEL, effort: str = EFFORT, max_tokens: int = 8000, top: int = 8,
                 fallbacks: bool = True, out: TextIO | None = None):
        self.client = client
        self.model, self.effort, self.max_tokens, self.top, self.fallbacks = model, effort, max_tokens, top, fallbacks
        self.out = out or sys.stdout
        self.messages: list[dict] = []
        self.last: object | None = None       # 最後の応答（テストと使用量の確認用）

    def _client(self):
        if self.client is None:
            self.client = make_client()
        return self.client

    def start(self, rec: dict, question: str | None = None) -> str | None:
        data = json.dumps(compact(rec, self.top), ensure_ascii=False, separators=(",", ":"))
        self.messages = []
        self._w(DISCLAIMER + "\n")
        return self._turn(f"いまの局面のデータです。\n```json\n{data}\n```\n\n{question or first_question(rec)}")

    def ask(self, question: str) -> str | None:
        if not self.messages:
            raise ExplainError("先に局面を渡してください（対話モードでは e）")
        return self._turn(question)

    # ------------------------------------------------------------

    def _w(self, s: str) -> None:
        self.out.write(s)
        self.out.flush()

    def _turn(self, user: str) -> str | None:
        self.messages.append({"role": "user", "content": user})
        try:
            text = self._send()
        except _api_errors() as e:
            self.messages.pop()
            self._w(f"\n（API のエラー: {getattr(e, 'message', e)}）\n")
            return None
        except ExplainError as e:            # anthropic が入っていないなど
            self.messages.pop()
            self._w(f"\n（{e}）\n")
            return None
        except TypeError as e:
            if "authentication" not in str(e):
                raise
            self.messages.pop()
            self._w("\n（API キーが見つかりません。環境変数 ANTHROPIC_API_KEY を設定するか、ant auth login してください）\n")
            return None
        if text is None:
            self.messages.pop()
        else:
            self.messages.append({"role": "assistant", "content": text})
        return text

    def _send(self) -> str | None:
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM, messages=self.messages,
                  thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                  cache_control={"type": "ephemeral"})          # 聞き返しのたびに同じ前半（system と局面）を読み直さないように
        if self.fallbacks:
            kw.update(betas=[FALLBACK_BETA], fallbacks="default")
        parts: list[str] = []
        with self._client().beta.messages.stream(**kw) as stream:
            for t in stream.text_stream:
                parts.append(t)
                self._w(t)
            msg = stream.get_final_message()
        self.last = msg
        self._w("\n")
        if msg.stop_reason == "refusal":
            self._w("（この質問には答えられないと返されました。" + ("上の途中までの文は使わないでください。" if parts else "")
                    + "聞き方を変えてみてください）\n")
            return None
        if any(getattr(b, "type", None) == "fallback" for b in msg.content):
            self._w(f"（{msg.model} が代わりに答えました）\n")
        if msg.stop_reason == "max_tokens":
            self._w("（長さの上限で途中で切れました）\n")
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return text or "".join(parts)


# ---------------------------------------------------------------- JSONL の1局面を後から聞く

def load_record(path: Path, step: int | None) -> dict:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "candidates" in r:
            recs.append(r)
    if not recs:
        raise ExplainError(f"{path} に判断の行がありません")
    if step is None:
        return recs[0]
    for r in recs:
        if r.get("step") == step:
            return r
    raise ExplainError(f"step {step} の判断がありません（ある step: {recs[0].get('step')}〜{recs[-1].get('step')}）")


def chat(ex: Explainer, rec: dict, inp: TextIO | None = None) -> None:
    inp = inp or sys.stdin
    ex.start(rec)
    while True:
        ex._w("質問> ")
        line = inp.readline()
        q = line.strip()
        if not line or not q or q == "q":
            return
        ex.ask(q)


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="hint.py の JSONL の1局面について、Claude に解説させて聞き返す")
    ap.add_argument("log", help="hint.py autoplay / interactive --out の JSONL")
    ap.add_argument("--step", type=int, default=None, help="どの手の判断か（JSONL の step。既定は最初の判断）")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT, choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--top", type=int, default=8, help="渡す候補の数")
    ap.add_argument("--no-fallbacks", action="store_true", help="断られたときにサーバー側で別のモデルに答え直させない")
    a = ap.parse_args(argv)
    try:
        rec = load_record(Path(a.log), a.step)
        chat(Explainer(model=a.model, effort=a.effort, top=a.top, fallbacks=not a.no_fallbacks), rec)
    except ExplainError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
