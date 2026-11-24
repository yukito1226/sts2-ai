# sts2-ai

Slay the Spire 2 のプレイ中の判断に、学習済みのAI「AlphaDefect」（[hmty-CHEN/AlphaDefect-Public](https://github.com/hmty-CHEN/AlphaDefect-Public)、MIT）を使ってヒントと解説を付けるためのツール群です。

| 段階 | 内容 | 場所 |
|---|---|---|
| Step 1 | ヒント生成：ヘッドレスエンジン上の各判断で、候補ごとの選択確率・価値・この戦闘に勝つ確率・エンジンが計算した事実を JSON に出す（探索なし、CPU） | [`tools/hint/`](tools/hint/README.md) |
| Step 2 | LLM 解説：Step 1 の JSON と盤面を Claude API に渡し、「なぜその手か」を日本語で説明する。聞き返せるチャット形式（対話モードの `e` と `?`） | [`tools/hint/explain.py`](tools/hint/README.md#claude-による解説step-2対話モードの-e-と-explainpy) |
| Step 3 | ゲーム画面にヒントを出す Mod（最小版：戦闘と地図）と、ローカルのヒントサーバー | [`tools/hintmod/`](tools/hintmod/README.md)、[`tools/hint/server.py`](tools/hint/server.py) |

AlphaDefect 本体のコードは書き換えません。本体のチェックアウトを `--ad-repo` か環境変数 `ALPHADEFECT_REPO` で指定して読み込みます。
重みは配布されている学習済みのものだけを使い、読み込む前に SHA-256 を本体の `weights/SHA256SUMS_*.txt` と照合し、`torch.load(..., weights_only=True)` で読みます。

確認に使った AlphaDefect-Public のコミット: `800db30`（`068953d` の次。違いは CITATION.cff の1行だけ）。

テスト（ゲーム本体と重みは不要）:

```bash
ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest discover -s tests -v
```
