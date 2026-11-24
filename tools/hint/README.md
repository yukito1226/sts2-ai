# AlphaDefect ヒントツール（Step 1：ヒント生成）

学習済みの Slay the Spire 2 用AI「AlphaDefect」（[hmty-CHEN/AlphaDefect-Public](https://github.com/hmty-CHEN/AlphaDefect-Public)、MIT）を使い、
ヘッドレスエンジン上の各判断で「候補ごとの選択確率」「価値と推定勝率」「エンジンが計算した事実」を JSON に出します。
探索なし、CPU だけで動きます。AlphaDefect 本体のコードは一切書き換えず、本体のチェックアウトを指定して読み込みます。

## 処理の流れ

```
エンジン（Sts2Headless.dll serve, .NET 9）
   │  標準入力: {"op":"reset", ...} / {"op":"step","action":i}（JSON 1行）
   │  標準出力: 4バイトの長さ + バイナリのフレーム（格式 28）
   ▼
sts2env.protocol.parse  → Obs（ヘッダ・記録・候補・候補の文字列）
   ▼
Ragged.from_obs(mode=整局) → tok_priv をゼロ → collate_ragged   （探索コード sim_pimc.ModelEvaluator と同じ）
   ▼
PolicyValueNet.forward_all → 候補のロジット / 価値 / 戦闘補助ヘッド / 候補ごとの先読み予測
   ▼
softmax → 方策の確率。1位がネットの手（評価と同じ貪欲）
```

| ファイル | 役割 |
|---|---|
| `hint.py` | コマンドの入口。`autoplay`（1局を自動で打って全判断を JSONL に書く）と `interactive`（局面ごとに止める） |
| `weights.py` | SHA-256 を `weights/SHA256SUMS_*.txt` と照合してから `torch.load(..., weights_only=True)` で読む |
| `engine.py` | エンジンとの通信（`sts2env/env.py` と同じ手順。起動コマンドを差し替えられる） |
| `infer.py` | 1局面をネットに通す |
| `record.py` | 1判断の JSON を組み立てる（盤面の要約、候補の名前と事実） |
| `names.py` | 語彙ID → 名前（ゲームの翻訳ファイルを使う。無ければキーのまま） |
| `winprob.py` | 価値 → 勝率の推定（概算と較正表） |
| `calibrate.py` | 自動プレイのログから較正表を作る |
| `mock_engine.py` | テスト用のモックエンジン（ゲーム本体なしで同じ形式のフレームを返す） |

## 出力（JSONL）

1行目が `meta`（重みのファイル名と SHA-256、版、obs_format、キャラ、シード、reset に付けた設定、勝率推定の設定）、
そのあと判断ごとに1行、最後に `result`（勝敗、幕、階、手数）です。判断の行はこうなります（値は例）:

```jsonc
{
  "decision": {"kind": 1, "screen": "戦闘", "phase": "combat", "num_candidates": 6},
  "state": {"act": 1, "floor": 2, "hp": 75, "max_hp": 80, "energy": 3, "turn": 1,
            "player": {"block": 0, "powers": [], "end_turn_preview": {"incoming_hp_loss": 6.0, ...}},
            "enemies": [{"name": "ニビット#0", "hp": 30, "attack": {"per_hit": 6, "hits": 1, "total": 6}, "intents": [...]}],
            "hand": [{"name": "ストライク", "cost": 1}, ...], "relics": [...], "potions": [...]},
  "net": {"value": 0.62, "head": "combat",
          "win_prob": 0.31, "win_prob_method": "linear", "win_prob_note": "未較正の概算 …",
          "combat_win_prob": 0.98, "combat_hp_loss_frac": 0.07},
  "candidates": [
    {"index": 3, "rank": 1, "prob": 0.41, "text": "強打 → ニビット#0", "action": "カードを使う",
     "facts": {"damage": {"per_hit": 8, "hits": 1, "total": 8, "kills": 0}, "hp_loss_if_end_turn_after": 6.0, "hp_after_end_turn": 69.0},
     "readout": {"advantage": 0.01, "hp_change_frac": -0.05, "enemy_left_change": -0.3, "win_within": 0.2, "lose_within": 0.0}},
    {"index": 0, "rank": 2, "prob": 0.19, "group_prob": 0.38, "same_as": 0, "text": "ストライク → ニビット#0", ...}
  ],
  "greedy": 3, "chosen": 3, "step": 1
}
```

各項目の意味:

- `prob` は方策の確率、`rank` はその順位です。同じカードが2枚あると候補が2つに分かれるので、見分けのつかない候補をまとめた確率を `group_prob` に出します（`same_as` はまとめ先の index）。
- `net.value` は価値ヘッドの値です。**勝率ではなく「この先にもらえる報酬の合計の予測」**です。戦闘中は戦闘用、戦闘外は戦闘外用のヘッドが使われます（`head`）。特権入力（山札の順番など）はゼロにしています。
- `net.win_prob` は勝率の推定です。`win_prob_method` で出し方を区別します。
  - `calibrated`：`calibrate.py` で作った較正表（実際に打った局の「価値 → 勝ったか」）を使った値です。
  - `linear`：較正表がないときの概算で、価値をクリア時の報酬で割った値です。里程碑報酬を撤去した重みでしか出しません。HPの形づけの分は仮定を置いて足し戻しています。**未較正なので、目安以上には使わないでください。**
  - 出せないとき（里程碑報酬が残っている重みなど）は `null` になり、理由が `win_prob_unavailable` に入ります。
- `net.combat_win_prob` と `combat_hp_loss_frac` は戦闘補助ヘッドの出力で、戦闘中だけ出ます。「この戦闘に勝つ確率」と「戦闘の終わりまでに失うHP（最大HPに対する割合）」です。学習の補助に使われていたヘッドで、行動の決定には使われていません。
- `facts` はエンジンが計算した数値（`engine/Sts2Headless/Engine/Obs/Candidates.cs`）です。
  - `damage`：修正値込みの1発のダメージ、回数、合計、倒せるか（全体攻撃なら倒せる数）
  - `block`：その場で得るブロック
  - `hp_loss_if_end_turn_after`：この手を打ってすぐターンを終えたら失う見込みのHP。ターン終了の候補では「今終えたら」の値です
  - `hp_after_end_turn`：そのあとに残るHP
  - `energy_gain`、`cards_to_hand`
  - `raw`：候補の数値すべて
- `readout` は候補ごとの先読み予測（`future_head` を持つ重みだけ）です。その手から12手先までの「優位（価値の変化）」「HPの変化」「敵の残りHP割合の変化」「その間に勝つ」「その間に負ける」を回帰したもので、確率として較正されたものではありません。戦闘中だけ出します。

## 動かし方

### A. モックで確認する（ゲームと重みは不要）

AlphaDefect-Public のチェックアウトと、Python の numpy・torch（CPU 版）だけで動きます。

```bash
git clone https://github.com/hmty-CHEN/AlphaDefect-Public ~/AlphaDefect-Public
python3.12 -m venv ~/.venvs/sts2 && ~/.venvs/sts2/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
export ALPHADEFECT_REPO=~/AlphaDefect-Public
cd sts2-ai
~/.venvs/sts2/bin/python -m unittest discover -s tests -v                  # テスト一式（18件、十数秒）

# 手で動かす場合（乱数で初期化した小さなネットなので、確率の値に意味はありません）
~/.venvs/sts2/bin/python -I tests/make_mock_ckpt.py --out /tmp/mock.pt
~/.venvs/sts2/bin/python -I tools/hint/hint.py autoplay --mock --allow-unlisted --ckpt /tmp/mock.pt --out /tmp/run.jsonl
~/.venvs/sts2/bin/python -I tools/hint/hint.py interactive --mock --allow-unlisted --ckpt /tmp/mock.pt
```

テストで確かめていること:

- 一覧にない重み、SHA-256 が合わない重みを読まないこと
- `weights_only=True` で、コードを実行する pickle を弾くこと
- 方策と価値が、評価コードと同じ経路（`collate_ragged` → `net(batch)`）や、探索の推論経路（`forward_infer`）と一致すること
- 特権入力が入ったフレームでも、出力が変わらないこと
- 自動プレイと対話モードが最後まで動くこと
- JSON の形（確率の合計が1、順位、同じ手のまとめ、ダメージ・ブロック・ターン終了時の被HP）
- 同じ入力なら結果が同じになること
- 画面の絞り込み
- 較正表の作成と、その適用
- 翻訳ファイルからの名前の引き当て

### B. 実機で確認する（WSL Ubuntu、Python 3.12、.NET 9）

1. **ゲームの版を合わせる。** Steam で Slay the Spire 2 のベータブランチを **v0.113.1** にします（アイアンクラッド、ネクロバインダー、フール）。
2. **ゲームファイルを取り出す。** AlphaDefect の README と同じ手順です。

   ```bash
   cd ~/AlphaDefect-Public
   V=v0.113.1
   G="/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2"
   mkdir -p ~/.local/share/sls2/game/$V
   rsync -a "$G/data_sts2_windows_x86_64/" ~/.local/share/sls2/game/$V/
   # 翻訳ファイル。まずどの言語があるかを一覧で確かめる（日本語のフォルダ名は未確認。jpn などのはず）
   python3 -I tools/extract_pck.py "$G/SlayTheSpire2.pck" | grep '^localization/' | cut -d/ -f2 | sort -u
   python3 -I tools/extract_pck.py "$G/SlayTheSpire2.pck" ~/.local/share/sls2/game/$V/res localization/eng/ localization/jpn/ localization/zhs/ localization/completion.json
   ```
3. **エンジンをビルドして自己診断する。**

   ```bash
   (cd engine/Sts2Headless && dotnet build -c Release -p:Sts2GameVersion=$V)
   dotnet engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll selfcheck
   ```
4. **重みを取ってきて、チェックサムを確かめる。**

   ```bash
   gh release download weights-v6 -R hmty-CHEN/AlphaDefect-Public -D weights/files_v6
   (cd weights/files_v6 && sha256sum -c ../SHA256SUMS_v6.txt)
   ```
   ツールも読み込む前に同じ照合をします。
5. **まず本家の評価が動くことを確かめる**（1局だけ）。

   ```bash
   .venv/bin/python -I tools/ability_test.py --character IRONCLAD --ckpt weights/files_v6/ironclad_ema_v24782.pt \
       --asc 10-10 --runs 1 --seeds-file seeds/R1.json --envs 1 --device cpu --out out/check_ability \
       --req-extra '{"obs_format": 28, "loop_cut": true, "stall_turns": 20}'
   ```
6. **ヒントツールで同じシードを打つ。**

   ```bash
   cd ~/sts2-ai
   export ALPHADEFECT_REPO=~/AlphaDefect-Public
   SEED=$(python3 -c "import json;print(json.load(open('$ALPHADEFECT_REPO/seeds/R1.json'))['seeds'][0])")
   ~/AlphaDefect-Public/.venv/bin/python -I tools/hint/hint.py autoplay --character IRONCLAD \
       --ckpt $ALPHADEFECT_REPO/weights/files_v6/ironclad_ema_v24782.pt --seed $SEED --out out/hint_$SEED.jsonl \
       --engine-log out/engine.log
   ```
   **確かめる点:** 最後の `result`（勝敗・幕・階）が、手順5の `out/check_ability/runs_s0.jsonl` の同じシードの結果と一致すること。一致すれば、フレームの要求・特権のゼロ化・ネットの通し方が評価と同じだと言えます。CPU とバッチの大きさの違いで、ごくまれに僅差の手が入れ替わることはあります（AlphaDefect の docs/26 §5）。
7. **対話モードを試す。**

   ```bash
   ~/AlphaDefect-Public/.venv/bin/python -I tools/hint/hint.py interactive --character IRONCLAD \
       --ckpt $ALPHADEFECT_REPO/weights/files_v6/ironclad_ema_v24782.pt --seed $SEED --screens COMBAT,CARD_REWARD,EVENT,SHOP,REST_SITE
   ```
   Enter でネットの手、番号でその候補、`j` で JSON、`a 20` で20判断ぶん自動、`q` で終了です。
8. **（任意）勝率の較正表を作る。** CPU で1局数分かかるので、夜間に回すことをおすすめします。

   ```bash
   ~/AlphaDefect-Public/.venv/bin/python -I tools/hint/hint.py autoplay --character IRONCLAD --ckpt ... \
       --seeds-file $ALPHADEFECT_REPO/seeds/RV1.json --runs 200 --quiet --out out/calib_runs
   python3 -I tools/hint/calibrate.py out/calib_runs/*.jsonl --out out/calib_ironclad_v24782.json
   # 以後 --calib out/calib_ironclad_v24782.json を付ける
   ```

主なオプション: `--screens`（記録・停止する画面を絞る）、`--merge-equiv`（同じ手をまとめた確率で1位を決める）、`--lang jpn,eng`（名前の言語）、
`--req-extra`（reset に足す設定。既定で `loop_cut`、`stall_turns: 20`、重みの `obs_format` を付けます）、`--threads`（CPU のスレッド数）。

## まだ確認できていないこと

- **実機での動作はまだ確認していません。** ゲーム本体と重みが無い環境で作ったので、確認はモックだけです。上の B の手順6で、本家の評価と結果が一致するかを最初に見てください。
- **配布されている重みの config の中身は未確認です。** 勝率の概算は、`milestone_from`（里程碑報酬の撤去）、`run_hp_shaping` などの報酬設定が重みに入っている前提で計算しています。入っていない、または前提と違う場合は `win_prob` が `null` になります（理由は `win_prob_unavailable` に出ます）。`meta.winprob` に読み取った設定が出るので、最初の実行で確認してください。どちらにしても、勝率は較正表を作るまで目安です。
- **候補の文字列（engine_label）を付けたまま打っています。** フレームの中身は変わらないはずですが（エンジンは文字列を末尾に足すだけ）、本家の評価は付けずに打っています。手順6の一致で確認できます。
- **日本語の翻訳フォルダ名**（`jpn` と仮定）と、翻訳ファイルのキーの形は、AlphaDefect の `tools/replay_viewer.py` と同じ前提です。見つからない名前は `CARD.STRIKE_IRONCLAD` のようなキーのまま出ます。
- **格式 29 は試していません。** アイアンクラッドの v24782 は格式 28 で学習されているので、ツールは重みの `obs_format`（28）をエンジンに要求します。
- **戦闘補助ヘッドと先読み予測は、学習の補助として付いていたヘッドです。** どのくらい当たるかは測っていません。
- 候補ごとの「その手を選んだ後の価値」（短いロールアウトや探索）は、次の段階で作ります。
