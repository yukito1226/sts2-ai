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
- 局全体の勝率は **既定では出しません**。ネットが直接出す値ではないためです。必要なときだけ、次のどちらかを付けると `net.win_prob` が出ます。
  - `--calib 表.json`：`calibrate.py` で作った較正表（実際に打った局の「価値 → 勝ったか」）を使います（`win_prob_method: calibrated`）。
  - `--winprob-estimate`：重みの報酬設定から概算します（`linear`、未較正）。里程碑報酬を撤去した重みでしか出ません。配布されている ironclad_ema_v24782 では出ません。
- `net.combat_win_prob` と `combat_hp_loss_frac` は戦闘補助ヘッドの出力で、戦闘中だけ出ます。ネット自身が確率として出す値です。「この戦闘に勝つ確率」と「戦闘の終わりまでに失うHP（最大HPに対する割合）」です。学習の補助に使われていたヘッドで、行動の決定には使われていません。
- `facts` はエンジンが計算した数値（`engine/Sts2Headless/Engine/Obs/Candidates.cs`）です。
  - `damage`：修正値込みの1発のダメージ、回数、合計、倒せるか（全体攻撃なら倒せる数）
  - `block`：その場で得るブロック
  - `hp_loss_if_end_turn_after`：この手を打ってすぐターンを終えたら失う見込みのHP。ターン終了の候補では「今終えたら」の値です
  - `hp_after_end_turn`：そのあとに残るHP
  - `energy_gain`、`cards_to_hand`
  - `raw`：候補の数値すべて
- `readout` は候補ごとの先読み予測（`future_head` を持つ重みだけ）です。その手から12手先までの「優位（価値の変化）」「HPの変化」「敵の残りHP割合の変化」「その間に勝つ」「その間に負ける」を回帰したもので、確率として較正されたものではありません。戦闘中だけ出します。

## 候補ごとの見込み（`--lookahead`、対話モードの `l`）

候補ごとに「その手を選んだあと、どのくらい良くなるか・悪くなるか」を短い先読みで比べます。探索ではないので、ネットの選ぶ手（`greedy`）は変えません。

1. 実際の対局とは別に、先読み用のエンジンを1つ起動します。同じ reset と、これまでに選んだ手の再生で、今の局面に合わせます。実際の対局には触りません。
2. 候補ごとに、見えない未来（山札の順番、この先の乱数、戦闘後の報酬）を `reseed` で引き直した「世界」を N 個作り、その手を打ちます。本当の未来は覗きません。どの候補にも同じ N 個の世界を使うので、候補どうしを比べやすくなります。
3. そこからネットの手で進めます。
   - 戦闘中：次の自分のターンが始まるまで（敵の行動を受けたあと）か、戦闘が終わるまで
   - 戦闘外：1手だけ
4. 行き着いた局面の価値に、途中でもらった報酬を足したものを「その手のあとの価値」とします。死んだら 0 です。

候補の `after` には次の値が入ります。

- `value_mean`：N 世界の平均
- `value_delta`：今の価値との差
- `value_sd`、`value_se`：ばらつきと標準誤差
- `hp_mean`、`died_frac`：先読みが終わったときの HP と、死んだ割合
- `enemy_hp_mean`、`combat_over_frac`：戦闘中だけ。敵の残り HP の合計と、戦闘が終わった割合
- `best_by_lookahead`：見込みがいちばん良い手（同じ手どうしはまとめて扱います）
- `horizon`：どこまで先読みしたか（`end_of_turn` か `one_step`）

見分けのつかない同じ手は、1回だけ計算して結果を写します（`same_as`）。

```bash
# 自動プレイで、戦闘とカード報酬の判断に 4 世界ずつ
hint.py autoplay ... --lookahead 4 --lookahead-screens COMBAT,CARD_REWARD
# 対話モードでは、止まった局面で l（または l 8 で 8 世界）
```

注意点:
- 先読みは短いです。戦闘中ならターンの終わりまでなので、数ターン先に効く手（パワー、強化、ドロー）の良さは出にくくなります。「価値」がその先をどれだけ見込めているかに頼っています。
- 第3幕の1体目のボスを倒した報酬（`act3_boss_reward`）は、途中の報酬に数えていません。第3幕のボス戦では、見込みが少なめに出ます。
- 戦闘中は、ターンの終わりまで進めると、打つ順番が違うだけの手が同じ局面に行き着きます。そのため、別の手でも見込みがまったく同じになることがよくあります（実機の1局で、774判断のうち435判断）。手の順番の良し悪しは、この先読みでは区別できません。
- 世界の数が少ないと、差が小さいときに順位が入れ替わりやすくなります。実機の1局（4 世界）では、見込みの1位とネットの1位の一致は 57%（444/774）でした。一致しなかった判断の 76% は、差が標準誤差の2倍以内です。
- 時間は「候補の数 × 世界の数 × 先読みの手数」に比例します。実機（WSL、4 世界）では、1判断あたり平均 0.75 秒、最大 5.4 秒でした。1局まるごと（774判断）では約10分です。

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
8. **（任意、使わなくてよい）局全体の勝率の較正表を作る。** CPU で1局数分かかるので、夜間に回すことをおすすめします。

   ```bash
   ~/AlphaDefect-Public/.venv/bin/python -I tools/hint/hint.py autoplay --character IRONCLAD --ckpt ... \
       --seeds-file $ALPHADEFECT_REPO/seeds/RV1.json --runs 200 --quiet --out out/calib_runs
   python3 -I tools/hint/calibrate.py out/calib_runs/*.jsonl --out out/calib_ironclad_v24782.json
   # 以後 --calib out/calib_ironclad_v24782.json を付ける
   ```

主なオプション: `--screens`（記録・停止する画面を絞る）、`--merge-equiv`（同じ手をまとめた確率で1位を決める）、`--lang jpn,eng`（名前の言語）、
`--req-extra`（reset に足す設定。既定で `loop_cut`、`stall_turns: 20`、重みの `obs_format` を付けます）、`--threads`（CPU のスレッド数）。

## 実機で確かめたこと（2026-11-24、Windows 11 上で直接、ゲーム v0.113.1、ironclad_ema_v24782.pt）

- ビルドと selfcheck は通り、重みの SHA-256 も一致しました。
- ヒントツールはシード 24VQ8UQR7J の1局を最後まで打てました。結果はクリア（第3幕49階、904手、839判断）、CPU で15秒でした。
- 日本語の翻訳フォルダは `jpn` でした。カード名、敵の名前、画面名が日本語で出ます。
- この重みの報酬設定は次のとおりでした。
  - `floor_reward` 0.02、`act1_boss_reward` 0.1、`act2_boss_reward` 0.5、`act3_boss_reward` 1.5、`victory_reward` 2.0、`act3_boss2_reward` 2.0
  - `milestone_from` はなく、里程碑報酬が残っています（ms_scale = 1）。`run_hp_shaping` は 0 です。
  - このため、価値から局全体の勝率は計算できません。勝率は既定で出さないことにしました（較正表を作れば出せます。B の手順8）。
- 終局フレームのヘッダの HP は、クリアしても 0 でした。そこで `result` には、そのままの値（`hp_end_frame`）と、最後の判断のときの HP（`hp_last_decision`）の両方を書くようにしました。

### Windows で直接動かす場合

- ヒントツールは動きます。.NET 9 は `%USERPROFILE%\.dotnet` に、ゲームファイルは `%USERPROFILE%\.local\share\sls2\game\v0.113.1` に置きます。
- 本家の `tools/ability_test.py` は動きません。パイプを `select()` で待つ作りで、Windows では使えないためです（`WinError 10038`）。本家との一致確認（B の手順6）には WSL が要ります。
- AlphaDefect-Public の `data/training/stats_archive/` に、名前にコロンを含むファイルが1つあり、Windows では取り出せません。学習記録なので、動作には関係ありません。

## まだ確認できていないこと

- **本家の評価（`ability_test.py`）と結果が一致するかは、まだ確かめていません。** WSL で手順6を行います。
- **候補の文字列（engine_label）を付けたまま打っています。** エンジンは文字列を末尾に足すだけなので、フレームの中身は変わらないはずです。これも手順6の一致で確認できます。
- **格式 29 は試していません。** アイアンクラッドの v24782 は格式 28 で学習されているので、ツールは重みの `obs_format`（28）をエンジンに要求します。
- **戦闘補助ヘッドと先読み予測は、学習の補助として付いていたヘッドです。** どのくらい当たるかは測っていません。
- 候補ごとの見込み（`--lookahead`）は、実機の1局（シード 24VQ8UQR7J、4 世界）で最後まで動きました。エラーはなく、選んだ手839件と結果は先読みなしのときとすべて同じでした。先読みの値が本家のクローン世界の評価とどのくらい合うかは、まだ比べていません。
