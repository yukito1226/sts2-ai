# Sts2Hint：ゲーム画面に AlphaDefect のヒントを出す Mod（Step 3）

Slay the Spire 2 のプレイ中に、学習済みの AI「AlphaDefect」が各候補を選ぶ確率と順位、価値、この戦闘に勝つ確率を、画面の右上に出します。
ゲームの状態は読むだけで、変えません（`affects_gameplay: false`）。

**ヒントが出る場面**：ヘッドレスエンジンで AI が判断するすべての画面です。

| 画面 | 読み取り元 |
|---|---|
| 戦闘（自分のターン） | 手札・ポーション・敵（ゲームの状態） |
| 戦闘中の選択（手札から選ぶ「武装」など、山札・捨て札から選ぶ画面、3枚から1枚） | 手札の選択モード（`NPlayerHand`）、選牌画面（`NCardGridSelectionScreen` 系、`NChooseACardSelectionScreen`） |
| 地図の移動先 | 地図画面が開いていて移動できるとき |
| 報酬画面・カード報酬・カードパック | `NRewardsScreen`、`NCardRewardSelectionScreen`、`NChooseABundleSelectionScreen` |
| デッキから選ぶ画面（強化・変化・削除・付与など） | 選牌画面（`NDeck…SelectScreen`） |
| 休息所 | `RestSiteSynchronizer` の選択肢 |
| ショップ（偽商人も） | 商品の一覧（`MerchantInventory`） |
| イベント | 今の選択肢（`EventModel.CurrentOptions`） |
| 宝箱 | 開けたあとの遺物（`TreasureRoomRelicSynchronizer`） |
| Crystal Sphere | ミニゲームの画面 |

デッキ一覧などの表示だけの画面を開いているあいだは「未対応の画面（画面の名前）」と出ます。閉じれば元に戻ります。

## 仕組み

```
ゲーム（Mod: Sts2Hint）                                    同じ PC の Python（tools/hint/server.py）
  今の状態（RunState）
   → 候補の一覧（LiveDecision.cs）
   → 観測フレーム（AlphaDefect のエンコーダそのもの）  ──127.0.0.1──→  ネット（探索なし、CPU、特権入力はゼロ）
  右上のパネルに表示  ←────────────────────────────── JSON ──  Step 1 と同じ形のヒント（record.py）
```

- **観測は AlphaDefect と同じコードで作ります。** `engine/Sts2Headless/Engine/` の `Observation.cs`、`Obs/**`、`Compat/**` などを、書き換えずにそのままコンパイルに取り込みます。
  本家の作者も配信で同じやり方（非公開の `Sts2Bridge` Mod、`engine/Directory.Build.targets` にその取り込み設定が残っています）をとっていて、
  ゲーム内で作ったフレームとヘッドレスエンジンのフレームは、実機 132 局・88,473 フレームで 99.891% がバイト単位で一致したと記録されています（`tools/clone/live_fidelity.py`）。
- エンコーダが参照するヘッドレスエンジン側の部品（`RunSession` など）は、`Code/Shims.cs` に最小限の代役を置いています。
- **候補の並びは、ヘッドレスエンジンの `RunSession.CombatDecision` / `MapDecision` と同じ順番で作ります**（`Code/LiveDecision.cs`）。順番や対象の付け方が違うとネットの入力が変わるためです。
- **候補の並びと作り方は、画面ごとにヘッドレスエンジン（`RunSession.cs`、`RewardsFlow.cs`）と同じにしています**（`Code/LiveDecision.cs`）。ゲームの画面のノードから、今出ている選択肢を読むだけです。
- 選牌の用途（見出しの文言、どのコマンドから開いたか、出どころのカード・ポーション）は、ヘッドレスの `HeadlessFixes.cs` と同じ場所で記録します（`Code/Hooks.cs`）。Harmony の前置き・後置きで値を読むだけで、ゲームの処理は変えません。
- 戦闘では、自分のターンで、行動の処理が終わっている（アクションキューが空）ときだけ送ります。局面が変わったとき（フレームのバイト列が変わったとき）だけ送るので、何度も計算しません。
- torch などはゲームに入れません。ネットの計算はヒントサーバー（Python）が行います。サーバーは 127.0.0.1 でしか待ち受けません。
- 局全体の勝率は出しません（価値が較正されていないため）。戦闘の勝率は戦闘補助ヘッドの値です。

## 使い方（Windows）

前提：Step 1 のヒントツールが Windows で動く状態（`tools/hint/README.md` の B。.NET 9、ゲームファイルと翻訳を `%USERPROFILE%\.local\share\sls2\game\v0.113.1`、重みは `weights/files_v6`）。
ゲームは **v0.113.1**、キャラは重みと同じ（例：アイアンクラッド）。

1. **Mod をビルドして入れる。** `mods\Sts2Hint\` に `Sts2Hint.dll`、`Sts2Hint.json`、`vocab.txt`、`aliases.txt`、`no_downside_relics.txt` がコピーされます。ほかの Mod やゲームのファイルは触りません。

   ```powershell
   cd C:\work\sts2-ai\tools\hintmod
   dotnet build -c Release
   # 場所が違うとき: dotnet build -c Release -p:Sts2Path="D:/SteamLibrary/steamapps/common/Slay the Spire 2" -p:AlphaDefectRepo=C:/work/AlphaDefect-Public
   ```
2. **ヒントサーバーを起動する**（ゲームより先でも後でもかまいません。Mod は3秒ごとにつなぎ直します）。

   ```powershell
   cd C:\work\sts2-ai
   $env:ALPHADEFECT_REPO = "C:\work\AlphaDefect-Public"
   C:\work\AlphaDefect-Public\.venv\Scripts\python.exe -I tools\hint\server.py --ckpt C:\work\AlphaDefect-Public\weights\files_v6\ironclad_ema_v24782.pt --dump out\live
   ```
   torch などは AlphaDefect-Public の仮想環境（`.venv`）に入っているので、その Python で起動します。素の `python` だと「No module named 'torch'」になります。
   `ヒントサーバー 127.0.0.1:8765 …` と出れば準備完了です。判断ごとに1行（画面、候補数、1位、かかった時間）が出ます。`--dump` は確認用の記録（`frames.bin` と `hints.jsonl`）で、なくてもかまいません。
3. **ゲームを起動して、アイアンクラッドで始める。** 右上にパネルが出ます。F8 で表示・非表示を切り替えます。
   パネル下の行に「サーバー: ironclad_ema_v24782.pt」と出ていれば接続できています。

設定（任意）：`mods\Sts2Hint\Sts2Hint.config.json` に `{"Port": 8765, "TopN": 6, "IntervalSec": 0.25}` のように書くと、ポート、表示する候補の数、確認の間隔を変えられます。

## テスト（ゲームと重みは不要）

```bash
ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest tests.test_server -v
```

Mod の代わりにテストがソケットでつなぎ、モックエンジンのフレームを Mod と同じ形式で送ります。確かめていること：あいさつ、返事が直接計算した結果と同じこと、通し番号、
壊れたフレームでも接続が切れないこと、長すぎる長さで切ること、127.0.0.1 以外では起動しないこと、記録の形式。

Mod（C#）は、v0.113.1 の `sts2.dll`、`GodotSharp.dll`、`0Harmony.dll` に対してビルドが通ることまで確かめています（警告 0、エラー 0）。ゲームの中では、まだ動かしていません。

## 実機で確認する手順

1. 上の「使い方」で、戦闘に入ってパネルに候補が出ることを確かめる。カードを1枚使うたびに、パネルが新しい局面に変わること。
2. **候補が実際の手と合っているか。** 手札で使えるカード×対象、使えるポーション×対象、ターン終了が並ぶはずです。1位の手を実際に選び続けて、おかしな手（使えないカードなど）が出ないか見てください。
3. **地図。** 戦闘のあと地図を開いたとき、行ける場所の候補が出ること。
4. **ほかの画面。** 報酬画面（ゴールドを取ったあと）、カード報酬、休息所、ショップ、イベント、宝箱、戦闘中の選択（「武装」など）、デッキから選ぶ画面で、
   画面の選択肢とパネルの候補が合っていること。合わない画面があれば、その画面の名前とサーバーのログ（`out\live\hints.jsonl` の最後の行）を教えてください。
5. **ヘッドレスエンジンとの一致（任意、WSL）。** `out\live\hints.jsonl` の候補の `engine_label`（`出 STRIKE_IRONCLAD → …` など）が、Step 1 のヒントツールで出る形と同じであること。
   バイト単位の一致は、本家の `live_fidelity.py` と同じく「同じシードで、同じ手を打ったヘッドレスエンジン」と比べる必要があり、まだ道具を作っていません。

## ヘッドレスとの違い

ヘッドレスエンジンは画面を自分で動かしますが、この Mod はゲームを操作しません。そのため、次の点が違います。

- **ゴールドと、マイナスのない遺物は、ヘッドレスでは自動で取られます**（`RewardsFlow` の AutoGold / AutoRelics、`no_downside_relics.txt`）。ネットはそれらを取ったあとの報酬画面しか見ていません。
  報酬画面にまだ残っているときは「先にゴールドと遺物を取ってください」と出して、ヒントは出しません。宝箱の中身がマイナスのない遺物1つだけのときも同じです。
- 報酬画面で、開いて取らずに閉じたカード報酬は、ヘッドレスでは二度と候補になりません。ここでは画面で「スキップ済み」になった報酬を同じ扱いにしています。
- 複数枚選ぶ画面で、ヘッドレスは選んだ順番を覚えていますが、ゲームの画面は選んだカードの集合しか持っていません（手札の選択は順番も持っています）。デッキから選ぶ画面では、選んだ順番の代わりに候補の並び順を使います。
- ヘッドレスは、休息所の選択をキャンセルしたり、ショップで買えなかったりした選択肢を、その部屋では候補から外します。休息所は同じようにしています（`AfterPlayerOptionChosen` で失敗を記録）。ショップの購入失敗はまれなので、外していません。
- 強化の選牌（休息所の鍛冶など）は、ヘッドレスではキャンセルできないので、候補に「取消」がありません。ゲームではキャンセルできますが、ヒントは出しません。
- 部屋ごとに1回だけキャンセルできる、というヘッドレスの決まりは入れていません。

## まだ確認できていないこと・制限

- **ゲームの中ではまだ動かしていません。** Mod の読み込み、`SceneTree.ProcessFrame` での更新、日本語フォントの読み込み、F8、各画面の読み取り（非公開フィールドの名前は v0.113.1 の `sts2.dll` で確かめただけ）は、実機での確認が要ります。
- 戦闘のヘッドレスとの一致は、作者の記録（99.891%）に頼っています。このModの候補作りが作者の非公開 Mod と同じかは、確かめようがありません（ヘッドレスエンジンと同じ規則にはしてあります）。
- Mod を読み込む前に開いた3枚から1枚の画面は、候補を読めません（開き直せば読めます）。
- マルチプレイは対象外です（1人用だけ）。
- 先読み（手ごとの見込み）と Claude の解説は、ゲーム内ではまだ出ません。
- 1つのサーバーで1つの重み（1キャラ）です。別のキャラで遊ぶと「この重みは … 用です」と出ます。
