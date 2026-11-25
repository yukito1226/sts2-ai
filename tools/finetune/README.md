# tools/finetune：追加学習（PPO の再開）の採集を速くする試み

AlphaDefect の追加学習で、採集進程の推論を速くするためのパッチと、その確認用の道具です。AlphaDefect 本体のファイルは書き換えません。
計測の結果は `out/ft_probe_report3.md` にあります（`out/` は git に入れていません）。

**結論を先に書くと、本番では今のところ既定（パッチなし）を推奨します。** 単独で測ると `fast,packed,compile` は 1.7 倍速く、答えもほぼ同じでした。ところが学習中は再コンパイルが止まらず、速さも安定しませんでした（レポート 3 を参照）。

| ファイル | 内容 |
|---|---|
| `fastinfer.py` | パッチ本体。環境変数 `STS2FT_ACTOR` で切り替える（`bf16` / `fast` / `packed` / `compile` / `stats`）。空なら本体と同じ動作 |
| `train_fast.py` | 起動用。`sts2rl.train.run_actor` を差し替えてから `sts2rl.train.main` を呼ぶ |
| `check_equiv.py` | 同じ観測の批で、本体の fp32 と高速版の候補確率・対数確率・価値を比べ、前向き計算の時間も測る |
| `../../tests/test_finetune.py` | 単体テスト（CPU、モック。spawn の子でパッチが効くことも確かめる） |

## 使い方（WSL）

```bash
# 学習（本体の python/train_defect.py と同じ引数。--actor-infer を付けなければ本体と同じ）
ALPHADEFECT_REPO=~/AlphaDefect-Public ~/ft_probe/venv/bin/python -I tools/finetune/train_fast.py \
    --actor-infer fast,packed,compile,stats --config ~/ft_probe/cfg.json

# 正しさと速さの確認（エンジンでランダムに打って 2,048 局面を集める）
ALPHADEFECT_REPO=~/AlphaDefect-Public ~/ft_probe/venv/bin/python -I tools/finetune/check_equiv.py \
    --weights ~/AlphaDefect-Public/weights/files_v6/ironclad_ema_v24782.pt --batch 16,32,r16 --out equiv.json

# テスト（GPU・重み・ゲームは不要）
ALPHADEFECT_REPO=~/AlphaDefect-Public python3 -m unittest tests.test_finetune -v
```

## 仕組み

- 採集進程は `multiprocessing` の spawn で起動されるので、親で当てたパッチは子に届きません。そこで `train_fast.py` は
  `sts2rl.train` の中の名前 `run_actor` を `fastinfer.run_actor_patched` に差し替えます。本体は `ctx.Process(target=run_actor)` で子を起動し、
  子はこの関数から始まって、自分の中でパッチを当ててから本体の `run_actor` を呼びます。
  パッチが効いたことは、子が標準エラーに出す `[fastinfer] 採集進程 rank=0 パッチ: …` の行で確かめられます。
- パッチが変えるのは、採集進程の `PolicyValueNet.forward_phi`（勾配なしの呼び出しのとき）と、`sts2rl.actor.collate_ragged`（`packed` のとき）だけです。
  学習進程の前向き計算（`forward_all`）には触りません。
- 行動の対数確率は、本体と同じく、このパッチ後の前向き計算の出力から計算されます（選んだ行動と一貫しています）。

## 注意

- `compile` は `fast` と一緒にしか使えません。普通の注意力は核心記録の数ごとの Python のループで、torch 2.15 の dynamo はここで落ちます。
- `compile` では、`model.py` の `_split` などでグラフが切れます。そのうえ採集の批は形が毎回変わるので、学習中は何分たっても再コンパイルが続きました（py-spy で確認）。
- `bf16` は、このネットと小さな批（〜16 行）ではかえって遅く（autocast の型変換の手間）、top1 の一致率も 98.8〜99.1% に下がりました。
