"""自動プレイのログから「価値 → 勝率」の較正表を作る。

価値ヘッドの値は勝率ではないので、実際に何局か打って「その価値の局面から実際に何割クリアしたか」を数える。
幕ごとに価値で並べて区間に分け、各区間の実際の勝率を単調回帰（価値が高いほど勝率が下がらないように）でならす。

  python3 -I tools/hint/hint.py autoplay --ckpt ... --seeds-file ~/AlphaDefect-Public/seeds/RV1.json --runs 200 --quiet --out out/calib_runs
  python3 -I tools/hint/calibrate.py out/calib_runs/*.jsonl --out out/calib_ironclad.json
  python3 -I tools/hint/hint.py interactive --ckpt ... --calib out/calib_ironclad.json

注意:
  - 同じ局の局面どうしは強く相関するので、局数が少ないと表は粗い（目安として 200 局以上）。表には局数と局面数を書いておく。
  - 較正した重みと別の重み・別の格式・探索ありのプレイには使えない（meta の sha256 と obs_format を表に書き、読み込み時には確認しない。使う側で合わせること）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winprob import pav  # noqa: E402


def read_runs(paths: list[str]) -> tuple[list[tuple[float, int, int, str]], list[dict]]:
    """[(価値, 幕(0始まり), 勝ったか, 局ID)], [meta, ...]"""
    rows, metas = [], []
    for p in paths:
        recs, meta, res = [], None, None
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "meta" in r:
                meta = r["meta"]
            elif "result" in r:
                res = r["result"]
            elif "net" in r:
                recs.append(r)
        if res is None or res.get("stuck") or not res.get("done"):
            continue        # 途中で止まった局・詰まった局は数えない
        if meta:
            metas.append(meta)
        for r in recs:
            rows.append((float(r["net"]["value"]), int(r["state"]["act"]) - 1, int(bool(res["win"])), p))
    return rows, metas


def fit(rows: list[tuple[float, int, int, str]], bins: int) -> dict:
    rows = sorted(rows)
    xs_all = [r[0] for r in rows]
    if not rows:
        return {"x": [], "y": [], "n": 0}
    # 局ごとの重み：1局の局面数で割り、どの局も合計 1 になるようにする（長い局に引っぱられないように）
    per_run: dict[str, int] = {}
    for r in rows:
        per_run[r[3]] = per_run.get(r[3], 0) + 1
    k = max(1, min(bins, len(rows)))
    xs, ys, ws = [], [], []
    for b in range(k):
        lo, hi = len(rows) * b // k, len(rows) * (b + 1) // k
        chunk = rows[lo:hi]
        if not chunk:
            continue
        w = [1.0 / per_run[r[3]] for r in chunk]
        sw = sum(w)
        xs.append(sum(r[0] * wi for r, wi in zip(chunk, w)) / sw)
        ys.append(sum(r[2] * wi for r, wi in zip(chunk, w)) / sw)
        ws.append(sw)
    ys = pav(ys, ws)
    return {"x": [round(x, 5) for x in xs], "y": [round(y, 4) for y in ys], "n": len(rows), "runs": len(per_run),
            "value_range": [round(xs_all[0], 4), round(xs_all[-1], 4)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="hint.py autoplay の JSONL（何局分でも）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=20)
    a = ap.parse_args()
    rows, metas = read_runs(a.logs)
    if not rows:
        raise SystemExit("使える局がありません（最後まで打った局のログを渡してください）")
    shas = sorted({m.get("sha256") for m in metas})
    if len(shas) > 1:
        print(f"警告: 違う重みのログが混ざっています: {shas}", flush=True)
    tables = {"all": fit(rows, a.bins)}
    for act in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == act]
        if len({r[3] for r in sub}) >= 20:     # 20 局未満の幕は全幕共通の表を使う
            tables[str(act)] = fit(sub, a.bins)
    runs = len({r[3] for r in rows})
    wins = len({r[3] for r in rows if r[2]})
    out = {"tool": "alphadefect-hint/calibrate-1", "sha256": shas, "obs_format": sorted({m.get("obs_format") for m in metas}, key=str),
           "runs": runs, "wins": wins, "positions": len(rows), "tables": tables}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{runs} 局（{wins} 勝）、{len(rows)} 局面から較正表を作りました → {a.out}")


if __name__ == "__main__":
    main()
