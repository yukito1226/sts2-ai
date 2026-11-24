"""AlphaDefect のヒント生成（Step 1）。探索なし、CPU で動く。

ヘッドレスエンジンで1局を進め、各判断で次を出す:
  判断の種類と画面、候補ごとの名前・方策の確率・順位、価値ヘッドの値と勝率の推定（特権入力はゼロ）、
  戦闘補助ヘッド（この戦闘に勝つ確率など）、候補ごとの先読み予測、エンジンが計算した事実（修正値込みのダメージ、ブロック、
  ターンを終えたら失うHP など）。

モード:
  autoplay     1局を最初から最後までネットの判断（貪欲）で進め、全判断を JSONL に書く
  interactive  局面ごとに止めて表示する。Enter でネットの手、番号で好きな手を選べる

例（実機）:
  python3 -I tools/hint/hint.py --ad-repo ~/AlphaDefect-Public autoplay \\
      --character IRONCLAD --ckpt ~/AlphaDefect-Public/weights/files_v6/ironclad_ema_v24782.pt --seed R1TEST --out out/hint/run.jsonl
  python3 -I tools/hint/hint.py --ad-repo ~/AlphaDefect-Public interactive --character IRONCLAD --ckpt ... --seed R1TEST
例（モック、ゲームと重みなし）: tests/test_hint_mock.py を参照
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrepo  # noqa: E402

DECISION_CODES = {"COMBAT": 1, "CARD_SELECT": 2, "CARD_REWARD": 3, "BUNDLE": 4, "REWARDS": 5, "EVENT": 6, "REST_SITE": 7, "SHOP": 8,
                  "TREASURE": 9, "MINIGAME": 10, "MAP": 11}
DEFAULT_REQ = {"loop_cut": True, "stall_turns": 20}     # ability_test の常規評価と同じ（README の例）


class Session:
    def __init__(self, a):
        self.repo = adrepo.setup(a.ad_repo)
        import torch
        torch.set_num_threads(max(1, a.threads))
        import weights
        from engine import Engine, default_cmd, mock_cmd
        from names import Namer
        from record import Builder
        from winprob import WinProb
        from sts2env.gamever import GAME_VERSION, localization_dir
        from sts2rl.catalog import Catalog

        self.a = a
        self.L = weights.load(a.ckpt, self.repo, allow_unlisted=a.allow_unlisted, device=a.device)
        self.vocab = Catalog().vocab
        loc_root = Path(a.loc_dir).expanduser() if a.loc_dir else localization_dir("eng").parent
        self.namer = Namer(self.vocab, loc_root, [x for x in a.lang.split(",") if x])
        self.builder = Builder(self.namer)
        self.wp = WinProb(self.L.config, self.L.version, a.calib)
        self.game_version = GAME_VERSION
        if a.mock:
            cmd, env = mock_cmd()
            env["ALPHADEFECT_REPO"] = str(self.repo)
        elif a.engine_cmd:
            cmd, env = a.engine_cmd.split(), None
        else:
            cmd, env = default_cmd(self.repo)
        log = open(a.engine_log, "ab") if a.engine_log else None
        import subprocess
        self.eng = Engine(cmd, env, log or subprocess.DEVNULL)
        self._engine_args = (cmd, env, log or subprocess.DEVNULL)
        self.max_hp0 = None
        self.steps = 0
        self.last_hp = None
        self.hist: list[int] = []          # これまでに選んだ手（先読み用のエンジンを今の局面に合わせるのに使う）
        self.base_req: dict | None = None
        self._look = None

    def meta(self) -> dict:
        c = self.L.config
        return {"tool": "alphadefect-hint/1", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ckpt": Path(self.L.path).name,
                "sha256": self.L.check["sha256"], "sha256_status": self.L.check["status"], "version": self.L.version,
                "obs_format": self.L.obs_format, "character": self.a.character, "ascension": self.a.ascension, "seed": self.a.seed,
                "game_version": self.game_version, "mock": bool(self.a.mock), "reset_extra": self.req_extra(),
                "localization": self.namer.langs_found, "winprob": self.wp.describe(),
                "model_flags": {k: v for k, v in (c.get("model") or {}).items() if isinstance(v, bool) and v},
                "privileged_inputs": "zeroed", "search": "none",
                "lookahead": ({"worlds": self.a.lookahead, "top": self.a.lookahead_top} if self.a.lookahead else None)}

    def req_extra(self) -> dict:
        extra = dict(DEFAULT_REQ)
        if self.L.obs_format is not None:
            extra["obs_format"] = self.L.obs_format     # 重みを学習した格式に合わせる（アイアンクラッドは 28 が必要）
        if self.a.req_extra:
            extra.update(json.loads(self.a.req_extra))
        return extra

    def reset(self):
        from sts2env.protocol import parse
        req = {"op": "reset", "mode": "run", "character": self.a.character, "ascension": self.a.ascension, "seed": self.a.seed,
               "labels": True, "snapshots": True, "route_summary": True, **self.req_extra()}
        o = parse(self.eng.request(req))
        self.base_req = req
        self.max_hp0 = int(o.max_hp) or None
        self.steps = 0
        self.last_hp = None
        self.hist = []
        return o

    def step(self, action: int):
        from sts2env.protocol import parse
        self.steps += 1
        self.hist.append(int(action))
        return parse(self.eng.request({"op": "step", "action": int(action)}))

    def lookahead(self, o, rec: dict, worlds: int | None = None) -> dict:
        """候補ごとの「その手を選んだあとの見込み」を rec の各候補の "after" に入れる（lookahead.py）。"""
        from engine import Engine
        from infer import evaluate
        from lookahead import Lookahead, LookaheadConfig
        if self._look is None:
            self._look = Lookahead(Engine(*self._engine_args), self.L.net, self.wp, self.base_req,
                                   LookaheadConfig(worlds=self.a.lookahead or 4, top=self.a.lookahead_top), self.a.device)
        if worlds:
            self._look.cfg.worlds = worlds
        self._look.base = self.base_req
        out = evaluate(self.L.net, o, self.a.device)
        t0 = time.perf_counter()
        from sts2rl.greedy import equivalence_classes
        res = self._look.run(o, out.probs, out.value, self.hist, len(self.hist), equivalence_classes(o))
        for c in rec["candidates"]:
            if c["index"] in res:
                c["after"] = res[c["index"]]
        rec["lookahead"] = {"worlds": self._look.cfg.worlds, "candidates": len(res), "ms": round((time.perf_counter() - t0) * 1000)}
        return rec

    def hint(self, o) -> dict:
        from infer import evaluate
        from sts2rl.greedy import equivalence_classes
        out = evaluate(self.L.net, o, self.a.device)
        self.last_hp = int(o.hp)
        if self.max_hp0 is None:
            self.max_hp0 = int(o.max_hp) or 1
        # 局全体の勝率は、ネットが直接出す値ではない。較正表（--calib）か、概算を明示的に頼んだとき（--winprob-estimate）だけ出す
        wp = (self.wp.estimate(out.value, int(o.act), float(o.hp), float(o.max_hp), float(self.max_hp0))
              if (self.a.calib or self.a.winprob_estimate) else None)
        classes = equivalence_classes(o) if o.num_actions else None
        rec = self.builder.build(o, out, wp, merge_classes=classes)
        rec["step"] = self.steps
        if self.a.merge_equiv and classes:
            from sts2rl.greedy import merged_argmax
            rec["greedy"] = int(merged_argmax(o, out.probs))
        return rec

    def close(self):
        self.eng.close()
        if self._look is not None:
            self._look.eng.close()


def wanted(rec: dict, screens: set[int] | None, only_multi: bool) -> bool:
    if only_multi and rec["decision"]["num_candidates"] <= 1:
        return False
    return screens is None or rec["decision"]["kind"] in screens


def parse_screens(s: str | None) -> set[int] | None:
    if not s:
        return None
    out = set()
    for x in s.split(","):
        x = x.strip().upper()
        if x:
            if x not in DECISION_CODES:
                raise SystemExit(f"--screens に知らない画面名: {x}（使えるもの: {', '.join(DECISION_CODES)}）")
            out.add(DECISION_CODES[x])
    return out


def finished(o) -> bool:
    return bool(o.done or o.stuck or o.num_actions == 0)


def result_of(o, steps: int, last_hp: int | None = None) -> dict:
    # 終局フレームのヘッダの HP は、クリアしても 0 になることがある（実機の v0.113.1 で確認。エンジンがプレイヤーの CurrentHp をそのまま出すため）。
    # そこで、最後の判断のときの HP を hp_last_decision として別に残す
    return {"result": {"win": bool(o.victory and not o.stuck), "done": bool(o.done), "stuck": bool(o.stuck), "act": int(o.act) + 1,
                       "floor": int(o.floor), "hp_end_frame": int(o.hp), "hp_last_decision": last_hp, "steps": steps}}


# ---------------------------------------------------------------- autoplay

def play_one(S: Session, out: Path, screens, a) -> dict:
    look_screens = parse_screens(a.lookahead_screens)
    n_rec = 0
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": S.meta()}, ensure_ascii=False) + "\n")
        o = S.reset()
        while not finished(o):
            rec = S.hint(o)
            act = rec["greedy"]
            if wanted(rec, screens, not a.all_decisions):
                if a.lookahead and rec["decision"]["num_candidates"] > 1 and wanted(rec, look_screens, True):
                    S.lookahead(o, rec)
                rec["chosen"] = act
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
                if not a.quiet:
                    print(one_line(rec), flush=True)
            o = S.step(act)
            if S.steps > a.max_steps:
                print(f"手数が上限（{a.max_steps}）を超えたので止めます", flush=True)
                break
        res = result_of(o, S.steps, S.last_hp)
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    r = res["result"]
    print(f"{S.a.seed}: {'クリア' if r['win'] else '敗北/停止'}、第{r['act']}幕 {r['floor']}階、{S.steps}手、記録 {n_rec} 件、"
          f"{time.time() - t0:.0f} 秒 → {out}", flush=True)
    return r


def autoplay(a) -> None:
    S = Session(a)
    screens = parse_screens(a.screens)
    try:
        if a.seeds_file:
            # 複数のシード：--out はフォルダ。<out>/<シード>.jsonl に1局ずつ書く（較正表を作るためのログ集め用）
            seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))
            seeds = (seeds["seeds"] if isinstance(seeds, dict) else seeds)[a.first:a.first + a.runs]
            wins = 0
            for sd in seeds:
                a.seed = sd
                dst = Path(a.out) / f"{sd}.jsonl"
                if dst.exists() and not a.overwrite:
                    continue
                wins += int(play_one(S, dst, screens, a)["win"])
            print(f"{len(seeds)} 局中 {wins} 勝（この実行で打った分）", flush=True)
        else:
            play_one(S, Path(a.out), screens, a)
    finally:
        S.close()


def one_line(rec: dict) -> str:
    d, net, st = rec["decision"], rec["net"], rec["state"]
    top = rec["candidates"][0] if rec["candidates"] else None
    wp = net.get("win_prob")
    cw = net.get("combat_win_prob")
    s = f"[{rec['step']:4d}] 第{st['act']}幕{st['floor']:2d}階 HP{st['hp']}/{st['max_hp']} {d['screen']:<6} " \
        f"価値 {net['value']:+.3f}" + (f" 勝率~{wp:.0%}" if wp is not None else "") + (f" 戦闘勝率 {cw:.0%}" if cw is not None else "")
    if top:
        s += f" → {top['text']} ({top['prob']:.0%})"
    return s


# ---------------------------------------------------------------- interactive

HELP = """  Enter  ネットの手（1位）で進む       番号   その候補で進む（index）
  l [N]  候補ごとの見込みを計算（N 世界） j      この局面の JSON を表示
  s      盤面を表示
  a N    ネットの手で N 判断ぶん進む     q      終わる
  e [質問]  Claude に解説させる（ANTHROPIC_API_KEY が要る。l のあとなら見込みも渡す）
  ? 質問    解説への聞き返し（例 ? なぜ防御じゃないの？）"""


def show(rec: dict, top_n: int) -> None:
    d, net, st = rec["decision"], rec["net"], rec["state"]
    print("=" * 78)
    print(f"手 {rec['step']}  第{st['act']}幕 {st['floor']}階  HP {st['hp']}/{st['max_hp']}  ゴールド {st['gold']}  "
          f"画面: {d['screen']}（{'戦闘' if d['phase'] == 'combat' else '戦闘外'}）")
    if st.get("in_combat"):
        pl = st.get("player", {})
        print(f"  エネルギー {st.get('energy')}/{st.get('max_energy')}  ブロック {pl.get('block', 0)}  ターン {st.get('turn')}")
        for e in st.get("enemies", []):
            atk = e.get("attack")
            print(f"  敵 {e['name']}: HP {e['hp']}/{e['max_hp']} ブロック {e['block']}"
                  + (f"  攻撃 {atk['per_hit']}×{atk['hits']}（計 {atk['total']}）" if atk else ""))
    line = f"  価値 {net['value']:+.4f}"
    if net.get("win_prob") is not None:
        line += f"  勝率（{'較正済み' if net['win_prob_method'] == 'calibrated' else '未較正の概算'}）{net['win_prob']:.1%}"
    if net.get("combat_win_prob") is not None:
        line += f"  この戦闘に勝つ {net['combat_win_prob']:.1%}  戦闘終了までの被HP 最大HPの {net['combat_hp_loss_frac']:.0%}"
    print(line)
    print(f"  {'順':>2} {'idx':>3} {'確率':>6}  候補")
    for c in rec["candidates"][:top_n]:
        f = c["facts"]
        bits = []
        if "damage" in f:
            dm = f["damage"]
            bits.append(f"{dm['per_hit']}×{dm['hits']}={dm['total']}" + ("（倒せる）" if dm["kills"] else ""))
        if "block" in f:
            bits.append(f"ブロック{f['block']}")
        if f.get("hp_loss_if_end_turn_after") is not None:
            bits.append(f"この後ターン終了で-{f['hp_loss_if_end_turn_after']:g}HP")
        if "group_prob" in c:
            bits.append(f"同じ手の合計 {c['group_prob']:.0%}")
        if "after" in c:
            af = c["after"]
            bits.append(f"見込み {af['value_delta']:+.3f}±{af['value_se']:.3f}" + (" ★" if af.get("best_by_lookahead") else "")
                        + (f" 死亡{af['died_frac']:.0%}" if af.get("died_frac") else ""))
        print(f"  {c['rank']:>2} {c['index']:>3} {c['prob']:>6.1%}  {c['text']}" + (f"   [{', '.join(bits)}]" if bits else ""))
    if len(rec["candidates"]) > top_n:
        print(f"  …ほか {len(rec['candidates']) - top_n} 件")


def interactive(a) -> None:
    S = Session(a)
    screens = parse_screens(a.screens)
    look_screens = parse_screens(a.lookahead_screens)
    fh = open(a.out, "w", encoding="utf-8") if a.out else None
    if fh:
        fh.write(json.dumps({"meta": S.meta()}, ensure_ascii=False) + "\n")
    print(HELP)
    auto_left = 0
    ex = None           # 解説の会話（explain.py）。局面が変わったら新しく始める
    try:
        o = S.reset()
        while not finished(o):
            rec = S.hint(o)
            stop = wanted(rec, screens, True) and auto_left <= 0
            act = rec["greedy"]
            if stop:
                if ex is not None:
                    ex.messages = []      # 前の局面の会話は持ち越さない
                if a.lookahead and wanted(rec, look_screens, True):
                    S.lookahead(o, rec)
                show(rec, a.top)
                while True:
                    try:
                        cmd = input("> ").strip()
                    except EOFError:
                        cmd = "q"
                    if cmd == "":
                        break
                    if cmd == "q":
                        raise KeyboardInterrupt
                    if cmd == "l" or cmd.startswith("l "):
                        try:
                            nw = int(cmd[1:].strip() or a.lookahead or 4)
                        except ValueError:
                            print("l の後には世界の数を書いてください")
                            continue
                        print(f"先読み中（{nw} 世界）…", flush=True)
                        S.lookahead(o, rec, nw)
                        show(rec, a.top)
                        print(f"  見込み＝その手のあと（戦闘はターンの終わりまで、戦闘外は1手）の価値の、今の価値からの差。±は標準誤差。"
                              f"★＝見込みが最大（{rec['lookahead']['ms']} ms）")
                        continue
                    if cmd == "e" or cmd.startswith("e "):
                        from explain import Explainer
                        if ex is None:
                            ex = Explainer(model=a.explain_model, effort=a.explain_effort, top=a.top,
                                           fallbacks=not a.explain_no_fallbacks)
                        rec.setdefault("glossary", S.builder.glossary(o))
                        ex.start(rec, cmd[1:].strip() or None)
                        continue
                    if cmd.startswith("?") or cmd.startswith("？"):
                        q = cmd[1:].strip()
                        if ex is None or not ex.messages:
                            print("先に e で解説を始めてください")
                        elif not q:
                            print("? の後に質問を書いてください")
                        else:
                            ex.ask(q)
                        continue
                    if cmd == "j":
                        print(json.dumps(rec, ensure_ascii=False, indent=1))
                        continue
                    if cmd == "s":
                        print(json.dumps(rec["state"], ensure_ascii=False, indent=1))
                        continue
                    if cmd.startswith("a"):
                        try:
                            auto_left = int(cmd[1:].strip() or "1")
                        except ValueError:
                            print("a の後に数を書いてください")
                            continue
                        break
                    if cmd.isdigit() and int(cmd) < rec["decision"]["num_candidates"]:
                        act = int(cmd)
                        break
                    print(HELP)
            elif wanted(rec, screens, True):
                auto_left -= 1
            rec["chosen"] = act
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            o = S.step(act)
        print(json.dumps(result_of(o, S.steps, S.last_hp), ensure_ascii=False))
        if fh:
            fh.write(json.dumps(result_of(o, S.steps, S.last_hp), ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("終了します")
    finally:
        if fh:
            fh.close()
        S.close()


# ---------------------------------------------------------------- 引数

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ad-repo", default=None, help="AlphaDefect-Public のチェックアウト（環境変数 ALPHADEFECT_REPO でも可）")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("autoplay", "interactive"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", required=True, help="重みファイル（.pt）。SHA-256 を weights/SHA256SUMS_*.txt と照合してから読む")
        p.add_argument("--character", default="IRONCLAD")
        p.add_argument("--ascension", type=int, default=10)
        p.add_argument("--seed", default="HINT0000")
        p.add_argument("--req-extra", default=None, help='reset に足す JSON（例 \'{"obs_format": 28}\'）。既定で loop_cut、stall_turns 20、重みの obs_format を付ける')
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--lang", default="jpn,eng", help="名前に使う翻訳の言語（ゲームの localization のフォルダ名、先にあるものを優先）")
        p.add_argument("--loc-dir", default=None, help="localization フォルダ（既定 ~/.local/share/sls2/game/<版>/res/localization）")
        p.add_argument("--calib", default=None, help="calibrate.py で作った較正表（価値 → 局全体の勝率）。渡したときだけ勝率を出す")
        p.add_argument("--winprob-estimate", action="store_true",
                       help="較正表なしで、重みの報酬設定から局全体の勝率を概算して出す（出せない設定の重みでは出ない。既定では出さない）")
        p.add_argument("--screens", default=None, help="この画面だけ記録・停止する（例 COMBAT,CARD_REWARD,EVENT,SHOP,REST_SITE,MAP）")
        p.add_argument("--merge-equiv", action="store_true", help="同じ手（同じカード2枚など）をまとめた確率で1位を決める（ability_test --merge-equiv と同じ）")
        p.add_argument("--engine-cmd", default=None, help="エンジンを起動するコマンド（既定は engine/Sts2Headless の Release ビルド）")
        p.add_argument("--engine-log", default=None, help="エンジンの標準エラーを書くファイル")
        p.add_argument("--mock", action="store_true", help="モックエンジンを使う（テスト用）")
        p.add_argument("--allow-unlisted", action="store_true", help="チェックサム一覧にない重みも読む（テスト用）")
        p.add_argument("--max-steps", type=int, default=6000)
        p.add_argument("--lookahead", type=int, default=0,
                       help="候補ごとの見込みを、この数の世界（見えない未来を引き直したもの）で計算する。0 は計算しない（対話モードでは l で都度）")
        p.add_argument("--lookahead-top", type=int, default=0, help="見込みを計算する候補の数（確率の高い順。0 は全部）")
        p.add_argument("--lookahead-screens", default=None, help="見込みを計算する画面（例 COMBAT,CARD_REWARD）。既定は全部")
        if name == "autoplay":
            p.add_argument("--out", required=True, help="JSONL の出力先（1行目 meta、各判断1行、最後に result）。--seeds-file のときはフォルダ")
            p.add_argument("--seeds-file", default=None, help="シードの一覧（seeds/R1.json などの {\"seeds\": [...]} か、ただの配列）。続けて何局も打つ")
            p.add_argument("--first", type=int, default=0, help="--seeds-file の何番目から打つか")
            p.add_argument("--runs", type=int, default=10, help="--seeds-file から何局打つか")
            p.add_argument("--overwrite", action="store_true", help="--seeds-file のとき、出力がすでにある局も打ち直す")
            p.add_argument("--all-decisions", action="store_true", help="候補が1つだけの判断も記録する")
            p.add_argument("--quiet", action="store_true")
        else:
            p.add_argument("--out", default=None, help="通った判断を JSONL に書く")
            p.add_argument("--top", type=int, default=12, help="表示する候補の数（解説に渡す候補の数も同じ）")
            p.add_argument("--explain-model", default="claude-opus-5-5", help="解説（e）に使う Claude のモデル")
            p.add_argument("--explain-effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"),
                           help="解説の考える深さ（高いほど遅く、料金も上がる）")
            p.add_argument("--explain-no-fallbacks", action="store_true",
                           help="解説を断られたときに、サーバー側で別のモデルに答え直させない（既定では答え直させる）")
    return ap


def utf8_stdio() -> None:
    """Windows のコンソールやパイプ（cp932）でも、簡体字を含むエンジンの文字列や JSON を書けるようにする。"""
    for f in (sys.stdout, sys.stderr):
        try:
            f.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    a = build_parser().parse_args(argv)
    {"autoplay": autoplay, "interactive": interactive}[a.mode](a)


if __name__ == "__main__":
    main()
