"""AlphaDefect の採集進程（sts2rl/actor.py）の推論を速くするパッチ。AlphaDefect 本体のファイルは書き換えない。

採集進程は 1 本のループで「エンジンの返事を待つ → まとめて前向き計算（net.forward_phi）→ 手を送る」を繰り返す。
本体では、この前向き計算は fp32（TF32）・普通の注意力・コンパイルなしのまま（actor.py:629 でネットを作り、991 行で呼ぶ）。
計測（out/ft_probe_report2.md）では、採集進程の時間の 84% がここだった。

切り替えは環境変数 STS2FT_ACTOR に、カンマ区切りで次を並べる（空か未設定なら何もしない＝本体と同じ動作）。
  bf16     torch.autocast(bfloat16) の中で前向き計算する。出力は float32 に戻す
  fast     net.fast=True（打包変長の注意力。本体の学習側が使っているもの）。cuda のときだけ効く
  packed   組批した配列を dtype ごとに 1 塊にまとめて 1 回で転送し（infer.batch_to_packed）、打包注意力の
           偏移（model.packed_meta）を CPU で先に計算して付ける。fast と一緒に使う
  compile  前向き計算を torch.compile(dynamic=True) する。fast と一緒に指定する（普通の注意力では dynamo が落ちる）。
           STS2FT_RECOMPILE_LIMIT で再コンパイル上限（既定 64）、STS2FT_COMPILE_THREADS でコンパイル用の子プロセス数（既定 1）
  stats    2,000 回ごとに、直近 2,000 回の前向き計算の平均の行数・平均ミリ秒・占める割合を標準エラーに出す
例: STS2FT_ACTOR=bf16,fast,packed

使い方:
  - 学習の起動は tools/finetune/train_fast.py を使う（sts2rl.train.run_actor を run_actor_patched に差し替えてから
    sts2rl.train.main を呼ぶ）。採集進程は multiprocessing の spawn で起動されるので、親でのパッチは子に届かない。
    そこで、子が最初に実行する関数（Process の target）そのものを、このモジュールの run_actor_patched にしている。
    子は spawn のときにこのモジュールを import し直し、そこで apply() してから本体の run_actor を呼ぶ。
  - パッチが効いたことは、子が標準エラーに出す「[fastinfer] 採集進程 rank=… パッチ: …」の行で確認できる。

PPO の比率との関係：採集が記録する行動の対数確率（logp）は、本体と同じく「この前向き計算の出力」から
Categorical で計算される。bf16 にすると logp も bf16 の出力から計算された値になり、選んだ行動と一貫している。
学習側は自分の前向き計算（learner_bf16）で新しい logp を出して比率を取るので、比率の初期値が 1 からわずかにずれる
（ずれの大きさは tools/finetune/check_equiv.py で測る）。
"""
from __future__ import annotations

import os
import sys
import time

ENV = "STS2FT_ACTOR"
KNOWN = ("bf16", "fast", "packed", "compile", "stats")
_STATE: dict = {"applied": None}


def parse_modes(s: str | None) -> frozenset[str]:
    modes = frozenset(m.strip().lower() for m in (s or "").split(",") if m.strip())
    bad = sorted(modes - set(KNOWN))
    if bad:
        raise ValueError(f"{ENV} に知らない指定があります: {bad}（使えるもの: {', '.join(KNOWN)}）")
    if "compile" in modes and "fast" not in modes:
        # 普通の注意力は核心記録の数ごとの Python のループで、torch 2.15 の dynamo はここで落ちる（IndexError、計測 2026-11-25）
        raise ValueError(f"{ENV}: compile は fast と一緒に指定してください")
    return modes


def _log(msg: str) -> None:
    print(f"[fastinfer] {msg}", file=sys.stderr, flush=True)


class _PackedBatch:
    """collate_ragged の結果（features.Batch）を包み、.to(device) で batch_to_packed（＋打包注意力の偏移）を返す。
    actor.py は `collate_ragged(...).to(device)` としか使わないので、.to 以外の属性は元の Batch に回す。"""

    __slots__ = ("_b", "_meta")

    def __init__(self, b, meta: bool):
        self._b, self._meta = b, meta

    def __getattr__(self, k):
        return getattr(self._b, k)

    def to(self, device):
        import torch
        from sts2rl.infer import batch_to_packed
        from sts2rl.model import packed_meta
        b = self._b
        if not str(device).startswith("cuda"):
            return b.to(device)
        if self._meta:
            for k, v in packed_meta(b.core_idx, b.core_mask).items():
                setattr(b, k, v)      # numpy の配列は batch_to_packed がまとめて転送し、int（pk_T, pk_max）はそのまま渡る
        out = batch_to_packed(b, device, pin=True)
        if self._meta:
            for k in ("pk_T", "pk_max"):
                out[k] = int(getattr(b, k))
        return out


def to_device(b, modes: frozenset[str], device):
    """組批の結果（features.Batch）を device に送る。packed なら 1 塊にまとめて転送（fast なら打包注意力の偏移も付ける）。"""
    if "packed" in modes:
        return _PackedBatch(b, "fast" in modes).to(device)
    return b.to(device)


def setup_compile() -> None:
    import torch._inductor.config as ic
    # inductor は既定で CPU 数ぶんのコンパイル用の子プロセスを立てる。採集進程 2 本 × 20 で RAM を数 GB 使い、
    # 12GB の WSL ではスワップした（計測 d3/d4）。既定は 1（子プロセスなし）、STS2FT_COMPILE_THREADS で変えられる
    ic.compile_threads = int(os.environ.get("STS2FT_COMPILE_THREADS", "1"))
    import torch._dynamo
    lim = int(os.environ.get("STS2FT_RECOMPILE_LIMIT", "64"))
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), lim))


def make_forward(orig_phi, modes: frozenset[str], where: str = ""):
    """本体の PolicyValueNet.forward_phi（orig_phi）を包んだ関数 f(net, b) → (logits, values, z)（どれも float32）を返す。"""
    import torch
    use_bf16, use_fast, use_compile, use_stats = ("bf16" in modes), ("fast" in modes), ("compile" in modes), ("stats" in modes)
    if use_compile:
        setup_compile()
    st = {"n": 0, "rows": 0, "t": 0.0, "t0": time.perf_counter()}
    ckey = "_ft_compiled_" + ",".join(sorted(modes))

    def forward_phi_fast(self, b):
        if torch.is_grad_enabled():         # 学習（勾配あり）には使わない：本体のまま
            return orig_phi(self, b)
        dev = b["core_idx"].device
        cuda = dev.type == "cuda"
        fn = orig_phi.__get__(self)
        if use_compile:
            fn = self.__dict__.get(ckey)
            if fn is None:
                fn = torch.compile(orig_phi.__get__(self), dynamic=True)
                self.__dict__[ckey] = fn
        fast_old = self.fast
        t0 = time.perf_counter() if use_stats else 0.0
        try:
            if use_fast and cuda:
                self.fast = True
            if use_bf16:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    out = fn(b)
            else:
                out = fn(b)
        finally:
            self.fast = fast_old
        out = tuple(o.float() for o in out)
        if use_stats:
            if cuda:
                torch.cuda.synchronize(dev)
            st["n"] += 1; st["k"] = st.get("k", 0) + 1
            st["rows"] += int(b["core_idx"].shape[0]); st["t"] += time.perf_counter() - t0
            if st["n"] % 2000 == 0:      # 直近 2,000 回の値（コンパイル直後の遅い分が後まで平均に残らないように）
                wall = time.perf_counter() - st["t0"]
                _log(f"{where} 前向き {st['n']} 回目まで：直近 {st['k']} 回の平均 {st['rows'] / st['k']:.1f} 行、"
                     f"{1000 * st['t'] / st['k']:.2f} ms、前向きの占める割合 {100 * st['t'] / wall:.0f}%")
                st.update(k=0, rows=0, t=0.0, t0=time.perf_counter())
        return out

    return forward_phi_fast


def apply(modes: frozenset[str] | str | None = None, where: str = "") -> frozenset[str]:
    """パッチを当てる（同じプロセスで 2 回目以降は何もしない）。当てた指定を返す。空なら何もしない。"""
    if isinstance(modes, str) or modes is None:
        modes = parse_modes(os.environ.get(ENV) if modes is None else modes)
    if _STATE["applied"] is not None:
        return _STATE["applied"]
    _STATE["applied"] = modes
    if not modes:
        return modes
    import torch
    from sts2rl import actor as A
    from sts2rl.model import PolicyValueNet

    if "packed" in modes:
        orig_collate = A.collate_ragged
        meta = "fast" in modes

        def collate_packed(*a, **k):
            return _PackedBatch(orig_collate(*a, **k), meta)
        A.collate_ragged = collate_packed

    PolicyValueNet.forward_phi = make_forward(PolicyValueNet.forward_phi, modes, where)
    _log(f"{where} パッチ: {','.join(sorted(modes))}（torch {torch.__version__}、pid {os.getpid()}）")
    return modes


def run_actor_patched(rank, cfg, weights_path, out_queue, stop_event):
    """sts2rl.actor.run_actor の代わりに Process の target にする関数。子の中でパッチを当ててから本体を呼ぶ。"""
    modes = apply(where=f"採集進程 rank={rank}")
    if os.environ.get("STS2FT_DRYRUN"):     # テスト用：パッチが子で効いたかだけを親に知らせて終わる
        from sts2rl.model import PolicyValueNet
        out_queue.put(("fastinfer", rank, sorted(modes), PolicyValueNet.forward_phi.__name__, os.getpid()))
        return None
    from sts2rl.actor import run_actor
    return run_actor(rank, cfg, weights_path, out_queue, stop_event)
