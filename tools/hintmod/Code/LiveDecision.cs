// ゲーム内の今の状態から、AlphaDefect のエンコーダに渡す Decision（候補の一覧）を作る。
// 候補の作り方と並び順は、ヘッドレスエンジンの RunSession.CombatDecision / MapDecision / UsablePotions / AddPotionActions（MIT）と同じにする。
// 候補の順番や対象の付け方が違うと、同じ局面でもネットの入力が変わるため。
// 違い：Run（実行する処理）は使わないので何もしない。ヘッドレス専用の打ち切り（stall_turns、loop_cut）は入れない。
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace Sts2Hint;

public static class LiveDecision
{
    static readonly Action Nop = () => { };

    /// <summary>今がヒントを出せる判断の場面なら Decision、そうでなければ null と理由。</summary>
    public static Decision? Current(out string why)
    {
        why = "";
        var rm = RunManager.Instance;
        var s = rm?.DebugOnlyGetState();
        if (s == null) { why = "ラン中ではありません"; return null; }
        if (s.Players.Count != 1) { why = "マルチプレイには対応していません"; return null; }
        var me = s.Players[0];
        if (s.IsGameOver) { why = "ランが終わりました"; return null; }

        bool inCombat = CombatManager.Instance.IsInProgress && me.PlayerCombatState != null;
        if (inCombat) return Combat(me, out why);

        if (NMapScreen.Instance is { IsOpen: true, IsTravelEnabled: true, IsTraveling: false })
            return Map(s, me, out why);

        why = "未対応の画面（カード報酬・ショップなどは次の段階で対応）";
        return null;
    }

    static bool Busy()
    {
        var rm = RunManager.Instance;
        return !rm.ActionQueueSet.IsEmpty || rm.ActionExecutor.IsRunning;
    }

    static Decision? Combat(Player me, out string why)
    {
        why = "";
        var pcs = me.PlayerCombatState!;
        if (pcs.Phase != PlayerTurnPhase.Play) { why = "自分のターンの操作待ちではありません"; return null; }
        if (Busy()) { why = "行動の処理中"; return null; }
        if (NOverlayStack.Instance is { ScreenCount: > 0 }) { why = "選択画面が開いています（次の段階で対応）"; return null; }
        var cs = me.Creature.CombatState!;

        var d = new Decision(DecisionKind.Combat) { Context = $"第 {pcs.TurnNumber} 回合" };
        var targets = new List<Creature?> { null };
        targets.AddRange(cs.Creatures.Where(c => c.IsAlive));
        foreach (CardModel c in PileType.Hand.GetPile(me).Cards)
        {
            if (!c.CanPlay(out _, out _)) continue;
            foreach (var t in targets)
                if (c.CanPlayTargeting(t))
                    d.Add(ActionKind.PlayCard, $"出 {c.Id.Entry}{TargetLabel(t)}", Nop, a: c, b: t);
        }
        foreach (var p in UsablePotions(me, inCombat: true))
            foreach (var t in targets)
                if (p.IsValidTarget(t))
                    d.Add(ActionKind.UsePotion, $"用 {p.Id.Entry}{TargetLabel(t)}", Nop, a: p, b: t);
        d.Add(ActionKind.EndTurn, "结束回合", Nop);
        return d;
    }

    static Decision? Map(RunState s, Player me, out string why)
    {
        why = "";
        if (Busy()) { why = "行動の処理中"; return null; }
        var cur = s.CurrentMapPoint;
        var d = new Decision(DecisionKind.Map) { Context = $"第 {s.CurrentActIndex + 1} 幕 {(cur == null ? "起点" : $"({cur.coord.col},{cur.coord.row})")}" };
        foreach (var p in RunSessionMap.Travelable(s).OrderBy(p => p.coord.col))
            d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", Nop, a: p);
        if (d.Choices.Count == 0) { why = "行ける場所がありません"; return null; }
        AddPotionActions(d, me);
        return d;
    }

    static string TargetLabel(Creature? t) => t == null ? "" : t.IsPlayer ? " → 自己" : $" → {t.Monster?.Id.Entry}";

    /// <summary>RunSession.UsablePotions と同じ（NPotionPopup の使用条件）。</summary>
    static IEnumerable<PotionModel> UsablePotions(Player me, bool inCombat) =>
        me.CanUseOrRemovePotions && !me.Creature.IsDead
            ? me.Potions.Where(p => !p.IsQueued && p.PassesCustomUsabilityCheck &&
                                   (p.Usage == PotionUsage.AnyTime || (inCombat && p.Usage == PotionUsage.CombatOnly)))
            : Enumerable.Empty<PotionModel>();

    /// <summary>RunSession.AddPotionActions と同じ（戦闘外の「いつでも使える」ポーション）。</summary>
    static void AddPotionActions(Decision d, Player me)
    {
        foreach (var p in UsablePotions(me, inCombat: false))
            if (p.IsValidTarget(me.Creature) || p.IsValidTarget(null))
                d.Add(ActionKind.UsePotion, "用 " + p.Id.Entry, Nop, a: p);
    }
}
