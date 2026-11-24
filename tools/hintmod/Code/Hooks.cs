// ゲームの処理を変えずに、エンコーダが必要とする情報だけを横から記録する Harmony パッチ。
// どれも前置き（prefix）か後置き（postfix）で値を読むだけで、引数・戻り値・処理の流れは変えない。
// ヘッドレスエンジンの HeadlessFixes.cs / GameShim.cs が同じ情報を取っている場所に合わせてある。
using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using Sts2Headless;

namespace Sts2Hint;

public static class Hooks
{
    /// <summary>直前に呼ばれた選牌コマンドの用途（見出しの文言、コマンド名、付与効果、出どころ）。次に開いた選牌画面に結び付ける。</summary>
    public static SelectInfo? LastInfo;
    /// <summary>選牌画面（ノード）ごとの用途。</summary>
    public static readonly ConditionalWeakTable<object, SelectInfo> InfoOf = new();
    /// <summary>NChooseACardSelectionScreen（3枚から1枚）の候補とスキップできるか。画面のフィールドには残らないので開くときに記録する。</summary>
    public static readonly ConditionalWeakTable<object, Tuple<IReadOnlyList<CardModel>, bool>> ChooseACard = new();
    /// <summary>手札からの選択が始まったときの候補（手札の順、絞り込み済み）と用途。選んだカードは手札から抜けるので、始めに写しておく。</summary>
    public static List<CardModel>? HandOptions;
    public static SelectInfo? HandInfo;
    /// <summary>休息所で選んで失敗した（キャンセルした）選択肢。ヘッドレスと同じく、その部屋では候補から外す。</summary>
    public static readonly HashSet<object> FailedRest = new(ReferenceEqualityComparer.Instance);
    static object? _restRoom;

    public static void Install(Harmony h)
    {
        foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
                     .Where(m => !m.ContainsGenericParameters && m.GetParameters().Any(p => p.ParameterType == typeof(CardSelectorPrefs))))
            TryPatch(h, m, prefix: nameof(SelectCmdPrefix));
        TryPatch(h, AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseACardScreen)), prefix: nameof(ChooseACardCmdPrefix));
        TryPatch(h, AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromHandForUpgrade)), prefix: nameof(HandUpgradeCmdPrefix));
        TryPatch(h, AccessTools.Method(typeof(NChooseACardSelectionScreen), nameof(NChooseACardSelectionScreen.ShowScreen)), postfix: nameof(ChooseACardShown));
        TryPatch(h, AccessTools.Method(typeof(NPlayerHand), nameof(NPlayerHand.SelectCards)), prefix: nameof(HandSelectPrefix));
        TryPatch(h, AccessTools.Method(typeof(PotionModel), nameof(PotionModel.OnUseWrapper)), prefix: nameof(PotionStart), postfix: nameof(PotionWrap));
    }

    /// <summary>休息所の結果を受け取る（RestSiteSynchronizer.AfterPlayerOptionChosen）。ランごとに作り直されるので、毎回つなぎ直す。</summary>
    static object? _restSync;
    public static void WatchRestSite()
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var sync = rm?.RestSiteSynchronizer;
        if (sync == null || ReferenceEquals(sync, _restSync)) return;
        _restSync = sync;
        sync.AfterPlayerOptionChosen += (opt, ok, _) => { if (!ok) FailedRest.Add(opt); };
    }

    public static void OnRoom(object? room)
    {
        if (ReferenceEquals(room, _restRoom)) return;
        _restRoom = room;
        FailedRest.Clear();
    }

    static void TryPatch(Harmony h, MethodBase? m, string? prefix = null, string? postfix = null)
    {
        if (m == null) { ModEntry.Logger.Error($"patch target missing (prefix {prefix}, postfix {postfix})"); return; }
        try
        {
            h.Patch(m, prefix: prefix == null ? null : new HarmonyMethod(typeof(Hooks), prefix),
                       postfix: postfix == null ? null : new HarmonyMethod(typeof(Hooks), postfix));
        }
        catch (Exception e) { ModEntry.Logger.Error($"patch {m.DeclaringType?.Name}.{m.Name}: {e.Message}"); }
    }

    /// <summary>HeadlessFixes.SetCancelable と同じ取り方。</summary>
    static void SelectCmdPrefix(object[] __args, MethodBase __originalMethod)
    {
        try
        {
            var p = __args.FirstOrDefault(a => a is CardSelectorPrefs) is CardSelectorPrefs pr ? pr : default;
            var ps = __originalMethod.GetParameters();
            int si = Array.FindIndex(ps, q => q.Name == "source");
            LastInfo = new SelectInfo(p.Prompt?.LocTable, p.Prompt?.LocEntryKey, __originalMethod.Name,
                __args.OfType<EnchantmentModel>().FirstOrDefault(),
                si >= 0 && si < __args.Length ? __args[si] as AbstractModel : null);
        }
        catch { }
    }

    /// <summary>GameShim.ChooseACardMinSelect と同じ用途。</summary>
    static void ChooseACardCmdPrefix() => LastInfo = new SelectInfo(null, null, "FromChooseACardScreen", null);

    static void HandUpgradeCmdPrefix(AbstractModel source) => LastInfo = new SelectInfo("gameplay_ui", "CHOOSE_CARD_UPGRADE_HEADER", "FromHandForUpgrade", null, source);

    static void ChooseACardShown(IReadOnlyList<CardModel> cards, bool canSkip, NChooseACardSelectionScreen? __result)
    {
        if (__result == null) return;
        ChooseACard.AddOrUpdate(__result, Tuple.Create(cards, canSkip));
        if (LastInfo != null) { InfoOf.AddOrUpdate(__result, LastInfo); LastInfo = null; }
    }

    static void HandSelectPrefix(NPlayerHand __instance, Func<CardModel, bool>? filter)
    {
        try
        {
            var me = MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState()!.Players[0];
            HandOptions = PileType.Hand.GetPile(me).Cards.Where(c => filter == null || filter(c)).ToList();
            HandInfo = LastInfo;
            LastInfo = null;
        }
        catch { HandOptions = null; }
    }

    /// <summary>HeadlessFixes.PotionStart / PotionWrap と同じ（使用中のポーション）。</summary>
    static void PotionStart(PotionModel __instance) => AgentCardSelector.UsingPotion = __instance;

    static void PotionWrap(PotionModel __instance, ref Task __result) => __result = PotionDone(__instance, __result);

    static async Task PotionDone(PotionModel p, Task t)
    {
        try { await t; }
        finally { if (AgentCardSelector.UsingPotion == p) AgentCardSelector.UsingPotion = null; }
    }

    /// <summary>新しく見つけた選牌画面に、直前の選牌コマンドの用途を結び付ける。</summary>
    public static SelectInfo? InfoFor(object screen)
    {
        if (InfoOf.TryGetValue(screen, out var info)) return info;
        if (LastInfo == null) return null;
        InfoOf.AddOrUpdate(screen, LastInfo);
        info = LastInfo;
        LastInfo = null;
        return info;
    }
}

/// <summary>非公開フィールドを読むだけの小さな助け。名前が変わっていたら null を返す（ゲーム更新で壊れても落ちない）。</summary>
public static class R
{
    static readonly Dictionary<(Type, string), FieldInfo?> Cache = new();

    public static T? F<T>(object? o, string name) where T : class
    {
        if (o == null) return null;
        var key = (o.GetType(), name);
        if (!Cache.TryGetValue(key, out var f)) Cache[key] = f = AccessTools.Field(o.GetType(), name);
        return f?.GetValue(o) as T;
    }

    public static T? V<T>(object? o, string name) where T : struct
    {
        if (o == null) return null;
        var key = (o.GetType(), name);
        if (!Cache.TryGetValue(key, out var f)) Cache[key] = f = AccessTools.Field(o.GetType(), name);
        return f?.GetValue(o) is T v ? v : null;
    }
}
