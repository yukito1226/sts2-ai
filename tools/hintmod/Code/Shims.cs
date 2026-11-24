// AlphaDefect の観測エンコーダ（engine/Sts2Headless/Engine/Obs/ など、MIT）をゲーム内でそのまま動かすための代役。
// エンコーダが参照するヘッドレスエンジン側の型を、同じ名前空間・同じ名前で最小限だけ用意する。
// 中身はヘッドレスエンジンの RunSession.cs / RewardsFlow.cs / UiFlows.cs / AgentCardSelector.cs / FakeScreens.cs の該当部分の写し、
// または実機では使わない値（0、null）。本家の Sts2Bridge（非公開）も同じ考え方で Obs/** と Compat/** を共有していた
// （engine/Directory.Build.targets）。
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>エンコーダが見る「今の局」。実機ではゲームの RunManager がそのまま持っている。</summary>
public sealed class RunSession
{
    public static RunState State => RunManager.Instance.DebugOnlyGetState()!;

    /// <summary>整局モード（実機は常に整局）。</summary>
    public bool CombatOnly => false;
    /// <summary>このModが見た判断の数（頭部 13。ネットの入力には使われない、記録用）。</summary>
    public int Steps { get; set; }
    public bool IsVictory { get; set; }
    /// <summary>選牌画面の今の選択（選牌画面の対応は後の段階。今は常に null）。</summary>
    public PendingCardChoice? Selecting { get; set; }
    internal IReadOnlyList<int> SelectedIndices { get; set; } = Array.Empty<int>();

    /// <summary>RunSession.RestHealInfo の写し（休息の回復量の見込み）。</summary>
    internal static (int actual, int nominal, int extra) RestHealInfo(Player p)
    {
        var cr = p.Creature;
        int nominal = (int)HealRestSiteOption.GetHealAmount(p);
        int actual = Math.Clamp(nominal, 0, Math.Max(0, cr.MaxHp - cr.CurrentHp));
        int extra = 0;
        foreach (var m in p.RunState.IterateHookListeners(null))
        {
            int n = m.ModifyExtraRestSiteHealText(p, Array.Empty<MegaCrit.Sts2.Core.Localization.LocString>()).Count;
            if (n > 0 && m.ModifyRestSiteHealAmount(cr, 100m) == 100m) extra += n;
        }
        return (actual, nominal, extra);
    }
}

/// <summary>AgentCardSelector.cs の SelectInfo（選牌の用途：見出しの文言、選牌コマンド、付ける付与効果）。</summary>
public sealed record SelectInfo(string? PromptTable, string? PromptKey, string Method, EnchantmentModel? Enchantment, AbstractModel? Source = null);

/// <summary>AgentCardSelector.cs の PendingCardChoice（選牌画面の候補）。ゲーム内では画面から組み立てる。</summary>
public sealed class PendingCardChoice(IReadOnlyList<CardModel> options, int min, int max, bool cancelable, SelectInfo? info = null)
{
    public SelectInfo? Info { get; } = info;
    public IReadOnlyList<CardModel> Options { get; } = options;
    public int Min { get; } = min;
    public int Max { get; } = max;
    public bool Cancelable { get; } = cancelable;
}

public static class AgentCardSelector
{
    /// <summary>使用中のポーション（ポーションから始まった選牌の出どころ）。</summary>
    public static MegaCrit.Sts2.Core.Models.PotionModel? UsingPotion { get; set; }
}

/// <summary>RewardsFlow.cs の PendingRewards（報酬画面）。</summary>
public sealed class PendingRewards(RewardsSet set, IReadOnlySet<Reward> attempted)
{
    public RewardsSet Set { get; } = set;
    public List<Reward> Available => Set.Rewards
        .Where(r => !r.SuccessfullySelected && !attempted.Contains(r))
        .Where(r => r is not PotionReward || Set.Player.HasOpenPotionSlots)
        .ToList();
}

public static class RewardsFlow
{
    public static PendingRewards? Pending { get; set; }
    public static RewardsSet? ActiveSet { get; set; }
    public static Reward? ActiveChoice { get; set; }
    public static IReadOnlySet<Reward>? ActiveAttempted { get; set; }

    public static IEnumerable<Reward> Unclaimed(RewardsSet set, IReadOnlySet<Reward>? attempted, Reward? except) =>
        set.Rewards.Where(r => !r.SuccessfullySelected && r != except && (attempted == null || !attempted.Contains(r)));
}

public static class UiFlows
{
    /// <summary>このプロセスで勝った戦闘の数（頭部 12。訓練側が戦闘の終わりを見分けるためのもので、ネットの入力ではない）。</summary>
    public static int CombatsWon { get; set; }
}

public static class FakeScreens
{
    /// <summary>Crystal Sphere のミニゲーム（後の段階で対応）。</summary>
    public static MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? CrystalSphere { get; set; }
}

/// <summary>RunSession.cs の RunSessionMap の写し（NMapScreen.RecalculateTravelability と同じ規則）。</summary>
public static class RunSessionMap
{
    public static IEnumerable<MegaCrit.Sts2.Core.Map.MapPoint> Travelable(RunState s)
    {
        var map = s.Map;
        if (s.VisitedMapCoords.Count == 0) return new[] { map.StartingMapPoint };
        var last = s.VisitedMapCoords[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new[] { map.SecondBossMapPoint };
        if (last.row == map.GetRowCount() - 1) return new[] { map.BossMapPoint };
        var point = map.GetPoint(last);
        return point == null ? new[] { map.StartingMapPoint } : MegaCrit.Sts2.Core.Map.MapTravel.GetTravelablePointsFrom(s, point);
    }
}
