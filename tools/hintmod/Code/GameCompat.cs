// ゲームの版の違いを吸収する小さな関数（AlphaDefect の Fool.cs・Gains26.cs を写すときに Sts2Hint.csproj が呼び出しを差し替える）。
// v0.114.0 で MelodyCmd.MelodyCardsPlayedThisTurn と FreeCardPower がなくなった。名前で探し、なければ 0 / false にする。
using System.Reflection;
using MegaCrit.Sts2.Core.Entities.Players;

namespace Sts2Hint;

public static class GameCompat
{
    static readonly MethodInfo? MelodyCards = typeof(MegaCrit.Sts2.Core.Commands.MelodyCmd)
        .GetMethod("MelodyCardsPlayedThisTurn", BindingFlags.Public | BindingFlags.Static, new[] { typeof(Player) });

    public static int MelodyCardsPlayedThisTurn(Player me)
    {
        try { return MelodyCards?.Invoke(null, new object[] { me }) is int n ? n : 0; }
        catch { return 0; }
    }

    public static bool IsFreeCardPower(object? o) => o?.GetType().Name == "FreeCardPower";
}
