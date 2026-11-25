// ゲーム内の今の状態から、AlphaDefect のエンコーダに渡す Decision（候補の一覧）を作る。
// 候補の作り方と並び順は、ヘッドレスエンジンの RunSession.cs（CombatDecision / MapDecision / CardSelectDecision / CardRewardDecision /
// BundleDecision / RewardsDecision / CrystalSphereDecision / EventDecision / RestDecision / ShopDecision / FakeMerchantDecision /
// TreasureDecision）と RewardsFlow.cs（MIT）に合わせる。候補の順番や対象の付け方が違うと、同じ局面でもネットの入力が変わるため。
// ヘッドレスは画面を自分で動かすが、ここでは画面のノードから「今出ている選択肢」を読むだけ。Run（実行する処理）は使わない。
// ヘッドレスとの違いは README の「ヘッドレスとの違い」にまとめてある。
using System.Collections;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace Sts2Hint;

public static class LiveDecision
{
    static readonly Action Nop = () => { };
    static HashSet<string>? _noDownside;

    /// <summary>エンコーダが読む画面の状態（RunSession.Selecting、RewardsFlow、FakeScreens.CrystalSphere）を空に戻す。毎回作り直す。</summary>
    public static void ResetContext(RunSession run)
    {
        run.Selecting = null;
        run.SelectedIndices = Array.Empty<int>();
        RewardsFlow.Pending = null;
        RewardsFlow.ActiveSet = null;
        RewardsFlow.ActiveChoice = null;
        RewardsFlow.ActiveAttempted = null;
        FakeScreens.CrystalSphere = null;
    }

    public static void LoadNoDownside(string path)
    {
        try { _noDownside = File.ReadAllLines(path).Select(l => l.Trim()).Where(l => l.Length > 0 && !l.StartsWith('#')).ToHashSet(); }
        catch { _noDownside = new HashSet<string>(); }
    }

    static bool IsNoDownside(RelicModel? r) => r != null && _noDownside != null && _noDownside.Contains(r.Id.Entry);

    /// <summary>今がヒントを出せる判断の場面なら Decision、そうでなければ null と理由。</summary>
    public static Decision? Current(RunSession run, out string why)
    {
        why = "";
        var rm = RunManager.Instance;
        var s = rm?.DebugOnlyGetState();
        if (s == null) { why = "ラン中ではありません"; return null; }
        if (s.Players.Count != 1) { why = "マルチプレイには対応していません"; return null; }
        var me = s.Players[0];
        if (s.IsGameOver) { why = "ランが終わりました"; return null; }
        Hooks.WatchRestSite();
        Hooks.OnRoom(s.CurrentRoom);

        // 0. 地図：地図を開くと、報酬画面などはオーバーレイに積まれたまま隠れる（NOverlayStack.ShowOverlays は地図が開いている間は出さない）。
        //    報酬を残して「進む」と、報酬画面が積まれたまま地図で次の部屋を選ぶことになるので、地図をオーバーレイより先に見る
        //    （見るだけで移動できないときは、下の画面のヒントをそのまま出す）
        if (!CombatManager.Instance.IsInProgress && NMapScreen.Instance is { IsOpen: true, IsTravelEnabled: true, IsTraveling: false })
            return Map(s, me, out why);

        // 1. いちばん上に出ている画面（ヘッドレスの GetDecisionOnce と同じ優先順：選牌 → カード報酬 → カードパック → 報酬 → Crystal Sphere）
        var overlays = R.F<IList>(NOverlayStack.Instance, "_overlays");
        object? top = overlays is { Count: > 0 } ? overlays[^1] : null;
        switch (top)
        {
            case NCardGridSelectionScreen grid: return GridSelect(run, grid, me, out why);
            case NChooseACardSelectionScreen choose: return ChooseACardSelect(run, choose, out why);
            case NCardRewardSelectionScreen cr: return CardReward(cr, overlays!, out why);
            case NChooseABundleSelectionScreen b: return Bundle(b, out why);
            case NRewardsScreen rs: return Rewards(rs, me, out why);
            case NCrystalSphereScreen cs: return Sphere(cs, out why);
            case null: break;
            default: why = $"未対応の画面（{top.GetType().Name}）"; return null;
        }

        bool inCombat = CombatManager.Instance.IsInProgress && me.PlayerCombatState != null;
        // 2. 戦闘中、手札から選ぶ場面（「武装」の強化、捨てる・廃棄するカードを選ぶなど）
        if (inCombat && NPlayerHand.Instance is { } hand && R.F<TaskCompletionSource<IEnumerable<CardModel>>>(hand, "_selectionCompletionSource") is { Task.IsCompleted: false })
            return HandSelect(run, hand, out why);
        if (inCombat) return Combat(me, out why);

        if (Busy()) { why = "処理中"; return null; }

        // 3. 部屋ごとの画面（ヘッドレスの s.CurrentRoom switch と同じ順）
        // イベント部屋に入った直後は、イベントがまだ用意されておらず GetLocalEvent が例外を投げることがある
        EventModel? localEvent = null;
        if (s.CurrentRoom is EventRoom)
            try { localEvent = rm!.EventSynchronizer.GetLocalEvent(); } catch (Exception) { why = "イベントの準備中"; return null; }
        switch (s.CurrentRoom)
        {
            case EventRoom { CanonicalEvent: FakeMerchant } when localEvent is FakeMerchant { StartedFight: false } fm:
                return Shop(fm.Inventory, me, "假商人 ", out why);
            case EventRoom when localEvent is { IsFinished: false } ev:
                return Event(ev, me, out why);
            case RestSiteRoom:
                return Rest(me, out why);
            case MerchantRoom:
                return Shop(rm!.MerchantSynchronizer.GetLocalInventory(), me, "", out why);
            case TreasureRoom:
                return Treasure(me, out why);
        }
        why = "判断の場面ではありません";
        return null;
    }

    static bool Busy()
    {
        var rm = RunManager.Instance;
        // ランの開始直後は ActionQueueSet などがまだ null のことがある（v0.114.0 のメインメニューで確認）
        if (rm?.ActionQueueSet == null || rm.ActionExecutor == null) return true;
        return !rm.ActionQueueSet.IsEmpty || rm.ActionExecutor.IsRunning;
    }

    // ---------------------------------------------------------------- 戦闘・地図

    static Decision? Combat(Player me, out string why)
    {
        why = "";
        var pcs = me.PlayerCombatState!;
        if (pcs.Phase != PlayerTurnPhase.Play) { why = "自分のターンの操作待ちではありません"; return null; }
        if (Busy()) { why = "行動の処理中"; return null; }
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

    // ---------------------------------------------------------------- 選牌

    static Decision? GridSelect(RunSession run, NCardGridSelectionScreen grid, Player me, out string why)
    {
        var cards = R.F<IReadOnlyList<CardModel>>(grid, "_cards");
        // 山札・捨て札・廃棄札から選ぶ画面（NCombatPileCardSelectScreen、CardSelectCmd.FromCombatPile）は _cards を空のままにし、
        // _pile と _filter から並べている。候補はヘッドレスのセレクターに渡るのと同じ「pile.Cards（_filter があれば絞ったもの）」
        if (R.F<CardPile>(grid, "_pile") is { } pile)
        {
            var filter = R.F<Func<CardModel, bool>>(grid, "_filter");
            cards = filter == null ? pile.Cards : pile.Cards.Where(filter).ToList();
        }
        var prefs = R.V<CardSelectorPrefs>(grid, "_prefs");
        var sel = R.F<IEnumerable<CardModel>>(grid, "_selectedCards");
        if (cards == null || prefs == null) { why = $"選牌画面を読めません（{grid.GetType().Name}）"; return null; }
        var p = prefs.Value;
        var pc = new PendingCardChoice(cards, p.MinSelect, p.MaxSelect, p.Cancelable, Hooks.InfoFor(grid));
        return CardSelect(run, pc, sel, out why);
    }

    static Decision? ChooseACardSelect(RunSession run, NChooseACardSelectionScreen screen, out string why)
    {
        if (!Hooks.ChooseACard.TryGetValue(screen, out var t)) { why = "カード選択の候補を読めません（Mod の読み込みより前に開いた画面）"; return null; }
        // GameShim.ChooseACardMinSelect：スキップできれば最少 0 枚、できなければ 1 枚。最大 1 枚
        var pc = new PendingCardChoice(t.Item1, t.Item2 ? 0 : 1, 1, false, Hooks.InfoFor(screen) ?? new SelectInfo(null, null, "FromChooseACardScreen", null));
        return CardSelect(run, pc, null, out why);
    }

    static Decision? HandSelect(RunSession run, NPlayerHand hand, out string why)
    {
        var prefs = R.V<CardSelectorPrefs>(hand, "_prefs");
        var opts = Hooks.HandOptions;
        if (prefs == null || opts == null) { why = "手札の選択を読めません"; return null; }
        var p = prefs.Value;
        var pc = new PendingCardChoice(opts, p.MinSelect, p.MaxSelect, p.Cancelable, Hooks.HandInfo);
        return CardSelect(run, pc, R.F<IEnumerable<CardModel>>(hand, "_selectedCards"), out why);
    }

    /// <summary>RunSession.CardSelectDecision と同じ。複数枚選ぶ画面は「1枚ずつ選ぶ＋確定」に分ける。</summary>
    static Decision? CardSelect(RunSession run, PendingCardChoice pc, IEnumerable<CardModel>? selectedCards, out string why)
    {
        why = "";
        var selected = new List<int>();
        if (selectedCards != null)
            foreach (var c in selectedCards)
            {
                int i = IndexOf(pc.Options, c);
                if (i >= 0 && !selected.Contains(i)) selected.Add(i);
            }
        int max = Math.Min(pc.Max, pc.Options.Count);
        int min = Math.Min(pc.Min, max);
        bool upgrade = pc.Info?.Method == "FromDeckForUpgrade" && max > 0;
        if (upgrade) min = Math.Max(min, 1);
        bool cancel = pc.Cancelable && !upgrade && selected.Count == 0;
        if (max == 0 || (min == max && max == pc.Options.Count && !cancel)) { why = "選ぶ余地がありません（全部選ばれます）"; return null; }
        var d = new Decision(DecisionKind.CardSelect) { Context = $"选 {min}..{max} 张，已选 {selected.Count}", Selected = selected.Count };
        bool upgradeAny = pc.Info?.Method is { } meth && meth.Contains("ForUpgrade");
        for (int i = 0; i < pc.Options.Count; i++)
        {
            if (selected.Contains(i)) continue;
            d.Add(ActionKind.SelectCard, "选 " + pc.Options[i].Id.Entry, Nop, a: pc.Options[i], b: upgradeAny ? Observation.UpgradedCopy(pc.Options[i]) : null);
        }
        if (selected.Count >= min) d.Add(ActionKind.ConfirmSelection, "确认", Nop);
        if (cancel) d.Add(ActionKind.Leave, "取消", Nop);
        run.Selecting = pc;
        run.SelectedIndices = selected;
        return d;
    }

    static int IndexOf(IReadOnlyList<CardModel> xs, CardModel c)
    {
        for (int i = 0; i < xs.Count; i++) if (ReferenceEquals(xs[i], c)) return i;
        return -1;
    }

    // ---------------------------------------------------------------- 報酬

    /// <summary>報酬画面で、開いて取らずに閉じたカード報酬（ヘッドレスの attempted）。画面では「スキップ済み」のボタンになる。</summary>
    static HashSet<Reward> Attempted(NRewardsScreen rs)
    {
        var set = new HashSet<Reward>();
        if (R.F<IEnumerable>(rs, "_skippedRewardButtons") is { } skipped)
            foreach (var b in skipped) if (b is NRewardButton { Reward: { } r }) set.Add(r);
        return set;
    }

    static Decision? Rewards(NRewardsScreen rs, Player me, out string why)
    {
        why = "";
        var set = R.F<RewardsSet>(rs, "_rewardsSet");
        if (set == null) { why = "報酬画面を読めません"; return null; }
        var attempted = Attempted(rs);
        // ヘッドレスは金貨と、マイナスのない遺物（リンクされた選択肢でないもの）を自動で取ってから判断を出す（RewardsFlow.Select）。
        // ここでは取らない（ゲームを操作しない）ので、残っていれば先に取ってもらう
        bool gold = set.Rewards.Any(r => r is GoldReward && !r.SuccessfullySelected && !attempted.Contains(r));
        bool relic = set.Rewards.Any(r => r is RelicReward rr && rr.ParentRewardSet == null && IsNoDownside(rr.Relic) && !r.SuccessfullySelected && !attempted.Contains(r));
        if (gold || relic) { why = $"先に{(gold ? "ゴールド" : "")}{(gold && relic ? "と" : "")}{(relic ? "遺物" : "")}を取ってください（AIはこれらを自動で取る前提です）"; return null; }

        var pr = new PendingRewards(set, attempted);
        var d = new Decision(DecisionKind.Rewards) { Context = set.Room is CombatRoom ? "战斗奖励" : "额外奖励" };
        var available = pr.Available;
        foreach (var r in available) d.Add(ActionKind.TakeReward, "领 " + RewardLabel(r), Nop, a: r);
        foreach (var r in available) if (r is CardReward cr) ShowRewardCards(d, set, cr);
        if (!set.DisallowSkipping || available.Count == 0) d.Add(ActionKind.Leave, "离开", Nop);
        AddPotionActions(d, me);
        if (!me.HasOpenPotionSlots && set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected))
        {
            AddPotionDiscards(d, me);
            foreach (var r in set.Rewards) if (r is PotionReward { Potion: not null } p && !p.SuccessfullySelected) d.Shown.Add((p, Zone.Offer, 0));
        }
        RewardsFlow.Pending = pr;
        return d;
    }

    static void ShowRewardCards(Decision d, RewardsSet set, CardReward cr)
    {
        var z = Observation.RewardZone(set, cr);
        foreach (var c in cr.Cards) d.Shown.Add((c, z, 0));
    }

    static Decision? CardReward(NCardRewardSelectionScreen screen, IList overlays, out string why)
    {
        why = "";
        var cards = R.F<IReadOnlyList<CardCreationResult>>(screen, "_options");
        var alts = R.F<IReadOnlyList<CardRewardAlternative>>(screen, "_extraOptions") ?? Array.Empty<CardRewardAlternative>();
        if (cards == null) { why = "カード報酬を読めません"; return null; }
        // このカード報酬を開いた報酬画面（下に重なっている）と、その中のどの報酬か（RewardsFlow.ActiveSet / ActiveChoice）
        var rs = overlays.OfType<NRewardsScreen>().LastOrDefault();
        var set = R.F<RewardsSet>(rs, "_rewardsSet");
        var d = new Decision(DecisionKind.CardReward);
        if (set != null)
        {
            var first = cards.FirstOrDefault()?.Card;
            var choice = set.Rewards.OfType<CardReward>().FirstOrDefault(r => first != null && r.Cards.Any(c => ReferenceEquals(c, first)));
            var attempted = Attempted(rs!);
            if (choice != null) attempted.Add(choice);
            RewardsFlow.ActiveSet = set; RewardsFlow.ActiveChoice = choice; RewardsFlow.ActiveAttempted = attempted;
            foreach (var r in RewardsFlow.Unclaimed(set, attempted, choice))
            {
                d.Shown.Add((r, Zone.Offer, 0));
                if (r is CardReward other) ShowRewardCards(d, set, other);
            }
        }
        for (int i = 0; i < cards.Count; i++) d.Add(ActionKind.PickCardReward, "拿 " + cards[i].Card.Id.Entry, Nop, a: cards[i].Card);
        for (int i = 0; i < alts.Count; i++) d.Add(ActionKind.CardRewardAlt, alts[i].OptionId, Nop, key: Vocab.Alt(alts[i].OptionId));
        return d;
    }

    static Decision? Bundle(NChooseABundleSelectionScreen screen, out string why)
    {
        why = "";
        var bundles = R.F<IReadOnlyList<IReadOnlyList<CardModel>>>(screen, "_bundles");
        if (bundles == null) { why = "カードパックを読めません"; return null; }
        var d = new Decision(DecisionKind.Bundle);
        for (int i = 0; i < bundles.Count; i++)
        {
            foreach (var c in bundles[i]) d.Shown.Add((c, i == 0 ? Zone.Pack1 : Zone.Pack2, 0));
            d.Add(ActionKind.PickBundle, "卡包 " + string.Join("+", bundles[i].Select(c => c.Id.Entry)), Nop, a: bundles[i].FirstOrDefault(), f0: i);
        }
        return d;
    }

    static Decision? Sphere(NCrystalSphereScreen screen, out string why)
    {
        why = "";
        var m = R.F<MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame>(screen, "_entity");
        if (m == null || m.IsFinished) { why = "Crystal Sphere は終わりました"; return null; }
        FakeScreens.CrystalSphere = m;
        var d = new Decision(DecisionKind.Minigame) { Context = $"Crystal Sphere 剩余 {m.DivinationCount} 次" };
        int w = m.GridSize.X, h = m.GridSize.Y;
        for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
        {
            if (!m.cells[x, y].IsHidden) continue;
            d.Add(ActionKind.MinigameCell, $"小 ({x},{y})", Nop, f0: x, f1: y, f2: 0);
            d.Add(ActionKind.MinigameCell, $"大 ({x},{y})", Nop, f0: x, f1: y, f2: 1);
        }
        return d;
    }

    // ---------------------------------------------------------------- 部屋

    static readonly HashSet<string> ExcludedEventOptions = new() { "TRIAL.pages.REJECT.options.DOUBLE_DOWN" };

    static Decision? Event(EventModel ev, Player me, out string why)
    {
        why = "";
        var d = new Decision(DecisionKind.Event) { Context = ev.Id.Entry };
        var opts = ev.CurrentOptions;
        for (int i = 0; i < opts.Count; i++)
        {
            if (opts[i].IsLocked || ExcludedEventOptions.Contains(opts[i].TextKey)) continue;
            d.Add(ActionKind.EventOption, opts[i].TextKey, Nop, a: opts[i], b: HoverObject(opts[i]));
        }
        if (d.Choices.Count == 0) { why = "イベントの選択肢がありません"; return null; }
        AddPotionActions(d, me);
        return d;
    }

    /// <summary>RunSession.HoverObject と同じ（選択肢のツールチップにあるカード・遺物・ポーション）。</summary>
    static object? HoverObject(EventOption opt)
    {
        if (opt.HoverTips == null) return null;
        foreach (var t in opt.HoverTips)
        {
            object? m = t is MegaCrit.Sts2.Core.HoverTips.CardHoverTip ct ? ct.Card
                : t is MegaCrit.Sts2.Core.HoverTips.HoverTip ht ? ht.CanonicalModel : null;
            if (m is CardModel or PotionModel || (m is RelicModel r && r.Id != opt.Relic?.Id)) return m;
        }
        return null;
    }

    static Decision? Rest(Player me, out string why)
    {
        why = "";
        var sync = RunManager.Instance.RestSiteSynchronizer;
        // 自分の休息所の記録（RestSiteSynchronizer.PlayerRestSite：終わったか、最後に選んだ選択肢）
        object? site = R.F<IList>(sync, "_restSites") is { Count: > 0 } list ? list[0] : null;
        var done = site == null ? null : AccessTools.Field(site.GetType(), "completionTaskSource")?.GetValue(site) as TaskCompletionSource;
        if (done is { Task.IsCompleted: true }) { why = "休息所は終わりました"; return null; }
        var opts = sync.GetLocalOptions();
        int? last = site == null ? null : AccessTools.Field(site.GetType(), "lastChosenOptionIndex")?.GetValue(site) is uint u ? (int)u : null;
        bool chosen = last is int li && li < opts.Count && !Hooks.FailedRest.Contains(opts[li]);
        var d = new Decision(DecisionKind.RestSite);
        bool canSmith = opts.Any(o => o is SmithRestSiteOption && o.IsEnabled && !Hooks.FailedRest.Contains(o));
        foreach (var o in opts)
        {
            if (!o.IsEnabled || Hooks.FailedRest.Contains(o)) continue;
            if (o is HealRestSiteOption && canSmith && RunSession.RestHealInfo(me) is { actual: 0, extra: 0 }) continue;
            d.Add(ActionKind.RestOption, o.OptionId, Nop, a: o);
        }
        if (d.Choices.Count == 0) { why = "休息所で選べるものがありません"; return null; }
        if (chosen) d.Add(ActionKind.Leave, "离开", Nop);
        if (chosen && d.Choices.Count == 1) { why = "休息所は終わりました"; return null; }
        AddPotionActions(d, me);
        return d;
    }

    static Decision? Shop(MerchantInventory? inv, Player me, string ctx, out string why)
    {
        why = "";
        if (inv == null) { why = "ショップを読めません"; return null; }
        var d = new Decision(DecisionKind.Shop) { Context = $"{ctx}金币 {me.Gold}" };
        foreach (MerchantEntry e in inv.AllEntries)
        {
            if (e.IsStocked && !e.EnoughGold && e is not MerchantCardRemovalEntry { Used: true }) d.Locked.Add((EntryObject(e), e.Cost));
            if (!e.IsStocked || !e.EnoughGold) continue;
            if (e is MerchantPotionEntry && !me.HasOpenPotionSlots) continue;
            if (e is MerchantCardRemovalEntry removal)
            {
                if (removal.Used) continue;
                d.Add(ActionKind.Buy, $"买 删牌 ({e.Cost})", Nop, a: removal, f0: e.Cost);
                continue;
            }
            d.Add(ActionKind.Buy, $"买 {EntryLabel(e)} ({e.Cost})", Nop, a: EntryObject(e), f0: e.Cost);
        }
        d.Add(ActionKind.Leave, "离开", Nop);
        AddPotionActions(d, me);
        if (ctx.Length == 0)
        {
            if (!me.HasOpenPotionSlots && inv.AllEntries.Any(e => e is MerchantPotionEntry && e.IsStocked && e.EnoughGold))
                AddPotionDiscards(d, me);
            if (!me.HasOpenPotionSlots)
                foreach (var e in inv.AllEntries) if (e is MerchantPotionEntry { Model: { } p } && e.IsStocked) d.Shown.Add((p, Zone.Offer, e.Cost));
        }
        return d;
    }

    static object EntryObject(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card,
        MerchantRelicEntry { Model: { } r } => r,
        MerchantPotionEntry { Model: { } p } => p,
        _ => e,
    };

    static string EntryLabel(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card.Id.Entry,
        MerchantRelicEntry r => r.Model?.Id.Entry ?? "遗物",
        MerchantPotionEntry p => p.Model?.Id.Entry ?? "药水",
        MerchantCardRemovalEntry => "删牌",
        _ => e.GetType().Name,
    };

    static Decision? Treasure(Player me, out string why)
    {
        why = "";
        var room = NTreasureRoom();
        if (room == null || R.V<bool>(room, "_isRelicCollectionOpen") != true) { why = "宝箱を開けてください"; return null; }
        var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
        if (relics == null || relics.Count == 0) { why = "宝箱は空です"; return null; }
        if (relics.Count == 1 && IsNoDownside(relics[0])) { why = "この遺物は取ってください（AIは自動で取る前提です）"; return null; }
        var d = new Decision(DecisionKind.Treasure);
        foreach (var r in relics) d.Add(ActionKind.PickRelic, "拿 " + r.Id.Entry, Nop, a: r);
        d.Add(ActionKind.SkipRelic, "跳过", Nop);
        AddPotionActions(d, me);
        return d;
    }

    static object? NTreasureRoom() => MegaCrit.Sts2.Core.Nodes.NRun.Instance?.TreasureRoom;

    // ---------------------------------------------------------------- 共通

    static string TargetLabel(Creature? t) => t == null ? "" : t.IsPlayer ? " → 自己" : $" → {t.Monster?.Id.Entry}";

    static string RewardLabel(Reward r) => r switch
    {
        GoldReward => "金币",
        PotionReward p => "药水 " + p.Potion?.Id.Entry,
        RelicReward rr => "遗物 " + rr.Relic?.Id.Entry,
        MegaCrit.Sts2.Core.Rewards.CardReward => "卡牌奖励",
        _ => r.GetType().Name,
    };

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

    /// <summary>RunSession.AddPotionDiscards と同じ（ポーションを捨てる）。</summary>
    static void AddPotionDiscards(Decision d, Player me)
    {
        if (!me.CanUseOrRemovePotions) return;
        var slots = me.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
            if (slots[i] is { IsQueued: false } p)
                d.Add(ActionKind.DiscardPotion, "丢弃 " + p.Id.Entry, Nop, a: p);
    }
}
