// Sts2Hint：AlphaDefect のヒントをゲーム画面に出す Mod。
// ゲームの状態は変えない（読むだけ）。観測は AlphaDefect と同じエンコーダで作り、ネットの計算はローカルのヒントサーバーが行う。
using System.Reflection;
using System.Text.Json.Nodes;
using Godot;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace Sts2Hint;

[ModInitializer(nameof(Initialize))]
public static class ModEntry
{
    public const string ModId = "Sts2Hint";
    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } = new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);
    public static string ModDir { get; private set; } = "";

    static HintClient? _client;
    static HintOverlay? _overlay;
    static readonly RunSession _run = new();
    static double _next;
    static uint _seq;
    static ulong _lastHash;
    static bool _toggleDown;
    static string _lastWhy = "";
    static Config _cfg = new();

    public static void Initialize()
    {
        ModDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "";
        _cfg = Config.Load(Path.Combine(ModDir, "Sts2Hint.config.json"));
        Vocab.Load(Path.Combine(ModDir, "vocab.txt"));
        LiveDecision.LoadNoDownside(Path.Combine(ModDir, "no_downside_relics.txt"));
        Observation.RouteSummary = true;        // tools/hint と同じ（reset の route_summary: true）
        Hooks.Install(new HarmonyLib.Harmony(ModId));
        _client = new HintClient("127.0.0.1", _cfg.Port);
        if (Engine.GetMainLoop() is SceneTree tree) tree.ProcessFrame += Tick;
        else Logger.Error("SceneTree が見つからないため、ヒントを更新できません");
        Logger.Info($"{ModId} loaded: vocab {Vocab.Count}, server 127.0.0.1:{_cfg.Port}");
    }

    static void Tick()
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree) return;
            _overlay ??= new HintOverlay(tree.Root) { TopN = _cfg.TopN };
            HandleKeys();
            Receive();
            double now = Time.GetTicksMsec() / 1000.0;
            if (now < _next) return;
            _next = now + _cfg.IntervalSec;
            Poll();
        }
        catch (Exception e) { Logger.Error($"tick: {e}"); _next += 5; }
    }

    static void HandleKeys()
    {
        bool down = Input.IsKeyPressed(Key.F8);
        if (down && !_toggleDown) _overlay!.Visible = !_overlay.Visible;
        _toggleDown = down;
    }

    static void Poll()
    {
        var c = _client!;
        var hello = c.Hello;
        _overlay!.SetStatus(c.Connected ? $"サーバー: {hello?["ckpt"]?.GetValue<string>()}（F8で隠す）" : c.Status);
        if (!c.Connected) { _lastHash = 0; return; }

        LiveDecision.ResetContext(_run);
        var d = LiveDecision.Current(_run, out string why);
        if (d == null)
        {
            if (why != _lastWhy) { _overlay.Clear(why); _lastWhy = why; }
            _lastHash = 0;
            return;
        }
        _lastWhy = "";
        // 重みを学習した観測格式に合わせる（アイアンクラッドは 28）。重みに記録がなければキャラの既定のまま
        Observation.FormatOverride = hello?["obs_format"] is JsonValue fmt && fmt.TryGetValue<int>(out int f) ? f : null;
        var me = RunManager.Instance.DebugOnlyGetState()!.Players[0];
        if (hello?["character"]?.GetValue<string>() is { Length: > 0 } ch && ch != Vocab.Model(me.Character))
        {
            _overlay.Clear($"この重みは {ch} 用です（今は {Vocab.Model(me.Character)}）");
            return;
        }
        var obs = Observation.Build(_run, d, labels: true);
        var bytes = obs.Serialize();
        ulong h = Hash(bytes);
        if (h == _lastHash) return;     // 局面が変わっていなければ送らない
        _lastHash = h;
        c.SendFrame(++_seq, bytes);
    }

    static void Receive()
    {
        while (_client!.TryReceive(out var j))
        {
            if (j == null) continue;
            if (j["type"]?.GetValue<string>() == "error") { _overlay!.SetStatus("サーバーのエラー: " + j["message"]); continue; }
            if (j["seq"]?.GetValue<uint>() != _seq) continue;     // 古い局面への返事は捨てる
            if (j["hint"] is { } rec) _overlay!.Show(rec);
        }
    }

    static ulong Hash(byte[] b)
    {
        ulong h = 1469598103934665603UL;
        foreach (byte x in b) { h ^= x; h *= 1099511628211UL; }
        return h;
    }
}

public sealed class Config
{
    public int Port { get; set; } = 8765;
    public int TopN { get; set; } = 6;
    public double IntervalSec { get; set; } = 0.25;

    public static Config Load(string path)
    {
        try
        {
            if (File.Exists(path))
                return System.Text.Json.JsonSerializer.Deserialize<Config>(File.ReadAllText(path)) ?? new Config();
        }
        catch (Exception e) { ModEntry.Logger.Error($"config: {e.Message}"); }
        return new Config();
    }
}
