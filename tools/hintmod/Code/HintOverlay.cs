// 画面右上のヒント表示。ゲームの操作は邪魔しない（マウスは素通り）。F8 で表示・非表示。
using System.Text.Json.Nodes;
using Godot;

namespace Sts2Hint;

public sealed class HintOverlay
{
    readonly CanvasLayer _layer;
    readonly PanelContainer _panel;
    readonly Label _title, _summary, _status;
    readonly VBoxContainer _rows;
    readonly Font? _font;
    public int TopN = 6;

    public HintOverlay(Node root)
    {
        _layer = new CanvasLayer { Layer = 110, Name = "Sts2HintOverlay" };
        _font = ResourceLoader.Exists("res://themes/fonts/jpn/noto_sans_cjkjp_bold_shared.tres")
            ? ResourceLoader.Load<Font>("res://themes/fonts/jpn/noto_sans_cjkjp_bold_shared.tres") : null;

        _panel = new PanelContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        var style = new StyleBoxFlat
        {
            BgColor = new Color(0.05f, 0.06f, 0.09f, 0.82f),
            BorderColor = new Color(0.45f, 0.75f, 0.95f, 0.9f),
            ContentMarginLeft = 10, ContentMarginRight = 10, ContentMarginTop = 6, ContentMarginBottom = 8,
        };
        style.SetBorderWidthAll(2);
        style.SetCornerRadiusAll(8);
        _panel.AddThemeStyleboxOverride("panel", style);
        _panel.SetAnchorsPreset(Control.LayoutPreset.TopRight);
        _panel.GrowHorizontal = Control.GrowDirection.Begin;
        _panel.OffsetRight = -16; _panel.OffsetTop = 96;
        _panel.CustomMinimumSize = new Vector2(380, 0);

        var box = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        box.AddThemeConstantOverride("separation", 2);
        _panel.AddChild(box);
        _title = MakeLabel(18, new Color(0.7f, 0.88f, 1f));
        _summary = MakeLabel(16, new Color(0.95f, 0.95f, 0.95f));
        _rows = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        _status = MakeLabel(13, new Color(0.7f, 0.7f, 0.7f));
        box.AddChild(_title); box.AddChild(_summary); box.AddChild(_rows); box.AddChild(_status);
        _layer.AddChild(_panel);
        root.CallDeferred(Node.MethodName.AddChild, _layer);
        _title.Text = "AI ヒント（AlphaDefect）";
    }

    Label MakeLabel(int size, Color color)
    {
        var l = new Label { MouseFilter = Control.MouseFilterEnum.Ignore, AutowrapMode = TextServer.AutowrapMode.Off };
        if (_font != null) l.AddThemeFontOverride("font", _font);
        l.AddThemeFontSizeOverride("font_size", size);
        l.AddThemeColorOverride("font_color", color);
        l.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0));
        l.AddThemeConstantOverride("outline_size", 4);
        return l;
    }

    public bool Visible { get => _layer.Visible; set => _layer.Visible = value; }

    public void SetTitle(string s) => _title.Text = s;
    public void SetStatus(string s) => _status.Text = s;

    /// <summary>判断の場面ではないとき：候補を消して理由だけ出す。</summary>
    public void Clear(string why)
    {
        _summary.Text = why;
        foreach (var c in _rows.GetChildren()) c.QueueFree();
    }

    /// <summary>サーバーが返したヒント（tools/hint/record.py の形）を表示する。</summary>
    public void Show(JsonNode rec)
    {
        foreach (var c in _rows.GetChildren()) c.QueueFree();
        var net = rec["net"];
        var parts = new List<string> { rec["decision"]?["screen"]?.GetValue<string>() ?? "" };
        if (net?["combat_win_prob"] is { } cw) parts.Add($"この戦闘に勝つ確率 {cw.GetValue<double>() * 100:F0}%");
        if (net?["combat_hp_loss_frac"] is { } hl && rec["state"]?["max_hp"] is { } mh)
            parts.Add($"失うHPの見込み {hl.GetValue<double>() * mh.GetValue<int>():F0}");
        if (net?["value"] is { } v) parts.Add($"価値 {v.GetValue<double>():F2}");
        _summary.Text = string.Join(" ・ ", parts.Where(p => p.Length > 0));

        var cands = rec["candidates"]?.AsArray();
        if (cands == null) return;
        var seen = new HashSet<int>();
        int shown = 0;
        foreach (var c in cands)
        {
            if (c == null) continue;
            // 同じ手（同じカード2枚など）は1行にまとめる
            if (c["same_as"] is { } sa && !seen.Add(sa.GetValue<int>())) continue;
            if (shown >= TopN) break;
            double p = (c["group_prob"] ?? c["prob"])?.GetValue<double>() ?? 0;
            string text = c["text"]?.GetValue<string>() ?? "?";
            var row = MakeLabel(16, shown == 0 ? new Color(1f, 0.85f, 0.35f) : new Color(0.9f, 0.9f, 0.9f));
            row.Text = $"{shown + 1}. {text}   {p * 100:F1}%{Facts(c["facts"])}";
            _rows.AddChild(row);
            shown++;
        }
    }

    static string Facts(JsonNode? f)
    {
        if (f == null) return "";
        var s = new List<string>();
        if (f["damage"]?["total"] is { } d && d.GetValue<int>() > 0) s.Add($"ダメ{d.GetValue<int>()}");
        if (f["block"] is { } b) s.Add($"ブロック{b.GetValue<int>()}");
        return s.Count > 0 ? "  (" + string.Join(" ", s) + ")" : "";
    }
}
