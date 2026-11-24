// ローカルのヒントサーバー（tools/hint/server.py）との通信。127.0.0.1 にだけつなぐ。
// 送る：[u32 長さ][u8 種類][本体]。種類 0 = あいさつ（JSON）、1 = 観測フレーム（[u32 通し番号] + エンジンと同じバイト列）。
// 受け取る：[u32 長さ][UTF-8 の JSON]。
// ゲームを止めないよう、通信は別スレッドで行い、結果はメインスレッドが Poll で受け取る。
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;

namespace Sts2Hint;

public sealed class HintClient
{
    readonly string _host;
    readonly int _port;
    readonly BlockingCollection<byte[]> _out = new(new ConcurrentQueue<byte[]>(), 4);
    readonly ConcurrentQueue<JsonNode> _in = new();
    volatile bool _connected;
    public bool Connected => _connected;
    public string Status { get; private set; } = "未接続";
    /// <summary>サーバーのあいさつ（重みの名前、観測格式、キャラ）。</summary>
    public JsonNode? Hello { get; private set; }

    public HintClient(string host, int port)
    {
        _host = host; _port = port;
        new Thread(Loop) { IsBackground = true, Name = "Sts2HintClient" }.Start();
    }

    /// <summary>最新のフレームだけ送ればよいので、詰まっていたら古いものを捨てる。</summary>
    public void SendFrame(uint seq, byte[] frame)
    {
        if (!_connected) return;
        var msg = new byte[1 + 4 + frame.Length];
        msg[0] = 1;
        BitConverter.TryWriteBytes(msg.AsSpan(1), seq);
        frame.CopyTo(msg, 5);
        while (!_out.TryAdd(msg)) _out.TryTake(out _);
    }

    public void SendJson(JsonNode j)
    {
        if (!_connected) return;
        var body = Encoding.UTF8.GetBytes(j.ToJsonString());
        var msg = new byte[1 + body.Length];
        msg[0] = 0;
        body.CopyTo(msg, 1);
        while (!_out.TryAdd(msg)) _out.TryTake(out _);
    }

    public bool TryReceive(out JsonNode? j) => _in.TryDequeue(out j);

    void Loop()
    {
        while (true)
        {
            try
            {
                using var tcp = new TcpClient();
                tcp.NoDelay = true;
                tcp.Connect(_host, _port);
                using var ns = tcp.GetStream();
                Status = "接続中";
                var hello = new JsonObject { ["type"] = "hello", ["client"] = "Sts2Hint/0.1" };
                Write(ns, 0, Encoding.UTF8.GetBytes(hello.ToJsonString()));
                Hello = JsonNode.Parse(Read(ns));
                while (_out.TryTake(out _)) { }
                _connected = true;
                Status = "接続済み";
                var reader = new Thread(() => ReadLoop(ns)) { IsBackground = true, Name = "Sts2HintReader" };
                reader.Start();
                while (_connected)
                {
                    if (!_out.TryTake(out var msg, 500)) continue;
                    var len = BitConverter.GetBytes(msg.Length);
                    ns.Write(len); ns.Write(msg);
                }
            }
            catch (Exception e)
            {
                Status = $"ヒントサーバーに接続できません（{_host}:{_port}）: {e.GetBaseException().Message}";
            }
            _connected = false;
            Thread.Sleep(3000);
        }
    }

    void ReadLoop(NetworkStream ns)
    {
        try
        {
            while (_connected)
            {
                var j = JsonNode.Parse(Read(ns));
                if (j != null) _in.Enqueue(j);
            }
        }
        catch (Exception e)
        {
            Status = "切断されました: " + e.GetBaseException().Message;
            _connected = false;
        }
    }

    static void Write(NetworkStream ns, byte type, byte[] body)
    {
        ns.Write(BitConverter.GetBytes(body.Length + 1));
        ns.WriteByte(type);
        ns.Write(body);
    }

    static byte[] Read(NetworkStream ns)
    {
        var head = ReadExact(ns, 4);
        int n = BitConverter.ToInt32(head);
        if (n < 0 || n > 64 << 20) throw new IOException($"不正な長さ {n}");
        return ReadExact(ns, n);
    }

    static byte[] ReadExact(NetworkStream ns, int n)
    {
        var buf = new byte[n];
        int got = 0;
        while (got < n)
        {
            int k = ns.Read(buf, got, n - got);
            if (k <= 0) throw new IOException("サーバーが接続を閉じました");
            got += k;
        }
        return buf;
    }
}
