import { useCallback, useEffect, useState } from "react";
import {
  ShieldCheck, AlertTriangle, Skull, Sparkles, Shuffle, Gauge,
  Network, Scissors, BookOpen, Cpu, Layers, FlaskConical, Lightbulb,
} from "lucide-react";
import { type Label, type Prediction, type CompareResponse, predict, compare, sample } from "./api";
import Walkthrough from "./Walkthrough";
import { PaperIdeal, WhyFRF, FutureWork } from "./Essays";

const LABEL_META: Record<Label, { vn: string; color: string; Icon: typeof ShieldCheck }> = {
  CLEAN: { vn: "An toàn", color: "#22a06b", Icon: ShieldCheck },
  OFFENSIVE: { vn: "Công kích", color: "#e0913d", Icon: AlertTriangle },
  HATE: { vn: "Thù ghét", color: "#d44747", Icon: Skull },
};

const EXAMPLES = [
  "Hôm nay thời tiết đẹp quá, mình đi chơi nhé các bạn",
  "Bạn viết bài hay quá, mình học được nhiều điều",
  "Thằng này ngu gì mà mãi không hiểu, bọn mày làm ăn kiểu gì vậy",
  "Đồ khùng, suốt ngày nói nhảm, mày có bị điên không",
  "M.n ơi cho mik hỏi cô này là ai vậy ạ",
];

export default function App() {
  const [text, setText] = useState("");
  const [pred, setPred] = useState<Prediction | null>(null);
  const [cmp, setCmp] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<0 | 1 | 2>(0);
  const [err, setErr] = useState("");

  const run = useCallback(async (t: string) => {
    if (!t.trim()) { setErr("Nhập một bình luận để phân tích."); return; }
    setErr(""); setLoading(true);
    try { setPred(await predict(t)); }
    catch (e) { setErr("Không kết nối được tới API. Đảm bảo backend đang chạy ở :8000."); }
    finally { setLoading(false); }
    compare(t).then(setCmp).catch(() => setCmp(null));
  }, []);

  useEffect(() => { run(EXAMPLES[2]); setText(EXAMPLES[2]); }, [run]);

  const pickSample = async () => {
    const s = await sample();
    if (s.text) { setText(s.text); run(s.text); }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">FRF</div>
        <div>
          <h1>FRF-MLP · Bộ lọc ngôn từ công kích tiếng Việt</h1>
          <div className="sub">
            Perceptron đa lớp kết hợp fuzzy logic Mamdani · phát hiện CLEAN / OFFENSIVE / HATE trên bình luận mạng xã hội
          </div>
        </div>
        <div className="badges">
          <span className="badge dot">API đang chạy</span>
          <span className="badge">ViHSD · 3 lớp</span>
        </div>
      </header>

      <section className="hero">
        <p className="hero-lede">
          Mô hình FRF-MLP chỉ <strong>10,3 triệu tham số</strong> với trọng số <strong>42 MB</strong>,
          suy diễn <strong>~3 mili-giây / bình luận trên CPU</strong> thông thường —
          đủ nhanh để kiểm duyệt <strong>thời gian thực</strong> trong đường ống mạng xã hội,
          chạy trên server giá rẻ hoặc ngay tại biên (edge), <strong>không cần GPU hay mô hình tiền huấn luyện</strong>.
          Để so sánh, m-BERT đa ngữ — baseline tốt nhất trên ViHSD — cần <em>178 triệu tham số</em>,
          <em>~700 MB</em> và thường yêu cầu GPU; FRF-MLP nhẹ hơn khoảng <strong>17×</strong> về tham số
          mà vẫn vượt m-BERT về macro-F1 (63,00% so với 62,69%), nhờ tận dụng tri thức ngôn ngữ mờ.
        </p>
        <div className="hero-stats">
          <div className="stat"><span className="stat-n">10,3M</span><span className="stat-l">tham số</span></div>
          <div className="stat"><span className="stat-n">42 MB</span><span className="stat-l">trọng số + vectorizer</span></div>
          <div className="stat"><span className="stat-n">~3 ms</span><span className="stat-l">suy diễn / bình luận (CPU)</span></div>
          <div className="stat"><span className="stat-n">0</span><span className="stat-l">GPU / tiền huấn luyện</span></div>
          <div className="stat vs"><span className="stat-n">17×</span><span className="stat-l">nhẹ hơn m-BERT</span></div>
        </div>
      </section>

      <nav className="tabs">
        <button className={tab === 0 ? "tab on" : "tab"} onClick={() => setTab(0)}><Cpu size={14} /> Bảng điều khiển</button>
        <button className={tab === 1 ? "tab on" : "tab"} onClick={() => setTab(1)}><FlaskConical size={14} /> Giải trình từng bước</button>
        <button className={tab === 2 ? "tab on" : "tab"} onClick={() => setTab(2)}><Lightbulb size={14} /> Ý tưởng paper &amp; cải tiến</button>
      </nav>
      {tab === 1 && <div className="tab-pane"><Walkthrough pred={pred} cmp={cmp} /></div>}
      {tab === 2 && <div className="tab-pane"><PaperIdeal /><WhyFRF /><FutureWork /></div>}

      {tab === 0 && (
      <div className="grid">
        {/* LEFT: composer + verdict */}
        <div className="card">
          <h2><Cpu size={15} /> Bình luận cần kiểm duyệt</h2>
          <div className="comment-shell">
            <div className="comment-head">
              <div className="avatar">U</div>
              <div>
                <div className="name">Người dùng ẩn danh</div>
                <div className="meta">bình luận · mạng xã hội</div>
              </div>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Dán vào một bình luận tiếng Việt… (ví dụ: mày là thằng ngu)"
              onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(text); }}
            />
          </div>
          <div className="actions">
            <button className="primary" onClick={() => run(text)} disabled={loading}>
              {loading ? <span className="spinner" /> : <Sparkles size={15} />}
              <span style={{ marginLeft: 7 }}>{loading ? "Đang phân tích…" : "Phân tích"}</span>
            </button>
            <button className="ghost" onClick={pickSample}><Shuffle size={14} style={{ marginRight: 5, verticalAlign: "-2px" }} />Mẫu ngẫu nhiên</button>
          </div>
          <div className="hint">Mẹo: Ctrl/⌘ + Enter để phân tích nhanh. Thử các mẫu dưới đây:</div>
          <div className="actions">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="ghost" onClick={() => { setText(ex); run(ex); }} style={{ maxWidth: "100%", textAlign: "left", opacity: 0.85 }}>
                {ex.length > 38 ? ex.slice(0, 38) + "…" : ex}
              </button>
            ))}
          </div>
          {err && <div style={{ color: "var(--hate)", fontSize: 13, marginTop: 12 }}>{err}</div>}

          {pred && <Verdict pred={pred} />}
        </div>

        {/* RIGHT: explainability */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div className="card">
            <h2><Network size={15} /> Phân bố xác suất 3 lớp</h2>
            <ProbBars pred={pred} />
            <FusionStrip pred={pred} />
          </div>

          <div className="card">
            <h2><Gauge size={15} /> Logic mờ — biến ngôn ngữ &amp; hàm thành viên</h2>
            <FuzzyPanel pred={pred} />
          </div>

          <div className="card">
            <h2><Layers size={15} /> Hệ 7 luật Mamdani</h2>
            <RulesPanel pred={pred} />
          </div>

          <div className="card">
            <h2><Scissors size={15} /> Token công kích (z-score log-odds)</h2>
            <TokensPanel pred={pred} />
          </div>
        </div>
      </div>
      )}

      <footer className="footer">
        <span><BookOpen size={12} style={{ verticalAlign: "-2px", marginRight: 4 }} />FRF-MLP · đồ án Học máy nâng cao</span>
        <span>MLP 256–128 + 22 đặc trưng mờ + fusion quyết định λ</span>
        <span>Demo giáo dục — kết quả mang tính tham khảo, không phải công cụ kiểm duyệt production.</span>
      </footer>
    </div>
  );
}

/* ---------- components ---------- */

function Verdict({ pred }: { pred: Prediction }) {
  const m = LABEL_META[pred.label];
  const Icon = m.Icon;
  const conf = pred.probabilities[pred.label];
  return (
    <div className="verdict" style={{ borderColor: m.color, boxShadow: `0 0 0 1px ${m.color}22, 0 8px 30px ${m.color}22` }}>
      <div className="ring" style={{ background: m.color, boxShadow: `0 0 22px ${m.color}66` }}>
        <Icon />
      </div>
      <div>
        <div className="vlabel" style={{ color: m.color }}>{m.vn.toUpperCase()}</div>
        <div className="vsub">{pred.label} · {pred.n_tokens} token · λ = {pred.lambda}</div>
      </div>
      <div className="conf">
        <div className="n" style={{ color: m.color }}>{(conf * 100).toFixed(1)}<span style={{ fontSize: 14, opacity: 0.7 }}>%</span></div>
        <div className="t">độ tinậy</div>
      </div>
    </div>
  );
}

function ProbBars({ pred }: { pred: Prediction | null }) {
  if (!pred) return null;
  const order: Label[] = ["HATE", "OFFENSIVE", "CLEAN"];
  return (
    <div className="probs">
      {order.map((l) => {
        const m = LABEL_META[l];
        return (
          <div className="prob" key={l}>
            <div className="top">
              <span className="lbl" style={{ color: m.color }}>{l} · {m.vn}</span>
              <span className="pct">{(pred.probabilities[l] * 100).toFixed(2)}%</span>
            </div>
            <div className="track">
              <div className="fill" style={{ width: `${pred.probabilities[l] * 100}%`, background: m.color }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function FusionStrip({ pred }: { pred: Prediction | null }) {
  if (!pred) return null;
  const row = (name: string, p: Record<Label, number>, tone: string) => {
    const top = Object.entries(p).sort((a, b) => b[1] - a[1])[0];
    return (
      <div className="row" key={name}>
        <span style={{ width: 46, color: "var(--text-faint)", fontSize: 11 }}>{name}</span>
        <div className="pill" style={{ background: `${tone}22`, color: tone }}>{top[0]} · {(top[1] * 100).toFixed(1)}%</div>
        <span style={{ color: "var(--text-faint)", fontSize: 11 }}>
          {(["CLEAN", "OFFENSIVE", "HATE"] as Label[]).map((l) => `${l[0]}:${(p[l] * 100).toFixed(0)}`).join("  ")}
        </span>
      </div>
    );
  };
  return (
    <div className="fusion">
      <strong style={{ color: "var(--text)" }}>Kết hợp 2 kênh</strong> &nbsp;·&nbsp; MLP thống kê (TF-IDF) + hệ luật mờ
      {row("MLP", pred.p_mlp, "#4c8dff")}
      {row("Mờ", pred.p_fuzzy, "#8b5cf6")}
      <div className="row">
        <span style={{ width: 46, color: "var(--text-faint)", fontSize: 11 }}>→ FRF</span>
        <span style={{ fontSize: 12 }}>
          p = (1−{pred.lambda})·p<sub>MLP</sub> + {pred.lambda}·p<sub>mờ</sub>
        </span>
      </div>
    </div>
  );
}

function FuzzyPanel({ pred }: { pred: Prediction | null }) {
  if (!pred) return null;
  return (
    <div>
      {pred.memberships.map((m) => (
        <div className="var" key={m.name}>
          <div className="head">
            <span>{m.name}</span>
            <span className="val">v = {m.value.toFixed(3)}</span>
          </div>
          <div className="mf-row">
            <span className="tag">LOW<br /><span style={{ color: "var(--clean)" }}>{(m.low * 100).toFixed(0)}</span></span>
            <div className="mf-bar"><div className="mf-low" style={{ width: `${m.low * 100}%` }} /></div>
            <span className="tag" style={{ opacity: 0 }}>·</span>
          </div>
          <div className="mf-row" style={{ marginTop: 3 }}>
            <span className="tag">MED<br /><span style={{ color: "var(--offensive)" }}>{(m.med * 100).toFixed(0)}</span></span>
            <div className="mf-bar"><div className="mf-med" style={{ width: `${m.med * 100}%` }} /></div>
            <span className="tag" style={{ opacity: 0 }}>·</span>
          </div>
          <div className="mf-row" style={{ marginTop: 3 }}>
            <span className="tag">HIGH<br /><span style={{ color: "var(--hate)" }}>{(m.high * 100).toFixed(0)}</span></span>
            <div className="mf-bar"><div className="mf-high" style={{ width: `${m.high * 100}%` }} /></div>
            <span className="tag" style={{ opacity: 0 }}>·</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function RulesPanel({ pred }: { pred: Prediction | null }) {
  if (!pred) return null;
  return (
    <div className="rules">
      {pred.rules.map((r, i) => {
        const fired = r.strength > 0.01;
        return (
          <div className={`rule ${fired ? "fired" : ""}`} key={i}>
            <span className="nm">{r.name}</span>
            <span className="str">{r.strength.toFixed(2)}</span>
            <div className="bar"><div style={{ width: `${r.strength * 100}%` }} /></div>
          </div>
        );
      })}
    </div>
  );
}

function TokensPanel({ pred }: { pred: Prediction | null }) {
  if (!pred) return null;
  if (!pred.highlights.length) return <div className="empty">Không có token vượt ngưỡng |z| ≥ 2 — bình luận không chứa từ công kích đặc trưng.</div>;
  return (
    <div className="tokens">
      {pred.highlights.map((h, i) => (
        <span key={i} className={`tok ${h.z > 0 ? "hate" : "clean"}`} title={`log-odds z = ${h.z}`}>
          {h.token}<span className="z">{h.z > 0 ? "+" : ""}{h.z}</span>
        </span>
      ))}
    </div>
  );
}
