import { useEffect, useRef, useState } from "react";
import {
  Filter, Type, BookMarked, Waves, GitBranch, Sigma, Network, Merge,
  Play, Pause, StepForward, List, FlaskConical, Lightbulb, Pi, GitCompare,
} from "lucide-react";
import {
  type Label, type Prediction, type CompareResponse, type StepContent,
  type S1, type S2, type S3, type S4, type S5, type S6, type S7, type S8,
  type MuDetail, type TokZ, type RuleLive, type ClassScore, type LogitPair, type FeatTerm,
} from "./api";

/* ============================================================
   Walkthrough — giải trình 8 bước FRF-MLP trên câu đang phân tích.
   - Mở hết (không accordion), TOC dính bên trái + scrollspy.
   - 2 chế độ: ▶ Tự chạy (auto-play) · Chạy từng bước (bấm mới mở).
   - Mỗi bước: mục tiêu + ý tưởng + công thức + BUNG SỐ THẬT + live data.
   - Bước phụ: So sánh trực tiếp 3 mô hình trên cùng câu.
   ============================================================ */

interface StepDef { n: number; title: string; icon: typeof Filter; goal: string; idea: string; formula: string }

const STEPS: StepDef[] = [
  { n: 1, title: "Tiền xử lý", icon: Filter,
    goal: "Đưa bình luận thô về dạng đều, giữ tín hiệu công kích.",
    idea: "Chuyển chữ thường, che URL/số/@mention, rút ký tự lặp (keooooo→keoo). GIỮ teencode & từ tục viết tắt (vcl, đm, cc) — đó là tín hiệu phân loại mạnh nhất. Cố ý không bỏ dấu câu teencode.",
    formula: "normalize(t) = lower( strip_urls( strip_nums( dedup(t) ) ) )" },
  { n: 2, title: "Vector TF-IDF", icon: Type,
    goal: "Biến văn bản thành vector số thưa để đưa vào MLP.",
    idea: "N-gram từ (1–2) + n-gram ký tự trong biên từ (2–4), trọng số TF-IDF sublinear. N-gram ký tự bắt teencode/biến thể chính tả — đặc trưng cố hữu của bình luận mạng xã hội tiếng Việt. Tổng 40.000 chiều.",
    formula: "tfidf(t,d) = (1+log tf) · log(N / df)" },
  { n: 3, title: "Lexicon + biến ngôn ngữ", icon: BookMarked,
    goal: "Định lượng mức công kích từng token KHÔNG cần từ điển thủ công.",
    idea: "So sánh tần suất token giữa nhóm {OFFENSIVE,HATE} với CLEAN bằng log-odds có tiên nghiệm Dirichlet → z-score (z>0=công kích, z<0=sạch). Rút 3 biến văn bản: S (độ công kích cực đại), D (mật độ từ công kích), T (độ nhắm đích — tỷ lệ đại từ mày/thằng/bọn/lũ). T là chìa phân biệt HATE (nhắm đích) với OFFENSIVE (tục nhưng không nhắm).",
    formula: "z_w = δ_w / √(1/(y_o+α) + 1/(y_c+α)),  δ=log-odds(off)−log-odds(clean)" },
  { n: 4, title: "Mờ hóa (hàm thành viên)", icon: Waves,
    goal: "Biến S, D, T (số) thành độ thuộc LOW/MED/HIGH — dạng mờ [0,1].",
    idea: "Hàm thành viên hình thang. Dùng logic mờ vì ranh giới CLEAN/OFFENSIVE/HATE vốn MỜ — kappa đồng thuận người gán nhãn ViHSD chỉ 0,52 — nên độ thuộc liên tục tự nhiên hơn ngưỡng cứng.",
    formula: "μ_[a,b,c,d](v) = clip( min((v−a)/(b−a), (d−v)/(d−c)), 0, 1 )" },
  { n: 5, title: "Suy diễn 7 luật Mamdani", icon: GitBranch,
    goal: "Mã hóa tri thức ngôn ngữ thành luật IF–THEN đọc được.",
    idea: "Vd R5: 'S HIGH ∧ (T MED∨HIGH) → HATE'. Antecedent kết bằng t-norm min (lấy giá trị nhỏ nhất — luật chỉ mạnh khi TẤT CẢ điều kiện thỏa). 7 luật phủ kín tổ hợp S/D/T.",
    formula: "r_k = w_k · min( μ_ant₁, μ_ant₂, … )" },
  { n: 6, title: "Giải mờ → p_mờ", icon: Sigma,
    goal: "Từ độ kích hoạt luật ra phân phối 3 lớp của kênh tri thức.",
    idea: "Cộng điểm cho mỗi lớp theo các luật kết luận về lớp đó (+ thiên lệch nhỏ về CLEAN khi không luật nào kích), rồi chuẩn hoá tổng=1. Đây là đầu ra của kênh logic mờ — diễn giải được, không cần huấn luyện.",
    formula: "p_mờ(c) = (Σ_{k: kq=c} r_k + b_c) / Σ_c′(…)" },
  { n: 7, title: "MLP 256–128 → p_MLP", icon: Network,
    goal: "Kênh thống kê: học tương tác phi tuyến từ 40.000 đặc trưng.",
    idea: "TF-IDF (40.000) ⊕ 22 đặc trưng mờ (3 biến + 9 μ + 7 luật + 3 p_mờ) = 40.022 đầu vào, qua 2 lớp ẩn 256–128 (ReLU + dropout), softmax ra p_MLP. Huấn luyện lan truyền ngược, entropy chéo có trọng số lớp xử lý mất cân bằng (CLEAN 83%).",
    formula: "h₁=ReLU(W₁x+b₁) · h₂=ReLU(W₂h₁+b₂) · p_MLP=softmax(W₃h₂+b₃)" },
  { n: 8, title: "Fusion quyết định", icon: Merge,
    goal: "Hợp nhất kênh thống kê (MLP) và kênh tri thức (mờ).",
    idea: "p = (1−λ)·p_MLP + λ·p_mờ, λ quét trên dev. λ nhỏ (≈0,05–0,25) = tin MLP chủ yếu, hệ mờ chỉ HIỆU CHỈNH — đúng kỳ vọng vì hệ mờ standalone yếu (macro-F1 41,6%) nhưng bổ chính xác các trường hợp MLP nhầm ở biên CLEAN/OFFENSIVE.",
    formula: "p = (1−λ)·p_MLP + λ·p_mờ,  λ*=argmax_λ macroF1(dev)" },
];

const NUM = (x: number, d = 2) => Number(x).toFixed(d);
const MODEL_NAME: Record<string, string> = {
  softmax: "Softmax regression", fuzzy: "Hệ mờ thuần", frf: "FRF-MLP (lai)",
};

export default function Walkthrough({ pred, cmp }: { pred: Prediction | null; cmp: CompareResponse | null }) {
  const [active, setActive] = useState(1);
  const [revealed, setRevealed] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const stepsRef = useRef<HTMLDivElement>(null);

  const scrollTo = (n: number) => {
    document.getElementById(`step-${n}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActive(n);
    setRevealed((r) => (r !== null && n > r ? n : r));
  };

  useEffect(() => {
    const root = stepsRef.current;
    if (!root) return;
    const secs = Array.from(root.querySelectorAll<HTMLElement>("[data-step]"));
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (vis) setActive(Number((vis.target as HTMLElement).dataset.step));
    }, { rootMargin: "-30% 0px -55% 0px", threshold: [0.15, 0.4] });
    secs.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, [pred]);

  useEffect(() => {
    if (!playing) return;
    if (active >= 9) { setPlaying(false); return; }
    const id = setTimeout(() => scrollTo(active + 1), 2200);
    return () => clearTimeout(id);
  }, [playing, active]);

  if (!pred)
    return <div className="empty">Phân tích một bình luận ở tab 'Bảng điều khiển' để xem giải trình từng bước trên chính câu đó.</div>;

  const s1 = pred.steps[0]?.content as S1 | undefined;
  const locked = (n: number) => revealed !== null && n > revealed;

  return (
    <div className="walk">
      <div className="walk-head">
        <h3><FlaskConical size={16} /> Đường ống FRF-MLP — 8 bước, bung số thật trên câu đang phân tích</h3>
        <p>
          Hai kênh chạy song song: <strong>thống kê</strong> (bước 2+7) và <strong>tri thức mờ</strong> (bước 3–6), hợp nhất ở bước 8.
          Đang giải cho: <em>“{(s1?.raw ?? "").slice(0, 80)}”</em>
        </p>
        <div className="walk-ctrl">
          <button className={`wc ${playing ? "on" : ""}`} onClick={() => { if (!playing) scrollTo(1); setPlaying(!playing); }}>
            {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? "Dừng" : "Tự chạy 8 bước"}
          </button>
          <button className="wc" onClick={() => { setRevealed(revealed === null ? 1 : null); setPlaying(false); }}>
            <StepForward size={14} /> {revealed === null ? "Chạy từng bước" : `Đang ở bước ${revealed}/9`}
          </button>
          {revealed !== null && (
            <button className="wc primary" onClick={() => scrollTo(Math.min(revealed + 1, 9))} disabled={revealed >= 9}>
              <StepForward size={14} /> Bước tiếp theo
            </button>
          )}
          {revealed !== null && (
            <button className="wc ghost" onClick={() => setRevealed(null)}><List size={14} /> Hiện tất cả</button>
          )}
        </div>
      </div>

      <div className="walk-body">
        <aside className="toc">
          {STEPS.map((s) => (
            <button key={s.n} className={`toc-item ${active === s.n ? "on" : ""}`} onClick={() => scrollTo(s.n)}>
              <span className="toc-n">{s.n}</span><span className="toc-t">{s.title}</span>
            </button>
          ))}
          <button className={`toc-item cmp ${active === 9 ? "on" : ""}`} onClick={() => scrollTo(9)}>
            <GitCompare size={13} /><span className="toc-t">So sánh các mô hình</span>
          </button>
        </aside>

        <div className="steps" ref={stepsRef}>
          {STEPS.map((s) => {
            const Icon = s.icon;
            const content = pred.steps[s.n - 1]?.content;
            const isLocked = locked(s.n);
            return (
              <section key={s.n} id={`step-${s.n}`} data-step={s.n}
                className={`wstep ${active === s.n ? "focus" : ""} ${isLocked ? "locked" : ""}`}>
                <header className="shead">
                  <span className="snum">{s.n}</span>
                  <Icon size={16} className="sicon" />
                  <div><div className="stitle">{s.title}</div><div className="sgoal">{s.goal}</div></div>
                </header>
                {isLocked ? (
                  <div className="slocked">Bước đang khoá — bấm <strong>“Bước tiếp theo”</strong> để mở và xem số liệu.</div>
                ) : content ? (
                  <div className="scontent">
                    <div className="sblock idea"><span className="slbl"><Lightbulb size={12} /> Ý tưởng</span><p>{s.idea}</p></div>
                    <div className="sblock"><span className="slbl"><Pi size={12} /> Công thức tổng quát</span><code className="formula">{s.formula}</code></div>
                    <Calc n={s.n} content={content} pred={pred} />
                    <div className="sblock"><span className="slbl"><FlaskConical size={12} /> Với câu này</span><StepData n={s.n} content={content} /></div>
                  </div>
                ) : null}
              </section>
            );
          })}

          <section id="step-9" data-step={9} className={`wstep cmp ${active === 9 ? "focus" : ""}`}>
            <header className="shead">
              <span className="snum"><GitCompare size={14} /></span>
              <div><div className="stitle">So sánh trực tiếp: softmax · fuzzy · FRF-MLP</div>
                <div className="sgoal">Cùng câu, 3 mô hình — bắt bẻ “tại sao không dùng cái kia”.</div></div>
            </header>
            <div className="scontent"><ComparePanel cmp={cmp} /></div>
          </section>
        </div>
      </div>
    </div>
  );
}

/* =================== BUNG SỐ (công thức thay số thật) =================== */
function Calc({ n, content, pred }: { n: number; content: StepContent; pred: Prediction }) {
  const lines = calcLines(n, content, pred);
  if (!lines.length) return null;
  return (
    <div className="sblock calc">
      <span className="slbl"><Pi size={12} /> Bung số (thay biến bằng giá trị của câu này)</span>
      <div className="calc-list">{lines.map((ln, i) => <code key={i} className="calc-line">{ln}</code>)}</div>
    </div>
  );
}

function calcLines(n: number, content: StepContent, pred: Prediction): string[] {
  if (n === 3) {
    const s = content as S3;
    const names = ["S", "D", "T"];
    return names.map((nm, j) => {
      const raw = s.crisp_raw[j], lo = s.bounds_lo[j], hi = s.bounds_hi[j];
      const den = hi - lo, r = den === 0 ? 0 : (raw - lo) / den;
      const clamped = Math.max(0, Math.min(1, r));
      return `${nm}_norm = clip((${NUM(raw)} − ${NUM(lo)}) / (${NUM(hi)} − ${NUM(lo)}), 0, 1) = clip(${NUM(raw - lo)}/${NUM(den)}, 0, 1) = ${NUM(clamped, 3)}`;
    });
  }
  if (n === 4) {
    const s = content as S4;
    return ["S", "D", "T"].map((v) => {
      const m: MuDetail | undefined = s.mu_detail.filter((x) => x.var === v).sort((a, b) => b.mu - a.mu)[0];
      if (!m || m.mu <= 0.001) return `${v}: không thuộc mức nào đáng kể (mọi μ ≈ 0).`;
      const lhs = (m.v - m.a) / Math.max(m.b - m.a, 1e-9);
      const rhs = (m.d - m.v) / Math.max(m.d - m.c, 1e-9);
      return `μ_${v}=${m.level}: clip(min((v−a)/(b−a),(d−v)/(d−c)),0,1) = clip(min((${NUM(m.v)}−${NUM(m.a)})/${NUM(m.b - m.a)},(${NUM(m.d)}−${NUM(m.v)})/${NUM(m.d - m.c)}),0,1) = clip(min(${NUM(lhs)},${NUM(rhs)}),0,1) = ${NUM(m.mu, 3)}`;
    });
  }
  if (n === 5) {
    const s = content as S5;
    const fired = s.rules.filter((r: RuleLive) => r.strength > 0.01);
    if (!fired.length) return ["Không luật nào kích hoạt (mọi antecedent = 0) → chỉ thiên lệch CLEAN."];
    return fired.map((r) => {
      const parts = r.antecedents.map((a) => NUM(a.value, 2));
      const mn = Math.min(...r.antecedents.map((a) => a.value));
      return `${r.name.split("→")[0].trim()} → min(${parts.join(", ")})·w=${NUM(r.weight)} = ${NUM(mn)}·${NUM(r.weight)} = ${NUM(r.strength, 3)} ⇒ ${r.conclusion}`;
    });
  }
  if (n === 6) {
    const s = content as S6;
    const tot = s.class_score.reduce((a: number, c: ClassScore) => a + c.score, 0) || 1;
    return s.class_score.map((c) => `p_mờ(${c.label}) = ${NUM(c.score, 3)} / ${NUM(tot, 3)} = ${NUM(c.score / tot, 4)}`);
  }
  if (n === 7) {
    const s = content as S7;
    const maxL = Math.max(...s.logits.map((l: LogitPair) => l.logit));
    const exps = s.logits.map((l) => Math.exp(l.logit - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);
    return s.logits.map((l, i) => `p_MLP(${l.label}) = exp(${NUM(l.logit)}) / Σexp = ${NUM(exps[i], 3)} / ${NUM(sum, 3)} = ${NUM(exps[i] / sum, 4)}`);
  }
  if (n === 8) {
    const s = content as S8;
    const lam = s.lambda;
    return (Object.keys(s.final) as Label[]).map((c) => {
      const mlp = pred.p_mlp[c] ?? 0, mu = pred.p_fuzzy[c] ?? 0;
      return `p(${c}) = (1−${NUM(lam)})·${NUM(mlp, 3)} + ${NUM(lam)}·${NUM(mu, 3)} = ${NUM((1 - lam) * mlp, 3)} + ${NUM(lam * mu, 3)} = ${NUM(s.final[c] * 100, 2)}%`;
    });
  }
  return [];
}

/* =================== BẢNG SỐ LIỆU LIVE =================== */
function StepData({ n, content }: { n: number; content: StepContent }) {
  if (n === 1) {
    const s = content as S1;
    return (
      <div className="live">
        <div className="kv"><span>Raw</span><code>{s.raw}</code></div>
        <div className="kv"><span>Chuẩn hoá</span><code>{s.normalized}</code></div>
        <div className="kv"><span>{s.n_tokens} token</span><code className="toks">{s.tokens_sample.join(" · ")}</code></div>
      </div>
    );
  }
  if (n === 2) {
    const s = content as S2;
    return (
      <div className="live">
        <div className="kv"><span>Đặc trưng ≠ 0</span><code>{s.nnz} / 40.000 chiều</code></div>
        <div className="feats">
          <div className="ftitle">Top n-gram từ (TF-IDF):</div>
          {s.top_word.map((f: FeatTerm, i) => <Feat key={i} {...f} />)}
          <div className="ftitle">Top n-gram ký tự:</div>
          {s.top_char.map((f: FeatTerm, i) => <Feat key={i} {...f} />)}
        </div>
      </div>
    );
  }
  if (n === 3) {
    const s = content as S3;
    const vars = ["S — độ công kích", "D — mật độ", "T — nhắm đích"];
    return (
      <div className="live">
        <div className="feats">
          <div className="ftitle">Token + z-score log-odds:</div>
          {s.tokens_z.map((t: TokZ, i) => (
            <span key={i} className={`chip ${t.z > 0 ? "pos" : t.z < 0 ? "neg" : ""}`}>{t.token}<b>{t.z > 0 ? "+" : ""}{NUM(t.z)}</b></span>
          ))}
        </div>
        <table className="vtable">
          <thead><tr><th>Biến</th><th>Raw</th><th>Phân vị train [lo, hi]</th><th>Chuẩn hoá [0,1]</th></tr></thead>
          <tbody>{[0, 1, 2].map((j) => (
            <tr key={j}><td>{vars[j]}</td><td>{NUM(s.crisp_raw[j], 3)}</td>
              <td>[{NUM(s.bounds_lo[j])}, {NUM(s.bounds_hi[j])}]</td><td><b>{NUM(s.crisp_norm[j], 3)}</b></td></tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  if (n === 4) {
    const s = content as S4;
    return (
      <div className="live mu-grid">
        {s.mu.map((m, i) => (
          <div key={i} className={`mu-cell ${m.value > 0.01 ? "on" : ""}`}>
            <div className="mu-lbl">{m.label}</div>
            <div className="mu-bar"><div style={{ width: `${m.value * 100}%` }} /></div>
            <div className="mu-val">{NUM(m.value, 3)}</div>
          </div>
        ))}
      </div>
    );
  }
  if (n === 5) {
    const s = content as S5;
    return (
      <div className="live rules-live">
        {s.rules.map((r: RuleLive, i) => (
          <div key={i} className={`rlive ${r.strength > 0.01 ? "fired" : ""}`}>
            <div className="rname">{r.name}</div>
            <div className="rante">{r.antecedents.map((a) => `${a.label.split("=")[0]}=${NUM(a.value, 2)}`).join("  ·  ")}</div>
            <div className="rbar"><div style={{ width: `${r.strength * 100}%` }} /></div>
            <div className="rstr">w={NUM(r.weight)} → {r.conclusion} ({NUM(r.strength, 3)})</div>
          </div>
        ))}
      </div>
    );
  }
  if (n === 6) {
    const s = content as S6;
    return (
      <div className="live">
        <div className="feats">{s.class_score.map((c: ClassScore) => (
          <div key={c.label} className="cscore">
            <span>{c.label}</span>
            <div className="mu-bar"><div style={{ width: `${Math.min(c.score, 1) * 100}%` }} /></div>
            <b>{NUM(c.score, 3)}</b>
          </div>
        ))}</div>
        <div className="kv"><span>p_mờ (chuẩn hoá)</span><code>{(Object.entries(s.p_fuzzy) as [string, number][]).map(([k, v]) => `${k}:${NUM(v * 100, 1)}%`).join("  ")}</code></div>
      </div>
    );
  }
  if (n === 7) {
    const s = content as S7;
    return (
      <div className="live">
        <div className="kv"><span>Lớp ẩn 1 (256)</span><code>{s.h1_active} neuron hoạt động (ReLU&gt;0)</code></div>
        <div className="kv"><span>Lớp ẩn 2 (128)</span><code>{s.h2_active} neuron hoạt động</code></div>
        <div className="feats">
          <div className="ftitle">Logit 3 lớp (trước softmax):</div>
          {s.logits.map((l: LogitPair) => (
            <div key={l.label} className="cscore">
              <span>{l.label}</span>
              <div className="mu-bar"><div className={l.logit >= 0 ? "pos" : "neg"} style={{ width: `${Math.min(Math.abs(l.logit) / 6, 1) * 100}%` }} /></div>
              <b>{l.logit > 0 ? "+" : ""}{NUM(l.logit)}</b>
            </div>
          ))}
          <div className="ftitle">p_MLP (sau softmax):</div>
          <code className="pp">{(Object.entries(s.p_mlp) as [string, number][]).map(([k, v]) => `${k}: ${NUM(v * 100, 2)}%`).join("   ")}</code>
        </div>
      </div>
    );
  }
  if (n === 8) {
    const s = content as S8;
    const fin = (Object.entries(s.final) as [Label, number][]).sort((a, b) => b[1] - a[1]);
    return (
      <div className="live">
        <div className="kv"><span>λ*</span><code>{NUM(s.lambda)} (chọn trên tập dev)</code></div>
        <div className="feats">{fin.map(([k, v]) => (
          <div key={k} className="cscore">
            <span>{k}{k === s.label ? " ✓" : ""}</span>
            <div className="mu-bar"><div className={k === s.label ? "win" : ""} style={{ width: `${v * 100}%` }} /></div>
            <b>{NUM(v * 100, 2)}%</b>
          </div>
        ))}</div>
        <div className="verdict-line">→ Nhãn dự đoán: <strong>{s.label}</strong></div>
      </div>
    );
  }
  return null;
}

function Feat({ term, weight }: FeatTerm) {
  return <span className="chip ghost">{term}<b>{NUM(weight, 3)}</b></span>;
}

/* =================== SO SÁNH 3 MÔ HÌNH =================== */
function ComparePanel({ cmp }: { cmp: CompareResponse | null }) {
  if (!cmp) return <div className="empty">Đang tải so sánh… phân tích một câu ở tab 'Bảng điều khiển'.</div>;
  return (
    <div className="cmp-panel">
      <p className="lede">Cùng một câu chạy qua 3 mô hình. Xem cột <strong>xác suất</strong> và <strong>nhãn</strong>:
        chỗ nào FRF-MLP khác softmax/fuzzy chính là chỗ kênh mờ hiệu chỉnh — đó là lý do thiết kế lai.</p>
      <div className="cmp-grid">
        {cmp.models.map((m) => (
          <div key={m.model} className={`cmp-card ${m.available === false ? "off" : ""}`}>
            <div className="cmp-h">{MODEL_NAME[m.model] ?? m.model}</div>
            {m.available === false ? (
              <div className="cmp-na">không khả dụng</div>
            ) : (
              <>
                <div className="cmp-label">{m.label_vn ?? m.label}</div>
                {m.probs && (Object.entries(m.probs) as [Label, number][]).map(([c, v]) => (
                  <div key={c} className="cmp-prob">
                    <span>{c}</span>
                    <div className="mu-bar"><div className={c === m.label ? "win" : ""} style={{ width: `${v * 100}%` }} /></div>
                    <b>{NUM(v * 100, 1)}%</b>
                  </div>
                ))}
                <p className="cmp-note">{m.note}</p>
              </>
            )}
          </div>
        ))}
      </div>
      <div className={`cmp-analysis ${cmp.agreed ? "ok" : "warn"}`}>
        <strong>{cmp.agreed ? "✓ Đồng thuận" : "⚠ Khác biệt"}</strong>
        <ul>{cmp.analysis.map((a, i) => <li key={i}>{a}</li>)}</ul>
      </div>
      <p className="cmp-foot">Tham chiếu paper: softmax 64,50% · fuzzy-only 41,6% · FRF-MLP 63,00% macro-F1.
        FRF-MLP chọn vì <strong>diễn giải được + nhẹ CPU + nâng lớp thiểu số</strong>, dù softmax nhỉnh hơn macro-F1 (xem tab 'Ý tưởng paper').</p>
    </div>
  );
}
