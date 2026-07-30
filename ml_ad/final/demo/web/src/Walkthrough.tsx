import { useState } from "react";
import {
  Filter, Type, BookMarked, Waves, GitBranch, Sigma, Network, Merge,
  ChevronDown, Lightbulb, Pi, FlaskConical,
} from "lucide-react";
import {
  type Label, type Prediction, type StepContent,
  type S1, type S2, type S3, type S4, type S5, type S6, type S7, type S8,
  type FeatTerm, type TokZ, type MuPair, type RuleLive, type ClassScore, type LogitPair,
} from "./api";

/* ============================================================
   Walkthrough — giải trình 8 bước thuật toán FRF-MLP trên chính
   câu đang phân tích. Mỗi bước: mục tiêu + ý tưởng + công thức +
   số liệu thật từ pred.steps[n] + giải thích diễn giải.
   ============================================================ */

interface Step {
  n: number;
  title: string;
  icon: typeof Filter;
  goal: string;
  idea: string;
  formula?: string;
}

const STEPS: Step[] = [
  {
    n: 1, title: "Tiền xử lý", icon: Filter,
    goal: "Đưa bình luận thô về dạng đều, giữ nguyên tín hiệu công kích.",
    idea: "Chuyển chữ thường, che URL / số / @mention (không mang ngữ nghĩa công kích), rút ký tự lặp (keooooo → keoo). Cố ý GIỮ teencode và từ tục viết tắt (vcl, đm, cc) vì đó chính là tín hiệu phân loại mạnh nhất trên mạng xã hội.",
    formula: "normalize(t) = lower( strip_urls( strip_nums( dedup(t) ) ) )",
  },
  {
    n: 2, title: "Vector TF-IDF", icon: Type,
    goal: "Biến văn bản thành vector số để đưa vào MLP.",
    idea: "Đếm n-gram từ (1–2) và n-gram ký tự trong biên từ (2–4), gắn trọng số TF-IDF để giảm tầm quan trọng của từ quá phổ biến. N-gram ký tự bắt được teencode và biến thể chính tả mà n-gram từ bỏ sót — đặc điểm cố hữu của bình luận mạng xã hội tiếng Việt. Tổng 40.000 chiều.",
    formula: "tfidf(t,d) = (1+log tf) · log(N / df)",
  },
  {
    n: 3, title: "Lexicon + biến ngôn ngữ", icon: BookMarked,
    goal: "Định lượng 'mức công kích' từng token mà không cần từ điển thủ công.",
    idea: "So sánh tần suất token giữa nhóm {OFFENSIVE, HATE} với CLEAN bằng log-odds có tiên nghiệm Dirichlet → ra z-score: z>0 là công kích, z<0 là sạch. Từ đó rút 3 biến mức văn bản: S (độ công kích cực đại), D (mật độ từ công kích), T (độ nhắm đích — tỷ lệ đại từ 'mày/thằng/bọn/lũ'). T chính là chìa phân biệt HATE (công kích NHẮM đích) với OFFENSIVE (tục nhưng không nhắm).",
    formula: "z_w = δ_w / √( 1/(y_o+α) + 1/(y_c+α) ),  δ = log-odds(off) − log-odds(clean)",
  },
  {
    n: 4, title: "Mờ hóa (hàm thành viên)", icon: Waves,
    goal: "Biến S, D, T (số) thành độ thuộc LOW / MED / HIGH — dạng mờ.",
    idea: "Dùng hàm thành viên hình thang. Lý do dùng logic mờ: ranh giới CLEAN/OFFENSIVE/HATE vốn MỜ — hệ số đồng thuận kappa giữa người gán nhãn ViHSD chỉ 0,52 — nên biểu diễn 'mức công kích' bằng độ thuộc liên tục [0,1] tự nhiên hơn là ngưỡng cứng.",
    formula: "μ_[a,b,c,d](v) = max(0, min((v−a)/(b−a), 1, (d−v)/(d−c)))",
  },
  {
    n: 5, title: "Suy diễn 7 luật Mamdani", icon: GitBranch,
    goal: "Mã hóa tri thức ngôn ngữ thành luật IF–THEN đọc được con người.",
    idea: "Ví dụ R5: 'NẾU S HIGH VÀ (T MED hoặc HIGH) THÌ HATE' — công kích mạnh CÙNG nhắm đích = thù ghét. Mỗi luật kết hợp antecedent bằng t-norm min (lấy giá trị nhỏ nhất — luật chỉ mạnh khi TẤT CẢ điều kiện thỏa). 7 luật phủ kín các tổ hợp S/D/T.",
    formula: "r_k = w_k · min( μ_antecedent_1, μ_antecedent_2, … )",
  },
  {
    n: 6, title: "Giải mờ → p_mờ", icon: Sigma,
    goal: "Từ độ kích hoạt luật ra phân phối xác suất 3 lớp của kênh tri thức.",
    idea: "Cộng điểm cho mỗi lớp theo các luật kết luận về lớp đó, cộng một thiên lệch nhỏ về CLEAN khi không luật nào kích hoạt, rồi chuẩn hoá để tổng bằng 1. Đây là đầu ra của kênh logic mờ — diễn giải được, không cần huấn luyện.",
    formula: "p_mờ(c) = ( Σ_{k: kết luận=c} r_k + b_c ) / Σ_c′ ( … )",
  },
  {
    n: 7, title: "MLP 256–128 → p_MLP", icon: Network,
    goal: "Kênh thống kê: học tương tác phi tuyến từ 40.000 đặc trưng TF-IDF.",
    idea: "Vectơ TF-IDF (40.000) được nối thêm 22 đặc trưng mờ (3 biến + 9 độ thuộc + 7 độ kích hoạt luật + 3 p_mờ) thành đầu vào 40.022 chiều, qua 2 lớp ẩn 256–128 với ReLU và dropout, rồi softmax ra p_MLP. Mạng học bằng lan truyền ngược với entropy chéo có trọng số lớp để xử lý mất cân bằng (CLEAN 83%).",
    formula: "h₁ = ReLU(W₁x+b₁) · h₂ = ReLU(W₂h₁+b₂) · p_MLP = softmax(W₃h₂+b₃)",
  },
  {
    n: 8, title: "Fusion quyết định", icon: Merge,
    goal: "Hợp nhất kênh thống kê (MLP) và kênh tri thức (mờ) ra quyết định cuối.",
    idea: "p = (1−λ)·p_MLP + λ·p_mờ, với λ quét trên tập phát triển. λ nhỏ (≈0,05–0,25) nghĩa là tin MLP chủ yếu, hệ mờ chỉ HIỆU CHỈNH — đúng kỳ vọng vì hệ mờ đứng một mình khá yếu (macro-F1 41,6%) nhưng lại bổ sung chính xác những trường hợp MLP nhầm ở biên CLEAN/OFFENSIVE.",
    formula: "p = (1−λ) p_MLP + λ p_mờ,   λ* = argmax_λ  macroF1(dev)",
  },
];

export default function Walkthrough({ pred }: { pred: Prediction | null }) {
  const [open, setOpen] = useState<number | null>(1);
  if (!pred)
    return <div className="empty">Phân tích một bình luận ở tab 'Bảng điều khiển' để xem giải trình từng bước trên chính câu đó.</div>;
  const raw0 = (pred.steps[0]?.content as S1 | undefined)?.raw ?? "";
  return (
    <div className="walk">
      <div className="walk-intro">
        <h3><FlaskConical size={16} /> Đường ống FRF-MLP — 8 bước, từ bình luận thô đến nhãn</h3>
        <p>
          Bấm từng bước để xem <strong>mục tiêu</strong>, <strong>ý tưởng</strong>, <strong>công thức</strong> và
          <strong> số liệu thật</strong> tính trên câu đang phân tích: <em>“{raw0.slice(0, 70)}”</em>.
          Hai kênh chạy song song: thống kê (bước 2 + 7) và tri thức mờ (bước 3–6), rồi hợp nhất ở bước 8.
        </p>
      </div>

      <div className="walk-rail">
        {STEPS.map((s) => {
          const live = pred.steps[s.n - 1]?.content;
          const isOpen = open === s.n;
          const Icon = s.icon;
          return (
            <div key={s.n} className={`wstep ${isOpen ? "open" : ""}`}>
              <button className="whead" onClick={() => setOpen(isOpen ? null : s.n)}>
                <span className="wnum">{s.n}</span>
                <Icon size={16} className="wicon" />
                <span className="wtitle">{s.title}</span>
                <span className="wgoal">{s.goal}</span>
                <ChevronDown size={16} className="wchev" />
              </button>
              {isOpen && live && (
                <div className="wbody">
                  <div className="wrow">
                    <div className="wcol idea"><Lightbulb size={13} /> Ý tưởng</div>
                    <div className="wcol txt">{s.idea}</div>
                  </div>
                  {s.formula && (
                    <div className="wrow">
                      <div className="wcol idea"><Pi size={13} /> Công thức</div>
                      <code className="wcol formula">{s.formula}</code>
                    </div>
                  )}
                  <div className="wrow">
                    <div className="wcol idea"><FlaskConical size={13} /> Với câu này</div>
                    <div className="wcol"><StepData n={s.n} content={live} /></div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---- live data render per step (cast tại boundary đã biết shape theo n) ---- */
function StepData({ n, content }: { n: number; content: StepContent }) {
  if (n === 1) {
    const live = content as S1;
    return (
      <div className="live">
        <div className="kv"><span>Raw</span><code>{live.raw}</code></div>
        <div className="kv"><span>Chuẩn hoá</span><code>{live.normalized}</code></div>
        <div className="kv"><span>{live.n_tokens} token</span><code className="toks">{live.tokens_sample.join(" · ")}</code></div>
      </div>
    );
  }
  if (n === 2) {
    const live = content as S2;
    return (
      <div className="live">
        <div className="kv"><span>Đặc trưng ≠ 0</span><code>{live.nnz} / 40.000 chiều</code></div>
        <div className="feats">
          <div className="ftitle">Top n-gram từ:</div>
          {live.top_word.map((f, i) => <Feat key={i} {...f} />)}
          <div className="ftitle">Top n-gram ký tự:</div>
          {live.top_char.map((f, i) => <Feat key={i} {...f} />)}
        </div>
      </div>
    );
  }
  if (n === 3) {
    const live = content as S3;
    const vars = ["S (độ công kích)", "D (mật độ)", "T (nhắm đích)"];
    return (
      <div className="live">
        <div className="feats">
          <div className="ftitle">Token + z-score log-odds:</div>
          {live.tokens_z.map((t: TokZ, i) => (
            <span key={i} className={`chip ${t.z > 0 ? "pos" : t.z < 0 ? "neg" : ""}`}>{t.token}<b>{t.z > 0 ? "+" : ""}{t.z}</b></span>
          ))}
        </div>
        <table className="vtable">
          <thead><tr><th>Biến</th><th>Raw</th><th>Phân vị train [lo, hi]</th><th>Chuẩn hoá [0,1]</th></tr></thead>
          <tbody>
            {[0, 1, 2].map((j) => (
              <tr key={j}><td>{vars[j]}</td><td>{live.crisp_raw[j]}</td>
                <td>[{live.bounds_lo[j]}, {live.bounds_hi[j]}]</td><td><b>{live.crisp_norm[j]}</b></td></tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (n === 4) {
    const live = content as S4;
    return (
      <div className="live mu-grid">
        {live.mu.map((m: MuPair, i) => (
          <div key={i} className={`mu-cell ${m.value > 0.01 ? "on" : ""}`}>
            <div className="mu-lbl">{m.label}</div>
            <div className="mu-bar"><div style={{ width: `${m.value * 100}%` }} /></div>
            <div className="mu-val">{m.value.toFixed(2)}</div>
          </div>
        ))}
      </div>
    );
  }
  if (n === 5) {
    const live = content as S5;
    return (
      <div className="live rules-live">
        {live.rules.map((r: RuleLive, i) => (
          <div key={i} className={`rlive ${r.strength > 0.01 ? "fired" : ""}`}>
            <div className="rname">{r.name}</div>
            <div className="rante">{r.antecedents.map((a) => `${a.label.split("=")[0]}=${a.value.toFixed(2)}`).join("  ·  ")}</div>
            <div className="rbar"><div style={{ width: `${r.strength * 100}%` }} /></div>
            <div className="rstr">w={r.weight} → {r.conclusion} ({r.strength.toFixed(2)})</div>
          </div>
        ))}
      </div>
    );
  }
  if (n === 6) {
    const live = content as S6;
    return (
      <div className="live">
        <div className="feats">
          {live.class_score.map((c: ClassScore) => (
            <div key={c.label} className="cscore">
              <span>{c.label}</span>
              <div className="mu-bar"><div style={{ width: `${Math.min(c.score, 1) * 100}%` }} /></div>
              <b>{c.score.toFixed(2)}</b>
            </div>
          ))}
        </div>
        <div className="kv"><span>p_mờ (chuẩn hoá)</span><code>{Object.entries(live.p_fuzzy).map(([k, v]) => `${k}:${(v * 100).toFixed(1)}%`).join("  ")}</code></div>
      </div>
    );
  }
  if (n === 7) {
    const live = content as S7;
    return (
      <div className="live">
        <div className="kv"><span>Lớp ẩn 1 (256 neuron)</span><code>{live.h1_active} neuron hoạt động (ReLU&gt;0)</code></div>
        <div className="kv"><span>Lớp ẩn 2 (128 neuron)</span><code>{live.h2_active} neuron hoạt động</code></div>
        <div className="feats">
          <div className="ftitle">Logit 3 lớp (trước softmax):</div>
          {live.logits.map((l: LogitPair) => (
            <div key={l.label} className="cscore">
              <span>{l.label}</span>
              <div className="mu-bar"><div className={l.logit >= 0 ? "pos" : "neg"} style={{ width: `${Math.min(Math.abs(l.logit) / 6, 1) * 100}%` }} /></div>
              <b>{l.logit > 0 ? "+" : ""}{l.logit.toFixed(2)}</b>
            </div>
          ))}
          <div className="ftitle">p_MLP (sau softmax):</div>
          <code className="pp">{Object.entries(live.p_mlp).map(([k, v]) => `${k}: ${(v * 100).toFixed(2)}%`).join("   ")}</code>
        </div>
      </div>
    );
  }
  if (n === 8) {
    const live = content as S8;
    const fin = Object.entries(live.final).sort((a, b) => b[1] - a[1]);
    return (
      <div className="live">
        <div className="kv"><span>λ*</span><code>{live.lambda} (chọn trên tập dev)</code></div>
        <div className="feats">
          {fin.map(([k, v]) => (
            <div key={k} className="cscore">
              <span>{k}{k === live.label ? " ✓" : ""}</span>
              <div className="mu-bar"><div className={k === live.label ? "win" : ""} style={{ width: `${v * 100}%` }} /></div>
              <b>{(v * 100).toFixed(2)}%</b>
            </div>
          ))}
        </div>
        <div className="verdict-line">→ Nhãn dự đoán: <strong>{live.label}</strong></div>
      </div>
    );
  }
  return null;
}

function Feat({ term, weight }: FeatTerm) {
  return <span className="chip ghost">{term}<b>{weight.toFixed(3)}</b></span>;
}
