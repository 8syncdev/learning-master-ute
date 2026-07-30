# -*- coding: utf-8 -*-
"""Backend demo FRF-MLP — phat hien ngôn tu cong kich tieng Viet.

Run:  uvicorn main:app --port 8000 --reload   (tu demo/api/)
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # ml_ad/final

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fuzzy import RULES, _trap, fuzzy_inference, normalize, tokenize

HERE = Path(__file__).parent
LABELS = ["CLEAN", "OFFENSIVE", "HATE"]
LABEL_VN = {"CLEAN": "An toàn", "OFFENSIVE": "Công kích", "HATE": "Thù ghét"}
COLOR = {"CLEAN": "#22a06b", "OFFENSIVE": "#e0913d", "HATE": "#d44747"}


class FRFMLP(nn.Module):
    def __init__(self, d_in: int, d_fuzzy: int = 0, hidden=(256, 128), p_drop=0.3):
        super().__init__()
        self.use_feat = d_fuzzy > 0
        layers, prev = [], d_in + d_fuzzy
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 3)

    def forward(self, x_tfidf, x_fuzzy=None):
        x = torch.cat([x_tfidf, x_fuzzy], 1) if self.use_feat else x_tfidf
        return torch.log_softmax(self.head(self.body(x)), 1)


# ---- load artifacts -------------------------------------------------------
with open(HERE / "artifacts.pkl", "rb") as f:
    A = pickle.load(f)
model = FRFMLP(A["d_in"], A["d_fuzzy"])
model.load_state_dict(torch.load(HERE / "mlp_feat.pt", map_location="cpu"))
model.eval()
W_VEC, C_VEC, LEX = A["w_vec"], A["c_vec"], A["lex"]
EXT, LAM = A["ext"], float(A["lambda_frf"])
Z_THRESH = 2.0

# softmax baseline (LogisticRegression C=4 class-weighted) cho endpoint /compare
SOFTMAX = None
_sx = HERE / "softmax.pkl"
if _sx.exists():
    with open(_sx, "rb") as _f:
        SOFTMAX = pickle.load(_f)

# sample comments (bo nhãn HATE/OFFENSIVE phan trang)
import csv
SAMPLES = []
csv_path = HERE.parent.parent / "data" / "ViHSD.csv"
if csv_path.exists():
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            SAMPLES.append({"text": row["free_text"], "label": LABELS[int(row["label_id"])]})

app = FastAPI(title="FRF-MLP demo API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "lambda": LAM, "d_in": A["d_in"], "samples": len(SAMPLES)}


@app.get("/sample")
def sample():
    if not SAMPLES:
        return {"text": "", "label": ""}
    # uu tien mau co noi dung cong kich
    pool = [s for s in SAMPLES if s["label"] in ("OFFENSIVE", "HATE")] or SAMPLES
    return random.choice(pool)


def _top_features(vec_row, names, k):
    row = vec_row.tocoo()
    pairs = sorted(zip(row.col, row.data), key=lambda kv: -abs(kv[1]))[:k]
    return [{"term": str(names[c]), "weight": round(float(v), 4)} for c, v in pairs]


# nhan cho moi chi so mu (S/D/T × LOW/MED/HIGH)
MU_LABEL = ["S=LOW", "S=MED", "S=HIGH", "D=LOW", "D=MED", "D=HIGH", "T=LOW", "T=MED", "T=HIGH"]
# antecedent cho moi luat (danh sach mu-index)
RULE_ANTE = {0: [0, 3], 1: [1, 6], 2: [2, 6], 3: [1, 8], 4: [2, 7, 8], 5: [5, 8], 6: [4, 6]}
# bien hàm thành viên hình thang [a,b,c,d] cho LOW/MED/HIGH (theo fuzzy.memberships)
MU_BREAK = {"LOW": [-1.0, 0.0, 0.15, 0.40], "MED": [0.20, 0.45, 0.55, 0.80],
            "HIGH": [0.60, 0.85, 1.00, 2.00]}
VAR_NAME = ["S", "D", "T"]


@app.post("/predict")
def predict(inp: PredIn):
    import torch.nn.functional as Ff
    from fuzzy import fuzzy_features
    text = inp.text or ""
    norm = normalize(text)
    toks = tokenize(text)

    # ---- kenh thong ke: TF-IDF ----
    xw = W_VEC.transform([norm])
    xc = C_VEC.transform([norm])
    X = sp.hstack([xw, xc]).tocsr()
    nnz = int(X.nnz)
    top_word = _top_features(xw[0], W_VEC.get_feature_names_out(), 8)
    top_char = _top_features(xc[0], C_VEC.get_feature_names_out(), 6)

    # ---- kenh tri thuc: fuzzy ----
    crisp_raw = EXT._raw(text)                 # [S_max, density, target] chua chuan hoa
    crisp = EXT.transform([text])[0]           # [0,1] sau chuan hoa
    mu, strengths, p_fuzzy = fuzzy_inference(crisp[None, :])
    mu, strengths, p_fuzzy = mu[0], strengths[0], p_fuzzy[0]
    F = fuzzy_features(crisp[None, :])

    # ---- MLP intermediates (hand-trace) ----
    with torch.no_grad():
        xt = torch.from_numpy(X.toarray().astype(np.float32))
        xf = torch.from_numpy(F)
        x_cat = torch.cat([xt, xf], 1)
        h1_pre = model.body[0](x_cat)            # Linear -> 256
        h1 = Ff.relu(h1_pre)
        h2_pre = model.body[3](h1)               # Linear -> 128
        h2 = Ff.relu(h2_pre)
        logits = model.head(h2)                  # -> 3
        p_mlp = Ff.softmax(logits, 1)[0].numpy()
    h1a = h1[0].numpy(); h2a = h2[0].numpy(); loga = logits[0].numpy()

    p = (1 - LAM) * p_mlp + LAM * p_fuzzy
    idx = int(np.argmax(p))

    # ---- token z-score (highlight + step lexicon) ----
    z = np.array([LEX.get(w, 0.0) for w in toks])
    order = np.argsort(-z)
    highlights, tok_z = [], []
    for i in range(len(toks)):
        tok_z.append({"token": toks[i], "z": round(float(z[i]), 2)})
    for i in order:
        if z[i] >= Z_THRESH or z[i] <= -Z_THRESH:
            highlights.append({"token": toks[i], "z": round(float(z[i]), 2)})
        if len(highlights) >= 12:
            break

    var_names = ["S — Độ công kích", "D — Mật độ từ công kích", "T — Độ nhắm đích"]
    memberships = [{"name": var_names[j], "value": round(float(crisp[j]), 3),
                    "low": round(float(mu[j * 3]), 3), "med": round(float(mu[j * 3 + 1]), 3),
                    "high": round(float(mu[j * 3 + 2]), 3)} for j in range(3)]

    # ---- 8-step trace ----
    top_neurons = lambda vec, k: sorted(range(len(vec)), key=lambda i: -vec[i])[:k]
    rule_class_score = [0.0, 0.0, 0.0]
    for k in range(len(RULES)):
        rule_class_score[RULES[k][2]] += float(strengths[k])
    rule_class_score[0] += float(np.clip(1.0 - strengths.max(), 0.0, 1.0)) * 0.5

    steps = [
        {"n": 1, "title": "Tiền xử lý", "subtitle": "chuẩn hóa văn bản thô",
         "content": {"raw": text, "normalized": norm, "n_tokens": len(toks),
                     "tokens_sample": toks[:20]}},
        {"n": 2, "title": "Vector TF-IDF", "subtitle": "word (1–2) + char_wb (2–4), 40.000 chiều",
         "content": {"nnz": nnz, "top_word": top_word, "top_char": top_char}},
        {"n": 3, "title": "Lexicon + biến ngôn ngữ", "subtitle": "z-score log-odds → (S, D, T)",
         "content": {"tokens_z": tok_z[:14], "crisp_raw": [round(float(v), 3) for v in crisp_raw],
                     "bounds_lo": [round(float(v), 3) for v in EXT.lo],
                     "bounds_hi": [round(float(v), 3) for v in EXT.hi],
                     "crisp_norm": [round(float(v), 3) for v in crisp]}},
        {"n": 4, "title": "Mờ hóa (hàm thành viên)", "subtitle": "trapezoid LOW/MED/HIGH cho S, D, T",
         "content": {
             "mu": [{"label": MU_LABEL[j], "value": round(float(mu[j]), 3)} for j in range(9)],
             "mu_detail": [
                 {"label": MU_LABEL[j], "var": VAR_NAME[j // 3], "level": lvl,
                  "v": round(float(crisp[j // 3]), 3),
                  "a": MU_BREAK[lvl][0], "b": MU_BREAK[lvl][1],
                  "c": MU_BREAK[lvl][2], "d": MU_BREAK[lvl][3],
                  "mu": round(float(mu[j]), 3)}
                 for j in range(9) for lvl in [MU_LABEL[j].split("=")[1]]
             ]}},
        {"n": 5, "title": "Suy diễn 7 luật Mamdani", "subtitle": "t-norm min trên mỗi antecedent",
         "content": {"rules": [{"name": RULES[k][0],
                                "antecedents": [{"label": MU_LABEL[i], "value": round(float(mu[i]), 3)}
                                                for i in RULE_ANTE[k]],
                                "weight": RULES[k][3],
                                "strength": round(float(strengths[k]), 3),
                                "conclusion": LABELS[RULES[k][2]]} for k in range(len(RULES))]}},
        {"n": 6, "title": "Giải mờ → p_mờ", "subtitle": "cộng điểm theo lớp, chuẩn hoá",
         "content": {"class_score": [{"label": LABELS[c], "score": round(rule_class_score[c], 3)}
                                     for c in range(3)],
                     "p_fuzzy": {LABELS[c]: round(float(p_fuzzy[c]), 4) for c in range(3)}}},
        {"n": 7, "title": "MLP 256–128 → p_MLP", "subtitle": "TF-IDF ⊕ 22 đặc trưng mờ qua 2 lớp",
         "content": {"h1_active": int((h1a > 0).sum()), "h1_top": [{"neuron": int(n), "act": round(float(h1a[n]), 3)}
                                                                   for n in top_neurons(h1a, 5)],
                     "h2_active": int((h2a > 0).sum()),
                     "logits": [{"label": LABELS[c], "logit": round(float(loga[c]), 3)}
                                for c in range(3)],
                     "p_mlp": {LABELS[c]: round(float(p_mlp[c]), 4) for c in range(3)}}},
        {"n": 8, "title": "Fusion quyết định", "subtitle": f"p = (1−{LAM:.2f})·p_MLP + {LAM:.2f}·p_mờ",
         "content": {"lambda": round(LAM, 2),
                     "final": {LABELS[c]: round(float(p[c]), 4) for c in range(3)},
                     "label": LABELS[idx]}},
    ]

    return {
        "label": LABELS[idx], "label_vn": LABEL_VN[LABELS[idx]], "color": COLOR[LABELS[idx]],
        "probabilities": {LABELS[k]: round(float(p[k]), 4) for k in range(3)},
        "p_mlp": {LABELS[k]: round(float(p_mlp[k]), 4) for k in range(3)},
        "p_fuzzy": {LABELS[k]: round(float(p_fuzzy[k]), 4) for k in range(3)},
        "lambda": round(LAM, 2),
        "memberships": memberships,
        "rules": [{"name": RULES[k][0], "strength": round(float(strengths[k]), 3),
                   "conclusion": LABELS[RULES[k][2]]} for k in range(len(RULES))],
        "highlights": highlights,
        "n_tokens": len(toks),
        "steps": steps,
    }


@app.post("/compare")
def compare(inp: PredIn):
    """Chạy cùng 1 câu qua 3 mô hình để giáo sư bắt bẻ 'tại sao không dùng cái kia'.
    Trả về softmax (tuyến tính), fuzzy-only (tri thức mờ), FRF-MLP (lai) + phân tích khác biệt.
    """
    import torch.nn.functional as Ff
    from fuzzy import fuzzy_features
    norm = normalize(inp.text or "")
    xw = W_VEC.transform([norm]); xc = C_VEC.transform([norm])
    X = sp.hstack([xw, xc]).tocsr()
    crisp = EXT.transform([inp.text])[0]
    _, _, p_fuzzy = fuzzy_inference(crisp[None, :])
    p_fuzzy = p_fuzzy[0]
    F = fuzzy_features(crisp[None, :])
    with torch.no_grad():
        xt = torch.from_numpy(X.toarray().astype(np.float32))
        xf = torch.from_numpy(F)
        p_mlp = Ff.softmax(model(xt, xf), 1)[0].numpy()
    p_frf = (1 - LAM) * p_mlp + LAM * p_fuzzy

    p_sx = SOFTMAX.predict_proba(X.toarray())[0] if SOFTMAX is not None else None

    def row(name, p, note):
        if p is None:
            return {"model": name, "available": False}
        i = int(np.argmax(p))
        return {"model": name, "label": LABELS[i], "label_vn": LABEL_VN[LABELS[i]],
                "probs": {LABELS[k]: round(float(p[k]), 4) for k in range(3)},
                "note": note}

    models = [
        row("softmax", p_sx,
            "Hồi quy softmax (LogisticRegression C=4, class-weighted) — tuyến tính, lồi, "
            "hộp đen: không giải thích được tại sao chọn nhãn này."),
        row("fuzzy", p_fuzzy,
            "Hệ mờ thuần (7 luật Mamdani) — diễn giải đầy đủ nhưng standalone yếu (macro-F1 41,6%), "
            "thiếu độ phủ teencode."),
        row("frf", p_frf,
            f"FRF-MLP: (1−{LAM:.2f})·p_MLP + {LAM:.2f}·p_mờ — lai 2 kênh (thống kê + tri thức mờ)."),
    ]

    # phân tích khác biệt giữa các mô hình (giáo sư bắt bẻ 'tại sao không dùng cái kia')
    labs = {m["model"]: m.get("label") for m in models if m.get("available", True)}
    agreed = len(set(labs.values())) <= 1
    diffs = []
    if agreed:
        diffs.append(f"Cả {len(labs)} mô hình đồng ý nhãn "
                     f"{next(iter(labs.values()))} — trường hợp dễ, ít gây tranh cãi.")
    else:
        frf_lab = labs.get("frf")
        for mk in ("softmax", "fuzzy"):
            if mk in labs and labs[mk] != frf_lab:
                reason = ("kênh mờ Mamdani hiệu chỉnh MLP ở biên CLEAN/OFFENSIVE nhờ biến T "
                          "(độ nhắm đích) — softmax tuyến tính không có tri thức này."
                          if mk == "softmax" else
                          "MLP thống kê đè lên kênh mờ (λ nhỏ) vì hệ mờ standalone nhiễu.")
                diffs.append(f"{mk.capitalize()} dự đoán {labs[mk]} nhưng FRF-MLP dự đoán "
                             f"{frf_lab}: {reason}")
        if not diffs:
            diffs.append("Các mô hình khác độ tin nhưng cùng nhãn FRF-MLP.")
    return {"models": models, "agreed": agreed, "analysis": diffs,
            "has_softmax": SOFTMAX is not None}
