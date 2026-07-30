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


@app.post("/predict")
def predict(inp: PredIn):
    text = inp.text or ""
    norm = normalize(text)
    # kenh thong ke: TF-IDF
    xw = W_VEC.transform([norm])
    xc = C_VEC.transform([norm])
    X = sp.hstack([xw, xc]).tocsr()
    # kenh tri thuc: fuzzy
    crisp = EXT.transform([text])[0]            # (3,)
    mu, strengths, p_fuzzy = fuzzy_inference(crisp[None, :])
    mu, strengths, p_fuzzy = mu[0], strengths[0], p_fuzzy[0]
    from fuzzy import fuzzy_features
    F = fuzzy_features(crisp[None, :])          # (1,22)
    with torch.no_grad():
        xt = torch.from_numpy(X.toarray().astype(np.float32))
        xf = torch.from_numpy(F)
        logp = model(xt, xf)
        p_mlp = torch.exp(logp)[0].numpy()
    p = (1 - LAM) * p_mlp + LAM * p_fuzzy
    idx = int(np.argmax(p))

    # token highlight: z-score cua moi token
    toks = tokenize(text)
    z = np.array([LEX.get(w, 0.0) for w in toks])
    order = np.argsort(-z)
    highlights = []
    for i in order:
        if z[i] >= Z_THRESH:
            highlights.append({"token": toks[i], "z": round(float(z[i]), 2)})
        elif z[i] <= -Z_THRESH:
            highlights.append({"token": toks[i], "z": round(float(z[i]), 2)})
        if len(highlights) >= 12:
            break

    # thanh vien theo bien (S,D,T)
    var_names = ["S — Độ công kích", "D — Mật độ từ công kích", "T — Độ nhắm đích"]
    memberships = []
    for j in range(3):
        memberships.append({
            "name": var_names[j], "value": round(float(crisp[j]), 3),
            "low": round(float(mu[j * 3]), 3), "med": round(float(mu[j * 3 + 1]), 3),
            "high": round(float(mu[j * 3 + 2]), 3),
        })

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
    }
