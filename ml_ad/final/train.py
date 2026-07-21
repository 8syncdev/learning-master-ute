# -*- coding: utf-8 -*-
"""Huan luyen day du cho bao cao cuoi ky: FRF-MLP (Fuzzy Rule-Fused MLP)
phan loai van ban cong kich tren mang xa hoi tieng Viet (ViHSD, 3 lop).

Cac mo hinh (cung TF-IDF word 1-2gram + char_wb 2-4gram):
  B0  softmax        : Hoi quy softmax (logistic da lop) - baseline tuyen tinh
  B1  mlp            : MLP thuan (khong fuzzy) - baseline chinh / goc ablation
  B2  fuzzy          : He luat mo thuan (argmax p_fuzzy) - baseline tri thuc
  A1  mlp_feat       : MLP + fusion dac trung mo (ablation: chi feature-level)
  A2  mlp_dec        : MLP + fusion quyet dinh voi lambda quet tren dev (ablation)
  FRF frf_mlp        : De xuat = fusion dac trung (A1) + fusion quyet dinh (A2)

Chay:  .venv/bin/python train.py            (toan bo, ~vai phut CPU)
Ket qua: outputs/metrics.json, outputs/curves.npz, outputs/confusions.npz,
         outputs/lexicon_top.json, outputs/fuzzy_stats.npz
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from fuzzy import (CrispExtractor, build_lexicon, fuzzy_features,
                   fuzzy_inference, normalize)

SEED = 42
HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
LABELS = ["CLEAN", "OFFENSIVE", "HATE"]

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(16)


# ---------------------------------------------------------------------------
# Du lieu + dac trung
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(HERE / "data" / "ViHSD.csv")
    df["free_text"] = df["free_text"].astype(str)
    parts = {}
    for s in ["train", "dev", "test"]:
        d = df[df["split"] == s]
        parts[s] = (d["free_text"].tolist(), d["label_id"].to_numpy())
    return parts


def build_features(parts):
    tr_x, tr_y = parts["train"]
    print("== TF-IDF ==")
    w_vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                            max_features=20000, min_df=2, sublinear_tf=True)
    c_vec = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                            ngram_range=(2, 4), max_features=20000, min_df=2,
                            sublinear_tf=True)
    Xw = {s: (w_vec.fit_transform(parts[s][0]) if s == "train" else w_vec.transform(parts[s][0]))
          for s in ["train", "dev", "test"]}
    Xc = {s: (c_vec.fit_transform(parts[s][0]) if s == "train" else c_vec.transform(parts[s][0]))
          for s in ["train", "dev", "test"]}
    X = {s: sp.hstack([Xw[s], Xc[s]]).tocsr() for s in Xw}
    print("   dims:", {s: X[s].shape for s in X})

    print("== Fuzzy ==")
    lex = build_lexicon(tr_x, tr_y)
    ext = CrispExtractor(lex).fit(tr_x)
    crisp = {s: ext.transform(parts[s][0]) for s in parts}
    F = {s: fuzzy_features(crisp[s]) for s in parts}
    pf = {s: fuzzy_inference(crisp[s])[2] for s in parts}

    top = sorted(lex.items(), key=lambda kv: -kv[1])[:30]
    bot = sorted(lex.items(), key=lambda kv: kv[1])[:10]
    (OUT / "lexicon_top.json").write_text(
        json.dumps({"offensive_top30": top, "clean_top10": bot},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(OUT / "fuzzy_stats.npz",
             crisp_train=crisp["train"], crisp_test=crisp["test"],
             y_train=tr_y, y_test=parts["test"][1])
    return X, F, pf


# ---------------------------------------------------------------------------
# Mo hinh torch
# ---------------------------------------------------------------------------
class FRFMLP(nn.Module):
    """MLP tren TF-IDF, tuy chon noi them vector dac trung mo vao dau vao."""

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
        logits = self.head(self.body(x))
        return torch.log_softmax(logits, 1)


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_torch(name, X, F, pf, parts, use_feat,
                epochs=40, bs=256, lr=1e-3, patience=5):
    tr_y = parts["train"][1]
    dv_y = parts["dev"][1]
    d_in = X["train"].shape[1]
    model = FRFMLP(d_in, d_fuzzy=F["train"].shape[1] if use_feat else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor(len(tr_y) / (3 * np.bincount(tr_y)), dtype=torch.float32)
    lossf = nn.NLLLoss(weight=w)
    rng = np.random.default_rng(SEED)

    def fwd(split, idx):
        xt = torch.from_numpy(X[split][idx].toarray().astype(np.float32))
        xf = torch.from_numpy(F[split][idx]) if use_feat else None
        return model(xt, xf)

    def evaluate(split):
        model.eval()
        preds = []
        with torch.no_grad():
            for idx in batches(X[split].shape[0], 1024, False, rng):
                preds.append(fwd(split, idx).argmax(1).numpy())
        return np.concatenate(preds)

    hist = {"loss": [], "dev_f1": []}
    best_f1, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for idx in batches(X["train"].shape[0], bs, True, rng):
            opt.zero_grad()
            logp = fwd("train", idx)
            loss = lossf(logp, torch.from_numpy(tr_y[idx]))
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        dev_pred = evaluate("dev")
        f1 = f1_score(dv_y, dev_pred, average="macro")
        hist["loss"].append(tot / nb); hist["dev_f1"].append(f1)
        star = ""
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            star = " *"
        else:
            bad += 1
        print(f"  [{name}] ep{ep:02d} loss={tot/nb:.4f} devF1={f1:.4f}{star}")
        if bad >= patience:
            print(f"  [{name}] early stop @ep{ep}")
            break
    model.load_state_dict(best_state)
    test_pred = evaluate("test")
    print(f"  [{name}] {time.time()-t0:.1f}s")
    return model, test_pred, hist


# ---------------------------------------------------------------------------
# Danh gia chung
# ---------------------------------------------------------------------------
def report(name, y_true, y_pred, results, confs):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    wf1 = f1_score(y_true, y_pred, average="weighted")
    rep = classification_report(y_true, y_pred, target_names=LABELS,
                                output_dict=True, zero_division=0)
    results[name] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
                     "per_class": {c: {k: rep[c][k] for k in
                                       ("precision", "recall", "f1-score", "support")}
                                   for c in LABELS}}
    confs[name] = confusion_matrix(y_true, y_pred)
    print(f">> {name}: acc={acc:.4f} macroF1={mf1:.4f} wF1={wf1:.4f}")


def main():
    parts = load_data()
    X, F, pf = build_features(parts)
    te_y = parts["test"][1]
    dv_y = parts["dev"][1]
    results, confs, curves = {}, {}, {}

    # B0: softmax regression
    print("== B0 softmax ==")
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                            random_state=SEED)
    lr.fit(X["train"], parts["train"][1])
    report("softmax", te_y, lr.predict(X["test"]), results, confs)

    # B2: fuzzy-only
    print("== B2 fuzzy-only ==")
    report("fuzzy", te_y, pf["test"].argmax(1), results, confs)

    # B1: MLP thuan
    print("== B1 mlp ==")
    _, pred, hist = train_torch("mlp", X, F, pf, parts, False)
    report("mlp", te_y, pred, results, confs); curves["mlp"] = hist

    # A1: MLP + fuzzy features (model duoc tai su dung cho FRF-MLP)
    print("== A1 mlp_feat ==")
    model_feat, pred, hist = train_torch("mlp_feat", X, F, pf, parts, True)
    report("mlp_feat", te_y, pred, results, confs); curves["mlp_feat"] = hist

    def probs(model, split, use_feat):
        preds = []
        model.eval()
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for idx in batches(X[split].shape[0], 1024, False, rng):
                xt = torch.from_numpy(X[split][idx].toarray().astype(np.float32))
                xf = torch.from_numpy(F[split][idx]) if use_feat else None
                preds.append(torch.exp(model(xt, xf)).numpy())
        return np.concatenate(preds)

    def sweep_lambda(p_dev):
        lambdas = np.linspace(0, 1, 21)
        scores = [f1_score(dv_y, ((1 - l) * p_dev + l * pf["dev"]).argmax(1),
                           average="macro") for l in lambdas]
        k = int(np.argmax(scores))
        return float(lambdas[k]), scores[k], lambdas.tolist(), scores

    # A2: MLP thuan + decision fusion (quet lambda tren dev)
    print("== A2 mlp_dec ==")
    model_b1, _, _ = train_torch("mlp_dec_base", X, F, pf, parts, False)
    l_best, dev_best, lams, scs = sweep_lambda(probs(model_b1, "dev", False))
    print(f"  lambda* = {l_best:.2f} (devF1={dev_best:.4f})")
    p_test = probs(model_b1, "test", False)
    report("mlp_dec", te_y, ((1 - l_best) * p_test + l_best * pf["test"]).argmax(1),
           results, confs)
    curves["lambda_sweep_mlp"] = {"lambdas": lams, "dev_f1": scs}
    results["mlp_dec"]["lambda"] = l_best

    # FRF-MLP (de xuat) = fusion dac trung (A1) + fusion quyet dinh (A2)
    print("== FRF-MLP (de xuat) ==")
    l_frf, dev_frf, lams, scs = sweep_lambda(probs(model_feat, "dev", True))
    print(f"  lambda* = {l_frf:.2f} (devF1={dev_frf:.4f})")
    p_test = probs(model_feat, "test", True)
    report("frf_mlp", te_y, ((1 - l_frf) * p_test + l_frf * pf["test"]).argmax(1),
           results, confs)
    curves["lambda_sweep_frf"] = {"lambdas": lams, "dev_f1": scs}
    results["frf_mlp"]["lambda"] = l_frf

    (OUT / "metrics.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez(OUT / "confusions.npz", **{k: v for k, v in confs.items()})
    (OUT / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
    print("saved -> outputs/")


if __name__ == "__main__":
    main()
