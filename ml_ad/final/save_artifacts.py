# -*- coding: utf-8 -*-
"""Luu artifact cua pipeline FRF-MLP de backend demo tai su dung (khong can retrain).

Output:
  demo/api/mlp_feat.pt       - trong so MLP (mlp_feat model)
  demo/api/artifacts.pkl     - {w_vec, c_vec, lex, ext, d_in, d_fuzzy, lambda_frf, rules}
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from fuzzy import RULES, fuzzy_inference
from train import FRFMLP, build_features, load_data, train_torch

HERE = Path(__file__).parent
API = HERE / "demo" / "api"


def main():
    parts = load_data()
    X, F, pf, fit = build_features(parts)

    # huan luyen lai mlp_feat (model de xuat dung cho demo)
    model, _, _ = train_torch("mlp_feat", X, F, pf, parts, True)
    d_in = X["train"].shape[1]
    d_fuzzy = F["train"].shape[1]

    # quet lambda tren dev cho fusion quyet dinh
    def probs(split):
        model.eval()
        rng = np.random.default_rng(0)
        ps = []
        with torch.no_grad():
            for i in range(0, X[split].shape[0], 1024):
                idx = slice(i, i + 1024)
                xt = torch.from_numpy(X[split][idx].toarray().astype(np.float32))
                xf = torch.from_numpy(F[split][idx])
                ps.append(torch.exp(model(xt, xf)).numpy())
        return np.concatenate(ps)

    p_dev = probs("dev")
    lams = np.linspace(0, 1, 21)
    sc = [f1_score(parts["dev"][1], ((1 - l) * p_dev + l * pf["dev"]).argmax(1),
                   average="macro") for l in lams]
    lam = float(lams[int(np.argmax(sc))])
    print(f"lambda* = {lam:.2f} (devF1={max(sc):.4f})")

    API.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), API / "mlp_feat.pt")
    with open(API / "artifacts.pkl", "wb") as f:
        pickle.dump(dict(
            w_vec=fit["w_vec"], c_vec=fit["c_vec"], lex=fit["lex"], ext=fit["ext"],
            d_in=d_in, d_fuzzy=d_fuzzy, lambda_frf=lam,
            rules=[(name, cls, w) for name, _, cls, w in RULES],
        ), f)
    print("saved ->", API / "mlp_feat.pt", API / "artifacts.pkl")


if __name__ == "__main__":
    main()
