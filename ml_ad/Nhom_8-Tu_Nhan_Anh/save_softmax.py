# -*- coding: utf-8 -*-
"""Lưu riêng mô hình softmax (sklearn LogisticRegression) cho endpoint /compare.

Dùng lại build_features của train.py để dựng đúng TF-IDF (20k word + 20k char_wb),
khớp C=4.0 / class_weight=balanced / seed 42 y hệt baseline B0 trong paper, rồi
pickle LR + fitten vào demo/api/softmax.pkl. Không retrain MLP (~20 giây).
"""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from train import SEED, build_features, load_data

HERE = Path(__file__).parent
API = HERE / "demo" / "api"


def main():
    parts = load_data()
    X, _F, _pf, _fit = build_features(parts)
    print("== fit softmax (LogisticRegression C=4.0 balanced) ==")
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                            random_state=SEED)
    lr.fit(X["train"], parts["train"][1])
    API.mkdir(parents=True, exist_ok=True)
    with open(API / "softmax.pkl", "wb") as f:
        pickle.dump(lr, f)
    print(f"saved -> {API / 'softmax.pkl'}  (classes={list(lr.classes_)})")


if __name__ == "__main__":
    main()
