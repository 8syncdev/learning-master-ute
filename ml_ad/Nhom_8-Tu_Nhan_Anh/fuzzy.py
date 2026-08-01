# -*- coding: utf-8 -*-
"""Thanh phan fuzzy logic cho FRF-MLP (Fuzzy Rule-Fused MLP) tren ViHSD.

Gom 3 khoi:
1. Lexicon: diem "cong kich" cua tung token, hoc tu tap TRAIN bang
   log-odds ratio voi informative Dirichlet prior (Monroe et al., 2008).
2. Bien ngon ngu (linguistic variables) o muc van ban:
   S  = do cong kich cuc dai (max token z-score, chuan hoa [0,1])
   D  = mat do tu cong kich  (ti le token co z > nguong)
   T  = do "nham dich" (ti le dai tu/danh xung chi doi tuong: may, thang, bon, lu...)
3. Ham thanh vien tam giac/hinh thang {LOW, MED, HIGH} + he luat Mamdani
   -> do kich hoat luat + phan phoi lop mo p_fuzzy(CLEAN, OFFENSIVE, HATE).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Tien xu ly
# ---------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NUM = re.compile(r"\d+")
_REPEAT = re.compile(r"(.)\1{2,}")
_TOKEN = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹ_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Chuan hoa nhe: giu nguyen teencode tuc tiu (la dac trung), che URL/mention/so."""
    t = str(text).lower()
    t = _URL.sub(" urltoken ", t)
    t = _MENTION.sub(" usertoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = _REPEAT.sub(r"\1\1", t)  # keooooo -> keoo
    return t.strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


# Danh xung/dai tu huong doi tuong (nham dich ca nhan/nhom) - bien T
TARGET_WORDS = {
    "mày", "may", "m", "mi", "chúng", "tụi", "bọn", "lũ", "đám",
    "thằng", "thang", "con", "nó", "no", "chúng_mày", "tui_bay", "bay",
    "họ", "quân", "loại", "giống", "đồ", "thứ",
}


# ---------------------------------------------------------------------------
# 1. Lexicon log-odds (informative Dirichlet prior)
# ---------------------------------------------------------------------------
def build_lexicon(texts: list[str], labels: np.ndarray, min_count: int = 3) -> dict[str, float]:
    """z-score log-odds cua token trong lop {OFFENSIVE+HATE} so voi CLEAN.

    z > 0: token nghieng ve cong kich; z < 0: nghieng ve sach.
    """
    cnt_off: Counter = Counter()
    cnt_cln: Counter = Counter()
    for t, y in zip(texts, labels):
        toks = tokenize(t)
        if y == 0:
            cnt_cln.update(toks)
        else:
            cnt_off.update(toks)
    vocab = {w for w in (set(cnt_off) | set(cnt_cln))
             if cnt_off[w] + cnt_cln[w] >= min_count}
    n_off = sum(cnt_off.values())
    n_cln = sum(cnt_cln.values())
    prior = cnt_off + cnt_cln          # prior = tan suat toan cuc (Monroe et al.)
    n_prior = sum(prior.values())
    alpha0 = 1000.0                    # tong khoi luong prior
    lex: dict[str, float] = {}
    for w in vocab:
        aw = alpha0 * prior[w] / n_prior
        yo, yc = cnt_off[w], cnt_cln[w]
        do = math.log((yo + aw) / (n_off + alpha0 - yo - aw))
        dc = math.log((yc + aw) / (n_cln + alpha0 - yc - aw))
        var = 1.0 / (yo + aw) + 1.0 / (yc + aw)
        lex[w] = (do - dc) / math.sqrt(var)
    return lex


# ---------------------------------------------------------------------------
# 2. Bien ngon ngu muc van ban
# ---------------------------------------------------------------------------
class CrispExtractor:
    """Tinh (S, D, T) tho roi chuan hoa [0,1] theo phan vi tap TRAIN."""

    def __init__(self, lexicon: dict[str, float], z_thresh: float = 2.0):
        self.lex = lexicon
        self.z_thresh = z_thresh
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def _raw(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        if not toks:
            return np.zeros(3)
        zs = np.array([self.lex.get(w, 0.0) for w in toks])
        s_max = float(zs.max()) if len(zs) else 0.0
        density = float((zs > self.z_thresh).mean())
        target = float(np.mean([w in TARGET_WORDS for w in toks]))
        return np.array([s_max, density, target])

    def fit(self, texts: list[str]) -> "CrispExtractor":
        raw = np.stack([self._raw(t) for t in texts])
        self.lo = np.percentile(raw, 5, axis=0)
        self.hi = np.percentile(raw, 95, axis=0)
        self.hi = np.where(self.hi - self.lo < 1e-9, self.lo + 1.0, self.hi)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        raw = np.stack([self._raw(t) for t in texts])
        x = (raw - self.lo) / (self.hi - self.lo)
        return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Ham thanh vien + he luat Mamdani
# ---------------------------------------------------------------------------
def _trap(x, a, b, c, d):
    return np.clip(np.minimum((x - a) / max(b - a, 1e-9), (d - x) / max(d - c, 1e-9)), 0.0, 1.0)


def memberships(x: np.ndarray) -> np.ndarray:
    """x: (N,3) trong [0,1] -> mu: (N,9) = {LOW,MED,HIGH} cho S, D, T."""
    out = []
    for j in range(x.shape[1]):
        v = x[:, j]
        out.append(_trap(v, -1.0, 0.0, 0.15, 0.40))   # LOW
        out.append(_trap(v, 0.20, 0.45, 0.55, 0.80))   # MED
        out.append(_trap(v, 0.60, 0.85, 1.00, 2.00))   # HIGH
    return np.stack(out, axis=1)


# (ten, cong thuc kich hoat theo mu, lop ket luan, trong so)
# mu index: S:0-2, D:3-5, T:6-8 (LOW, MED, HIGH)
RULES = [
    ("R1: S LOW ∧ D LOW → CLEAN",            lambda m: np.minimum(m[:, 0], m[:, 3]), 0, 1.0),
    ("R2: S MED ∧ T LOW → OFFENSIVE",        lambda m: np.minimum(m[:, 1], m[:, 6]), 1, 1.0),
    ("R3: S HIGH ∧ T LOW → OFFENSIVE",       lambda m: np.minimum(m[:, 2], m[:, 6]), 1, 1.0),
    ("R4: S MED ∧ T HIGH → HATE",            lambda m: np.minimum(m[:, 1], m[:, 8]), 2, 0.9),
    ("R5: S HIGH ∧ T MED∨HIGH → HATE",       lambda m: np.minimum(m[:, 2], np.maximum(m[:, 7], m[:, 8])), 2, 1.0),
    ("R6: D HIGH ∧ T HIGH → HATE",           lambda m: np.minimum(m[:, 5], m[:, 8]), 2, 0.8),
    ("R7: D MED ∧ T LOW → OFFENSIVE",        lambda m: np.minimum(m[:, 4], m[:, 6]), 1, 0.7),
]


def fuzzy_inference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: (N,3) -> (mu (N,9), rule_strength (N,7), p_fuzzy (N,3))."""
    mu = memberships(x)
    strengths = np.stack([w * f(mu) for _, f, _, w in RULES], axis=1)
    scores = np.zeros((x.shape[0], 3))
    for k, (_, _, cls, _) in enumerate(RULES):
        scores[:, cls] += strengths[:, k]
    # khong luat nao kich hoat -> CLEAN
    scores[:, 0] += np.clip(1.0 - strengths.max(axis=1), 0.0, 1.0) * 0.5
    p = scores / scores.sum(axis=1, keepdims=True)
    return mu, strengths, p


def fuzzy_features(x: np.ndarray) -> np.ndarray:
    """Vector dac trung mo day du dua vao MLP: [crisp(3) | mu(9) | rules(7) | p_fuzzy(3)] = 22 chieu."""
    mu, st, p = fuzzy_inference(x)
    return np.concatenate([x, mu, st, p], axis=1).astype(np.float32)
