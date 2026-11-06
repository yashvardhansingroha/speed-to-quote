"""Evaluation metrics, written out by hand (no sklearn) so each formula can be explained and checked.

Conventions
  * accuracy        = correct / total.
  * per-class F1    = 2PR / (P + R); defined as 0 when P + R = 0.
  * macro-F1        = unweighted mean of per-class F1 over the classes that occur in the labels OR the predictions
                      (a class that never occurs anywhere cannot be judged and is left out).
  * missing-field   POSITIVE class = the field is `missing` or `unclear` (i.e. something must still be requested).
    detection       precision = TP / (TP + FP), recall = TP / (TP + FN), pooled over every (inquiry, field) pair.
                    `not_applicable` and `provided` are both negatives. A ratio with a zero denominator is None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

POSITIVE_STATUSES = ("missing", "unclear")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> Optional[float]:
    if not y_true:
        return None
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


@dataclass(frozen=True)
class ClassScore:
    label: str
    precision: Optional[float]
    recall: Optional[float]
    f1: float
    support: int          # number of labelled examples of this class
    predicted: int        # number of times it was predicted


def per_class(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[ClassScore]:
    out = []
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        support = sum(t == c for t in y_true)
        predicted = sum(p == c for p in y_pred)
        prec = tp / predicted if predicted else None
        rec = tp / support if support else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        out.append(ClassScore(c, prec, rec, f1, support, predicted))
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Optional[float]:
    scores = [s for s in per_class(y_true, y_pred, labels) if s.support or s.predicted]
    return sum(s.f1 for s in scores) / len(scores) if scores else None


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    """Rows = labelled (true) class, columns = predicted class."""
    m = pd.DataFrame(0, index=list(labels), columns=list(labels))
    for t, p in zip(y_true, y_pred):
        if t in m.index and p in m.columns:
            m.loc[t, p] += 1
    m.index.name, m.columns.name = "labelled", "predicted"
    return m


@dataclass(frozen=True)
class Detection:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def detection(true_statuses: Sequence[str], pred_statuses: Sequence[str]) -> Detection:
    tp = fp = fn = tn = 0
    for t, p in zip(true_statuses, pred_statuses):
        tpos, ppos = t in POSITIVE_STATUSES, p in POSITIVE_STATUSES
        tp += tpos and ppos
        fp += (not tpos) and ppos
        fn += tpos and (not ppos)
        tn += (not tpos) and (not ppos)
    return Detection(tp, fp, fn, tn)


def median(values: Sequence[float]) -> Optional[float]:
    return float(pd.Series(list(values)).median()) if len(values) else None
