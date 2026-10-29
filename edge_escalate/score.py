"""A turn escalates when its score reaches the threshold chosen on the train split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Weights live here. The threshold does not. Train picks it.
UNRESOLVED = 2
NOVEL = 2
HIGH_RISK = 3
LONG = 1
LONG_TOKENS = 80
# Illustrative price of one cloud call, in thousandths of a dollar.
CLOUD_MILLI_USD = 2


@dataclass(frozen=True)
class Turn:
    id: str
    split: str
    label: str
    unresolved: bool
    novel: bool
    high_risk: bool
    tokens: int
    text: str

    def points(self) -> int:
        total = 0
        if self.unresolved:
            total += UNRESOLVED
        if self.novel:
            total += NOVEL
        if self.high_risk:
            total += HIGH_RISK
        if self.tokens >= LONG_TOKENS:
            total += LONG
        return total


@dataclass(frozen=True)
class Report:
    path: str
    threshold: int
    holdout: int
    precision: float
    recall: float
    escalations: int
    cloud_milli: int
    saved_milli: int
    false_escalate: tuple[str, ...]
    false_stay: tuple[str, ...]


def score_path(path: Path) -> Report:
    turns = _load(path)
    train = tuple(turn for turn in turns if turn.split == "train")
    holdout = tuple(turn for turn in turns if turn.split == "holdout")
    if not train or not holdout:
        raise ValueError(f"{path}: need both a train split and a holdout split")
    threshold = _choose_threshold(train)
    judged = [(turn, turn.points() >= threshold) for turn in holdout]
    tp = sum(turn.label == "escalate" and pred for turn, pred in judged)
    fp = sum(turn.label == "stay" and pred for turn, pred in judged)
    fn = sum(turn.label == "escalate" and not pred for turn, pred in judged)
    escalations = sum(pred for _, pred in judged)
    return Report(
        path=str(path),
        threshold=threshold,
        holdout=len(holdout),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        escalations=escalations,
        cloud_milli=escalations * CLOUD_MILLI_USD,
        saved_milli=(len(holdout) - escalations) * CLOUD_MILLI_USD,
        false_escalate=tuple(turn.id for turn, pred in judged if pred and turn.label == "stay"),
        false_stay=tuple(turn.id for turn, pred in judged if not pred and turn.label == "escalate"),
    )


def format_table(report: Report) -> str:
    header = ("split", "n", "precision", "recall", "escalations", "cloud usd", "saved vs always")
    row = (
        "holdout",
        str(report.holdout),
        f"{report.precision:.2f}",
        f"{report.recall:.2f}",
        str(report.escalations),
        _usd(report.cloud_milli),
        _usd(report.saved_milli),
    )
    widths = [max(len(header[index]), len(row[index])) for index in range(7)]

    def line(cells: tuple[str, ...]) -> str:
        parts = []
        for index, cell in enumerate(cells):
            if index == 0:
                parts.append(cell.ljust(widths[index]))
            else:
                parts.append(cell.rjust(widths[index]))
        return "  ".join(parts)

    note = (
        f"Threshold {report.threshold} was chosen on train by escalate F1. "
        f"A cloud call is {_usd(CLOUD_MILLI_USD)} illustrative."
    )
    body = [
        line(header),
        line(tuple("-" * width for width in widths)),
        line(row),
    ]
    return "\n".join(body) + "\n\n" + note


def format_details(report: Report) -> str:
    lines = ["holdout misses"]
    if report.false_escalate:
        lines.append("  false escalate: " + ", ".join(report.false_escalate))
    if report.false_stay:
        lines.append("  false stay: " + ", ".join(report.false_stay))
    if len(lines) == 1:
        lines.append("  none")
    return "\n".join(lines)


def _choose_threshold(train: tuple[Turn, ...]) -> int:
    best_t = 0
    best_f1 = -1.0
    for threshold in range(0, UNRESOLVED + NOVEL + HIGH_RISK + LONG + 1):
        score = _f1(train, threshold)
        if score > best_f1 or (score == best_f1 and threshold > best_t):
            best_f1 = score
            best_t = threshold
    return best_t


def _f1(turns: tuple[Turn, ...], threshold: int) -> float:
    tp = sum(turn.label == "escalate" and turn.points() >= threshold for turn in turns)
    fp = sum(turn.label == "stay" and turn.points() >= threshold for turn in turns)
    fn = sum(turn.label == "escalate" and turn.points() < threshold for turn in turns)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ratio(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return part / whole


def _usd(milli: int) -> str:
    return f"${milli / 1000:.3f}"


def _load(path: Path) -> tuple[Turn, ...]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        raise
    if not lines:
        raise ValueError(f"{path}: empty file")
    turns = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        turns.append(_turn(record, path, line_number))
    return tuple(turns)


def _turn(record: object, path: Path, line_number: int) -> Turn:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: each line must be an object")
    label = record.get("label")
    split = record.get("split")
    if split not in {"train", "holdout"}:
        raise ValueError(f"{path}:{line_number}: split must be train or holdout")
    if label not in {"stay", "escalate"}:
        raise ValueError(f"{path}:{line_number}: label must be stay or escalate")
    tokens = record.get("tokens")
    if not isinstance(tokens, int) or tokens < 0:
        raise ValueError(f"{path}:{line_number}: tokens must be a non-negative integer")
    flags = []
    for name in ("unresolved", "novel", "high_risk"):
        value = record.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"{path}:{line_number}: {name} must be true or false")
        flags.append(value)
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
    return Turn(str(record.get("id") or line_number), split, label, flags[0], flags[1], flags[2], tokens, text)
