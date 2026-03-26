from __future__ import annotations

from collections import Counter
from typing import Dict, List


def calculate_risk(detections: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Calculate cargo risk classification from model detections.

    Returns a dict with:
    - level: CLEAR | SUSPICIOUS | PROHIBITED
    - score: 0-100
    - reason: human-readable explanation
    - flags: list of items that triggered risk
    """
    total_threat_categories = 12
    if not detections:
        return {
            "level": "CLEAR",
            "score": 0,
            "reason": (
                f"No prohibited or suspicious items detected across all "
                f"{total_threat_categories} threat categories. Shipment appears consistent "
                "with declared contents."
            ),
            "flags": [],
        }

    prohibited_terms = {"gun", "bullet", "knife"}
    suspicious_terms = {
        "baton",
        "plier",
        "hammer",
        "powerbank",
        "scissors",
        "wrench",
        "sprayer",
        "handcuffs",
        "lighter",
    }

    prohibited_hits: List[Dict[str, object]] = []
    suspicious_hits: List[Dict[str, object]] = []

    for detection in detections:
        label = str(detection.get("label", "")).strip()
        confidence = float(detection.get("confidence", 0.0))
        normalized = label.lower()

        if normalized in prohibited_terms:
            prohibited_hits.append({"label": label, "confidence": confidence})
            continue

        if normalized in suspicious_terms:
            suspicious_hits.append({"label": label, "confidence": confidence})
            continue

    if prohibited_hits:
        top_hit = max(prohibited_hits, key=lambda item: float(item["confidence"]))
        max_conf = float(top_hit["confidence"])
        score = _scaled_score(max_conf, 90, 100)
        counts = _count_labels(prohibited_hits)
        class_summary = _format_class_summary(counts)
        total_instances = sum(counts.values())
        flags = sorted(counts.keys())
        reason = (
            f"{class_summary} detected in shipment ({total_instances} total instance"
            f"{'' if total_instances == 1 else 's'}). Highest confidence: {max_conf * 100:.1f}%. "
            "Prohibited threat item(s) found — immediate physical inspection and seizure protocol required."
        )
        return {
            "level": "PROHIBITED",
            "score": score,
            "reason": reason,
            "flags": flags,
        }

    if suspicious_hits:
        top_hit = max(suspicious_hits, key=lambda item: float(item["confidence"]))
        max_conf = float(top_hit["confidence"])
        score = _scaled_score(max_conf, 50, 75)
        counts = _count_labels(suspicious_hits)
        class_summary = _format_class_summary(counts)
        total_instances = sum(counts.values())
        flags = sorted(counts.keys())
        reason = (
            f"{class_summary} detected in shipment ({total_instances} total instance"
            f"{'' if total_instances == 1 else 's'}). Highest confidence: {max_conf * 100:.1f}%. "
            "Restricted/suspicious item(s) flagged — recommend secondary screening and manifest verification."
        )
        return {
            "level": "SUSPICIOUS",
            "score": score,
            "reason": reason,
            "flags": flags,
        }

    # If no prohibited/suspicious labels are found, treat shipment as clear.
    avg_conf = sum(float(d.get("confidence", 0.0)) for d in detections) / len(detections)
    clear_score = int(round((1.0 - _clamp01(avg_conf)) * 20))
    return {
        "level": "CLEAR",
        "score": clear_score,
        "reason": (
            f"No prohibited or suspicious items detected across all {total_threat_categories} "
            "threat categories. Shipment appears consistent with declared contents."
        ),
        "flags": [],
    }


def _scaled_score(confidence: float, low: int, high: int) -> int:
    confidence = _clamp01(confidence)
    return int(round(low + (high - low) * confidence))


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _count_labels(hits: List[Dict[str, object]]) -> Counter[str]:
    labels = [str(item.get("label", "unknown")).strip() for item in hits]
    return Counter(labels)


def _format_class_summary(counts: Counter[str]) -> str:
    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(label)
        else:
            parts.append(f"{count} {label}")

    if not parts:
        return "No flagged classes"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
