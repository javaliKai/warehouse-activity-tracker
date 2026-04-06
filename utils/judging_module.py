import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


# --------------------------------------------------
# Label alias configuration
# --------------------------------------------------
UNSAFE_FOOTWEAR_ALIASES = {
    "sandal",
    "sandals",
    "slipper",
    "slippers",
    "flip flop",
    "flip-flop",
    "open footwear",
    "open-toe footwear",
}

EQUIPMENT_ALIASES = {
    "pallet jack",
    "forklift",
    "trolley",
    "cart",
    "hand truck",
    "pallet truck",
    "jack",
}

PERSON_ALIASES = {
    "person",
    "worker",
    "man",
    "woman",
    "human",
}


# --------------------------------------------------
# Basic utilities
# --------------------------------------------------
def normalize_label(label: str) -> str:
    """
    Normalize a raw label string from Grounding DINO.
    """
    if label is None:
        return ""
    label = str(label).strip().lower()
    label = label.replace("_", " ").replace("-", " ")
    label = label.replace(".", "").replace(",", "")
    label = " ".join(label.split())
    return label


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.astype(float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def overlap_ratio(box_a: np.ndarray, box_b: np.ndarray, denom: str = "b") -> float:
    """
    Overlap ratio using intersection area divided by area of:
    - 'a' => area(box_a)
    - 'b' => area(box_b)
    """
    inter = intersection_area(box_a, box_b)
    if denom == "a":
        area = box_area(box_a)
    else:
        area = box_area(box_b)

    if area <= 1e-6:
        return 0.0
    return inter / area


def compute_box_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Edge-to-edge distance between two boxes.
    Returns 0 if they overlap.
    """
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)

    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)

    return float(np.hypot(dx, dy))


def get_box_center(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box.astype(float)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def group_boxes_by_label(detections, labels: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert supervision detections + raw labels into a grouped dict.

    Returns:
    {
        "person": [
            {"box": np.ndarray([x1,y1,x2,y2]), "det_idx": 0, "raw_label": "person"},
            ...
        ],
        "sandal": [...],
        ...
    }
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    xyxy = np.asarray(detections.xyxy)
    for i, raw_label in enumerate(labels):
        norm_label = normalize_label(raw_label)
        item = {
            "box": xyxy[i].astype(float),
            "det_idx": i,
            "raw_label": raw_label,
            "label": norm_label,
        }
        grouped.setdefault(norm_label, []).append(item)

    return grouped


def collect_by_aliases(grouped: Dict[str, List[Dict[str, Any]]], aliases: set) -> List[Dict[str, Any]]:
    """
    Collect grouped entries whose normalized label matches any alias exactly.
    """
    out = []
    for label, items in grouped.items():
        if label in aliases:
            out.extend(items)
    return out


def infer_persons(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return collect_by_aliases(grouped, PERSON_ALIASES)


def infer_footwear(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return collect_by_aliases(grouped, UNSAFE_FOOTWEAR_ALIASES)


def infer_equipment(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return collect_by_aliases(grouped, EQUIPMENT_ALIASES)


# --------------------------------------------------
# Rule-specific geometry
# --------------------------------------------------
def make_foot_region(person_box: np.ndarray,
                     foot_height_ratio: float = 0.22,
                     foot_width_ratio: float = 0.60) -> np.ndarray:
    """
    Approximate foot region from the bottom of a person's bounding box.

    foot_height_ratio:
        Bottom percentage of person's height considered as foot/lower-leg region.

    foot_width_ratio:
        Central width percentage used for the foot region.
    """
    x1, y1, x2, y2 = person_box.astype(float)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    foot_h = h * foot_height_ratio
    foot_w = w * foot_width_ratio

    cx = (x1 + x2) / 2.0
    fx1 = cx - foot_w / 2.0
    fx2 = cx + foot_w / 2.0
    fy1 = y2 - foot_h
    fy2 = y2

    return np.array([fx1, fy1, fx2, fy2], dtype=float)


# --------------------------------------------------
# Rule 1: improper footwear
# --------------------------------------------------
def check_improper_footwear(grouped: Dict[str, List[Dict[str, Any]]],
                            overlap_threshold: float = 0.20) -> List[Dict[str, Any]]:
    """
    Rule:
    - find persons
    - create a bottom foot zone for each person
    - if a sandal/slipper overlaps the foot zone enough, trigger violation

    overlap_threshold:
        ratio of sandal box covered by the foot region
    """
    persons = infer_persons(grouped)
    footwear_items = infer_footwear(grouped)

    violations = []

    for p_idx, person in enumerate(persons):
        person_box = person["box"]
        foot_box = make_foot_region(person_box)

        best_match = None
        best_overlap = 0.0

        for fw in footwear_items:
            fw_box = fw["box"]
            ov = overlap_ratio(foot_box, fw_box, denom="b")  # how much of footwear sits in foot zone
            if ov > best_overlap:
                best_overlap = ov
                best_match = fw

        if best_match is not None and best_overlap >= overlap_threshold:
            violations.append({
                "type": "improper_footwear",
                "risk_score": 0.55,
                "risk_level": "Medium",
                "reason": f"Unsafe footwear '{best_match['raw_label']}' detected in a worker's foot region.",
                "person_det_idx": person["det_idx"],
                "related_det_indices": [person["det_idx"], best_match["det_idx"]],
                "person_box": person_box,
                "foot_box": foot_box,
                "matched_box": best_match["box"],
                "matched_label": best_match["raw_label"],
                "match_value": float(best_overlap),
            })

    return violations


# --------------------------------------------------
# Rule 2: unsafe proximity
# --------------------------------------------------
def check_unsafe_proximity(grouped: Dict[str, List[Dict[str, Any]]],
                           distance_threshold_px: float = 40.0) -> List[Dict[str, Any]]:
    """
    Rule:
    - find persons
    - find equipment
    - if box distance is below threshold => unsafe proximity

    For your 640x360 resized frames, a threshold around 35~60 px is a good starting point.
    """
    persons = infer_persons(grouped)
    equipment_items = infer_equipment(grouped)

    violations = []

    for person in persons:
        p_box = person["box"]

        best_eq = None
        best_dist = float("inf")

        for eq in equipment_items:
            eq_box = eq["box"]
            dist = compute_box_distance(p_box, eq_box)
            if dist < best_dist:
                best_dist = dist
                best_eq = eq

        if best_eq is not None and best_dist <= distance_threshold_px:
            risk_score = 0.75 if best_dist > distance_threshold_px * 0.35 else 0.85
            risk_level = "High" if risk_score < 0.85 else "Critical"

            violations.append({
                "type": "unsafe_proximity",
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "reason": f"Worker is too close to equipment '{best_eq['raw_label']}' (distance={best_dist:.1f}px).",
                "person_det_idx": person["det_idx"],
                "related_det_indices": [person["det_idx"], best_eq["det_idx"]],
                "person_box": p_box,
                "matched_box": best_eq["box"],
                "matched_label": best_eq["raw_label"],
                "match_value": float(best_dist),
            })

    return violations


# --------------------------------------------------
# Summarization
# --------------------------------------------------
def summarize_violations(violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(violations) == 0:
        return {
            "is_violation": False,
            "overall_risk_score": 0.0,
            "overall_risk_level": "Safe",
            "top_reason": "No safety violation detected.",
        }

    max_score = max(v["risk_score"] for v in violations)

    # If both rule types appear together, slightly boost overall risk
    violation_types = {v["type"] for v in violations}
    if "improper_footwear" in violation_types and "unsafe_proximity" in violation_types:
        max_score = max(max_score, 0.90)

    if max_score >= 0.85:
        level = "Critical"
    elif max_score >= 0.60:
        level = "High"
    else:
        level = "Medium"

    # Use highest-score violation as the main explanation
    top_violation = sorted(violations, key=lambda x: x["risk_score"], reverse=True)[0]

    return {
        "is_violation": True,
        "overall_risk_score": float(max_score),
        "overall_risk_level": level,
        "top_reason": top_violation["reason"],
    }


# --------------------------------------------------
# Public API
# --------------------------------------------------
def judge_frame(detections,
                labels: List[str],
                frame_idx: int,
                proximity_threshold_px: float = 40.0,
                footwear_overlap_threshold: float = 0.20) -> Dict[str, Any]:
    """
    Main entry point for one frame.

    Returns a dict like:
    {
        "frame_idx": 12,
        "grouped": ...,
        "violations": [...],
        "is_violation": True/False,
        "overall_risk_score": ...,
        "overall_risk_level": ...,
        "top_reason": ...
    }
    """
    grouped = group_boxes_by_label(detections, labels)

    violations = []
    violations.extend(
        check_improper_footwear(
            grouped,
            overlap_threshold=footwear_overlap_threshold
        )
    )
    violations.extend(
        check_unsafe_proximity(
            grouped,
            distance_threshold_px=proximity_threshold_px
        )
    )

    summary = summarize_violations(violations)

    return {
        "frame_idx": frame_idx,
        "grouped": grouped,
        "violations": violations,
        **summary,
    }


# --------------------------------------------------
# Optional rendering helper
# --------------------------------------------------
def draw_judge_overlay(image: np.ndarray,
                       judge_result: Dict[str, Any],
                       pause_badge: bool = True) -> np.ndarray:
    """
    Draw warning banner + risk info on frame.
    Keeps your main script cleaner.
    """
    out = image.copy()
    h, w = out.shape[:2]

    if not judge_result["is_violation"]:
        return out

    # Top red banner
    banner_h = max(50, int(0.12 * h))
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 255), thickness=-1)

    title = "VIOLATION DETECTED"
    risk_text = f"Risk: {judge_result['overall_risk_level']} ({judge_result['overall_risk_score']:.2f})"
    reason_text = judge_result["top_reason"]

    cv2.putText(out, title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, risk_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    # Reason text under banner
    y_reason = banner_h + 28
    cv2.putText(out, reason_text[:90], (15, y_reason), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # Mark involved boxes
    for v in judge_result["violations"]:
        if "person_box" in v:
            x1, y1, x2, y2 = map(int, v["person_box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if "matched_box" in v:
            x1, y1, x2, y2 = map(int, v["matched_box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)

        if v["type"] == "improper_footwear" and "foot_box" in v:
            x1, y1, x2, y2 = map(int, v["foot_box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)

    if pause_badge:
        badge_text = "PAUSE ALERT"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx2 = w - 15
        bx1 = bx2 - tw - 20
        by1 = 10
        by2 = by1 + th + 16
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
        cv2.putText(out, badge_text, (bx1 + 10, by2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return out