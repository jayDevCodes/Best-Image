"""Optional local COCO object detection and simple 2-D relation checks.

Enable explicitly with BEST_IMAGE_ENABLE_OBJECT_DETECTION=1 and install the optional
requirements-vision.txt. This detector covers a fixed COCO class set; unsupported nouns,
small/occluded objects, and depth relations such as "behind" are not reliably checked.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import re
import threading
from typing import Any

_ENABLED = os.getenv("BEST_IMAGE_ENABLE_OBJECT_DETECTION", "").strip().lower() in {"1", "true", "yes", "on"}
_MODEL_NAME = os.getenv("BEST_IMAGE_OBJECT_MODEL", "yolov8n.pt")
_MODEL: Any = None
_LOCK = threading.RLock()

# Common prompt words mapped to YOLO's COCO labels. Only these categories can be
# objectively checked by this optional signal.
_ALIASES = {
    "person": "person", "people": "person", "persons": "person", "man": "person", "men": "person",
    "woman": "person", "women": "person", "boy": "person", "girl": "person", "child": "person",
    "children": "person", "human": "person", "humans": "person",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "car": "car", "cars": "car", "motorcycle": "motorcycle", "motorcycles": "motorcycle", "motorbike": "motorcycle",
    "airplane": "airplane", "airplanes": "airplane", "plane": "airplane", "planes": "airplane",
    "bus": "bus", "buses": "bus", "train": "train", "trains": "train", "truck": "truck", "trucks": "truck",
    "boat": "boat", "boats": "boat", "bird": "bird", "birds": "bird", "cat": "cat", "cats": "cat",
    "dog": "dog", "dogs": "dog", "horse": "horse", "horses": "horse", "sheep": "sheep", "cow": "cow", "cows": "cow",
    "elephant": "elephant", "elephants": "elephant", "bear": "bear", "bears": "bear", "zebra": "zebra", "zebras": "zebra",
    "giraffe": "giraffe", "giraffes": "giraffe", "backpack": "backpack", "backpacks": "backpack",
    "umbrella": "umbrella", "umbrellas": "umbrella", "handbag": "handbag", "suitcase": "suitcase",
    "frisbee": "frisbee", "skis": "skis", "snowboard": "snowboard", "kite": "kite", "kites": "kite",
    "skateboard": "skateboard", "skateboards": "skateboard", "surfboard": "surfboard", "bottle": "bottle",
    "bottles": "bottle", "glass": "wine glass", "wine glass": "wine glass", "cup": "cup", "cups": "cup",
    "fork": "fork", "knife": "knife", "spoon": "spoon", "bowl": "bowl", "banana": "banana", "bananas": "banana",
    "apple": "apple", "apples": "apple", "sandwich": "sandwich", "broccoli": "broccoli", "carrot": "carrot",
    "hot dog": "hot dog", "pizza": "pizza", "donut": "donut", "doughnut": "donut", "cake": "cake",
    "chair": "chair", "chairs": "chair", "couch": "couch", "sofa": "couch", "plant": "potted plant",
    "potted plant": "potted plant", "bed": "bed", "table": "dining table", "dining table": "dining table",
    "toilet": "toilet", "tv": "tv", "television": "tv", "laptop": "laptop", "mouse": "mouse", "remote": "remote",
    "keyboard": "keyboard", "phone": "cell phone", "cell phone": "cell phone", "microwave": "microwave",
    "oven": "oven", "toaster": "toaster", "sink": "sink", "refrigerator": "refrigerator", "fridge": "refrigerator",
    "book": "book", "books": "book", "clock": "clock", "vase": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear", "toothbrush": "toothbrush",
}
_COUNT_WORDS = {
    "no": 0, "zero": 0, "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_RELATIONS = (
    ("to the left of", "left"), ("left of", "left"),
    ("to the right of", "right"), ("right of", "right"),
    ("above", "above"), ("below", "below"), ("under", "below"),
)
_ALIAS_PATTERNS = sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
_ALIAS_REGEX = [(alias, canonical, re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", re.I)) for alias, canonical in _ALIAS_PATTERNS]
_NUMBER = re.compile(r"\b(no|zero|one|a|an|single|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b", re.I)
_RELATION_REGEX = [(phrase, relation, re.compile(r"\b" + re.escape(phrase) + r"\b", re.I)) for phrase, relation in _RELATIONS]


def detector_status() -> dict[str, Any]:
    installed = importlib.util.find_spec("ultralytics") is not None
    available = _ENABLED and installed
    return {
        "enabled": _ENABLED,
        "package_installed": installed,
        "available": available,
        "loaded": _MODEL is not None,
        "model": _MODEL_NAME,
        "message": (
            "Optional local object detection is enabled." if available else
            "Object detection is off by default. Install requirements-vision.txt and set BEST_IMAGE_ENABLE_OBJECT_DETECTION=1 to enable it."
        ),
    }


def _read_expected_count(prompt: str, position: int) -> int | None:
    prefix = prompt[max(0, position - 55):position]
    matches = list(_NUMBER.finditer(prefix))
    if not matches:
        return None
    match = matches[-1]
    if len(prefix[match.end():].split()) > 2:
        return None
    token = match.group(1).casefold()
    if token.isdigit():
        return min(int(token), 100)
    return _COUNT_WORDS.get(token)


def extract_object_mentions(prompt: str) -> list[dict[str, Any]]:
    """Extract COCO-supported class mentions and nearby explicit count targets."""
    found: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for alias, canonical, pattern in _ALIAS_REGEX:
        for match in pattern.finditer(prompt or ""):
            start, end = match.span()
            if any(start < other_end and end > other_start for other_start, other_end in occupied):
                continue
            occupied.append((start, end))
            found.append({
                "label": canonical,
                "start": start,
                "end": end,
                "expected_count": _read_expected_count(prompt, start),
                "source": match.group(0),
            })
    found.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    # Drop duplicates caused by aliases with overlapping canonical forms while preserving
    # repeated mentions of the same class for relation parsing.
    return found


def extract_count_targets(prompt: str) -> list[dict[str, Any]]:
    mentions = extract_object_mentions(prompt)
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []
    for mention in mentions:
        label = mention["label"]
        if label in seen:
            continue
        seen.add(label)
        targets.append({
            "label": label,
            "expected_count": mention["expected_count"],
            "position": mention["start"],
        })
    return targets


def extract_spatial_targets(prompt: str) -> list[dict[str, Any]]:
    mentions = extract_object_mentions(prompt)
    relations: list[dict[str, Any]] = []
    for phrase, relation, pattern in _RELATION_REGEX:
        for match in pattern.finditer(prompt or ""):
            before = [item for item in mentions if item["end"] <= match.start()]
            after = [item for item in mentions if item["start"] >= match.end()]
            if not before or not after:
                continue
            left = max(before, key=lambda item: item["end"])
            right = min(after, key=lambda item: item["start"])
            relations.append({"left_label": left["label"], "right_label": right["label"], "relation": relation, "position": match.start()})
    relations.sort(key=lambda item: item["position"])
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (relation["left_label"], relation["right_label"], relation["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return unique[:3]


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        from ultralytics import YOLO
        _MODEL = YOLO(_MODEL_NAME)
        return _MODEL


def _predict(image: Any) -> list[dict[str, Any]]:
    import torch
    model = _load_model()
    mps = getattr(torch.backends, "mps", None)
    device = "mps" if mps is not None and mps.is_available() else "cpu"
    results = model.predict(source=image, conf=0.25, imgsz=640, verbose=False, device=device, max_det=100)
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    names = getattr(result, "names", {}) or {}
    for coordinates, class_id, confidence in zip(xyxy, classes, confidences):
        label = str(names.get(int(class_id), "")).casefold()
        if not label:
            continue
        x1, y1, x2, y2 = [float(value) for value in coordinates]
        detections.append({
            "label": label,
            "confidence": float(confidence),
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
        })
    return detections


def _object_presence_score(targets: list[dict[str, Any]], detections: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for target in targets:
        count = sum(1 for item in detections if item["label"] == target["label"])
        expected = target["expected_count"]
        if expected is None:
            scores.append(100.0 if count else 0.0)
        else:
            # Each missed/extra instance reduces the score; this is intentionally a coarse signal.
            scores.append(max(0.0, 100.0 - 40.0 * abs(count - expected)))
    return round(sum(scores) / len(scores), 2) if scores else 50.0


def _spatial_score(relations: list[dict[str, Any]], detections: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for relation in relations:
        left_boxes = [item for item in detections if item["label"] == relation["left_label"]]
        right_boxes = [item for item in detections if item["label"] == relation["right_label"]]
        pairs = [(left, right) for left, right in itertools.product(left_boxes, right_boxes) if left is not right]
        if not pairs:
            scores.append(0.0)
            continue
        def matches(pair: tuple[dict[str, Any], dict[str, Any]]) -> bool:
            left, right = pair
            if relation["relation"] == "left":
                return left["center_x"] < right["center_x"]
            if relation["relation"] == "right":
                return left["center_x"] > right["center_x"]
            if relation["relation"] == "above":
                return left["center_y"] < right["center_y"]
            return left["center_y"] > right["center_y"]
        scores.append(100.0 if any(matches(pair) for pair in pairs) else 0.0)
    return round(sum(scores) / len(scores), 2) if scores else 50.0


def score_object_requirements(prompt: str, images: list[Any]) -> dict[str, Any]:
    """Return optional detector scores, degrading gracefully when unavailable."""
    status = detector_status()
    empty = {"available": status["available"], "used": False, "object_scores": None, "spatial_scores": None, "message": status["message"], "warning": None}
    if not status["available"]:
        if re.search(r"\b(exactly|at least|at most|number of|one|two|three|four|five|six|\d+|left|right|above|below|under|next to|beside|behind)\b", prompt, re.I):
            empty["warning"] = "Object-count or spatial terms were found, but the optional COCO object detector is not enabled; verify those details manually."
        return empty

    count_targets = extract_count_targets(prompt)
    spatial_targets = extract_spatial_targets(prompt)
    if not count_targets and not spatial_targets:
        empty["message"] = "Object detection skipped because no supported object or 2-D relation target was found."
        return empty

    try:
        object_scores: list[float] = []
        spatial_scores: list[float] = []
        for image in images:
            detections = _predict(image)
            if count_targets:
                object_scores.append(_object_presence_score(count_targets, detections))
            if spatial_targets:
                spatial_scores.append(_spatial_score(spatial_targets, detections))
        used_object = bool(count_targets)
        used_spatial = bool(spatial_targets)
        messages: list[str] = []
        if used_object:
            messages.append("Local COCO object presence/count signal was added.")
        if used_spatial:
            messages.append("Left/right/above/below relations were estimated from detected box centers.")
        return {
            "available": True,
            "used": used_object or used_spatial,
            "object_scores": object_scores if used_object else None,
            "spatial_scores": spatial_scores if used_spatial else None,
            "object_targets": count_targets,
            "spatial_targets": spatial_targets,
            "message": " ".join(messages),
            "warning": "Object detection covers COCO classes only and can miss small or occluded objects. Spatial scores use 2-D box centers; they do not reliably verify depth (behind/in front of) or nuanced relations.",
        }
    except Exception as exc:
        return {
            "available": False,
            "used": False,
            "object_scores": None,
            "spatial_scores": None,
            "message": "Optional object detection failed: " + type(exc).__name__,
            "warning": "The optional detector failed to load or run. The baseline similarity ranking was used; review counts and placement manually.",
        }
