import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_VIDEO_OUT = "output_violations.mp4"
DEFAULT_VEHICLE_MODEL_WEIGHTS = "yolo12l.pt"
DEFAULT_HELMET_MODEL_WEIGHTS = "runs_detect/runs/yolov8/helmet_v8s/weights/best.pt"
DEFAULT_TRACKER_CFG = "bytetrack.yaml"

HSV_RED1_LO, HSV_RED1_HI = (0, 80, 80), (10, 255, 255)
HSV_RED2_LO, HSV_RED2_HI = (170, 80, 80), (180, 255, 255)
HSV_YELLOW_LO, HSV_YELLOW_HI = (20, 90, 100), (32, 255, 255)
HSV_GREEN_LO, HSV_GREEN_HI = (40, 80, 80), (85, 255, 255)
MIN_PIX_COUNT = 50

VEH_OK = {"car", "bus", "truck", "motorcycle"}
MOTORCYCLE_CLASS = "motorcycle"
HELMET_OK_CLASSES = {"with_helmet", "with helmet"}
HELMET_BAD_CLASSES = {"without_helmet", "without helmet", "no_helmet", "no helmet"}

COLOR_LINE = (0, 0, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_OK = (0, 255, 0)
COLOR_BAD = (0, 0, 255)
COLOR_ROI = (150, 255, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ProcessingOptions:
    conf: float = 0.35
    iou: float = 0.45
    traffic_light_refresh: int = 5


@dataclass
class HelmetProcessingOptions:
    vehicle_conf: float = 0.35
    helmet_conf: float = 0.35
    iou: float = 0.45


@dataclass
class ProcessingProgress:
    processed_frames: int
    total_frames: int
    percent: float
    violations_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    module: str
    video_in: str
    video_out: str
    total_violations: int
    processed_frames: int
    total_frames: int
    fps: float
    width: int
    height: int
    elapsed_seconds: float
    config: dict[str, Any]
    options: dict[str, Any]
    violations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[ProcessingProgress], None]


def put_text(img, txt, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def normalize_line(line: list[int] | tuple[int, int, int, int]) -> list[int]:
    if not isinstance(line, (list, tuple)) or len(line) != 4:
        raise ValueError("Line must contain exactly 4 integers.")
    values = [int(v) for v in line]
    if values[0] == values[2] and values[1] == values[3]:
        raise ValueError("Line start and end points must be different.")
    return values


def normalize_rois(rois: list[list[int]] | tuple[tuple[int, int, int, int], ...]) -> list[list[int]]:
    if not isinstance(rois, (list, tuple)) or len(rois) < 1:
        raise ValueError("At least one traffic-light ROI is required.")

    normalized = []
    for roi in rois:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("Each ROI must contain exactly 4 integers.")
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x1 == x2 or y1 == y2:
            raise ValueError("ROI width and height must be greater than zero.")
        normalized.append([x1, y1, x2, y2])
    return normalized


def validate_geometry(line: list[int], rois: list[list[int]]) -> dict[str, Any]:
    return {"line": normalize_line(line), "rois": normalize_rois(rois)}


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return validate_geometry(payload.get("line"), payload.get("rois", []))


def save_config_file(config_path: str | Path, config: dict[str, Any]) -> None:
    validated = validate_geometry(config.get("line"), config.get("rois", []))
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2), encoding="utf-8")


def extract_first_frame(video_path: str | Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    return frame


def encode_frame_jpeg(frame) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return buffer.tobytes()


def read_video_metadata(video_path: str | Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if total_frames and fps else 0.0
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
    }


def hsv_major_color(bgr):
    if bgr is None or bgr.size == 0:
        return "unknown"
    h, w = bgr.shape[:2]
    if h * w > 160 * 160:
        scale = min(1.0, np.sqrt((160 * 160) / (h * w)))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    vch = clahe.apply(vch)
    hsv = cv2.merge([hch, sch, vch])

    red1 = cv2.inRange(hsv, HSV_RED1_LO, HSV_RED1_HI)
    red2 = cv2.inRange(hsv, HSV_RED2_LO, HSV_RED2_HI)
    yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    green = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red1 = cv2.morphologyEx(red1, cv2.MORPH_OPEN, kernel, iterations=1)
    red2 = cv2.morphologyEx(red2, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    red1 = cv2.morphologyEx(red1, cv2.MORPH_CLOSE, kernel, iterations=1)
    red2 = cv2.morphologyEx(red2, cv2.MORPH_CLOSE, kernel, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=1)

    red = red1 + red2
    vnorm = vch.astype(np.float32) / 255.0
    r = float((red / 255.0 * vnorm).sum())
    y = float((yellow / 255.0 * vnorm).sum())
    g = float((green / 255.0 * vnorm).sum())

    total_px = bgr.shape[0] * bgr.shape[1]
    ratio_thresh = 0.002
    valid = (
        (r > MIN_PIX_COUNT or r / total_px > ratio_thresh)
        or (y > MIN_PIX_COUNT or y / total_px > ratio_thresh)
        or (g > MIN_PIX_COUNT or g / total_px > ratio_thresh)
    )
    if not valid:
        return "unknown"

    vals = {"red": r, "yellow": y, "green": g}
    best = max(vals, key=vals.get)
    second = sorted(vals.values(), reverse=True)[1]
    if second > 0 and (vals[best] / (second + 1e-6)) < 1.15:
        return "unknown"
    return best


def crop_roi(frame, roi):
    x1, y1, x2, y2 = roi
    xa, ya = max(0, min(x1, x2)), max(0, min(y1, y2))
    xb, yb = min(frame.shape[1] - 1, max(x1, x2)), min(frame.shape[0] - 1, max(y1, y2))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()


def infer_all_roi_colors(frame, rois):
    colors = []
    for roi in rois:
        patch = crop_roi(frame, roi)
        colors.append(hsv_major_color(patch))
    return colors


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def normalized_class_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def is_no_helmet_class(name: str) -> bool:
    normalized = normalized_class_name(name)
    return normalized in {normalized_class_name(value) for value in HELMET_BAD_CLASSES}


def is_helmet_class(name: str) -> bool:
    normalized = normalized_class_name(name)
    return normalized in {normalized_class_name(value) for value in HELMET_OK_CLASSES}


def violation_event(
    *,
    module: str,
    frame_idx: int,
    fps: float,
    label: str,
    track_id: int | None = None,
    class_name: str | None = None,
    box: tuple[int, int, int, int] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seconds = frame_idx / fps if fps > 0 else 0.0
    event: dict[str, Any] = {
        "id": f"{module}-{track_id if track_id is not None else 'event'}-{frame_idx}",
        "module": module,
        "label": label,
        "frame": int(frame_idx),
        "time_seconds": round(float(seconds), 3),
    }
    if track_id is not None:
        event["track_id"] = int(track_id)
    if class_name is not None:
        event["class_name"] = class_name
    if box is not None:
        event["box"] = [int(v) for v in box]
    if details:
        event["details"] = details
    return event


def draw_stats_panel(frame, total_violations):
    panel_w, panel_h = 260, 70
    x0, y0 = 15, 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 25, 25), -1)
    frame[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(
        overlay[y0 : y0 + panel_h, x0 : x0 + panel_w],
        0.45,
        frame[y0 : y0 + panel_h, x0 : x0 + panel_w],
        0.55,
        0,
    )
    put_text(frame, "Violations:", (x0 + 12, y0 + 28), COLOR_TEXT, 0.7, 2)
    put_text(frame, f"{total_violations}", (x0 + 12, y0 + 58), COLOR_BAD, 0.95, 2)


def make_browser_friendly_mp4(path: str | Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return

    video_path = Path(path)
    marker_path = video_path.with_suffix(f"{video_path.suffix}.browser-ready")
    if marker_path.exists() and marker_path.stat().st_mtime >= video_path.stat().st_mtime:
        return

    temp_path = video_path.with_name(f"{video_path.stem}.browser{video_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.replace(video_path)
        marker_path.write_text("ok\n", encoding="utf-8")
    except Exception:
        temp_path.unlink(missing_ok=True)


def _progress(processed_frames: int, total_frames: int, violations_count: int) -> ProcessingProgress:
    if total_frames > 0:
        percent = min(100.0, round((processed_frames / total_frames) * 100.0, 2))
    else:
        percent = 0.0
    return ProcessingProgress(
        processed_frames=processed_frames,
        total_frames=total_frames,
        percent=percent,
        violations_count=violations_count,
    )


def process_red_light_video(
    video_in: str | Path,
    video_out: str | Path,
    line: list[int],
    rois: list[list[int]],
    *,
    weights: str | Path = DEFAULT_VEHICLE_MODEL_WEIGHTS,
    tracker: str | Path = DEFAULT_TRACKER_CFG,
    options: ProcessingOptions | None = None,
    model: YOLO | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = False,
) -> ProcessingResult:
    options = options or ProcessingOptions()
    geometry = validate_geometry(line, rois)
    video_path = Path(video_in)
    output_path = Path(video_out)
    weights_path = Path(weights)
    tracker_path = Path(tracker)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if model is None and not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata["fps"] or 30.0)
    total_frames = int(metadata["total_frames"] or 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    detector = model or YOLO(str(weights_path))
    start_time = time.time()
    stream = detector.track(
        source=str(video_path),
        conf=options.conf,
        iou=options.iou,
        tracker=str(tracker_path),
        stream=True,
        persist=True,
        verbose=verbose,
    )

    p1 = (geometry["line"][0], geometry["line"][1])
    p2 = (geometry["line"][2], geometry["line"][3])
    names_cache = None
    frame_idx = -1
    roi_colors: list[str] = []
    violated_ids: set[int] = set()
    prev_center: dict[int, tuple[int, int]] = {}
    violations: list[dict[str, Any]] = []
    violations_count = 0

    try:
        for res in stream:
            frame_idx += 1
            frame = res.orig_img.copy()

            if names_cache is None and hasattr(res, "names") and isinstance(res.names, dict):
                names_cache = {int(k): v for k, v in res.names.items()}

            if frame_idx % options.traffic_light_refresh == 0 or not roi_colors:
                roi_colors = infer_all_roi_colors(frame, geometry["rois"])

            any_red = any(color == "red" for color in roi_colors)
            cv2.line(frame, p1, p2, COLOR_LINE, 2)

            for idx, roi in enumerate(geometry["rois"]):
                x1, y1, x2, y2 = roi
                rx1, ry1, rx2, ry2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_ROI, 2)
                color_name = roi_colors[idx] if idx < len(roi_colors) else "unknown"
                put_text(frame, color_name.upper(), (rx1, max(20, ry1 - 8)), COLOR_ROI, 0.7, 2)

            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
                clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

                for bb, tid, cidx, conf in zip(xyxy, ids, clses, confs):
                    cls_name = names_cache.get(int(cidx), str(cidx)) if names_cache else str(cidx)
                    if cls_name not in VEH_OK or conf < options.conf:
                        continue

                    x1, y1, x2, y2 = map(int, bb)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if tid not in prev_center:
                        prev_center[tid] = (cx, cy)

                    crossed_now = segments_intersect(prev_center[tid], (cx, cy), p1, p2)
                    prev_center[tid] = (cx, cy)

                    if any_red and crossed_now and tid not in violated_ids:
                        violations.append(
                            violation_event(
                                module="red_light",
                                frame_idx=frame_idx,
                                fps=fps,
                                label="Red-light violation",
                                track_id=int(tid),
                                class_name=cls_name,
                                box=(x1, y1, x2, y2),
                                details={
                                    "traffic_light_colors": roi_colors,
                                    "line": geometry["line"],
                                },
                            )
                        )
                        violated_ids.add(tid)

                    if tid in violated_ids:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BAD, 3)
                        put_text(frame, "VIOLATION", (x1, max(22, y1 - 10)), COLOR_BAD, 0.8, 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
                        put_text(frame, f"{cls_name} ID{tid}", (x1, max(22, y1 - 10)), (180, 255, 180), 0.6, 2)

                violations_count = len(violated_ids)

            draw_stats_panel(frame, violations_count)
            out.write(frame)

            processed_frames = frame_idx + 1
            if progress_callback:
                progress_callback(_progress(processed_frames, total_frames, violations_count))
    finally:
        out.release()
        make_browser_friendly_mp4(output_path)

    processed_frames = frame_idx + 1 if frame_idx >= 0 else 0
    elapsed_seconds = time.time() - start_time
    if progress_callback:
        progress_callback(
            ProcessingProgress(
                processed_frames=processed_frames,
                total_frames=total_frames,
                percent=100.0,
                violations_count=violations_count,
            )
        )

    return ProcessingResult(
        module="red_light",
        video_in=str(video_path),
        video_out=str(output_path),
        total_violations=violations_count,
        processed_frames=processed_frames,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        elapsed_seconds=elapsed_seconds,
        config=geometry,
        options=asdict(options),
        violations=violations,
    )


def process_video(*args, **kwargs) -> ProcessingResult:
    return process_red_light_video(*args, **kwargs)


def _association_zone(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expand = int(width * 0.45)
    zx1 = max(0, x1 - expand)
    zx2 = min(frame_width - 1, x2 + expand)
    zy1 = max(0, y1 - int(height * 0.85))
    zy2 = min(frame_height - 1, y1 + int(height * 0.9))
    return zx1, zy1, zx2, zy2


def _point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return _box_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))


def _x_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, _ay1, ax2, _ay2 = a
    bx1, _by1, bx2, _by2 = b
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    return overlap / max(1, ax2 - ax1)


def _match_no_helmet_to_motorcycle(
    detection: dict[str, Any],
    motorcycle: dict[str, Any],
) -> float:
    if not is_no_helmet_class(detection["class"]):
        return 0.0

    helmet_box = detection["box"]
    helmet_area = max(1, _box_area(helmet_box))
    zone = motorcycle["zone"]
    motorcycle_box = motorcycle["box"]
    center_in_zone = _point_in_box(detection["center"], zone)
    zone_overlap = _intersection_area(helmet_box, zone) / helmet_area
    x_overlap = _x_overlap_ratio(helmet_box, motorcycle_box)

    mx1, my1, mx2, my2 = motorcycle_box
    _hx, hy = detection["center"]
    motorcycle_height = max(1, my2 - my1)
    plausible_vertical = (my1 - motorcycle_height * 0.95) <= hy <= (my1 + motorcycle_height * 0.95)

    if not (center_in_zone or zone_overlap >= 0.12 or (x_overlap >= 0.25 and plausible_vertical)):
        return 0.0

    score = zone_overlap + x_overlap
    if center_in_zone:
        score += 1.0
    return score


def process_helmet_video(
    video_in: str | Path,
    video_out: str | Path,
    *,
    vehicle_weights: str | Path = DEFAULT_VEHICLE_MODEL_WEIGHTS,
    helmet_weights: str | Path = DEFAULT_HELMET_MODEL_WEIGHTS,
    tracker: str | Path = DEFAULT_TRACKER_CFG,
    options: HelmetProcessingOptions | None = None,
    vehicle_model: YOLO | None = None,
    helmet_model: YOLO | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = False,
) -> ProcessingResult:
    options = options or HelmetProcessingOptions()
    video_path = Path(video_in)
    output_path = Path(video_out)
    vehicle_weights_path = Path(vehicle_weights)
    helmet_weights_path = Path(helmet_weights)
    tracker_path = Path(tracker)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if vehicle_model is None and not vehicle_weights_path.exists():
        raise FileNotFoundError(f"Vehicle model weights not found: {vehicle_weights_path}")
    if helmet_model is None and not helmet_weights_path.exists():
        raise FileNotFoundError(f"Helmet model weights not found: {helmet_weights_path}")
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata["fps"] or 30.0)
    total_frames = int(metadata["total_frames"] or 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    vehicle_detector = vehicle_model or YOLO(str(vehicle_weights_path))
    helmet_detector = helmet_model or YOLO(str(helmet_weights_path))
    start_time = time.time()
    stream = vehicle_detector.track(
        source=str(video_path),
        conf=options.vehicle_conf,
        iou=options.iou,
        tracker=str(tracker_path),
        stream=True,
        persist=True,
        verbose=verbose,
    )

    vehicle_names_cache = None
    helmet_names_cache = getattr(helmet_detector, "names", None)
    if isinstance(helmet_names_cache, dict):
        helmet_names_cache = {int(k): v for k, v in helmet_names_cache.items()}

    frame_idx = -1
    violated_ids: set[int] = set()
    violations: list[dict[str, Any]] = []
    violations_count = 0

    try:
        for res in stream:
            frame_idx += 1
            frame = res.orig_img.copy()

            if vehicle_names_cache is None and hasattr(res, "names") and isinstance(res.names, dict):
                vehicle_names_cache = {int(k): v for k, v in res.names.items()}

            motorcycles: list[dict[str, Any]] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
                clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

                for bb, tid, cidx, conf in zip(xyxy, ids, clses, confs):
                    cls_name = vehicle_names_cache.get(int(cidx), str(cidx)) if vehicle_names_cache else str(cidx)
                    if cls_name != MOTORCYCLE_CLASS or conf < options.vehicle_conf:
                        continue
                    x1, y1, x2, y2 = map(int, bb)
                    zone = _association_zone((x1, y1, x2, y2), width, height)
                    motorcycles.append({"id": int(tid), "box": (x1, y1, x2, y2), "zone": zone})

            helmet_results = helmet_detector.predict(
                source=frame,
                conf=options.helmet_conf,
                iou=options.iou,
                verbose=False,
            )
            helmet_boxes = getattr(helmet_results[0], "boxes", None) if helmet_results else None
            helmet_detections: list[dict[str, Any]] = []
            if helmet_boxes is not None and len(helmet_boxes) > 0:
                h_xyxy = helmet_boxes.xyxy.cpu().numpy()
                h_clses = (
                    helmet_boxes.cls.cpu().numpy().astype(int)
                    if helmet_boxes.cls is not None
                    else np.zeros(len(h_xyxy), dtype=int)
                )
                h_confs = helmet_boxes.conf.cpu().numpy() if helmet_boxes.conf is not None else np.ones(len(h_xyxy))

                for bb, cidx, conf in zip(h_xyxy, h_clses, h_confs):
                    cls_name = helmet_names_cache.get(int(cidx), str(cidx)) if helmet_names_cache else str(cidx)
                    if conf < options.helmet_conf:
                        continue
                    x1, y1, x2, y2 = map(int, bb)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    helmet_detections.append(
                        {
                            "class": cls_name,
                            "box": (x1, y1, x2, y2),
                            "center": center,
                        }
                    )

                    color = COLOR_BAD if is_no_helmet_class(cls_name) else COLOR_OK
                    label = "NO HELMET" if is_no_helmet_class(cls_name) else "HELMET"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    put_text(frame, label, (x1, max(22, y1 - 8)), color, 0.55, 2)

            no_helmet_by_motorcycle: dict[int, tuple[float, dict[str, Any]]] = {}
            for detection in helmet_detections:
                if not is_no_helmet_class(detection["class"]):
                    continue
                best_score = 0.0
                best_motorcycle = None
                for motorcycle in motorcycles:
                    score = _match_no_helmet_to_motorcycle(detection, motorcycle)
                    if score > best_score:
                        best_score = score
                        best_motorcycle = motorcycle
                if best_motorcycle is None:
                    continue
                tid = int(best_motorcycle["id"])
                previous = no_helmet_by_motorcycle.get(tid)
                if previous is None or best_score > previous[0]:
                    no_helmet_by_motorcycle[tid] = (best_score, detection)

            motorcycles_by_id = {int(motorcycle["id"]): motorcycle for motorcycle in motorcycles}
            for tid, (matched_score, matched_no_helmet) in no_helmet_by_motorcycle.items():
                motorcycle = motorcycles_by_id[tid]
                if tid not in violated_ids:
                    violations.append(
                        violation_event(
                            module="helmet",
                            frame_idx=frame_idx,
                            fps=fps,
                            label="Helmet violation",
                            track_id=tid,
                            class_name=MOTORCYCLE_CLASS,
                            box=motorcycle["box"],
                            details={
                                "helmet_class": matched_no_helmet["class"],
                                "helmet_box": [int(v) for v in matched_no_helmet["box"]],
                                "association_zone": [int(v) for v in motorcycle["zone"]],
                                "match_score": round(float(matched_score), 3),
                            },
                        )
                    )
                    violated_ids.add(tid)

            for motorcycle in motorcycles:
                x1, y1, x2, y2 = motorcycle["box"]
                zx1, zy1, zx2, zy2 = motorcycle["zone"]
                tid = motorcycle["id"]
                if tid in violated_ids:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BAD, 3)
                    put_text(frame, f"HELMET VIOLATION ID{tid}", (x1, max(22, y1 - 10)), COLOR_BAD, 0.75, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
                    put_text(frame, f"motorcycle ID{tid}", (x1, max(22, y1 - 10)), (180, 255, 180), 0.6, 2)
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 180, 60), 1)

            violations_count = len(violated_ids)
            draw_stats_panel(frame, violations_count)
            out.write(frame)

            processed_frames = frame_idx + 1
            if progress_callback:
                progress_callback(_progress(processed_frames, total_frames, violations_count))
    finally:
        out.release()
        make_browser_friendly_mp4(output_path)

    processed_frames = frame_idx + 1 if frame_idx >= 0 else 0
    elapsed_seconds = time.time() - start_time
    if progress_callback:
        progress_callback(
            ProcessingProgress(
                processed_frames=processed_frames,
                total_frames=total_frames,
                percent=100.0,
                violations_count=violations_count,
            )
        )

    return ProcessingResult(
        module="helmet",
        video_in=str(video_path),
        video_out=str(output_path),
        total_violations=violations_count,
        processed_frames=processed_frames,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        elapsed_seconds=elapsed_seconds,
        config={},
        options=asdict(options),
        violations=violations,
    )
