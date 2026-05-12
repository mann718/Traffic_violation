import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from ultralytics import YOLO

from .detector import (
    DEFAULT_HELMET_MODEL_WEIGHTS,
    DEFAULT_SPEED_MODEL_WEIGHTS,
    DEFAULT_TRACKER_CFG,
    DEFAULT_VEHICLE_MODEL_WEIGHTS,
    HelmetProcessingOptions,
    ProcessingOptions,
    SpeedLaneProcessingOptions,
    encode_frame_jpeg,
    extract_first_frame,
    load_config_file,
    make_browser_friendly_mp4,
    process_helmet_video,
    process_red_light_video,
    process_speed_lane_video,
    read_video_metadata,
    save_config_file,
    validate_geometry,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = ROOT_DIR / "inputs"
HELMET_INPUTS_DIR = ROOT_DIR / "inputs_helmet"
APP_DATA_DIR = ROOT_DIR / "app_data"
UPLOADS_DIR = APP_DATA_DIR / "uploads"
CONFIGS_DIR = APP_DATA_DIR / "configs"
RUNS_DIR = APP_DATA_DIR / "runs"
def resolve_existing_path(*relative_candidates: str) -> Path:
    for candidate in relative_candidates:
        path = ROOT_DIR / candidate
        if path.exists():
            return path
    return ROOT_DIR / relative_candidates[0]


WEIGHTS_PATH = resolve_existing_path(
    DEFAULT_VEHICLE_MODEL_WEIGHTS,
    f"backend/{DEFAULT_VEHICLE_MODEL_WEIGHTS}",
)
HELMET_WEIGHTS_PATH = resolve_existing_path(
    DEFAULT_HELMET_MODEL_WEIGHTS,
    f"backend/{DEFAULT_HELMET_MODEL_WEIGHTS}",
)
TRACKER_PATH = ROOT_DIR / DEFAULT_TRACKER_CFG
SPEED_WEIGHTS_PATH = resolve_existing_path(
    DEFAULT_SPEED_MODEL_WEIGHTS,
    f"backend/{DEFAULT_SPEED_MODEL_WEIGHTS}",
)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ProcessingModule = Literal["red_light", "helmet", "speed_lane"]


class GeometryPayload(BaseModel):
    line: list[int] = Field(..., min_length=4, max_length=4)
    rois: list[list[int]]


class JobCreatePayload(BaseModel):
    video_id: str
    module: ProcessingModule = "red_light"
    line: list[int] | None = Field(None, min_length=4, max_length=4)
    rois: list[list[int]] = Field(default_factory=list)
    conf: float = Field(0.35, ge=0.01, le=1.0)
    vehicle_conf: float = Field(0.35, ge=0.01, le=1.0)
    helmet_conf: float = Field(0.35, ge=0.01, le=1.0)
    iou: float = Field(0.45, ge=0.01, le=1.0)
    traffic_light_refresh: int = Field(5, ge=1, le=120)
    line1_ratio: float = Field(0.70, ge=0.1, le=0.99)
    line2_ratio: float = Field(0.85, ge=0.1, le=0.99)
    meters_between_lines: float = Field(8.0, gt=0.1, le=200.0)
    speed_persist_frames: int = Field(150, ge=1, le=1000)
    speed_limit_kmh: float = Field(50.0, ge=1.0, le=300.0)
    violation_frame_threshold: int = Field(8, ge=1, le=240)


jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
model_lock = threading.Lock()
cached_vehicle_model: YOLO | None = None
cached_helmet_model: YOLO | None = None
cached_speed_model: YOLO | None = None

def allowed_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Red-Light Violation Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_storage() -> None:
    for path in (INPUTS_DIR, HELMET_INPUTS_DIR, UPLOADS_DIR, CONFIGS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem or "video"
    suffix = Path(filename).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "video"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload must be a supported video file.")
    return f"{stem}{suffix}"


def safe_config_name(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)


def video_id_for(kind: str, path: Path) -> str:
    return f"{kind}__{path.name}"


def resolve_video(video_id: str) -> tuple[Path, str]:
    if video_id.startswith("sample__"):
        name = video_id.removeprefix("sample__")
        path = INPUTS_DIR / Path(name).name
        kind = "sample"
    elif video_id.startswith("helmet__"):
        name = video_id.removeprefix("helmet__")
        path = HELMET_INPUTS_DIR / Path(name).name
        kind = "helmet"
    elif video_id.startswith("upload__"):
        name = video_id.removeprefix("upload__")
        path = UPLOADS_DIR / Path(name).name
        kind = "upload"
    else:
        raise HTTPException(status_code=404, detail="Unknown video id.")

    if not path.exists() or path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Video not found.")
    return path, kind


def writable_config_path(video_id: str) -> Path:
    return CONFIGS_DIR / f"{safe_config_name(video_id)}.json"


def readable_config_path(video_id: str, video_path: Path, kind: str) -> Path | None:
    saved = writable_config_path(video_id)
    if saved.exists():
        return saved
    sidecar = video_path.with_suffix(video_path.suffix + ".setup.json")
    if kind in {"sample", "helmet"} and sidecar.exists():
        return sidecar
    return None


def describe_video(path: Path, kind: str) -> dict[str, Any]:
    video_id = video_id_for(kind, path)
    metadata = read_video_metadata(path)
    config_path = readable_config_path(video_id, path, kind)
    return {
        "id": video_id,
        "name": path.name,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "has_config": config_path is not None,
        **metadata,
    }


def validate_geometry_or_400(line: list[int], rois: list[list[int]]) -> dict[str, Any]:
    try:
        return validate_geometry(line, rois)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def list_video_paths() -> list[tuple[Path, str]]:
    sample_paths = sorted(INPUTS_DIR.glob("*.mp4")) if INPUTS_DIR.exists() else []
    helmet_paths = sorted(HELMET_INPUTS_DIR.glob("*.mp4")) if HELMET_INPUTS_DIR.exists() else []
    upload_paths = sorted(path for path in UPLOADS_DIR.glob("*") if path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS)
    return (
        [(path, "sample") for path in sample_paths]
        + [(path, "helmet") for path in helmet_paths]
        + [(path, "upload") for path in upload_paths]
    )


def get_vehicle_model() -> YOLO:
    global cached_vehicle_model
    with model_lock:
        if cached_vehicle_model is None:
            if not WEIGHTS_PATH.exists():
                raise RuntimeError(f"Model weights not found: {WEIGHTS_PATH}")
            cached_vehicle_model = YOLO(str(WEIGHTS_PATH))
        return cached_vehicle_model


def get_helmet_model() -> YOLO:
    global cached_helmet_model
    with model_lock:
        if cached_helmet_model is None:
            if not HELMET_WEIGHTS_PATH.exists():
                raise RuntimeError(
                    f"Helmet model weights not found: {HELMET_WEIGHTS_PATH}. "
                    "Provide trained helmet weights (with/without helmet classes) and try again."
                )
            cached_helmet_model = YOLO(str(HELMET_WEIGHTS_PATH))
        return cached_helmet_model


def get_speed_model() -> YOLO:
    global cached_speed_model
    with model_lock:
        if cached_speed_model is None:
            model_path = SPEED_WEIGHTS_PATH if SPEED_WEIGHTS_PATH.exists() else WEIGHTS_PATH
            if not model_path.exists():
                raise RuntimeError(
                    "Speed model weights not found. Checked "
                    f"{SPEED_WEIGHTS_PATH} and fallback {WEIGHTS_PATH}."
                )
            cached_speed_model = YOLO(str(model_path))
        return cached_speed_model


def persist_job(job: dict[str, Any]) -> None:
    job_dir = Path(job["run_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    public = public_job(job)
    (job_dir / "job.json").write_text(json.dumps(public, indent=2), encoding="utf-8")


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "module": job["module"],
        "video_id": job["video_id"],
        "video_name": job["video_name"],
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "result": job.get("result"),
    }


def persisted_job_path(job_id: str) -> Path:
    return RUNS_DIR / Path(job_id).name / "job.json"


def load_persisted_job(job_id: str) -> dict[str, Any] | None:
    path = persisted_job_path(job_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("id") != job_id:
        return None
    return payload


def iter_persisted_jobs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    persisted = []
    for path in RUNS_DIR.glob("*/job.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("id"):
            persisted.append(payload)
    return persisted


def latest_persisted_job(module: ProcessingModule) -> dict[str, Any] | None:
    candidates = [
        job
        for job in iter_persisted_jobs()
        if job.get("module") == module and job.get("status") == "succeeded" and job.get("result")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda job: job.get("finished_at") or job.get("started_at") or job.get("created_at") or "")


def output_path_for_public_job(job: dict[str, Any]) -> Path | None:
    job_id = job.get("id")
    if not job_id:
        return None
    run_dir = RUNS_DIR / Path(str(job_id)).name
    module = job.get("module")
    if module == "helmet":
        preferred = "output_helmet_violations.mp4"
    elif module == "speed_lane":
        preferred = "output_speed_lane_violations.mp4"
    else:
        preferred = "output_red_light_violations.mp4"
    preferred_path = run_dir / preferred
    if preferred_path.exists():
        return preferred_path
    outputs = sorted(run_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return outputs[0] if outputs else None


def active_job_exists() -> bool:
    return any(job["status"] in {"queued", "running"} for job in jobs.values())


def run_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
        job["started_at"] = now_iso()
        job["progress"] = {"processed_frames": 0, "total_frames": job["total_frames"], "percent": 0.0, "violations_count": 0}
        persist_job(job)

    def on_progress(progress):
        with jobs_lock:
            current = jobs[job_id]
            current["progress"] = progress.to_dict()

    try:
        if job["module"] == "helmet":
            result = process_helmet_video(
                video_in=job["video_path"],
                video_out=job["output_path"],
                vehicle_weights=WEIGHTS_PATH,
                helmet_weights=HELMET_WEIGHTS_PATH,
                tracker=TRACKER_PATH,
                options=HelmetProcessingOptions(
                    vehicle_conf=job["options"]["vehicle_conf"],
                    helmet_conf=job["options"]["helmet_conf"],
                    iou=job["options"]["iou"],
                ),
                vehicle_model=get_vehicle_model(),
                helmet_model=get_helmet_model(),
                progress_callback=on_progress,
                verbose=False,
            )
        elif job["module"] == "speed_lane":
            result = process_speed_lane_video(
                video_in=job["video_path"],
                video_out=job["output_path"],
                weights=SPEED_WEIGHTS_PATH,
                tracker=TRACKER_PATH,
                options=SpeedLaneProcessingOptions(
                    conf=job["options"]["conf"],
                    iou=job["options"]["iou"],
                    line1_ratio=job["options"]["line1_ratio"],
                    line2_ratio=job["options"]["line2_ratio"],
                    meters_between_lines=job["options"]["meters_between_lines"],
                    speed_persist_frames=job["options"]["speed_persist_frames"],
                    speed_limit_kmh=job["options"]["speed_limit_kmh"],
                    violation_frame_threshold=job["options"]["violation_frame_threshold"],
                ),
                model=get_speed_model(),
                progress_callback=on_progress,
                verbose=False,
            )
        else:
            result = process_red_light_video(
                video_in=job["video_path"],
                video_out=job["output_path"],
                line=job["config"]["line"],
                rois=job["config"]["rois"],
                weights=WEIGHTS_PATH,
                tracker=TRACKER_PATH,
                options=ProcessingOptions(
                    conf=job["options"]["conf"],
                    iou=job["options"]["iou"],
                    traffic_light_refresh=job["options"]["traffic_light_refresh"],
                ),
                model=get_vehicle_model(),
                progress_callback=on_progress,
                verbose=False,
            )
        with jobs_lock:
            current = jobs[job_id]
            current["status"] = "succeeded"
            current["finished_at"] = now_iso()
            current["progress"] = {
                "processed_frames": result.processed_frames,
                "total_frames": result.total_frames,
                "percent": 100.0,
                "violations_count": result.total_violations,
            }
            current["result"] = {
                "module": result.module,
                "total_violations": result.total_violations,
                "processed_frames": result.processed_frames,
                "total_frames": result.total_frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "elapsed_seconds": result.elapsed_seconds,
                "config": result.config,
                "options": result.options,
                "violations": result.violations,
                "video_url": f"/api/jobs/{job_id}/video",
            }
            persist_job(current)
    except Exception as exc:
        with jobs_lock:
            current = jobs[job_id]
            current["status"] = "failed"
            current["finished_at"] = now_iso()
            current["error"] = str(exc)
            persist_job(current)


@app.on_event("startup")
def startup() -> None:
    ensure_storage()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/videos")
def list_videos() -> dict[str, list[dict[str, Any]]]:
    ensure_storage()
    videos = []
    for path, kind in list_video_paths():
        try:
            videos.append(describe_video(path, kind))
        except Exception:
            continue
    return {"videos": videos}


@app.post("/api/videos/upload", status_code=201)
def upload_video(module: ProcessingModule = "red_light", file: UploadFile = File(...)) -> dict[str, Any]:
    ensure_storage()
    filename = safe_filename(file.filename or "video.mp4")
    target_dir = HELMET_INPUTS_DIR if module == "helmet" else INPUTS_DIR
    kind = "helmet" if module == "helmet" else "sample"
    target = target_dir / filename
    if target.exists():
        target = target_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return describe_video(target, kind)


@app.get("/api/videos/{video_id}/frame")
def first_frame(video_id: str) -> Response:
    path, _kind = resolve_video(video_id)
    frame = extract_first_frame(path)
    return Response(content=encode_frame_jpeg(frame), media_type="image/jpeg")


@app.get("/api/videos/{video_id}/config")
def get_config(video_id: str) -> dict[str, Any]:
    path, kind = resolve_video(video_id)
    config_path = readable_config_path(video_id, path, kind)
    if config_path is None:
        return {"line": None, "rois": []}
    try:
        return load_config_file(config_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/videos/{video_id}/config")
def put_config(video_id: str, payload: GeometryPayload) -> dict[str, Any]:
    path, _kind = resolve_video(video_id)
    _ = path
    config = validate_geometry_or_400(payload.line, payload.rois)
    save_config_file(writable_config_path(video_id), config)
    return config


@app.post("/api/jobs", status_code=202)
def create_job(payload: JobCreatePayload) -> dict[str, Any]:
    ensure_storage()
    video_path, _kind = resolve_video(payload.video_id)
    config = {}
    if payload.module == "red_light":
        if payload.line is None:
            raise HTTPException(status_code=400, detail="Line is required for red-light processing.")
        config = validate_geometry_or_400(payload.line, payload.rois)
    metadata = read_video_metadata(video_path)

    with jobs_lock:
        if active_job_exists():
            raise HTTPException(status_code=409, detail="A processing job is already running. Please wait for it to finish.")

        job_id = uuid.uuid4().hex
        run_dir = RUNS_DIR / job_id
        if payload.module == "helmet":
            output_name = "output_helmet_violations.mp4"
        elif payload.module == "speed_lane":
            output_name = "output_speed_lane_violations.mp4"
        else:
            output_name = "output_red_light_violations.mp4"
        output_path = run_dir / output_name
        job = {
            "id": job_id,
            "module": payload.module,
            "video_id": payload.video_id,
            "video_name": video_path.name,
            "video_path": str(video_path),
            "output_path": str(output_path),
            "run_dir": str(run_dir),
            "status": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
            "total_frames": metadata["total_frames"],
            "progress": {"processed_frames": 0, "total_frames": metadata["total_frames"], "percent": 0.0, "violations_count": 0},
            "config": config,
            "options": {
                "conf": payload.conf,
                "vehicle_conf": payload.vehicle_conf,
                "helmet_conf": payload.helmet_conf,
                "iou": payload.iou,
                "traffic_light_refresh": payload.traffic_light_refresh,
                "line1_ratio": payload.line1_ratio,
                "line2_ratio": payload.line2_ratio,
                "meters_between_lines": payload.meters_between_lines,
                "speed_persist_frames": payload.speed_persist_frames,
                "speed_limit_kmh": payload.speed_limit_kmh,
                "violation_frame_threshold": payload.violation_frame_threshold,
            },
        }
        jobs[job_id] = job
        persist_job(job)

    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()
    return public_job(job)


@app.get("/api/jobs/latest")
def get_latest_job(module: ProcessingModule) -> dict[str, Any] | None:
    with jobs_lock:
        memory_candidates = [
            public_job(job)
            for job in jobs.values()
            if job.get("module") == module and job.get("status") == "succeeded" and job.get("result")
        ]
    persisted = latest_persisted_job(module)
    candidates = memory_candidates + ([persisted] if persisted else [])
    if not candidates:
        return None
    return max(candidates, key=lambda job: job.get("finished_at") or job.get("started_at") or job.get("created_at") or "")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            return public_job(job)
    persisted = load_persisted_job(job_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return persisted


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            if job["status"] != "succeeded":
                raise HTTPException(status_code=409, detail="Output video is not ready yet.")
            output_path = Path(job["output_path"])
            video_name = job["video_name"]
        else:
            persisted = load_persisted_job(job_id)
            if persisted is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            if persisted.get("status") != "succeeded":
                raise HTTPException(status_code=409, detail="Output video is not ready yet.")
            output_path = output_path_for_public_job(persisted)
            video_name = str(persisted.get("video_name") or "video.mp4")
            if output_path is None:
                raise HTTPException(status_code=404, detail="Output video not found.")

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output video not found.")
    make_browser_friendly_mp4(output_path)
    return FileResponse(output_path, media_type="video/mp4", filename=f"annotated_{video_name}")
