import {
  CheckCircle2,
  Film,
  Minus,
  Play,
  RefreshCw,
  Save,
  Square,
  Trash2,
  Undo2,
  Upload
} from "lucide-react";
import { ChangeEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type VideoItem = {
  id: string;
  name: string;
  kind: "sample" | "helmet" | "upload";
  size_bytes: number;
  has_config: boolean;
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  duration_seconds: number;
};

type Point = { x: number; y: number };
type Rect = [number, number, number, number];
type ProcessingModule = "red_light" | "helmet" | "speed_lane";

type ApiConfig = {
  line: number[] | null;
  rois: number[][];
};

type JobProgress = {
  processed_frames: number;
  total_frames: number;
  percent: number;
  violations_count: number;
};

type ViolationEvent = {
  id: string;
  module: ProcessingModule;
  label: string;
  frame: number;
  time_seconds: number;
  track_id?: number;
  class_name?: string;
  box?: number[];
  details?: Record<string, unknown>;
};

type JobResult = {
  module: ProcessingModule;
  total_violations: number;
  processed_frames: number;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  elapsed_seconds: number;
  violations?: ViolationEvent[];
  video_url: string;
};

type Job = {
  id: string;
  module: ProcessingModule;
  video_id: string;
  video_name: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: JobProgress;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  result: JobResult | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.error || message;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function uploadVideoWithProgress(
  data: FormData,
  module: ProcessingModule,
  onProgress: (percent: number) => void
): Promise<VideoItem> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(`/api/videos/upload?module=${module}`));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      try {
        const payload = JSON.parse(xhr.responseText || "{}");
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress(100);
          resolve(payload as VideoItem);
        } else {
          reject(new Error(payload.detail || payload.error || `Upload failed with ${xhr.status}`));
        }
      } catch {
        reject(new Error(`Upload failed with ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed."));
    xhr.send(data);
  });
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0:00";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTimestamp(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0:00";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  const tenths = Math.floor((seconds % 1) * 10);
  return tenths > 0 ? `${mins}:${secs}.${tenths}` : `${mins}:${secs}`;
}

function lineToPoints(line: number[] | null): Point[] {
  if (!line || line.length !== 4) {
    return [];
  }
  return [
    { x: line[0], y: line[1] },
    { x: line[2], y: line[3] }
  ];
}

function normalizeRect(a: Point, b: Point): Rect {
  return [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)];
}

function videoMatchesModule(video: VideoItem, module: ProcessingModule) {
  return module === "helmet" ? video.kind === "helmet" : video.kind === "sample";
}

function videoReadyForModule(video: VideoItem, module: ProcessingModule) {
  return module === "helmet" || module === "speed_lane" || video.has_config;
}

function runDisabledReason(selectedModule: ProcessingModule, selectedVideo: VideoItem | null, redLightReady: boolean) {
  if (!selectedVideo) {
    return "Select a video first.";
  }
  if (selectedModule === "red_light" && !redLightReady) {
    return "For Red Light, draw one Line and at least one ROI, then click Run.";
  }
  return "";
}

function App() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [line, setLine] = useState<Point[]>([]);
  const [rois, setRois] = useState<Rect[]>([]);
  const [draft, setDraft] = useState<Point[]>([]);
  const [mode, setMode] = useState<"line" | "roi" | null>(null);
  const [selectedModule, setSelectedModule] = useState<ProcessingModule>("red_light");
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [latestJob, setLatestJob] = useState<Job | null>(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const resultVideoRef = useRef<HTMLVideoElement | null>(null);

  const visibleVideos = useMemo(
    () => videos.filter((video) => videoMatchesModule(video, selectedModule)),
    [selectedModule, videos]
  );

  const selectedVideo = useMemo(
    () => visibleVideos.find((video) => video.id === selectedVideoId) ?? null,
    [selectedVideoId, visibleVideos]
  );

  const frameUrl = selectedVideo ? apiUrl(`/api/videos/${encodeURIComponent(selectedVideo.id)}/frame`) : "";
  const redLightReady = line.length === 2 && rois.length > 0;
  const canRun = Boolean(
    selectedVideo && !running && (selectedModule === "helmet" || selectedModule === "speed_lane" || redLightReady)
  );
  const runHint = runDisabledReason(selectedModule, selectedVideo, redLightReady);
  const violationLabel = selectedModule === "helmet" ? "Helmet Violations" : "Violations";
  const displayJob = activeJob?.result ? activeJob : latestJob;
  const displayResult = displayJob?.result ?? null;
  const resultVideoUrl = displayResult
    ? apiUrl(`${displayResult.video_url}?t=${encodeURIComponent(displayJob?.finished_at ?? displayJob?.id ?? "")}`)
    : "";

  const loadVideos = useCallback(async (preferredId?: string) => {
    const payload = await requestJson<{ videos: VideoItem[] }>("/api/videos");
    setVideos(payload.videos);
    const moduleVideos = payload.videos.filter((video) => videoMatchesModule(video, selectedModule));
    const nextId = preferredId || selectedVideoId || moduleVideos[0]?.id || "";
    setSelectedVideoId(moduleVideos.some((video) => video.id === nextId) ? nextId : moduleVideos[0]?.id || "");
  }, [selectedModule, selectedVideoId]);

  const loadConfig = useCallback(async (videoId: string) => {
    const config = await requestJson<ApiConfig>(`/api/videos/${encodeURIComponent(videoId)}/config`);
    setLine(lineToPoints(config.line));
    setRois((config.rois || []).map((roi) => normalizeRect({ x: roi[0], y: roi[1] }, { x: roi[2], y: roi[3] })));
    setDraft([]);
    setMode(null);
  }, []);

  const loadLatestJob = useCallback(async (module: ProcessingModule) => {
    const job = await requestJson<Job | null>(`/api/jobs/latest?module=${module}`);
    setLatestJob(job);
  }, []);

  useEffect(() => {
    loadVideos().catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    loadLatestJob(selectedModule).catch((err: Error) => setError(err.message));
  }, [loadLatestJob, selectedModule]);

  useEffect(() => {
    if (!visibleVideos.length) {
      setSelectedVideoId("");
      return;
    }
    if (!selectedVideo || selectedVideo.kind !== visibleVideos.find((video) => video.id === selectedVideoId)?.kind) {
      setSelectedVideoId(visibleVideos[0].id);
    }
  }, [selectedModule, selectedVideo, selectedVideoId, visibleVideos]);

  useEffect(() => {
    if (!selectedVideoId || !selectedVideo) {
      return;
    }
    setActiveJob(null);
    if (selectedModule === "helmet" || selectedModule === "speed_lane") {
      setLine([]);
      setRois([]);
      setDraft([]);
      setMode(null);
      setStatus(selectedModule === "helmet" ? "Helmet video loaded" : "Speed/Lane video loaded");
      return;
    }
    setStatus("Setup loaded");
    loadConfig(selectedVideoId).catch((err: Error) => {
      setLine([]);
      setRois([]);
      setError(err.message);
    });
  }, [loadConfig, selectedModule, selectedVideo, selectedVideoId]);

  useEffect(() => {
    if (!frameUrl || !canvasRef.current) {
      imageRef.current = null;
      return;
    }

    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      drawCanvas();
    };
    image.src = frameUrl;
  }, [frameUrl]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const image = imageRef.current;
    if (!canvas || !ctx || !image) {
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 4;
    ctx.strokeStyle = "#d63f31";
    ctx.fillStyle = "#d63f31";
    if (line.length > 0) {
      ctx.beginPath();
      ctx.arc(line[0].x, line[0].y, 6, 0, Math.PI * 2);
      ctx.fill();
      if (line.length === 2) {
        ctx.beginPath();
        ctx.moveTo(line[0].x, line[0].y);
        ctx.lineTo(line[1].x, line[1].y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(line[1].x, line[1].y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#19945f";
    ctx.fillStyle = "#19945f";
    ctx.font = "20px system-ui, sans-serif";
    rois.forEach((roi, index) => {
      ctx.strokeRect(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]);
      ctx.fillText(`ROI ${index + 1}`, roi[0] + 8, Math.max(24, roi[1] - 8));
    });

    if (draft.length === 1) {
      ctx.beginPath();
      ctx.arc(draft[0].x, draft[0].y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [draft, line, rois]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  useEffect(() => {
    if (!activeJob || !["queued", "running"].includes(activeJob.status)) {
      return;
    }
    const poll = async () => {
      try {
        const next = await requestJson<Job>(`/api/jobs/${activeJob.id}`);
        setActiveJob(next);
        if (next.status === "failed") {
          setStatus("Processing failed");
          if (next.error) {
            setError(next.error);
          }
        } else {
          setStatus(next.status === "succeeded" ? "Processing complete" : "Processing");
        }
        if (next.status === "succeeded") {
          setLatestJob(next);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    };
    const timer = window.setInterval(poll, 1000);
    poll();
    return () => window.clearInterval(timer);
  }, [activeJob?.id, activeJob?.status]);

  function canvasPoint(event: MouseEvent<HTMLCanvasElement>): Point {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.round((event.clientX - rect.left) * (canvas.width / rect.width)),
      y: Math.round((event.clientY - rect.top) * (canvas.height / rect.height))
    };
  }

  function handleCanvasClick(event: MouseEvent<HTMLCanvasElement>) {
    const point = canvasPoint(event);
    setError(null);

    if (mode === "line") {
      setLine((current) => {
        const next = current.length >= 2 ? [point] : [...current, point];
        if (next.length === 2) {
          setMode(null);
          setStatus("Line set");
        }
        return next;
      });
      return;
    }

    if (mode === "roi") {
      if (draft.length === 0) {
        setDraft([point]);
      } else {
        setRois((current) => [...current, normalizeRect(draft[0], point)]);
        setDraft([]);
        setMode(null);
        setStatus("ROI added");
      }
    }
  }

  function geometryPayload() {
    if (line.length !== 2 || rois.length < 1) {
      throw new Error("Set one line and at least one ROI before saving or running.");
    }
    return {
      line: [line[0].x, line[0].y, line[1].x, line[1].y],
      rois
    };
  }

  async function saveSetup() {
    if (!selectedVideo) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const payload = geometryPayload();
      await requestJson<ApiConfig>(`/api/videos/${encodeURIComponent(selectedVideo.id)}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setStatus("Setup saved");
      await loadVideos(selectedVideo.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function runProcessing() {
    if (!selectedVideo) {
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const payload = selectedModule === "red_light" ? geometryPayload() : null;
      if (payload) {
        await requestJson<ApiConfig>(`/api/videos/${encodeURIComponent(selectedVideo.id)}/config`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      }
      const job = await requestJson<Job>("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          module: selectedModule,
          video_id: selectedVideo.id,
          ...(payload ?? {}),
          conf: selectedModule === "speed_lane" ? 0.25 : 0.35,
          vehicle_conf: 0.35,
          helmet_conf: 0.35,
          iou: 0.45,
          traffic_light_refresh: 5,
          line1_ratio: 0.7,
          line2_ratio: 0.85,
          meters_between_lines: 8.0,
          speed_persist_frames: 150,
          speed_limit_kmh: 50.0,
          violation_frame_threshold: 8
        })
      });
      setActiveJob(job);
      setLatestJob(null);
      setStatus("Processing");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    try {
      const data = new FormData();
      data.append("file", file);
      const uploaded = await uploadVideoWithProgress(data, selectedModule, setUploadProgress);
      setStatus("Video uploaded");
      await loadVideos(uploaded.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }

  function undo() {
    setError(null);
    if (draft.length) {
      setDraft([]);
      return;
    }
    if (rois.length) {
      setRois((current) => current.slice(0, -1));
      return;
    }
    setLine([]);
  }

  function resetSetup() {
    setLine([]);
    setRois([]);
    setDraft([]);
    setMode(null);
    setActiveJob(null);
    setLatestJob(null);
    setStatus("Setup cleared");
  }

  function seekToViolation(event: ViolationEvent) {
    const video = resultVideoRef.current;
    if (!video) {
      return;
    }
    video.currentTime = Math.max(0, event.time_seconds);
    video.play().catch(() => {
      // Some browsers block autoplay until the user presses play.
    });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Traffic Violation Detection</h1>
          <span>{status}</span>
        </div>
        <button className="ghost" onClick={() => loadVideos().catch((err: Error) => setError(err.message))}>
          <RefreshCw size={18} />
          Refresh
        </button>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <div className="sidebar-head">
            <h2>Videos</h2>
            <label className={`upload-button ${uploading ? "disabled" : ""}`}>
              <Upload size={18} />
              {uploading ? "Uploading" : "Upload"}
              <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} />
            </label>
          </div>

          {uploading && (
            <div className="upload-progress">
              <div className="progress-head">
                <span>Uploading to {selectedModule === "helmet" ? "inputs_helmet" : "inputs"}</span>
                <strong>{uploadProgress}%</strong>
              </div>
              <div className="progress-track">
                <div style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}

          <div className="video-list">
            {visibleVideos.map((video) => (
              <button
                key={video.id}
                className={`video-row ${video.id === selectedVideoId ? "selected" : ""}`}
                onClick={() => setSelectedVideoId(video.id)}
              >
                <Film size={18} />
                <span>
                  <strong>{video.name}</strong>
                  <small>
                    {video.kind} · {formatDuration(video.duration_seconds)} · {formatBytes(video.size_bytes)}
                  </small>
                </span>
                {videoReadyForModule(video, selectedModule) ? <CheckCircle2 size={17} className="ok" /> : <span />}
              </button>
            ))}
            {!visibleVideos.length && (
              <div className="empty-video-list">
                {selectedModule === "helmet"
                  ? "No helmet videos found"
                  : selectedModule === "speed_lane"
                    ? "No speed/lane videos found"
                    : "No red-light videos found"}
              </div>
            )}
          </div>
        </aside>

        <main className="workspace">
          <section className="stage-panel">
            <div className="toolstrip">
              <div className="segmented module-selector">
                <button
                  className={selectedModule === "red_light" ? "active" : ""}
                  onClick={() => { setSelectedModule("red_light"); setActiveJob(null); setLatestJob(null); setStatus("Red-light module selected"); }}
                  title="Run red-light violation detection"
                >
                  Red Light
                </button>
                <button
                  className={selectedModule === "helmet" ? "active" : ""}
                  onClick={() => { setSelectedModule("helmet"); setMode(null); setDraft([]); setActiveJob(null); setLatestJob(null); setStatus("Helmet module selected"); }}
                  title="Run helmet violation detection"
                >
                  Helmet
                </button>
                <button
                  className={selectedModule === "speed_lane" ? "active" : ""}
                  onClick={() => {
                    setSelectedModule("speed_lane");
                    setMode(null);
                    setDraft([]);
                    setActiveJob(null);
                    setLatestJob(null);
                    setStatus("Speed/Lane module selected");
                  }}
                  title="Run speed estimation and lane violation detection"
                >
                  Speed + Lane
                </button>
              </div>
              {selectedModule === "red_light" && (
                <>
                  <div className="segmented">
                    <button className={mode === "line" ? "active" : ""} onClick={() => { setMode("line"); setDraft([]); }} title="Draw line">
                      <Minus size={18} />
                      Line
                    </button>
                    <button className={mode === "roi" ? "active" : ""} onClick={() => { setMode("roi"); setDraft([]); }} title="Add traffic-light ROI">
                      <Square size={18} />
                      ROI
                    </button>
                  </div>
                  <button className="ghost" onClick={undo} title="Undo last mark">
                    <Undo2 size={18} />
                    Undo
                  </button>
                  <button className="ghost danger" onClick={resetSetup} title="Clear setup">
                    <Trash2 size={18} />
                    Clear
                  </button>
                  <button onClick={saveSetup} disabled={!selectedVideo || saving || line.length !== 2 || rois.length < 1}>
                    <Save size={18} />
                    {saving ? "Saving" : "Save"}
                  </button>
                </>
              )}
              <button className="primary" onClick={runProcessing} disabled={!canRun}>
                <Play size={18} />
                Run
              </button>
            </div>
            {!canRun && runHint && <div className="error-banner">{runHint}</div>}

            <div className="canvas-shell">
              {selectedVideo ? (
                <canvas ref={canvasRef} onClick={handleCanvasClick} />
              ) : (
                <div className="empty-state">No videos found</div>
              )}
            </div>
          </section>

          <section className="status-panel">
            <div className="metric-grid">
              <div>
                <span>{selectedModule === "helmet" ? "Module" : selectedModule === "speed_lane" ? "Speed Lines" : "Line"}</span>
                <strong>
                  {selectedModule === "helmet"
                    ? "Helmet"
                    : selectedModule === "speed_lane"
                      ? "Auto"
                      : line.length === 2
                        ? "Ready"
                        : "Missing"}
                </strong>
              </div>
              <div>
                <span>{selectedModule === "helmet" ? "Setup" : selectedModule === "speed_lane" ? "Lanes" : "ROIs"}</span>
                <strong>{selectedModule === "helmet" || selectedModule === "speed_lane" ? "Auto" : rois.length}</strong>
              </div>
              <div>
                <span>Frames</span>
                <strong>{selectedVideo?.total_frames ?? 0}</strong>
              </div>
              <div>
                <span>{violationLabel}</span>
                <strong>{activeJob?.progress.violations_count ?? activeJob?.result?.total_violations ?? latestJob?.result?.total_violations ?? 0}</strong>
              </div>
            </div>

            {activeJob && (
              <div className="job-panel">
                <div className="progress-head">
                  <span>{activeJob.status}</span>
                  <strong>{Math.round(activeJob.progress.percent)}%</strong>
                </div>
                <div className="progress-track">
                  <div style={{ width: `${Math.min(100, activeJob.progress.percent)}%` }} />
                </div>
                <small>
                  {activeJob.progress.processed_frames} / {activeJob.progress.total_frames} frames
                </small>
              </div>
            )}

            {displayJob?.status === "succeeded" && displayResult && (
              <div className="result-panel">
                <video ref={resultVideoRef} key={resultVideoUrl} controls preload="metadata" src={resultVideoUrl} />
                <div className="result-summary">
                  <span>{displayJob === latestJob ? "Last saved result" : "Current result"} · {displayJob.video_name}</span>
                  <strong>
                    {displayResult.total_violations}{" "}
                    {displayResult.module === "helmet"
                      ? "helmet violations"
                      : displayResult.module === "speed_lane"
                        ? "speed/lane violations"
                        : "violations"}
                  </strong>
                  <small>{displayResult.elapsed_seconds.toFixed(1)}s processing time</small>
                  <a href={resultVideoUrl} target="_blank" rel="noreferrer">Open annotated video</a>
                  <div className="violation-list">
                    <span>Violation Log</span>
                    {(displayResult.violations ?? []).length > 0 ? (
                      (displayResult.violations ?? []).map((event, index) => (
                        <button key={event.id} type="button" onClick={() => seekToViolation(event)}>
                          <strong>{index + 1}. {formatTimestamp(event.time_seconds)}</strong>
                          <small>
                            {event.label}
                            {event.track_id !== undefined ? ` · ID${event.track_id}` : ""}
                            {event.class_name ? ` · ${event.class_name}` : ""}
                          </small>
                        </button>
                      ))
                    ) : (
                      <small>No detailed violation events saved for this run.</small>
                    )}
                  </div>
                </div>
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
