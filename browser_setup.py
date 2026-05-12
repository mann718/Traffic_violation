import argparse
import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile

import cv2


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Red-Light Setup</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #101418;
      color: #f3f5f7;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 24px;
      border-right: 1px solid #23303b;
      background: linear-gradient(180deg, #131c24, #0d1318);
    }}
    .sidebar h1 {{
      margin-top: 0;
      font-size: 1.5rem;
    }}
    .sidebar p {{
      color: #b9c3cb;
      line-height: 1.5;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin: 20px 0;
    }}
    button {{
      border: 0;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 0.95rem;
      cursor: pointer;
      background: #1c8f63;
      color: white;
    }}
    button.secondary {{
      background: #2a3641;
    }}
    button.warn {{
      background: #b5542b;
    }}
    button:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    .status {{
      margin-top: 16px;
      padding: 12px;
      border-radius: 12px;
      background: #162029;
      color: #dce4ea;
      min-height: 48px;
    }}
    .stage {{
      padding: 24px;
      overflow: auto;
    }}
    canvas {{
      max-width: 100%;
      height: auto;
      border-radius: 18px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      background: #000;
      cursor: crosshair;
    }}
    code {{
      background: #1a232c;
      border-radius: 8px;
      padding: 2px 6px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Scene Setup</h1>
      <p>Click once to start a line or ROI, then click again to finish it. Draw one red violation line and one or more traffic-light ROIs.</p>
      <div class="controls">
        <button id="lineBtn">Draw Line</button>
        <button id="roiBtn">Add ROI</button>
        <button class="secondary" id="undoBtn">Undo Last</button>
        <button class="secondary" id="clearRoiBtn">Clear ROIs</button>
        <button class="secondary" id="resetBtn">Reset All</button>
        <button class="warn" id="saveBtn" disabled>Save And Continue</button>
      </div>
      <div class="status" id="status">Choose <code>Draw Line</code> or <code>Add ROI</code> to begin.</div>
      <p>Config will be saved to:<br><code>{config_path}</code></p>
      <p>Frame size: <code>{width}x{height}</code></p>
    </aside>
    <main class="stage">
      <canvas id="canvas" width="{width}" height="{height}"></canvas>
    </main>
  </div>
  <script>
    const image = new Image();
    image.src = "data:image/jpeg;base64,{image_b64}";

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const saveBtn = document.getElementById("saveBtn");

    let mode = null;
    let line = [];
    let rois = [];
    let draft = [];

    const setStatus = (text) => {{
      statusEl.textContent = text;
    }};

    const syncSaveState = () => {{
      saveBtn.disabled = !(line.length === 2 && rois.length >= 1);
    }};

    const normalizeRect = (a, b) => {{
      return [
        Math.min(a.x, b.x),
        Math.min(a.y, b.y),
        Math.max(a.x, b.x),
        Math.max(a.y, b.y)
      ];
    }};

    const draw = () => {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

      if (line.length >= 1) {{
        ctx.strokeStyle = "#ff4d4d";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(line[0].x, line[0].y);
        const endPoint = line[1] || draft[0];
        if (endPoint) {{
          ctx.lineTo(endPoint.x, endPoint.y);
        }}
        ctx.stroke();
      }}

      rois.forEach((roi, idx) => {{
        ctx.strokeStyle = "#53d18b";
        ctx.lineWidth = 3;
        ctx.strokeRect(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]);
        ctx.fillStyle = "#53d18b";
        ctx.font = "16px sans-serif";
        ctx.fillText(`ROI ${{idx + 1}}`, roi[0] + 6, roi[1] + 22);
      }});

      if (mode === "roi" && draft.length === 1) {{
        const [x1, y1, x2, y2] = normalizeRect(draft[0], draft[0]);
        ctx.strokeStyle = "#9ce7bb";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }}
    }};

    image.onload = draw;

    document.getElementById("lineBtn").onclick = () => {{
      mode = "line";
      draft = [];
      line = [];
      setStatus("Click two points to place the violation line.");
      draw();
      syncSaveState();
    }};

    document.getElementById("roiBtn").onclick = () => {{
      mode = "roi";
      draft = [];
      setStatus("Click two corners for a traffic-light ROI.");
      draw();
    }};

    document.getElementById("undoBtn").onclick = () => {{
      if (draft.length) {{
        draft = [];
      }} else if (rois.length) {{
        rois.pop();
      }} else if (line.length) {{
        line = [];
      }}
      draw();
      syncSaveState();
    }};

    document.getElementById("clearRoiBtn").onclick = () => {{
      rois = [];
      draft = [];
      draw();
      syncSaveState();
    }};

    document.getElementById("resetBtn").onclick = () => {{
      line = [];
      rois = [];
      draft = [];
      mode = null;
      setStatus("Reset complete. Choose Draw Line or Add ROI.");
      draw();
      syncSaveState();
    }};

    canvas.addEventListener("click", (event) => {{
      const rect = canvas.getBoundingClientRect();
      const x = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
      const y = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));

      if (mode === "line") {{
        line.push({{ x, y }});
        if (line.length === 2) {{
          mode = null;
          setStatus("Line saved. Add one or more ROIs, then save.");
        }}
      }} else if (mode === "roi") {{
        draft.push({{ x, y }});
        if (draft.length === 2) {{
          rois.push(normalizeRect(draft[0], draft[1]));
          draft = [];
          mode = null;
          setStatus("ROI saved. Add more ROIs or save the config.");
        }}
      }} else {{
        setStatus("Choose Draw Line or Add ROI first.");
      }}

      draw();
      syncSaveState();
    }});

    saveBtn.onclick = async () => {{
      const payload = {{
        line: [line[0].x, line[0].y, line[1].x, line[1].y],
        rois: rois
      }};
      const response = await fetch("/save", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      const result = await response.json();
      if (response.ok) {{
        setStatus("Saved. You can return to the terminal; inference will continue automatically.");
      }} else {{
        setStatus(result.error || "Failed to save config.");
      }}
    }};
  </script>
</body>
</html>
"""


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    return frame


def _encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode setup frame")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def run_browser_setup(video_path, output_config, host="127.0.0.1", port=8765, open_browser=True):
    frame = extract_first_frame(video_path)
    height, width = frame.shape[:2]
    image_b64 = _encode_frame(frame)
    output_config = str(Path(output_config).resolve())
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = HTML_TEMPLATE.format(
                image_b64=image_b64,
                width=width,
                height=height,
                config_path=output_config,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                line = payload.get("line")
                rois = payload.get("rois", [])
                if not isinstance(line, list) or len(line) != 4:
                    raise ValueError("Line must contain exactly 4 numbers.")
                if not isinstance(rois, list) or len(rois) < 1:
                    raise ValueError("At least one ROI is required.")
                for roi in rois:
                    if not isinstance(roi, list) or len(roi) != 4:
                        raise ValueError("Each ROI must contain exactly 4 numbers.")
                output_path = Path(output_config)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "line": [int(v) for v in line],
                    "rois": [[int(v) for v in roi] for roi in rois],
                }
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
                    json.dump(payload, handle, indent=2)
                    handle.flush()
                    temp_name = handle.name
                Path(temp_name).replace(output_path)
                self._json({"ok": True, "output_config": output_config})
                done.set()
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=400)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Browser setup ready at {url}")
    print(f"Config will be saved to {output_config}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        while not done.is_set():
            server.handle_request()
    finally:
        server.server_close()
    return output_config


def main():
    parser = argparse.ArgumentParser(description="Browser-based setup for line and ROI selection.")
    parser.add_argument("video_in", help="Input video path")
    parser.add_argument("--output-config", default="setup.json", help="Where to save the setup JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser-open", action="store_true")
    args = parser.parse_args()
    run_browser_setup(
        video_path=args.video_in,
        output_config=args.output_config,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser_open,
    )


if __name__ == "__main__":
    main()
