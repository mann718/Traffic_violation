# Traffic Violation Detection

## Repository Link

Project repository: `https://github.com/deepmbhatt/StreetSentry`

## Objective And Application Description

This application is a traffic surveillance system for detecting three kinds of violations from road videos:

1. `Red-light violation`
2. `Helmet violation`
3. `Speed + lane violation`

The objective is to reduce manual video review by automatically processing traffic footage, identifying violations, generating annotated output videos, and saving structured results that can be reviewed later.

The system has a web interface built with `React + Vite` and a backend built with `FastAPI`. It also includes a Python CLI flow for local processing.

### Red-light violation mode

- Detects and tracks vehicles using YOLO and ByteTrack.
- Lets the user draw a stop line and one or more traffic-light ROIs.
- Detects a violation when a tracked vehicle crosses the stop line while the traffic light is red.
- Saves an annotated output video and a result record in `app_data/runs/`.

### Helmet violation mode

- Detects motorcycles and helmet / no-helmet cases from video.
- Associates helmet detections with tracked motorcycle riders.
- Marks a motorcycle as a violation when the rider is detected without a helmet.
- Saves an annotated output video and a result record in `app_data/runs/`.

### Speed + lane mode

- Uses YOLO + ByteTrack tracking on road videos.
- Computes speed between two horizontal reference lines (`70%` and `85%` of frame height by default).
- Flags overspeed using `speed_limit_kmh` (default `50 km/h`).
- Uses curved lane polygons and flags lane violations when a tracked vehicle stays outside its assigned lane for multiple frames.
- Saves an annotated output video and a result record in `app_data/runs/`.

## Main Features

- Three processing modes in one application: `Red Light`, `Helmet`, and `Speed + Lane`
- Web-based video selection and upload
- Mode-aware video organization:
  - `inputs/` for red-light videos
  - `inputs_helmet/` for helmet videos
- Annotated output video generation
- Saved run history in `app_data/runs/`
- Latest saved result view for each mode
- Violation log with clickable timestamps that seek the output video

## Installation And Setup

### Prerequisites

- Python `3.10+` recommended
- `pip`
- Node.js and `npm`
- Git
- Optional GPU support with a CUDA-compatible PyTorch installation

### Clone the repository

```bash
git clone https://github.com/deepmbhatt/StreetSentry.git
cd Red-Light-Violation-Detection
```

### Create and activate a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Model files required

Make sure these model files are present:

- Vehicle model: `yolo12l.pt`
- Speed/lane model: `detection_track.pt` (falls back to `yolo12l.pt` if missing)
- Helmet model: `best.pt` (must be a valid trained checkpoint, not a Git LFS pointer file)

The backend resolves model files from both the project root and `backend/` when available.

## How To Run The Application

### Start the backend

From the project root:

```bash
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Start the frontend

In a second terminal:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### Open the web application

Open:

```text
http://127.0.0.1:5173/
```

## How To Use The Application

### Red-light mode

1. Select `Red Light` in the top module selector.
2. Choose a video from the sidebar. Red-light videos come from `inputs/`.
3. Draw:
   - one stop line
   - one or more traffic-light ROIs
4. Save the setup if needed.
5. Click `Run`.
6. After processing finishes, review:
   - the annotated output video
   - the violation count
   - the violation log with clickable timestamps

### Helmet mode

1. Select `Helmet` in the top module selector.
2. Choose a video from the sidebar. Helmet videos come from `inputs_helmet/`.
3. No line or ROI setup is required.
4. Click `Run`.
5. After processing finishes, review:
   - the annotated output video
   - the violation count
   - the violation log with clickable timestamps

### Speed + lane mode

1. Select `Speed + Lane` in the top module selector.
2. Choose a video from the sidebar (from `inputs/`).
3. No manual line/ROI setup is required; lane polygons and speed lines are auto-configured.
4. Click `Run`.
5. After processing finishes, review:
   - the annotated output video
   - speed/lane violation count
   - the violation log with clickable timestamps

### Upload behavior

- If the app is in `Red Light` mode, uploaded videos are saved into `inputs/`.
- If the app is in `Helmet` mode, uploaded videos are saved into `inputs_helmet/`.
- If the app is in `Speed + Lane` mode, uploaded videos are saved into `inputs/`.
- Upload progress is shown in the sidebar.
- After upload completes, the uploaded video appears in the sidebar automatically.

### Reviewing saved results

- The app stores each run in `app_data/runs/<job_id>/`.
- Each run includes a `job.json` file and an annotated output video.
- For each mode, the UI can show the latest saved successful result.
- Clicking a violation entry in the log jumps the output video to the recorded violation time.

## CLI Usage

The project also supports command-line execution.

### Red-light mode

```bash
.venv/bin/python main.py path/to/video.mp4 --module red_light
```

### Helmet mode

```bash
.venv/bin/python main.py path/to/video.mp4 --module helmet
```

### Speed + lane mode

```bash
.venv/bin/python main.py path/to/video.mp4 --module speed_lane
```

### Optional CLI arguments

- `--weights` for the vehicle model path
- `--helmet-weights` for the helmet model path
- `--speed-weights` for speed/lane model path (if exposed in your CLI version)
- `--conf`, `--vehicle-conf`, `--helmet-conf`, `--iou`
- `--browser-setup` for browser-based setup
- `--config` and `--save-config` for reusable red-light geometry

## Project Structure

```text
Red-Light-Violation-Detection/
├── backend/                     # FastAPI backend
├── frontend/                    # React + Vite frontend
├── inputs/                      # Red-light and speed/lane videos
├── inputs_helmet/               # Helmet videos
├── app_data/runs/               # Saved run outputs and job.json files
├── detector.py                  # Core detection logic
├── main.py                      # CLI entrypoint
├── browser_setup.py             # Browser-based red-light setup
├── bytetrack.yaml               # ByteTrack configuration
├── yolo12l.pt                   # Vehicle model
├── detection_track.pt           # Speed/lane model (optional if using vehicle fallback)
├── best.pt                      # Helmet model
└── README.md
```

## Additional Relevant Information

### Current output and persistence behavior

- Each completed run writes a result record to `app_data/runs/<job_id>/job.json`.
- Output videos are converted to a browser-friendly MP4 format when possible.
- Old runs created before newer logging features may not contain detailed violation-event entries.

### GPU note

GPU processing depends on the local PyTorch and CUDA setup. If the installed Torch build is not compatible with the NVIDIA driver, the system will fall back to CPU.

### Important submission note

This repository now includes all required README sections explicitly:

- Objective and description of the application
- Instructions to install and run the code
- Details on how to use the application
- Additional relevant information
#   T r a f f i c _ v i o l a t i o n  
 