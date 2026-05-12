import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2

from browser_setup import run_browser_setup
from detector import (
    COLOR_LINE,
    COLOR_OK,
    COLOR_ROI,
    DEFAULT_HELMET_MODEL_WEIGHTS,
    DEFAULT_TRACKER_CFG,
    DEFAULT_VEHICLE_MODEL_WEIGHTS,
    DEFAULT_VIDEO_OUT,
    FONT,
    HelmetProcessingOptions,
    ProcessingOptions,
    extract_first_frame,
    load_config_file,
    process_helmet_video,
    process_red_light_video,
    save_config_file,
)


line_pts = []
roi_pts_list = []
drawing_mode = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Red-light violation detection with YOLO + ByteTrack + HSV traffic-light ROIs."
    )
    parser.add_argument("video_in", help="Input video path")
    parser.add_argument(
        "--module",
        choices=("red_light", "helmet"),
        default="red_light",
        help="Processing module to run. Default: red_light",
    )
    parser.add_argument(
        "--video-out",
        default=DEFAULT_VIDEO_OUT,
        help=f"Output annotated video path. Default: {DEFAULT_VIDEO_OUT}",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_VEHICLE_MODEL_WEIGHTS,
        help=f"YOLO weights path. Default: {DEFAULT_VEHICLE_MODEL_WEIGHTS}",
    )
    parser.add_argument(
        "--helmet-weights",
        default=DEFAULT_HELMET_MODEL_WEIGHTS,
        help=f"Helmet YOLO weights path. Default: {DEFAULT_HELMET_MODEL_WEIGHTS}",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER_CFG,
        help=f"ByteTrack config path. Default: {DEFAULT_TRACKER_CFG}",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Vehicle confidence threshold")
    parser.add_argument("--vehicle-conf", type=float, default=0.35, help="Vehicle confidence threshold for helmet module")
    parser.add_argument("--helmet-conf", type=float, default=0.35, help="Helmet confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Vehicle IOU threshold")
    parser.add_argument(
        "--traffic-light-refresh",
        type=int,
        default=5,
        help="Refresh interval in frames for HSV traffic-light classification",
    )
    parser.add_argument(
        "--line",
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        type=int,
        help="Provide the violation line directly and skip line drawing",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        type=int,
        action="append",
        help="Provide a traffic-light ROI directly. Repeat to add multiple ROIs.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file containing {'line':[x1,y1,x2,y2],'rois':[[x1,y1,x2,y2],...]}",
    )
    parser.add_argument(
        "--save-config",
        help="Optional JSON path where the final line/ROI setup will be saved",
    )
    parser.add_argument(
        "--browser-setup",
        action="store_true",
        help="Launch browser-based line/ROI setup before inference.",
    )
    parser.add_argument(
        "--setup-config",
        help="Where to save or read browser-generated setup JSON. Defaults to <video>.setup.json",
    )
    parser.add_argument(
        "--setup-host",
        default="127.0.0.1",
        help="Host for the browser setup server.",
    )
    parser.add_argument(
        "--setup-port",
        type=int,
        default=8765,
        help="Port for the browser setup server.",
    )
    parser.add_argument(
        "--no-browser-open",
        action="store_true",
        help="Do not auto-open the browser setup URL.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the output video after processing",
    )
    return parser.parse_args()


def reset_state():
    global line_pts, roi_pts_list, drawing_mode
    line_pts = []
    roi_pts_list = []
    drawing_mode = None


def put_text(img, txt, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_setup_hud(disp):
    put_text(disp, "Press L then click 2 points to draw LINE", (18, 28), (200, 200, 200), 0.7, 2)
    put_text(disp, "Press R then click 2 points to add TL ROI (multi allowed)", (18, 56), (200, 200, 200), 0.7, 2)
    put_text(disp, "U: undo last ROI/LINE   C: clear ROIs   SPACE: start   Q/Esc: quit", (18, 84), (200, 200, 200), 0.7, 2)
    if len(line_pts) == 1:
        cv2.circle(disp, line_pts[0], 4, COLOR_LINE, -1)
    if len(line_pts) == 2:
        cv2.line(disp, line_pts[0], line_pts[1], COLOR_LINE, 2)
        put_text(disp, "Line OK", (line_pts[0][0] + 6, line_pts[0][1] + 18), COLOR_OK, 0.7, 2)
    for idx, roi in enumerate(roi_pts_list):
        if len(roi) == 1:
            cv2.circle(disp, roi[0], 4, COLOR_ROI, -1)
        elif len(roi) == 2:
            (x1, y1), (x2, y2) = roi
            cv2.rectangle(disp, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), COLOR_ROI, 2)
            put_text(disp, f"ROI #{idx + 1}", (min(x1, x2) + 6, min(y1, y2) + 20), COLOR_OK, 0.6, 2)


def setup_mouse_cb(event, x, y, flags, param):
    global drawing_mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    h, w = param["img_shape"]
    if not (0 <= x < w and 0 <= y < h):
        return
    if drawing_mode == "line":
        if len(line_pts) < 2:
            line_pts.append((x, y))
        if len(line_pts) == 2:
            drawing_mode = None
    elif drawing_mode == "roi":
        if not roi_pts_list or len(roi_pts_list[-1]) == 2:
            roi_pts_list.append([])
        roi_pts_list[-1].append((x, y))
        if len(roi_pts_list[-1]) == 2:
            drawing_mode = None


def auto_open_video(path):
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def load_config(config_path):
    payload = load_config_file(config_path)
    line_pts.extend([(payload["line"][0], payload["line"][1]), (payload["line"][2], payload["line"][3])])
    for roi in payload["rois"]:
        roi_pts_list.append([(roi[0], roi[1]), (roi[2], roi[3])])


def apply_cli_geometry(args):
    if args.line:
        x1, y1, x2, y2 = args.line
        line_pts.extend([(x1, y1), (x2, y2)])
    for roi in args.roi or []:
        x1, y1, x2, y2 = roi
        roi_pts_list.append([(x1, y1), (x2, y2)])


def current_geometry():
    return {
        "line": [line_pts[0][0], line_pts[0][1], line_pts[1][0], line_pts[1][1]],
        "rois": [[roi[0][0], roi[0][1], roi[1][0], roi[1][1]] for roi in roi_pts_list if len(roi) == 2],
    }


def save_config(save_path):
    save_config_file(save_path, current_geometry())


def ensure_ready(args):
    if not os.path.exists(args.video_in):
        raise FileNotFoundError(f"Input video not found: {args.video_in}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}\n"
            "Download or copy a YOLO weights file into the repo and pass it with --weights."
        )
    if args.module == "helmet" and not os.path.exists(args.helmet_weights):
        raise FileNotFoundError(
            f"Helmet model weights not found: {args.helmet_weights}\n"
            "Copy the helmet YOLO weights into the repo and pass them with --helmet-weights."
        )
    if not os.path.exists(args.tracker):
        raise FileNotFoundError(f"Tracker config not found: {args.tracker}")


def has_display():
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def default_setup_config_path(video_in):
    video_path = Path(video_in).resolve()
    return str(video_path.with_suffix(video_path.suffix + ".setup.json"))


def interactive_setup(first_frame):
    global drawing_mode
    h, w = first_frame.shape[:2]
    cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("setup", setup_mouse_cb, param={"img_shape": (h, w)})
    while True:
        disp = first_frame.copy()
        draw_setup_hud(disp)
        cv2.imshow("setup", disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("l"):
            drawing_mode = "line"
            if len(line_pts) == 2:
                line_pts.clear()
        elif key == ord("r"):
            drawing_mode = "roi"
        elif key == ord("u"):
            if roi_pts_list:
                if len(roi_pts_list[-1]) == 1:
                    roi_pts_list[-1].clear()
                else:
                    roi_pts_list.pop()
            elif line_pts:
                line_pts.clear()
        elif key == ord("c"):
            roi_pts_list.clear()
        elif key == ord(" ") and len(line_pts) == 2 and any(len(roi) == 2 for roi in roi_pts_list):
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return False
    cv2.destroyWindow("setup")
    return True


def main():
    args = parse_args()
    reset_state()
    ensure_ready(args)

    if args.module == "helmet":
        result = process_helmet_video(
            video_in=args.video_in,
            video_out=args.video_out,
            vehicle_weights=args.weights,
            helmet_weights=args.helmet_weights,
            tracker=args.tracker,
            options=HelmetProcessingOptions(
                vehicle_conf=args.vehicle_conf,
                helmet_conf=args.helmet_conf,
                iou=args.iou,
            ),
            verbose=True,
        )
        print(
            f"Processed {result.processed_frames} frames. "
            f"Helmet violations: {result.total_violations}. "
            f"Output: {result.video_out}"
        )
        if not args.no_open:
            auto_open_video(args.video_out)
        return

    if args.config:
        load_config(args.config)
    apply_cli_geometry(args)

    first = extract_first_frame(args.video_in)

    if not (len(line_pts) == 2 and any(len(roi) == 2 for roi in roi_pts_list)):
        if args.browser_setup or not has_display():
            setup_config = args.setup_config or default_setup_config_path(args.video_in)
            run_browser_setup(
                video_path=args.video_in,
                output_config=setup_config,
                host=args.setup_host,
                port=args.setup_port,
                open_browser=not args.no_browser_open,
            )
            load_config(setup_config)
        else:
            if not interactive_setup(first):
                return

    if args.save_config:
        save_config(args.save_config)

    geometry = current_geometry()
    result = process_red_light_video(
        video_in=args.video_in,
        video_out=args.video_out,
        line=geometry["line"],
        rois=geometry["rois"],
        weights=args.weights,
        tracker=args.tracker,
        options=ProcessingOptions(
            conf=args.conf,
            iou=args.iou,
            traffic_light_refresh=args.traffic_light_refresh,
        ),
        verbose=True,
    )
    print(
        f"Processed {result.processed_frames} frames. "
        f"Violations: {result.total_violations}. "
        f"Output: {result.video_out}"
    )
    if not args.no_open:
        auto_open_video(args.video_out)


if __name__ == "__main__":
    main()

