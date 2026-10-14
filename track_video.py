"""
track_video.py -- follow one player's feet through a fixed-camera video.

The camera must stay still, and one person is in frame.  You click the four
court corners on the first frame, then press 3.  MediaPipe Pose Landmarker (lite) finds the
heels in every frame.  Each contact pixel is mapped with the same homography
main.py uses, and the result is a court diagram with a red dot, plus a CSV of
positions in feet.

The original frames are not warped.  The diagram is the drawn court from
perspective.draw_court_diagram.

Examples:
    python track_video.py drill.mp4
    python track_video.py drill.mp4 --load output/video_calibration.json
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request

import cv2
import numpy as np

import contact
import perspective as P

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO, "models", "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

DOT_COLOR = (0, 0, 255)          # BGR red
TRAIL_COLOR = (40, 40, 180)
TRAIL_LENGTH = 30
C_CORNER = (0, 255, 255)
C_EDGE = (0, 200, 255)
C_REPROJ = (0, 255, 0)
C_TEXT = (255, 255, 255)
WIN_CLICK = "1 - first frame (click here)"
WIN_DIAGRAM = "3 - court diagram"

HELP_TEXT = """
------------------------------------------------------------------
 CONTROLS   (click and type in the image window, not the terminal)
------------------------------------------------------------------
 mouse left click : add the next court corner
 u : undo the last corner
 r : reset all corners
 h : print this help
 3 : start the court-diagram animation, after 4 valid corners
 q or ESC : quit
------------------------------------------------------------------
 CORNER ORDER: far-left, far-right, near-right, near-left.
 "far/near" and "left/right" are as the court appears in the frame,
 going clockwise around the court on screen.
------------------------------------------------------------------
"""


def gui_available():
    """True if this OpenCV build can open windows."""
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__gui_probe__")
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def ensure_model(path=MODEL_PATH):
    """Download the lite pose model once.  Returns the local path."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    print(f"downloading pose model to {path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        os.replace(temporary, path)
    except OSError as exc:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise SystemExit(f"ERROR: could not download the pose model: {exc}") from exc
    return path


class PoseContacts:
    """Heel contact pixel for each frame, from one Pose Landmarker pass.

    Call close() when finished.  The landmarker is created on first use so
    importing this module does not require mediapipe.
    """

    def __init__(self, model_path, min_visibility=contact.DEFAULT_MIN_VISIBILITY):
        import mediapipe as mp

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._min_visibility = min_visibility
        self._last_timestamp_ms = -1

    def __call__(self, frame, index, fps):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(round(index * 1000.0 / float(fps)))
        if timestamp <= self._last_timestamp_ms:
            timestamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp
        result = self._landmarker.detect_for_video(mp_image, timestamp)
        if not result.pose_landmarks:
            return None
        height, width = frame.shape[:2]
        return contact.contact_from_landmarks(
            result.pose_landmarks[0], width, height, self._min_visibility
        )

    def close(self):
        self._landmarker.close()


def court_point(calib, pixel):
    """Image pixel -> (x_ft, y_ft), or None."""
    if pixel is None:
        return None
    xy = calib.image_to_court([pixel])[0]
    return (float(xy[0]), float(xy[1]))


def render_track(base, points_ft, ppf, margin_ft, trail=TRAIL_LENGTH):
    """Copy of the court diagram with a red dot and a short trail.

    points_ft is the recent history, oldest first.  Entries are (x_ft, y_ft)
    or None for a frame with no visible heel.  The trail breaks at those gaps.
    """
    img = base.copy()
    window = list(points_ft)[-trail:]

    def to_px(point_ft):
        p = P.feet_to_output_px([point_ft], ppf, margin_ft)[0]
        return (int(round(p[0])), int(round(p[1])))

    run = []

    def flush():
        if len(run) >= 2:
            cv2.polylines(img, [np.asarray(run, dtype=np.int32)], False, TRAIL_COLOR, 2, cv2.LINE_AA)
        run.clear()

    for point in window:
        if point is None:
            flush()
        else:
            run.append(to_px(point))
    flush()

    current = window[-1] if window else None
    if current is not None:
        center = to_px(current)
        cv2.circle(img, center, 9, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"({current[0]:.1f}, {current[1]:.1f}) ft"
        if P.is_out_of_bounds(current):
            label += " OUT"
        P.put_label(img, label, (center[0] + 12, center[1] - 10), DOT_COLOR, 0.5)
    return img


def _with_prompt(image, text):
    """A bar along the bottom of a copy of image."""
    img = image.copy()
    top = max(0, img.shape[0] - 36)
    bar = img.copy()
    cv2.rectangle(bar, (0, top), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
    cv2.putText(img, text, (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                C_TEXT, 1, cv2.LINE_AA)
    return img


def play_animation(base, history, ppf, margin, trail, fps):
    """Play the stored court positions, then wait for replay or quit.

    r starts the animation again.  q or ESC closes the window.  The pose model
    is not run again.
    """
    if not history:
        return
    delay = max(1, int(round(1000.0 / float(fps))))
    prompt = "r replay    q quit"
    print(f"\nAnimation.  {prompt}")
    cv2.namedWindow(WIN_DIAGRAM, cv2.WINDOW_AUTOSIZE)
    first = render_track(base, history[:1], ppf, margin, trail)

    while True:
        replay = False
        last = first
        for index in range(len(history)):
            last = render_track(base, history[:index + 1], ppf, margin, trail)
            cv2.imshow(WIN_DIAGRAM, last)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(WIN_DIAGRAM)
                return
            if key == ord("r"):
                replay = True
                break
        if replay:
            continue
        held = _with_prompt(last, prompt)
        while True:
            cv2.imshow(WIN_DIAGRAM, held)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(WIN_DIAGRAM)
                return
            if key == ord("r"):
                break
            try:
                if cv2.getWindowProperty(WIN_DIAGRAM, cv2.WND_PROP_VISIBLE) < 1:
                    return
            except cv2.error:
                return


def process_video(video_path, calib, contact_at, out_dir, trail=TRAIL_LENGTH, preview=False):
    """Map every frame and write output/court_path.mp4 and positions.csv.

    contact_at(frame, index, fps) returns an (x, y) image pixel or None.
    Returns a small dict describing what was written.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps < 1.0:
        fps = 30.0

    base, ppf, margin = P.draw_court_diagram()
    height, width = base.shape[:2]
    video_out = os.path.join(out_dir, "court_path.mp4")
    csv_out = os.path.join(out_dir, "positions.csv")
    writer = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"ERROR: could not write {video_out}")

    history = []
    n = 0
    n_hit = 0
    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as handle:
            rows = csv.writer(handle)
            rows.writerow(["frame", "time_s", "x_ft", "y_ft"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                feet = court_point(calib, contact_at(frame, n, fps))
                if feet is None:
                    rows.writerow([n, f"{n / fps:.3f}", "", ""])
                else:
                    n_hit += 1
                    rows.writerow([n, f"{n / fps:.3f}", f"{feet[0]:.3f}", f"{feet[1]:.3f}"])
                history.append(feet)
                writer.write(render_track(base, history, ppf, margin, trail))
                n += 1
                if n % 30 == 0:
                    print(f"  frame {n}, feet found on {n_hit}")
    finally:
        cap.release()
        writer.release()

    if n == 0:
        raise SystemExit(f"ERROR: {video_path} has no frames")
    if preview:
        play_animation(base, history, ppf, margin, trail, fps)
    return {
        "frames": n,
        "with_feet": n_hit,
        "video": video_out,
        "csv": csv_out,
        "fps": fps,
    }


def click_corners(frame, max_display=(1280, 800)):
    """Click the four corners, then press 3.  Returns the corners, or None if cancelled.

    The window stays open after the fourth click so the closing edge, and the
    green court lines, are actually visible.  OpenCV does not refresh a window
    until waitKey, so accepting the quad in the same turn as the click hides
    that edge.
    """
    scaler = P.DisplayScaler(frame.shape, max_display[0], max_display[1])
    state = {
        "corners": [],
        "calib": None,
        "message": "Click the 4 court corners: far-left, far-right, near-right, near-left.",
    }

    def consider():
        pts = state["corners"]
        if len(pts) < 4:
            state["calib"] = None
            if pts:
                state["message"] = f"Clicked {len(pts)}/4. Next: {P.CORNER_LABELS[len(pts)]}."
            else:
                state["message"] = ("Click the 4 court corners: far-left, far-right, "
                                    "near-right, near-left.")
            return
        problems = P.validate_corners(pts, frame.shape)
        if problems:
            state["calib"] = None
            print("\nREJECTED corner selection:")
            for problem in problems:
                print(f"  - {problem}")
            print("Press 'u' to undo or 'r' to reset.\n")
            state["message"] = "Invalid selection: " + problems[0][:70]
            return
        state["calib"] = P.Calibration.from_corners(pts, frame.shape, "first-frame")
        state["message"] = "Corners set. Press 3 to start the animation."
        print("\nCorners set. The closing edge and the green court lines stay up.")
        print("Press 3 in the image window to start the animation.\n")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(state["corners"]) >= 4:
            state["message"] = "4 corners already selected. Press 'u' to undo or 'r' to reset."
            return
        pt = scaler.to_original(x, y)
        state["corners"].append(pt)
        n = len(state["corners"])
        print(f"corner {n} ({P.CORNER_LABELS[n - 1]}): ({pt[0]:.1f}, {pt[1]:.1f}) px")
        consider()

    def render():
        img = scaler.resize(frame)
        calib = state["calib"]
        if calib is not None:
            def seg(a, b, color=C_REPROJ, thick=1):
                mapped = calib.court_to_image([a, b])
                cv2.line(img, scaler.to_display_int(mapped[0]), scaler.to_display_int(mapped[1]),
                         color, thick, cv2.LINE_AA)

            width_ft, length_ft = P.COURT_WIDTH_FT, P.COURT_LENGTH_FT
            for y_ft in (P.KITCHEN_FAR_Y_FT, P.KITCHEN_NEAR_Y_FT):
                seg((0, y_ft), (width_ft, y_ft))
            seg((P.CENTERLINE_X_FT, 0), (P.CENTERLINE_X_FT, P.KITCHEN_FAR_Y_FT))
            seg((P.CENTERLINE_X_FT, P.KITCHEN_NEAR_Y_FT), (P.CENTERLINE_X_FT, length_ft))
            seg((0, P.NET_Y_FT), (width_ft, P.NET_Y_FT), (200, 200, 0), 1)

        pts = [scaler.to_display_int(p) for p in state["corners"]]
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, C_CORNER, -1)
            cv2.circle(img, p, 9, (0, 0, 0), 1)
            P.put_label(img, f"{i + 1}. {P.CORNER_LABELS[i]}", (p[0] + 10, p[1] - 8),
                        C_CORNER, 0.5)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], C_EDGE, 2, cv2.LINE_AA)
        if len(pts) == 4:
            cv2.line(img, pts[3], pts[0], C_EDGE, 2, cv2.LINE_AA)

        status = f"corners: {len(pts)}/4    press 3 to start the animation"
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
        img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, status, (10, 20), font, 0.5, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, state["message"][:110], (10, 42), font, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
        if img.shape[1] > 760:
            cv2.putText(img, "h = help   3 = start   q = quit", (img.shape[1] - 280, 20),
                        font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    print(HELP_TEXT)
    print("The camera must stay still for the rest of the video.\n")
    cv2.namedWindow(WIN_CLICK, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_CLICK, on_mouse)

    while True:
        cv2.imshow(WIN_CLICK, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(WIN_CLICK)
            return None
        if key == ord("u") and state["corners"]:
            state["corners"].pop()
            print(f"undid a corner; {len(state['corners'])}/4 selected")
            consider()
        elif key == ord("r"):
            state["corners"].clear()
            print("reset corners")
            consider()
        elif key == ord("h"):
            print(HELP_TEXT)
        elif key == ord("3"):
            if state["calib"] is None:
                state["message"] = "Click 4 valid corners first, then press 3."
                print(state["message"])
            else:
                cv2.destroyWindow(WIN_CLICK)
                return list(state["corners"])
        try:
            if cv2.getWindowProperty(WIN_CLICK, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Map one player's feet in a fixed-camera video onto a court diagram.",
    )
    parser.add_argument("video", help="path to the drill video")
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    parser.add_argument("--calibration", default=None,
                        help="calibration JSON to write, or to read with --load "
                             "(default: <out>/video_calibration.json)")
    parser.add_argument("--load", action="store_true",
                        help="load the calibration instead of clicking corners")
    parser.add_argument("--max-display", default="1280x800",
                        help="largest corner-click window, WxH (default: 1280x800)")
    parser.add_argument("--no-preview", action="store_true",
                        help="write the diagram video without opening a window during processing")
    parser.add_argument("--model", default=MODEL_PATH, help="path to pose_landmarker_lite.task")
    return parser.parse_args(argv)


def _max_display(text):
    try:
        w_str, h_str = text.lower().split("x")
        return (int(w_str), int(h_str))
    except ValueError:
        raise SystemExit(f"ERROR: --max-display must look like 1280x800, got '{text}'")


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.video):
        raise SystemExit(f"ERROR: video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise SystemExit(f"ERROR: could not read a frame from {args.video}")

    calib_path = args.calibration or os.path.join(args.out, "video_calibration.json")
    if args.load:
        try:
            calib = P.Calibration.load(calib_path, first.shape)
        except P.CalibrationError as exc:
            raise SystemExit(f"ERROR: {exc}")
        print(f"loaded calibration from {calib_path}")
    else:
        if not gui_available():
            raise SystemExit(
                "ERROR: this OpenCV build cannot open windows.\n"
                "       Install the GUI build:  pip install opencv-python"
            )
        corners = click_corners(first, _max_display(args.max_display))
        if corners is None:
            raise SystemExit("cancelled")
        try:
            calib = P.Calibration.from_corners(corners, first.shape, args.video)
        except P.CalibrationError as exc:
            raise SystemExit(f"ERROR: {exc}")
        calib.save(calib_path)
        print(f"saved calibration to {calib_path}")

    model_path = ensure_model(args.model)
    tracker = PoseContacts(model_path)
    try:
        summary = process_video(
            args.video, calib, tracker, args.out, preview=not args.no_preview
        )
    finally:
        tracker.close()

    print(
        f"\n{summary['with_feet']} of {summary['frames']} frames had a foot contact\n"
        f"  {summary['video']}\n"
        f"  {summary['csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
