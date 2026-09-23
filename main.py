"""
main.py -- interactive top-down court mapper.

Workflow:
    1. load an angled photograph of a pickleball court,
    2. click the four outside court corners to calibrate,
    3. click players' ground-contact points to read their court coordinates,
    4. optionally click known landmarks to measure the accuracy,
    5. save the calibration, the rectified image, the diagram and CSV exports.

All of the geometry lives in perspective.py.  This file is only the command
line, the OpenCV windows, and the mouse and keyboard handling.

Examples:
    python main.py demo/demo_angled.png
    python main.py demo/demo_angled.png --load
    python main.py photo.jpg --calibration output/my_court.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

import perspective as P

WIN_MAIN = "1 - original image (click here)"
WIN_RECT = "2 - rectified top-down view"
WIN_DIAG = "3 - court diagram"

MODE_CORNERS = "corners"
MODE_PLAYERS = "players"
MODE_VALIDATE = "validate"

# BGR colours
C_CORNER = (0, 255, 255)
C_EDGE = (0, 200, 255)
C_PLAYER = (0, 220, 255)
C_PLAYER_OOB = (0, 0, 255)
C_LANDMARK = (255, 160, 0)
C_REPROJ = (0, 255, 0)
C_TEXT = (255, 255, 255)

HELP_TEXT = """
------------------------------------------------------------------
 CONTROLS   (click and type in the image window, not the terminal)
------------------------------------------------------------------
 mouse left click : add a point for the current mode
 c : corner mode      -- (re)select the 4 court corners
 p : player mode      -- click players' ground-contact points
 v : validation mode  -- click a known landmark, then name it in the terminal
 u : undo the last point of the current mode
 r : reset all points of the current mode
 2 : show / hide the rectified top-down window
 3 : show / hide the court diagram window
 s : save everything (calibration, CSVs, images) to the output folder
 l : reload the calibration file from disk
 h : print this help
 q or ESC : quit
------------------------------------------------------------------
 CORNER ORDER: far-left, far-right, near-right, near-left.
 "far/near" and "left/right" are as the court appears in the photo,
 going clockwise around the court on screen.
------------------------------------------------------------------
"""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_image(path):
    """Read an image with clear errors.  Exits the program on failure."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: image not found: {path}\n"
                 f"       Check the path, or run 'python generate_demo.py' and use "
                 f"demo/demo_angled.png")
    if os.path.isdir(path):
        sys.exit(f"ERROR: {path} is a directory, not an image file.")
    if os.path.getsize(path) == 0:
        sys.exit(f"ERROR: {path} is empty (0 bytes).")
    # np.fromfile + imdecode instead of cv2.imread so non-ASCII paths work on Windows.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError as exc:
        sys.exit(f"ERROR: could not read {path}: {exc}")
    if image is None:
        sys.exit(f"ERROR: could not decode {path} as an image.\n"
                 f"       Supported formats include .png, .jpg, .bmp, .tif. "
                 f"The file may be corrupt or a renamed non-image file.")
    return image


def gui_available():
    """True if this OpenCV build can open windows (headless wheels cannot)."""
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__gui_probe__")
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


# --------------------------------------------------------------------------
# The application
# --------------------------------------------------------------------------
class CourtMapper:
    def __init__(self, image, image_path, out_dir, calib_path, ppf=P.DEFAULT_PPF,
                 max_display=(1280, 800)):
        self.image = image
        self.image_path = image_path
        self.out_dir = out_dir
        self.calib_path = calib_path
        self.ppf = ppf
        self.scaler = P.DisplayScaler(image.shape, max_display[0], max_display[1])

        self.calib = None
        self.corner_pts = []       # original-image pixels, click order
        self.players = []          # {label, image, court, oob}
        self.landmarks = []        # {label, image, measured, truth, error_ft}
        self.mode = MODE_CORNERS
        self.message = "Click the 4 court corners: far-left, far-right, near-right, near-left."

        self.diagram_base, _, self.margin_ft = P.draw_court_diagram(self.ppf)
        self.show_rect = False
        self.show_diag = False
        self.rectified = None

    # -- state changes -----------------------------------------------------
    def set_mode(self, mode):
        if mode in (MODE_PLAYERS, MODE_VALIDATE) and self.calib is None:
            self.message = "Calibrate first: press 'c' and click the 4 court corners."
            return
        self.mode = mode
        self.message = {
            MODE_CORNERS: "Corner mode: click far-left, far-right, near-right, near-left.",
            MODE_PLAYERS: "Player mode: click the point between each player's feet.",
            MODE_VALIDATE: "Validation mode: click a known landmark, then choose it in the terminal.",
        }[mode]
        if mode == MODE_CORNERS:
            self.corner_pts = []

    def add_corner(self, pt):
        self.corner_pts.append(pt)
        n = len(self.corner_pts)
        print(f"corner {n} ({P.CORNER_LABELS[n - 1]}): ({pt[0]:.1f}, {pt[1]:.1f}) px")
        if n < 4:
            self.message = f"Clicked {n}/4. Next: {P.CORNER_LABELS[n]}."
            return
        self.try_calibrate()

    def try_calibrate(self):
        problems = P.validate_corners(self.corner_pts, self.image.shape)
        if problems:
            print("\nREJECTED corner selection:")
            for p in problems:
                print(f"  - {p}")
            print("Press 'r' to reset or 'u' to undo the last point.\n")
            self.message = "Invalid selection: " + problems[0][:70]
            return
        try:
            self.calib = P.Calibration.from_corners(self.corner_pts, self.image.shape,
                                                    self.image_path)
        except P.CalibrationError as exc:
            print(f"\nREJECTED: {exc}\n")
            self.message = f"Invalid selection: {exc}"[:90]
            return
        self.recompute()
        self.mode = MODE_PLAYERS
        self.message = "Calibrated. Player mode: click players' ground-contact points."
        print("\nCalibrated. Homography (image pixels -> court feet):")
        print(np.array2string(self.calib.H_image_to_court, precision=6, suppress_small=True))
        rt = self.calib.court_to_image(self.calib.image_to_court(self.corner_pts))
        residual = float(np.max(np.linalg.norm(np.asarray(self.corner_pts) - rt, axis=1)))
        print(f"round-trip residual on the 4 corners: {residual:.2e} px "
              f"(numerical only -- this is NOT an accuracy measurement)")
        print("Measure real accuracy with validation mode ('v') on other landmarks.\n")

    def recompute(self):
        """Re-map every stored click after the calibration changes."""
        if self.calib is None:
            return
        self.rectified = P.rectify(self.image, self.calib, self.ppf)
        for pl in self.players:
            pl["court"] = self.calib.image_to_court([pl["image"]])[0]
            pl["oob"] = P.is_out_of_bounds(pl["court"])
        for lm in self.landmarks:
            lm["measured"] = self.calib.image_to_court([lm["image"]])[0]
            lm["error_ft"] = float(np.linalg.norm(lm["measured"] - np.asarray(lm["truth"])))

    def add_player(self, pt):
        court = self.calib.image_to_court([pt])[0]
        label = f"P{len(self.players) + 1}"
        oob = P.is_out_of_bounds(court)
        self.players.append({"label": label, "image": pt, "court": court, "oob": oob})
        flag = "  [OUT OF COURT -- reported, not clamped]" if oob else ""
        print(f"{label}: image ({pt[0]:.1f}, {pt[1]:.1f}) px  ->  court "
              f"({court[0]:.2f}, {court[1]:.2f}) ft{flag}")
        self.message = f"{label} at ({court[0]:.1f}, {court[1]:.1f}) ft" + (" OUT" if oob else "")

    def add_landmark(self, pt):
        """Ask the terminal which landmark was clicked, then record its error."""
        print("\nWhich landmark did you click?  (court coordinates in feet)")
        for i, (label, x, y) in enumerate(P.VALIDATION_LANDMARKS, start=1):
            print(f"  {i:2d}. {label:<38} ({x:.1f}, {y:.1f})")
        print("   x,y  -- type your own court coordinates, e.g. 10,15")
        print("   s    -- skip this click")
        choice = input("choice: ").strip()

        if choice.lower() in ("s", "skip", ""):
            print("skipped.\n")
            return
        if "," in choice:
            try:
                xs, ys = choice.split(",")[:2]
                truth = (float(xs), float(ys))
                label = f"custom ({truth[0]:.1f}, {truth[1]:.1f})"
            except ValueError:
                print("could not parse those coordinates -- click skipped.\n")
                return
        else:
            try:
                idx = int(choice)
                if not 1 <= idx <= len(P.VALIDATION_LANDMARKS):
                    raise IndexError(idx)
                label, tx, ty = P.VALIDATION_LANDMARKS[idx - 1]
                truth = (tx, ty)
            except (ValueError, IndexError):
                print("not a valid choice -- click skipped.\n")
                return

        measured = self.calib.image_to_court([pt])[0]
        err = float(np.linalg.norm(measured - np.asarray(truth)))
        self.landmarks.append({"label": label, "image": pt, "measured": measured,
                               "truth": truth, "error_ft": err})
        print(f"  measured ({measured[0]:.2f}, {measured[1]:.2f}) ft, "
              f"true ({truth[0]:.2f}, {truth[1]:.2f}) ft, error {err:.3f} ft")
        self.print_validation_summary()

    def print_validation_summary(self):
        errs = [lm["error_ft"] for lm in self.landmarks]
        s = P.error_summary(errs)
        note = "" if s["n"] >= 5 else f"  (click at least 5; you have {s['n']})"
        print(f"  running: n={s['n']}  mean={s['mean_ft']:.3f} ft  "
              f"RMSE={s['rmse_ft']:.3f} ft  max={s['max_ft']:.3f} ft{note}\n")
        self.message = f"validation n={s['n']} mean={s['mean_ft']:.2f} ft RMSE={s['rmse_ft']:.2f} ft"

    def undo(self):
        target = {MODE_CORNERS: self.corner_pts, MODE_PLAYERS: self.players,
                  MODE_VALIDATE: self.landmarks}[self.mode]
        if not target:
            self.message = f"nothing to undo in {self.mode} mode"
            return
        target.pop()
        if self.mode == MODE_CORNERS:
            self.message = f"undid a corner; {len(self.corner_pts)}/4 selected"
        else:
            self.message = f"undid the last {self.mode[:-1]}"
        print(self.message)

    def reset(self):
        if self.mode == MODE_CORNERS:
            self.corner_pts = []
        elif self.mode == MODE_PLAYERS:
            self.players = []
        else:
            self.landmarks = []
        self.message = f"reset {self.mode}"
        print(self.message)

    # -- rendering ---------------------------------------------------------
    def render_main(self):
        """Draw the overlay in DISPLAY coordinates on a resized copy.

        Points are stored in original-image pixels, so every one is converted
        through the scaler here.  Only the display copy is annotated.
        """
        img = self.scaler.resize(self.image)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # reprojected court lines -- visual feedback on how good the fit is
        if self.calib is not None:
            def seg(a, b, color=C_REPROJ, thick=1):
                pts = self.calib.court_to_image([a, b])
                p0 = self.scaler.to_display_int(pts[0])
                p1 = self.scaler.to_display_int(pts[1])
                cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)

            W, L = P.COURT_WIDTH_FT, P.COURT_LENGTH_FT
            for y in (P.KITCHEN_FAR_Y_FT, P.KITCHEN_NEAR_Y_FT):
                seg((0, y), (W, y))
            seg((P.CENTERLINE_X_FT, 0), (P.CENTERLINE_X_FT, P.KITCHEN_FAR_Y_FT))
            seg((P.CENTERLINE_X_FT, P.KITCHEN_NEAR_Y_FT), (P.CENTERLINE_X_FT, L))
            seg((0, P.NET_Y_FT), (W, P.NET_Y_FT), (200, 200, 0), 1)

        # corner clicks
        pts_disp = [self.scaler.to_display_int(p) for p in self.corner_pts]
        for i, p in enumerate(pts_disp):
            cv2.circle(img, p, 6, C_CORNER, -1)
            cv2.circle(img, p, 9, (0, 0, 0), 1)
            P.put_label(img, f"{i + 1}. {P.CORNER_LABELS[i]}", (p[0] + 10, p[1] - 8),
                        C_CORNER, 0.5)
        for i in range(len(pts_disp) - 1):
            cv2.line(img, pts_disp[i], pts_disp[i + 1], C_EDGE, 2, cv2.LINE_AA)
        if len(pts_disp) == 4:
            cv2.line(img, pts_disp[3], pts_disp[0], C_EDGE, 2, cv2.LINE_AA)

        # players
        for pl in self.players:
            p = self.scaler.to_display_int(pl["image"])
            c = C_PLAYER_OOB if pl["oob"] else C_PLAYER
            cv2.drawMarker(img, p, c, cv2.MARKER_CROSS, 16, 2)
            cv2.circle(img, p, 9, c, 2)
            txt = f"{pl['label']} ({pl['court'][0]:.1f}, {pl['court'][1]:.1f})"
            if pl["oob"]:
                txt += " OUT"
            P.put_label(img, txt, (p[0] + 12, p[1] - 10), c, 0.5)

        # validation landmarks
        for i, lm in enumerate(self.landmarks, start=1):
            p = self.scaler.to_display_int(lm["image"])
            cv2.drawMarker(img, p, C_LANDMARK, cv2.MARKER_TILTED_CROSS, 14, 2)
            P.put_label(img, f"L{i} err {lm['error_ft']:.2f} ft", (p[0] + 10, p[1] + 16),
                        C_LANDMARK)

        # heads-up display
        status = f"mode: {self.mode}   calibrated: {'yes' if self.calib else 'NO'}   " \
                 f"players: {len(self.players)}   landmarks: {len(self.landmarks)}"
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
        img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
        cv2.putText(img, status, (10, 20), font, 0.5, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, self.message[:110], (10, 42), font, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
        if img.shape[1] > 760:      # only if it will not collide with the status line
            cv2.putText(img, "h = help   q = quit", (img.shape[1] - 190, 20), font, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def render_diagram(self):
        pts = [pl["court"] for pl in self.players]
        labels = [pl["label"] for pl in self.players]
        img = P.draw_markers_on_diagram(self.diagram_base, pts, labels,
                                        self.ppf, self.margin_ft)
        if self.landmarks:
            lm_pts = [lm["measured"] for lm in self.landmarks]
            lm_labels = [f"L{i}" for i in range(1, len(self.landmarks) + 1)]
            img = P.draw_markers_on_diagram(img, lm_pts, lm_labels, self.ppf,
                                            self.margin_ft, color=C_LANDMARK,
                                            show_coords=False)
        return img

    # -- mouse -------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # The window shows a scaled copy -- convert the click back to the
        # ORIGINAL image pixel grid before it is used for any geometry.
        pt = self.scaler.to_original(x, y)
        if self.mode == MODE_CORNERS:
            if len(self.corner_pts) >= 4:
                self.message = "4 corners already selected. Press 'r' to reset or 'u' to undo."
                return
            self.add_corner(pt)
        elif self.calib is None:
            self.message = "Calibrate first: press 'c' and click the 4 court corners."
        elif self.mode == MODE_PLAYERS:
            self.add_player(pt)
        else:
            self.add_landmark(pt)

    # -- saving ------------------------------------------------------------
    def save_all(self):
        os.makedirs(self.out_dir, exist_ok=True)
        written = []

        if self.calib is None:
            print("nothing to save yet -- calibrate first.")
            self.message = "nothing to save; calibrate first"
            return
        self.calib.save(self.calib_path)
        written.append(self.calib_path)

        players_csv = os.path.join(self.out_dir, "players.csv")
        with open(players_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "image_x_px", "image_y_px", "court_x_ft", "court_y_ft",
                        "out_of_court"])
            for pl in self.players:
                w.writerow([pl["label"], f"{pl['image'][0]:.2f}", f"{pl['image'][1]:.2f}",
                            f"{pl['court'][0]:.3f}", f"{pl['court'][1]:.3f}",
                            "yes" if pl["oob"] else "no"])
        written.append(players_csv)

        if self.landmarks:
            val_csv = os.path.join(self.out_dir, "validation.csv")
            with open(val_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["landmark", "image_x_px", "image_y_px", "measured_x_ft",
                            "measured_y_ft", "true_x_ft", "true_y_ft", "error_ft"])
                for lm in self.landmarks:
                    w.writerow([lm["label"], f"{lm['image'][0]:.2f}", f"{lm['image'][1]:.2f}",
                                f"{lm['measured'][0]:.3f}", f"{lm['measured'][1]:.3f}",
                                f"{lm['truth'][0]:.3f}", f"{lm['truth'][1]:.3f}",
                                f"{lm['error_ft']:.4f}"])
            written.append(val_csv)

            summary = P.error_summary([lm["error_ft"] for lm in self.landmarks])
            summary["source_image"] = os.path.basename(self.image_path)
            summary["note"] = ("Errors are measured at landmarks NOT used to fit the "
                               "homography. The 4 calibration corners are excluded by "
                               "construction, since they fit exactly.")
            sum_path = os.path.join(self.out_dir, "validation_summary.json")
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            written.append(sum_path)

        rect_path = os.path.join(self.out_dir, "rectified.png")
        cv2.imwrite(rect_path, self.rectified)
        written.append(rect_path)

        diag_path = os.path.join(self.out_dir, "court_diagram.png")
        cv2.imwrite(diag_path, self.render_diagram())
        written.append(diag_path)

        annotated = os.path.join(self.out_dir, "annotated_original.png")
        cv2.imwrite(annotated, self.render_main())
        written.append(annotated)

        print("\nsaved:")
        for path in written:
            print(f"  {path}")
        print()
        self.message = f"saved {len(written)} files to {self.out_dir}/"

    def load_calibration(self):
        try:
            calib = P.Calibration.load(self.calib_path, self.image.shape)
        except P.CalibrationError as exc:
            print(f"\nERROR loading calibration: {exc}\n")
            self.message = f"load failed: {exc}"[:100]
            return False
        self.calib = calib
        self.corner_pts = [tuple(c) for c in calib.image_corners]
        self.recompute()
        self.mode = MODE_PLAYERS
        self.message = f"loaded calibration from {self.calib_path}"
        print(f"\nloaded calibration from {self.calib_path}")
        print(f"  made for {calib.image_width}x{calib.image_height} "
              f"image '{calib.source_image}' on {calib.created_utc}")
        print("  NOTE: matching image size does not prove the calibration fits this "
              "photo.\n        The camera position, zoom and crop must match too -- "
              "check that the\n        green reprojected court lines land on the "
              "painted lines.\n")
        return True

    # -- main loop ---------------------------------------------------------
    def run(self):
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_MAIN, self.on_mouse)
        print(HELP_TEXT)

        while True:
            cv2.imshow(WIN_MAIN, self.render_main())
            if self.show_rect and self.rectified is not None:
                cv2.imshow(WIN_RECT, self.rectified)
            if self.show_diag:
                cv2.imshow(WIN_DIAG, self.render_diagram())

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                self.set_mode(MODE_CORNERS)
            elif key == ord("p"):
                self.set_mode(MODE_PLAYERS)
            elif key == ord("v"):
                self.set_mode(MODE_VALIDATE)
            elif key == ord("u"):
                self.undo()
            elif key == ord("r"):
                self.reset()
            elif key == ord("h"):
                print(HELP_TEXT)
            elif key == ord("s"):
                self.save_all()
            elif key == ord("l"):
                self.load_calibration()
            elif key == ord("2"):
                if self.calib is None:
                    self.message = "calibrate first to see the rectified view"
                else:
                    self.show_rect = not self.show_rect
                    if not self.show_rect:
                        cv2.destroyWindow(WIN_RECT)
            elif key == ord("3"):
                self.show_diag = not self.show_diag
                if not self.show_diag:
                    cv2.destroyWindow(WIN_DIAG)

            # closing the main window with the X button should also exit
            if cv2.getWindowProperty(WIN_MAIN, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        if self.players:
            print("\nfinal player positions (court feet):")
            for pl in self.players:
                flag = "  OUT OF COURT" if pl["oob"] else ""
                print(f"  {pl['label']}: ({pl['court'][0]:.2f}, {pl['court'][1]:.2f}){flag}")
        if self.landmarks:
            self.print_validation_summary()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Map an angled pickleball court photo to a top-down view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_TEXT,
    )
    ap.add_argument("image", nargs="?", help="path to the court photograph")
    ap.add_argument("--demo", action="store_true",
                    help="use demo/demo_angled.png (generating it first if needed)")
    ap.add_argument("--calibration", default=None,
                    help="calibration JSON path (default: <output>/calibration.json)")
    ap.add_argument("--load", action="store_true",
                    help="load the calibration file at startup instead of clicking corners")
    ap.add_argument("--out", default="output", help="output folder (default: output)")
    ap.add_argument("--ppf", type=float, default=P.DEFAULT_PPF,
                    help=f"pixels per foot for rendered views (default: {P.DEFAULT_PPF:g})")
    ap.add_argument("--max-display", default="1280x800",
                    help="largest window size, WxH (default: 1280x800)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    image_path = args.image
    if args.demo:
        image_path = os.path.join("demo", "demo_angled.png")
        if not os.path.exists(image_path):
            print("demo image missing -- generating it now...")
            import generate_demo
            generate_demo.main([])
    if not image_path:
        sys.exit("ERROR: no image given.\n"
                 "       usage: python main.py IMAGE   (or: python main.py --demo)")

    try:
        w_str, h_str = args.max_display.lower().split("x")
        max_display = (int(w_str), int(h_str))
    except ValueError:
        sys.exit(f"ERROR: --max-display must look like 1280x800, got '{args.max_display}'")
    if args.ppf <= 0:
        sys.exit("ERROR: --ppf must be positive")

    image = load_image(image_path)
    h, w = image.shape[:2]
    calib_path = args.calibration or os.path.join(args.out, "calibration.json")

    print(f"loaded {image_path}  ({w} x {h} px)")
    app = CourtMapper(image, image_path, args.out, calib_path, args.ppf, max_display)
    if app.scaler.scale < 1.0:
        print(f"window shows it at {app.scaler.display_size[0]}x{app.scaler.display_size[1]} "
              f"(scale {app.scaler.scale:.3f}); clicks are converted back to original pixels")

    if args.load:
        if not app.load_calibration():
            sys.exit(1)

    if not gui_available():
        sys.exit("ERROR: this OpenCV build cannot open windows (headless install?).\n"
                 "       Install the GUI build:  pip install opencv-python\n"
                 "       Meanwhile, run 'python verify_demo.py' for the non-interactive check.")

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
