"""
verify_demo.py -- run the whole workflow on a synthetic demo, with no windows.

This is the script to run on a machine that cannot open OpenCV windows (a
server, a container, a remote shell), and the quickest way to check nothing is
broken.  It performs every step main.py performs except the mouse clicking: it
"clicks" the four corners using the positions recorded by generate_demo.py,
then checks the results against known ground truth and writes the same output
files.

It runs in one of two modes, chosen automatically:

  EXACT MODE   -- an undistorted demo image and no click noise.  A homography
                  fits it perfectly, so every landmark must come back within
                  1e-6 ft or the run FAILS.  This proves the code is correct.
  REPORT MODE  -- a distorted image (generate_demo.py --distortion) and/or
                  simulated click noise.  The errors are now real, so the
                  script reports mean/RMSE instead of demanding zero.  The
                  structural checks (round trips, save/reload, rejection of bad
                  input) still have to pass.

Neither mode measures accuracy on a REAL photograph, where the distortion is
unknown, the clicks are yours, the court is not perfectly flat, and no ground
truth exists.  Report those numbers separately.

Examples:
    python verify_demo.py
    python verify_demo.py --image demo/demo_angled_distorted.png
    python verify_demo.py --image demo/demo_low.png --click-noise 2
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

DEMO_DIR = "demo"
DEFAULT_IMAGE = os.path.join(DEMO_DIR, "demo_angled.png")

TOL_FT = 1e-6          # tolerance for "exact" synthetic results
TOL_PX = 1e-6

# cv2.getPerspectiveTransform takes float32 point arrays, so corner coordinates
# are rounded to ~1e-4 px before the solve.  Over a 44 ft court that shows up as
# ~1e-6 ft (a third of a micrometre) when the corners are not exactly
# representable in float32.  This is the arithmetic floor, not an error worth
# chasing, so the fit self-check uses a slightly looser bound.
TOL_FIT_FT = 1e-4


def ok(passed):
    return "PASS" if passed else "FAIL"


def truth_path_for(image_path):
    stem, _ = os.path.splitext(image_path)
    return stem + "_ground_truth.json"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEFAULT_IMAGE,
                    help=f"demo image to verify (default: {DEFAULT_IMAGE})")
    ap.add_argument("--truth", default=None,
                    help="ground-truth JSON (default: <image>_ground_truth.json)")
    ap.add_argument("--click-noise", type=float, default=0.0, metavar="PX",
                    help="simulate imprecise corner clicks: Gaussian noise of this "
                         "standard deviation in pixels (default: 0)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the click noise")
    ap.add_argument("--out-dir", default=None,
                    help="where to write results (default: output/<image name>)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    failures = []

    # ---------------------------------------------------------------- setup
    image_path = args.image
    truth_file = args.truth or truth_path_for(image_path)

    if not (os.path.exists(image_path) and os.path.exists(truth_file)):
        if image_path == DEFAULT_IMAGE:
            print("demo files missing -- generating them now")
            import generate_demo
            generate_demo.main(["--out-dir", DEMO_DIR])
        else:
            sys.exit(f"ERROR: missing {image_path} or {truth_file}\n"
                     f"       generate it first, e.g.:\n"
                     f"       python generate_demo.py --all")

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"ERROR: could not read {image_path}")
    with open(truth_file, "r", encoding="utf-8") as f:
        truth = json.load(f)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = args.out_dir or os.path.join("output", stem)
    os.makedirs(out_dir, exist_ok=True)

    k1 = float(truth.get("distortion_k1", 0.0))
    noise = float(args.click_noise)
    exact = bool(truth.get("exact", True)) and noise == 0.0

    print("=" * 76)
    print("SYNTHETIC VERIFICATION -- known ground truth, no GUI, no human clicks")
    print("=" * 76)
    print(f"image      : {image_path}  ({image.shape[1]} x {image.shape[0]} px)")
    print(f"view       : {truth.get('view', 'default')}")
    print(f"distortion : " + ("none (ideal pinhole)" if not k1 else f"radial, k1 = {k1:g}"))
    print(f"click noise: " + ("none (exact corner positions)"
                              if noise == 0 else f"sigma = {noise:g} px"))
    if exact:
        print("mode       : EXACT -- landmark errors must be < 1e-6 ft or this run fails")
    else:
        print("mode       : REPORT -- errors are real; structural checks still must pass")
    print()

    # ------------------------------------------------- 1. calibrate
    corners_true = np.asarray(truth["corners_image"], dtype=np.float64)
    if noise > 0:
        rng_clicks = np.random.default_rng(args.seed)
        corners_clicked = corners_true + rng_clicks.normal(0.0, noise, corners_true.shape)
    else:
        corners_clicked = corners_true.copy()

    print("STEP 1  calibrate from the four court corners")
    for label, (x, y), (tx, ty) in zip(P.CORNER_LABELS, corners_clicked, corners_true):
        extra = f"   (true {tx:8.2f}, {ty:8.2f})" if noise > 0 else ""
        print(f"        {label:<10} clicked ({x:8.2f}, {y:8.2f}) px{extra}")
    calib = P.Calibration.from_corners(corners_clicked, image.shape, image_path)
    print("        homography (image px -> court ft):")
    for row in calib.H_image_to_court:
        print("          [" + "  ".join(f"{v: .6e}" for v in row) + "]")

    # The clicked corners map to the court rectangle exactly -- they define it.
    mapped = calib.image_to_court(corners_clicked)
    corner_err = float(np.max(np.abs(mapped - P.court_corners_ft())))
    passed = corner_err < TOL_FIT_FT
    failures += [] if passed else ["corners do not map to the court rectangle"]
    print(f"        clicked corners -> (0,0) (20,0) (20,44) (0,44): max error "
          f"{corner_err:.2e} ft  [{ok(passed)}]")
    if noise > 0:
        print(f"        ^ still ~zero even though every click was off by ~{noise:g} px.")
        print("          The fit always reproduces its own inputs, which is exactly why")
        print("          the corner residual is never evidence of accuracy.")
    else:
        print("        (a self-check of the fit, NOT independent accuracy)")
    print()

    # ------------------------------------------------- 2. round trip
    print("STEP 2  image -> court -> image round trip on 200 random pixels")
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(220, 1120, 200), rng.uniform(240, 700, 200)])
    back = calib.court_to_image(calib.image_to_court(pts))
    rt_err = float(np.max(np.linalg.norm(pts - back, axis=1)))
    passed = rt_err < 1e-6
    failures += [] if passed else ["round trip exceeded tolerance"]
    print(f"        max round-trip error {rt_err:.2e} px  [{ok(passed)}]")
    print("        (pure matrix algebra: exact whatever the image looks like)\n")

    # ------------------------------------------------- 3. landmarks
    print("STEP 3  independent landmarks (NOT used to fit the homography)")
    print(f"        {'landmark':<38} {'measured (ft)':>17} {'true (ft)':>14} {'err ft':>9}")
    rows, errors = [], []
    for lm in truth["landmarks"]:
        img_pt = [lm["image_x"], lm["image_y"]]
        measured = calib.image_to_court([img_pt])[0]
        true_ft = np.array([lm["x_ft"], lm["y_ft"]])
        err = float(np.linalg.norm(measured - true_ft))
        errors.append(err)
        rows.append([lm["label"], f"{img_pt[0]:.2f}", f"{img_pt[1]:.2f}",
                     f"{measured[0]:.4f}", f"{measured[1]:.4f}",
                     f"{true_ft[0]:.3f}", f"{true_ft[1]:.3f}",
                     f"{err:.3e}" if exact else f"{err:.4f}"])
        fmt = f"{err:9.2e}" if exact else f"{err:9.3f}"
        print(f"        {lm['label']:<38} ({measured[0]:7.3f},{measured[1]:7.3f}) "
              f"({true_ft[0]:5.1f},{true_ft[1]:5.1f}) {fmt}")

    summary = P.error_summary(errors)
    if exact:
        passed = summary["max_ft"] < TOL_FT and summary["n"] >= 5
        failures += [] if passed else ["landmark errors exceeded tolerance"]
        print(f"        n={summary['n']}  mean={summary['mean_ft']:.3e} ft  "
              f"RMSE={summary['rmse_ft']:.3e} ft  max={summary['max_ft']:.3e} ft  [{ok(passed)}]")
        print("        errors are ~1e-14 ft because the image is an ideal pinhole view")
        print("        and the clicks are exact. Add --distortion or --click-noise for")
        print("        realistic numbers.\n")
    else:
        worst = truth["landmarks"][int(np.argmax(errors))]["label"]
        print(f"        n={summary['n']}  mean={summary['mean_ft']:.3f} ft  "
              f"RMSE={summary['rmse_ft']:.3f} ft  max={summary['max_ft']:.3f} ft  "
              f"median={summary['median_ft']:.3f} ft")
        print(f"        worst landmark: {worst}")
        causes = []
        if k1:
            causes.append("lens distortion the homography cannot represent")
        if noise > 0:
            causes.append(f"corner clicks off by ~{noise:g} px")
        print(f"        these errors are REAL, caused by: {', and '.join(causes)}.")
        print("        RMSE above mean means the error is uneven across the court --")
        print("        compare where the worst landmarks sit in the frame.\n")

    val_csv = os.path.join(out_dir, "validation.csv")
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["landmark", "image_x_px", "image_y_px", "measured_x_ft", "measured_y_ft",
                    "true_x_ft", "true_y_ft", "error_ft"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "source_image": os.path.basename(image_path),
                   "distortion_k1": k1, "click_noise_px": noise, "exact_mode": exact,
                   "note": "Landmarks exclude the 4 calibration corners by construction."},
                  f, indent=2)

    # ------------------------------------------------- 4. players
    print("STEP 4  planted 'players' mapped to court coordinates")
    player_rows, p_errors = [], []
    players_ft, player_labels = [], []
    for pl in truth["players"]:
        img_pt = [pl["image_x"], pl["image_y"]]
        court = calib.image_to_court([img_pt])[0]
        true_ft = np.array([pl["x_ft"], pl["y_ft"]])
        err = float(np.linalg.norm(court - true_ft))
        p_errors.append(err)
        players_ft.append(court)
        player_labels.append(pl["label"])
        oob = P.is_out_of_bounds(court)
        player_rows.append([pl["label"], f"{img_pt[0]:.2f}", f"{img_pt[1]:.2f}",
                            f"{court[0]:.3f}", f"{court[1]:.3f}", "yes" if oob else "no"])
        fmt = f"{err:.2e}" if exact else f"{err:.3f}"
        print(f"        {pl['label']}: image ({img_pt[0]:7.1f},{img_pt[1]:7.1f}) px -> "
              f"court ({court[0]:6.2f},{court[1]:6.2f}) ft   true "
              f"({true_ft[0]:5.1f},{true_ft[1]:5.1f})   err {fmt} ft")
    if exact:
        passed = max(p_errors) < TOL_FT
        failures += [] if passed else ["player positions exceeded tolerance"]
        print(f"        max player error {max(p_errors):.2e} ft  [{ok(passed)}]")
    else:
        print(f"        max player error {max(p_errors):.3f} ft (real, see above)")

    # a point deliberately off the court: it must be reported, not clamped
    off_court_img = calib.court_to_image([[25.0, 48.0]])[0]
    off_court = calib.image_to_court([off_court_img])[0]
    passed = P.is_out_of_bounds(off_court) and abs(off_court[0] - 25.0) < 1e-6
    failures += [] if passed else ["out-of-court point was not preserved"]
    print(f"        out-of-court check: ({off_court[0]:.2f}, {off_court[1]:.2f}) ft "
          f"flagged out, not clamped  [{ok(passed)}]\n")

    players_csv = os.path.join(out_dir, "players.csv")
    with open(players_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "image_x_px", "image_y_px", "court_x_ft", "court_y_ft",
                    "out_of_court"])
        w.writerows(player_rows)

    # ------------------------------------------------- 5. save / reload
    print("STEP 5  save the calibration, reload it, and compare")
    calib_path = os.path.join(out_dir, "calibration.json")
    calib.save(calib_path)
    reloaded = P.Calibration.load(calib_path, image.shape)
    h_diff = float(np.max(np.abs(reloaded.H_image_to_court - calib.H_image_to_court)))
    test_pts = np.column_stack([rng.uniform(220, 1120, 50), rng.uniform(240, 700, 50)])
    map_diff = float(np.max(np.abs(reloaded.image_to_court(test_pts)
                                   - calib.image_to_court(test_pts))))
    passed = h_diff == 0.0 and map_diff == 0.0
    failures += [] if passed else ["reloaded calibration differs from the original"]
    print(f"        matrix difference {h_diff:.2e}, mapped-point difference "
          f"{map_diff:.2e} ft  [{ok(passed)}]")

    try:
        P.Calibration.load(calib_path, (100, 100, 3))
        rejected, reason = False, ""
    except P.CalibrationError as exc:
        rejected, reason = True, str(exc)
    failures += [] if rejected else ["mismatched image size was not rejected"]
    print(f"        100x100 image rejected: {ok(rejected)}")
    if rejected:
        print(f"          -> {reason}")
    print("        NOTE: passing the size check does not prove a calibration fits a")
    print("        different photo -- camera position, zoom and crop must match too.\n")

    # ------------------------------------------------- 6. bad selections
    print("STEP 6  invalid corner selections are rejected")
    h_img, w_img = image.shape[:2]
    cases = {
        "duplicate corner": [[300, 200], [300, 200], [1100, 700], [200, 700]],
        "bow-tie (self-intersecting)": [[300, 200], [1100, 700], [1000, 200], [350, 760]],
        "collinear / zero area": [[300, 200], [600, 200], [900, 200], [1200, 200]],
        "tiny quadrilateral": [[300, 200], [340, 200], [340, 240], [300, 240]],
        "reversed click order": [[300, 200], [200, 700], [1100, 700], [1000, 200]],
    }
    for name, quad in cases.items():
        problems = P.validate_corners(quad, (h_img, w_img, 3))
        good = len(problems) > 0
        failures += [] if good else [f"bad selection accepted: {name}"]
        print(f"        {name:<30} rejected [{ok(good)}]  "
              f"{problems[0][:58] if problems else ''}")
    good_problems = P.validate_corners(corners_clicked, image.shape)
    passed = not good_problems
    failures += [] if passed else ["the correct selection was wrongly rejected"]
    print(f"        {'the real corners':<30} accepted [{ok(passed)}]\n")

    # ------------------------------------------------- 7. display scaling
    print("STEP 7  display-scaled clicks map back to original pixels")
    scaler = P.DisplayScaler(image.shape, 640, 400)
    disp_corners = [scaler.to_display(x, y) for x, y in corners_true]
    recovered = np.array([scaler.to_original(x, y) for x, y in disp_corners])
    scale_err = float(np.max(np.abs(recovered - corners_true)))
    passed = scale_err < 1e-9
    failures += [] if passed else ["display scaling round trip failed"]
    print(f"        {image.shape[1]}x{image.shape[0]} shown at "
          f"{scaler.display_size[0]}x{scaler.display_size[1]} (scale {scaler.scale:.4f})")
    print(f"        max error converting clicks back: {scale_err:.2e} px  [{ok(passed)}]\n")

    # ------------------------------------------------- 8. render outputs
    print("STEP 8  render and save the top-down view and the diagram")
    rect = P.rectify(image, calib, P.DEFAULT_PPF)
    exp_w, exp_h = P.rectified_size(P.DEFAULT_PPF)
    passed = rect.shape[1] == exp_w and rect.shape[0] == exp_h
    failures += [] if passed else ["rectified image has the wrong size"]
    print(f"        rectified image {rect.shape[1]}x{rect.shape[0]} px, aspect "
          f"{rect.shape[1] / rect.shape[0]:.4f} (20/44 = {20 / 44:.4f})  [{ok(passed)}]")
    if k1:
        print("        the rectified court will still show curved lines: warping cannot")
        print("        undo lens distortion, only the perspective.")

    diagram, ppf, margin = P.draw_court_diagram(P.DEFAULT_PPF)
    diagram = P.draw_markers_on_diagram(diagram, players_ft, player_labels, ppf, margin)
    rect_path = os.path.join(out_dir, "rectified.png")
    diag_path = os.path.join(out_dir, "court_diagram.png")
    cv2.imwrite(rect_path, rect)
    cv2.imwrite(diag_path, diagram)

    probe_ft = [10.0, 22.0]
    probe_px = P.feet_to_output_px([probe_ft], P.DEFAULT_PPF)[0]
    print(f"        court ({probe_ft[0]:.0f}, {probe_ft[1]:.0f}) ft is rectified pixel "
          f"({probe_px[0]:.0f}, {probe_px[1]:.0f}) at {P.DEFAULT_PPF:g} px/ft\n")

    for path in (val_csv, players_csv, calib_path, rect_path, diag_path):
        print(f"        wrote {path}")

    # ------------------------------------------------- result
    print("\n" + "=" * 76)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for f_ in failures:
            print(f"  - {f_}")
        print("=" * 76)
        return 1

    if exact:
        print("RESULT: all structural and exactness checks passed.")
        print("This verifies the CODE. It is not a measurement of accuracy on a real")
        print("photograph. For realistic errors, try:")
        print("  python generate_demo.py --distortion")
        print("  python verify_demo.py --image demo/demo_angled_distorted.png")
    else:
        print("RESULT: all structural checks passed; accuracy reported above.")
        print(f"        mean {summary['mean_ft']:.3f} ft, RMSE {summary['rmse_ft']:.3f} ft "
              f"over {summary['n']} independent landmarks.")
        print("These are synthetic errors from a KNOWN distortion and/or simulated click")
        print("noise. A real photograph adds unknown distortion and no ground truth.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
