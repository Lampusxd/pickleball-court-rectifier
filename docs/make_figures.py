"""
docs/make_figures.py -- regenerate the images embedded in README.md.

Every figure in the README comes from actually running the project, not from a
drawing tool, so it can be reproduced from scratch:

    python generate_demo.py --all
    python docs/make_figures.py

Writes into docs/images/.  Uses the DISTORTED demo image on purpose: the
reprojection errors are visible there, which is what a reader should see.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as M          # noqa: E402
import perspective as P   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(REPO, "docs", "images")
DEMO = os.path.join(REPO, "demo")

EXACT_IMG = os.path.join(DEMO, "demo_angled.png")
DIST_IMG = os.path.join(DEMO, "demo_angled_distorted.png")
DIST_TRUTH = os.path.join(DEMO, "demo_angled_distorted_ground_truth.json")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def need(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} is missing. Run:  python generate_demo.py --all")
    return path


def build_app():
    """A CourtMapper calibrated on the distorted demo, with players and landmarks."""
    with open(need(DIST_TRUTH), encoding="utf-8") as f:
        truth = json.load(f)
    image = M.load_image(need(DIST_IMG))

    app = M.CourtMapper(image, DIST_IMG, os.path.join(REPO, "output", "figures"),
                        os.path.join(REPO, "output", "figures", "calibration.json"))
    for x, y in truth["corners_image"]:
        app.on_mouse(cv2.EVENT_LBUTTONDOWN, int(round(x)), int(round(y)), 0, None)
    assert app.calib is not None, "calibration failed while building figures"

    for pl in truth["players"]:
        app.on_mouse(cv2.EVENT_LBUTTONDOWN,
                     int(round(pl["image_x"])), int(round(pl["image_y"])), 0, None)

    # validation landmarks, added directly (the interactive path asks the terminal)
    for lm in truth["landmarks"][:6]:
        pt = (lm["image_x"], lm["image_y"])
        measured = app.calib.image_to_court([pt])[0]
        true_ft = (lm["x_ft"], lm["y_ft"])
        app.landmarks.append({
            "label": lm["label"], "image": pt, "measured": measured, "truth": true_ft,
            "error_ft": float(np.linalg.norm(measured - np.asarray(true_ft))),
        })
    app.message = "player mode: click each player's ground-contact point"
    return app, truth


def caption(img, text, height=34):
    """Add a caption bar under an image."""
    bar = np.full((height, img.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 11), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def fit_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), height),
                      interpolation=cv2.INTER_AREA)


def save(name, img, max_width=1100):
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    path = os.path.join(IMAGES, name)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"  {os.path.relpath(path, REPO)}  ({img.shape[1]}x{img.shape[0]}, "
          f"{os.path.getsize(path) / 1024:.0f} KB)")


def main():
    os.makedirs(IMAGES, exist_ok=True)
    app, truth = build_app()
    print("writing figures:")

    # 1. the overlay on the original photo
    overlay = app.render_main()
    save("overlay.png", overlay)

    # 2. the rectified top-down view
    save("rectified.png", app.rectified)

    # 3. the court diagram with player markers
    save("diagram.png", app.render_diagram())

    # 4. the three-panel workflow strip used at the top of the README
    h = 430
    panels = [
        caption(fit_height(overlay, h), "1. angled image + clicked corners"),
        caption(fit_height(app.rectified, h), "2. warpPerspective"),
        caption(fit_height(app.render_diagram(), h), "3. positions in feet"),
    ]
    gap = np.full((panels[0].shape[0], 12, 3), (28, 28, 28), dtype=np.uint8)
    save("workflow.png", np.hstack([panels[0], gap, panels[1], gap, panels[2]]))

    # 5. exact vs distorted, side by side, with the measured error on each
    exact = M.load_image(need(EXACT_IMG))
    dist = M.load_image(need(DIST_IMG))
    left = caption(fit_height(exact, 420),
                   "demo_angled.png  --  ideal pinhole, mean error 1.5e-14 ft")
    right = caption(fit_height(dist, 420),
                    "demo_angled_distorted.png  --  known lens, mean error 0.238 ft")
    save("distortion_comparison.png", np.hstack([left, gap[:left.shape[0]], right]))

    # 6. a zoom on the reprojected lines, where the distortion error is visible
    crop = overlay[150:520, 150:1130]
    zoom = cv2.resize(crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    save("reprojection.png", caption(
        zoom, "green lines are drawn FROM court feet: where they miss the paint is the error"))

    print("\ndone. These are embedded in README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
