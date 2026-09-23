"""
generate_demo.py -- build synthetic pickleball courts and angled "photos" of them.

Why this exists: to learn the workflow you need an angled court image, and to
CHECK your work you need to know the right answer.  A real photograph gives you
the first but not the second.  So this script

  1. draws a true top-down court (where feet -> pixels is an exact scale),
  2. warps it with a KNOWN homography into an angled, camera-like view,
  3. optionally bends it with a KNOWN radial lens distortion,
  4. writes a ground-truth JSON recording exactly where every court landmark and
     every planted "player" ended up in the final image.

Two kinds of demo image come out of this, and they teach different things:

  EXACT (no distortion)     -- a perfect pinhole view.  A homography fits it
                               perfectly, so any error you measure is
                               floating-point noise.  Use it to prove the CODE
                               is right.
  DISTORTED (--distortion)  -- a lens bends the straight lines, exactly as a
                               phone or action camera does.  A homography
                               CANNOT fit it perfectly, so validation errors
                               become real, in tenths of a foot.  Use it to
                               practise measuring and explaining accuracy.

Neither is a real photograph.  Both have perfect knowledge of the truth, which
no real photo ever gives you.  Say which one you used when you report numbers.

Examples:
    python generate_demo.py
    python generate_demo.py --view low
    python generate_demo.py --view corner --distortion 0.18
    python generate_demo.py --all
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import perspective as P

# Layout of the synthetic top-down source image ---------------------------
SRC_PPF = 14.0          # pixels per foot in the top-down source image
SRC_MARGIN_FT = 8.0     # surrounding apron drawn around the court

# The angled "photograph" we produce
OUT_WIDTH = 1280
OUT_HEIGHT = 800

# Where the four court corners land in the angled image.  These four points ARE
# the known homography: any convex quadrilateral here defines a perspective
# view.  Order matches P.CORNER_LABELS: far-left, far-right, near-right,
# near-left -- clockwise on screen.
VIEWS = {
    # a typical spectator view from behind and above the near baseline
    "default": [[472.0, 232.0], [846.0, 226.0], [1128.0, 706.0], [214.0, 724.0]],
    # camera low and close: the far end compresses into very few pixels
    "low": [[540.0, 306.0], [744.0, 302.0], [1246.0, 772.0], [36.0, 784.0]],
    # camera high on a pole: much closer to a top-down view already
    "high": [[330.0, 122.0], [950.0, 118.0], [1082.0, 758.0], [198.0, 764.0]],
    # camera off to one side, behind the near-left corner
    "corner": [[286.0, 268.0], [742.0, 196.0], [1232.0, 548.0], [402.0, 768.0]],
    # a strong oblique angle, the hardest case here
    "oblique": [[252.0, 318.0], [826.0, 206.0], [1214.0, 512.0], [386.0, 754.0]],
}

# Players planted at known court positions (x_ft, y_ft), ground contact point.
DEMO_PLAYERS = (
    ("P1", 5.0, 9.0),
    ("P2", 15.5, 12.5),
    ("P3", 6.5, 32.0),
    ("P4", 14.0, 36.5),
)

# Default radial distortion strength used by --distortion with no value.
DEFAULT_K1 = 0.16


def _px(x_ft, y_ft, ppf=SRC_PPF, margin_ft=SRC_MARGIN_FT):
    """Court feet -> top-down source-image pixels (an exact scale + offset)."""
    p = P.feet_to_output_px([[x_ft, y_ft]], ppf, margin_ft)[0]
    return (int(round(p[0])), int(round(p[1])))


# --------------------------------------------------------------------------
# The synthetic court
# --------------------------------------------------------------------------
def draw_topdown_court(ppf=SRC_PPF, margin_ft=SRC_MARGIN_FT, seed=7):
    """Draw a plain top-down court with a little texture, so the warp looks real."""
    rng = np.random.default_rng(seed)
    w = int(round((P.COURT_WIDTH_FT + 2 * margin_ft) * ppf))
    h = int(round((P.COURT_LENGTH_FT + 2 * margin_ft) * ppf))

    img = np.full((h, w, 3), (78, 96, 72), dtype=np.uint8)          # surrounding apron
    cv2.rectangle(img, _px(0, 0, ppf, margin_ft),
                  _px(P.COURT_WIDTH_FT, P.COURT_LENGTH_FT, ppf, margin_ft),
                  (128, 92, 48), -1)                                 # blue-ish court paint

    # faint noise so the warped image is not perfectly flat colour
    noise = rng.normal(0.0, 4.0, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    line_w = max(1, int(round(0.2 * ppf)))
    white = (245, 245, 245)

    # boundary, kitchen lines, and the two service centerlines
    cv2.rectangle(img, _px(0, 0, ppf, margin_ft),
                  _px(P.COURT_WIDTH_FT, P.COURT_LENGTH_FT, ppf, margin_ft), white, line_w)
    for y in (P.KITCHEN_FAR_Y_FT, P.KITCHEN_NEAR_Y_FT):
        cv2.line(img, _px(0, y, ppf, margin_ft),
                 _px(P.COURT_WIDTH_FT, y, ppf, margin_ft), white, line_w)
    cv2.line(img, _px(P.CENTERLINE_X_FT, 0, ppf, margin_ft),
             _px(P.CENTERLINE_X_FT, P.KITCHEN_FAR_Y_FT, ppf, margin_ft), white, line_w)
    cv2.line(img, _px(P.CENTERLINE_X_FT, P.KITCHEN_NEAR_Y_FT, ppf, margin_ft),
             _px(P.CENTERLINE_X_FT, P.COURT_LENGTH_FT, ppf, margin_ft), white, line_w)

    # the net, as a band across y = 22
    cv2.line(img, _px(-1.5, P.NET_Y_FT, ppf, margin_ft),
             _px(P.COURT_WIDTH_FT + 1.5, P.NET_Y_FT, ppf, margin_ft), (45, 45, 45), max(2, line_w))

    # Players are drawn as flat ground discs, NOT standing figures.  The whole
    # method assumes points lie on the court plane, and a drawn body would warp
    # as if it were painted on the ground, which would be misleading.
    for name, x, y in DEMO_PLAYERS:
        c = _px(x, y, ppf, margin_ft)
        cv2.circle(img, c, int(0.8 * ppf), (30, 30, 210), -1)
        cv2.circle(img, c, int(0.8 * ppf), (255, 255, 255), 1)
        cv2.putText(img, name, (c[0] + int(ppf), c[1] - int(0.4 * ppf)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_angled_view(topdown, ppf=SRC_PPF, margin_ft=SRC_MARGIN_FT,
                      out_size=(OUT_WIDTH, OUT_HEIGHT), corners=None):
    """Warp the top-down court into an angled camera-like view.

    Returns (angled_image, H_feet_to_image).  Two homographies are composed:
    feet -> top-down pixels (a plain scale) and top-down pixels -> angled pixels
    (the perspective we invent here).  Composing them gives exact ground truth
    for any court coordinate.
    """
    if corners is None:
        corners = VIEWS["default"]
    corners = np.asarray(corners, dtype=np.float64)

    src = np.array([_px(x, y, ppf, margin_ft) for x, y in P.court_corners_ft()],
                   dtype=np.float32)
    dst = corners.astype(np.float32)
    H_td_to_angled = cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    angled = cv2.warpPerspective(
        topdown, H_td_to_angled, out_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(35, 45, 40),
    )

    # Anything above the far baseline in a real photo would be background, not
    # more court.  Painting a simple backdrop avoids the mirrored smear a
    # perspective warp produces beyond the horizon line.
    horizon = max(0, int(min(corners[0][1], corners[1][1])) - 60)
    angled[:horizon] = (120, 105, 85)
    cv2.rectangle(angled, (0, horizon), (out_size[0], horizon + 12), (90, 80, 65), -1)

    H_feet_to_image = H_td_to_angled @ P.pixel_scale_matrix(ppf, margin_ft)
    return angled, H_feet_to_image


# --------------------------------------------------------------------------
# Radial lens distortion (the thing that makes a homography imperfect)
# --------------------------------------------------------------------------
def distortion_frame(image_shape):
    """Centre and radius used to normalise pixel coordinates for distortion."""
    h, w = image_shape[:2]
    centre = np.array([w / 2.0, h / 2.0])
    radius = 0.5 * float(np.hypot(w, h))   # r ~= 1 at the image corners
    return centre, radius


def undistorted_from_distorted(points_d, k1, k2, centre, radius):
    """Distorted pixel -> ideal (pinhole) pixel.

    This is the direction cv2.remap needs: for each pixel of the OUTPUT
    (distorted) image it asks where to sample in the input (ideal) image.
    """
    p = (np.asarray(points_d, dtype=np.float64) - centre) / radius
    r2 = np.sum(p * p, axis=-1, keepdims=True)
    return centre + p * (1.0 + k1 * r2 + k2 * r2 * r2) * radius


def max_invertible_ideal_radius(k1, k2=0.0):
    """Largest ideal (normalised) radius this distortion can represent.

    The radial map s -> s*(1 + k1*s^2 + k2*s^4) is only invertible while it is
    increasing.  With a negative k1 (barrel distortion) it turns over at
    s* = sqrt(-1 / (3*k1)), and every ideal radius beyond its value there folds
    back -- no distorted pixel corresponds to it.  Positive coefficients never
    turn over, so the limit is infinite.
    """
    if k2 == 0.0:
        if k1 >= 0.0:
            return float("inf")
        s_star = float(np.sqrt(-1.0 / (3.0 * k1)))
        return float(s_star + k1 * s_star ** 3)
    s = np.linspace(0.0, 8.0, 80001)[1:]           # general case: scan for the turn
    turning = np.nonzero(1.0 + 3.0 * k1 * s ** 2 + 5.0 * k2 * s ** 4 <= 0.0)[0]
    if turning.size == 0:
        return float("inf")
    s_star = float(s[turning[0]])
    return float(s_star + k1 * s_star ** 3 + k2 * s_star ** 5)


def distorted_from_undistorted(points_i, k1, k2, centre, radius, iterations=25):
    """Ideal pixel -> distorted pixel, by inverting the polynomial numerically.

    There is no closed form.  The distortion is purely radial, so it never
    changes a point's direction from the centre -- only its distance.  That
    turns the 2-D inverse into one scalar equation per point:

        find s  such that  s * (1 + k1*s^2 + k2*s^4) = t

    where t is the ideal normalised radius and s the distorted one.  Newton's
    method solves it to machine precision in a handful of steps, for positive
    (pincushion) and negative (barrel) coefficients alike.  This is how we know
    exactly where each landmark ends up in the distorted image.
    """
    p_i = (np.asarray(points_i, dtype=np.float64).reshape(-1, 2) - centre) / radius
    t = np.linalg.norm(p_i, axis=1)

    t_max = max_invertible_ideal_radius(k1, k2)
    if np.any(t > t_max):
        n = int(np.sum(t > t_max))
        raise ValueError(
            f"radial distortion k1={k1:g}, k2={k2:g} folds beyond normalised radius "
            f"{t_max:.4f}, and {n} of {t.size} point(s) lie past it (max {t.max():.4f}). "
            f"No distorted pixel corresponds to them. Use a smaller |k1|, or a "
            f"positive k1, or keep the points nearer the image centre."
        )

    s = t.copy()
    for _ in range(iterations):
        g = s + k1 * s ** 3 + k2 * s ** 5 - t
        dg = 1.0 + 3.0 * k1 * s ** 2 + 5.0 * k2 * s ** 4
        dg = np.where(np.abs(dg) < 1e-12, 1e-12, dg)     # never divide by ~0
        s = s - g / dg
    ratio = np.where(t > 1e-12, s / np.where(t > 1e-12, t, 1.0), 1.0)
    return centre + p_i * ratio[:, None] * radius


def apply_distortion(image, k1, k2=0.0):
    """Bend an ideal pinhole image with radial distortion, as a lens would."""
    h, w = image.shape[:2]
    centre, radius = distortion_frame(image.shape)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    src = undistorted_from_distorted(grid, k1, k2, centre, radius).reshape(h, w, 2)
    return cv2.remap(image, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def ground_truth(H_feet_to_image, view="default", k1=0.0, k2=0.0,
                 image_shape=(OUT_HEIGHT, OUT_WIDTH, 3)):
    """Exact image positions of corners, landmarks and players.

    When k1 != 0 the ideal positions are pushed through the same distortion the
    image received, so the recorded pixels are where the features really are in
    the file you can open.
    """
    centre, radius = distortion_frame(image_shape)
    distorted = bool(k1 or k2)

    def to_image(points_ft):
        px = P.apply_homography(H_feet_to_image, points_ft)
        if distorted:
            px = distorted_from_undistorted(px, k1, k2, centre, radius)
        return px

    corners_px = to_image(P.court_corners_ft())

    landmarks = []
    for label, x, y in P.VALIDATION_LANDMARKS:
        px = to_image([[x, y]])[0]
        landmarks.append({"label": label, "x_ft": x, "y_ft": y,
                          "image_x": float(px[0]), "image_y": float(px[1])})

    players = []
    for name, x, y in DEMO_PLAYERS:
        px = to_image([[x, y]])[0]
        players.append({"label": name, "x_ft": x, "y_ft": y,
                        "image_x": float(px[0]), "image_y": float(px[1])})

    if distorted:
        note = (
            "This image was bent by a KNOWN radial distortion, so straight court "
            "lines are curved. A single homography cannot fit a curved image "
            "exactly, so landmark errors here are REAL (tenths of a foot), not "
            "floating-point noise. This is still synthetic: a real photograph "
            "adds unknown distortion, imprecise clicks and an imperfectly flat "
            "court, and gives you no ground truth at all."
        )
    else:
        note = (
            "This image was made by warping a true top-down court with the known "
            "homography below, with no lens distortion, so these positions are "
            "exact and a homography fits them to floating-point precision. Use it "
            "to verify the code, not to claim accuracy on a real photograph."
        )

    return {
        "description": note,
        "view": view,
        "exact": not distorted,
        "distortion_k1": float(k1),
        "distortion_k2": float(k2),
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "corner_order": list(P.CORNER_LABELS),
        "corners_image": [[float(x), float(y)] for x, y in corners_px],
        "H_feet_to_image": [[float(v) for v in row] for row in H_feet_to_image],
        "H_note": ("maps court feet to IDEAL pinhole pixels; when distortion_k1 is "
                   "non-zero the recorded corner/landmark pixels have the distortion "
                   "applied on top, so they will NOT match this matrix exactly"),
        "landmarks": landmarks,
        "players": players,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def base_name(view, k1):
    """demo_angled for the default exact view; descriptive names otherwise."""
    name = "demo_angled" if view == "default" else f"demo_{view}"
    return name + ("_distorted" if k1 else "")


def build_one(view, k1, k2, out_dir, topdown=None, quiet=False):
    """Generate one demo image plus its ground truth.  Returns (img_path, gt_path)."""
    if topdown is None:
        topdown = draw_topdown_court()
    angled, H_feet_to_image = build_angled_view(topdown, corners=VIEWS[view])
    if k1 or k2:
        angled = apply_distortion(angled, k1, k2)
    truth = ground_truth(H_feet_to_image, view, k1, k2, angled.shape)

    stem = base_name(view, k1)
    img_path = os.path.join(out_dir, stem + ".png")
    gt_path = os.path.join(out_dir, stem + "_ground_truth.json")
    cv2.imwrite(img_path, angled)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    if not quiet:
        kind = "EXACT pinhole" if truth["exact"] else f"DISTORTED k1={k1:g}"
        print(f"wrote {img_path:<42} ({angled.shape[1]}x{angled.shape[0]}, "
              f"view '{view}', {kind})")
        print(f"      {gt_path}")
    return img_path, gt_path, truth


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate synthetic demo court images with known ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="views: " + ", ".join(VIEWS),
    )
    ap.add_argument("--out-dir", default="demo", help="where to write the files (default: demo)")
    ap.add_argument("--view", default="default", choices=sorted(VIEWS),
                    help="camera viewpoint (default: default)")
    ap.add_argument("--distortion", nargs="?", type=float, const=DEFAULT_K1, default=0.0,
                    metavar="K1",
                    help=f"add radial lens distortion (bare flag = {DEFAULT_K1:g}); "
                         f"try 0.08 for mild, 0.25 for an action camera")
    ap.add_argument("--k2", type=float, default=0.0, help="second distortion coefficient")
    ap.add_argument("--all", action="store_true",
                    help="generate every view, exact and distorted (12 files)")
    ap.add_argument("--show", action="store_true", help="display the result in a window")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    topdown = draw_topdown_court()
    td_path = os.path.join(args.out_dir, "demo_topdown.png")
    cv2.imwrite(td_path, topdown)
    print(f"wrote {td_path:<42} ({topdown.shape[1]}x{topdown.shape[0]}, true top-down source)")

    if args.all:
        for view in sorted(VIEWS):
            for k1 in (0.0, DEFAULT_K1):
                build_one(view, k1, 0.0, args.out_dir, topdown)
        print()
        print("\nAll views generated. Compare 'low' with 'high' to see how much the "
              "camera angle\nmatters, and any *_distorted.png with its exact twin to "
              "see what a lens does.")
        print("\nNext:  python verify_demo.py --image demo/demo_angled_distorted.png")
        return 0

    try:
        img_path, gt_path, truth = build_one(args.view, args.distortion, args.k2,
                                             args.out_dir, topdown)
    except ValueError as exc:
        # strong barrel distortion can fold part of the court out of existence
        raise SystemExit(f"ERROR: {exc}")

    print("\nCourt corners in the image (far-left, far-right, near-right, near-left):")
    for label, (x, y) in zip(P.CORNER_LABELS, truth["corners_image"]):
        print(f"  {label:<10} ({x:8.2f}, {y:8.2f})")

    if truth["exact"]:
        print("\nThis view has NO lens distortion: a homography will fit it to ~1e-14 ft.")
        print("For realistic, non-zero errors, add --distortion.")
    else:
        print(f"\nThis view is distorted (k1={args.distortion:g}): the court lines are "
              f"curved, so a\nsingle homography cannot fit it exactly and validation "
              f"errors will be real.")

    print(f"\nNext:  python verify_demo.py --image {img_path}")
    print(f"  or:  python main.py {img_path}   (interactive)")

    if args.show:
        cv2.imshow(os.path.basename(img_path), cv2.imread(img_path))
        print("\nPress any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
