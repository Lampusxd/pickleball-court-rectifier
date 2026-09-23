"""
perspective.py -- geometry for the pickleball court rectifier.

Everything in this file is pure computation: no OpenCV windows, no mouse
callbacks.  That separation is deliberate, so the math can be unit-tested and
run on a machine with no display (see verify_demo.py and tests/).

THREE COORDINATE SYSTEMS ARE USED.  Keeping them apart is most of the work:

  1. IMAGE PIXELS   (x right, y down) -- pixels of the original photograph.
  2. COURT FEET     (x right, y down-court) -- physical positions on the court
                    plane.  The far-left outside corner is (0, 0), x grows
                    toward the right sideline (0 .. 20 ft), and y grows toward
                    the near baseline (0 .. 44 ft).  "far/near" and
                    "left/right" are as the court appears in the photograph,
                    not any official end of a real court.
  3. OUTPUT PIXELS  (x right, y down) -- pixels of a rendered top-down image.
                    These are COURT FEET multiplied by a pixels-per-foot scale
                    (plus a margin, for the diagram).

The homography we solve for maps 1 -> 2.  Rendering (2 -> 3) is a separate,
trivial scale, and it is kept separate on purpose: changing the rendering
resolution must never change a reported player coordinate in feet.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Court constants (USA Pickleball dimensions, measured to the outside of lines)
# --------------------------------------------------------------------------
COURT_WIDTH_FT = 20.0          # sideline to sideline
COURT_LENGTH_FT = 44.0         # baseline to baseline
NET_Y_FT = 22.0                # the net bisects the 44 ft length
KITCHEN_FAR_Y_FT = 15.0        # non-volley-zone line on the far side
KITCHEN_NEAR_Y_FT = 29.0       # non-volley-zone line on the near side
CENTERLINE_X_FT = 10.0         # service centerline
DEFAULT_PPF = 20.0             # pixels per foot for rendered top-down images

# The four outside corners must be clicked in this order.  The order matters:
# cv2.getPerspectiveTransform pairs src[i] with dst[i], so a different click
# order silently produces a rotated or mirrored court.
CORNER_LABELS = ("far-left", "far-right", "near-right", "near-left")

# Landmarks that are NOT used to fit the homography, and are therefore valid
# independent evidence of accuracy.  (label, x_ft, y_ft)
VALIDATION_LANDMARKS = (
    ("far kitchen line / left sideline", 0.0, KITCHEN_FAR_Y_FT),
    ("far kitchen line / right sideline", 20.0, KITCHEN_FAR_Y_FT),
    ("near kitchen line / left sideline", 0.0, KITCHEN_NEAR_Y_FT),
    ("near kitchen line / right sideline", 20.0, KITCHEN_NEAR_Y_FT),
    ("far kitchen line / centerline", 10.0, KITCHEN_FAR_Y_FT),
    ("near kitchen line / centerline", 10.0, KITCHEN_NEAR_Y_FT),
    ("far baseline / centerline", 10.0, 0.0),
    ("near baseline / centerline", 10.0, COURT_LENGTH_FT),
    ("net line / left sideline", 0.0, NET_Y_FT),
    ("net line / right sideline", 20.0, NET_Y_FT),
)

MIN_CORNER_SEPARATION_PX = 10.0   # two clicks closer than this are "duplicates"
MIN_QUAD_AREA_FRACTION = 0.005    # quad must cover >= 0.5% of the image


class CalibrationError(Exception):
    """Raised for bad corner selections or unusable calibration files."""


# --------------------------------------------------------------------------
# Corner geometry and validation
# --------------------------------------------------------------------------
def court_corners_ft():
    """The four court corners in feet, in CORNER_LABELS order."""
    return np.array(
        [
            [0.0, 0.0],                          # far-left
            [COURT_WIDTH_FT, 0.0],               # far-right
            [COURT_WIDTH_FT, COURT_LENGTH_FT],   # near-right
            [0.0, COURT_LENGTH_FT],              # near-left
        ],
        dtype=np.float64,
    )


def signed_area(points):
    """Shoelace signed area of a polygon.

    In image coordinates y points DOWN, so a polygon listed clockwise on screen
    (far-left, far-right, near-right, near-left) has a POSITIVE signed area
    here.  A negative value means the corners were clicked in the opposite
    order, which would mirror the court.
    """
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def validate_corners(points, image_shape=None):
    """Return a list of human-readable problems with a 4-corner selection.

    An empty list means the selection is usable.  Checks, in order: duplicate
    or near-duplicate clicks, near-zero area, collinearity, non-convex or
    self-intersecting ("bow-tie") quadrilaterals, and reversed click order.
    """
    problems = []
    p = np.asarray(points, dtype=np.float64)

    if p.ndim != 2 or p.shape != (4, 2):
        return [f"need exactly 4 points, got shape {p.shape}"]
    if not np.all(np.isfinite(p)):
        return ["points contain non-finite values"]

    # 1. duplicate / near-duplicate clicks
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(p[i] - p[j]))
            if d < MIN_CORNER_SEPARATION_PX:
                problems.append(
                    f"points {i + 1} ({CORNER_LABELS[i]}) and {j + 1} "
                    f"({CORNER_LABELS[j]}) are only {d:.1f} px apart -- duplicate click?"
                )

    area = signed_area(p)

    # 2. degenerate / near-zero area (all four corners nearly collinear)
    min_area = 1000.0
    if image_shape is not None:
        h, w = image_shape[:2]
        min_area = max(min_area, MIN_QUAD_AREA_FRACTION * w * h)
    if abs(area) < min_area:
        problems.append(
            f"quadrilateral area is {abs(area):.0f} px^2, below the minimum "
            f"{min_area:.0f} px^2 -- the points are nearly collinear or far too "
            f"close together to define a stable homography"
        )

    # 3. convexity.  For a simple convex polygon every consecutive edge pair
    #    turns the same way, so all cross products share a sign.  A self-
    #    intersecting bow-tie produces mixed signs, so this catches both cases.
    crosses = []
    for i in range(4):
        v1 = p[(i + 1) % 4] - p[i]
        v2 = p[(i + 2) % 4] - p[(i + 1) % 4]
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
    crosses = np.array(crosses)
    if np.any(np.abs(crosses) < 1e-9):
        problems.append("three of the points are collinear -- the quadrilateral is degenerate")
    elif not (np.all(crosses > 0) or np.all(crosses < 0)):
        problems.append(
            "the quadrilateral is non-convex or self-intersecting (its edges cross) -- "
            "click the corners in order around the court, not diagonally"
        )
    # 4. click order.  Only meaningful once we know the shape is convex.
    elif area < 0:
        problems.append(
            "the corners appear to be in reverse order (counter-clockwise on screen). "
            "Click far-left, far-right, near-right, near-left, going around the court"
        )

    return problems


# --------------------------------------------------------------------------
# The homography itself
# --------------------------------------------------------------------------
def compute_homography(image_corners):
    """Solve for the 3x3 matrix mapping IMAGE PIXELS -> COURT FEET.

    cv2.getPerspectiveTransform takes exactly 4 point pairs and solves for the
    8 degrees of freedom of a homography (the 9th entry is fixed by scale,
    H[2, 2] = 1).  src[i] must correspond to dst[i], which is why the corner
    click order is fixed.
    """
    src = np.asarray(image_corners, dtype=np.float32).reshape(4, 2)
    dst = court_corners_ft().astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise CalibrationError("the four points produced a singular (non-invertible) homography")
    return np.asarray(H, dtype=np.float64)


def apply_homography(H, points):
    """Apply a 3x3 homography to an (N, 2) array of points.

    cv2.perspectiveTransform does the homogeneous-coordinate bookkeeping: it
    treats each (x, y) as (x, y, 1), multiplies by H, then divides the result
    by its third component w.  That final division by w is exactly what makes
    the mapping perspective rather than affine.  It expects an (N, 1, 2) array.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if p.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    out = cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64))
    return out.reshape(-1, 2)


def is_out_of_bounds(point_ft, tol_ft=1e-6):
    """True if a court coordinate lies outside the 20 x 44 ft playing surface.

    The default tolerance is one micro-foot: it only absorbs floating-point
    noise, so a point computed as exactly on a line is reported as in.  It is
    far too small to hide a real out-of-court position.  Note this flags the
    point; nothing in this project ever clamps a coordinate to the boundary.
    """
    x, y = float(point_ft[0]), float(point_ft[1])
    return (
        x < -tol_ft
        or x > COURT_WIDTH_FT + tol_ft
        or y < -tol_ft
        or y > COURT_LENGTH_FT + tol_ft
    )


# --------------------------------------------------------------------------
# Calibration: hold it, save it, load it
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    """Everything needed to turn image pixels of ONE photo into court feet."""

    image_width: int
    image_height: int
    image_corners: np.ndarray            # (4, 2) float, CORNER_LABELS order
    H_image_to_court: np.ndarray         # (3, 3) float64
    court_width_ft: float = COURT_WIDTH_FT
    court_length_ft: float = COURT_LENGTH_FT
    source_image: str = ""
    created_utc: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_corners(cls, image_corners, image_shape, source_image=""):
        problems = validate_corners(image_corners, image_shape)
        if problems:
            raise CalibrationError("; ".join(problems))
        h, w = image_shape[:2]
        corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
        return cls(
            image_width=int(w),
            image_height=int(h),
            image_corners=corners,
            H_image_to_court=compute_homography(corners),
            source_image=os.path.basename(source_image),
        )

    # -- derived matrices --------------------------------------------------
    @property
    def H_court_to_image(self):
        """Inverse mapping, COURT FEET -> IMAGE PIXELS (used for drawing)."""
        return np.linalg.inv(self.H_image_to_court)

    # -- point mapping -----------------------------------------------------
    def image_to_court(self, points):
        """Image pixels -> court feet.  Valid only for points ON the court plane."""
        return apply_homography(self.H_image_to_court, points)

    def court_to_image(self, points_ft):
        """Court feet -> image pixels."""
        return apply_homography(self.H_court_to_image, points_ft)

    # -- persistence -------------------------------------------------------
    def to_dict(self):
        return {
            "format": "pickleball-court-calibration",
            "version": 1,
            "created_utc": self.created_utc,
            "source_image": self.source_image,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "court_width_ft": float(self.court_width_ft),
            "court_length_ft": float(self.court_length_ft),
            "corner_order": list(CORNER_LABELS),
            "image_corners": [[float(x), float(y)] for x, y in self.image_corners],
            "court_corners_ft": [[float(x), float(y)] for x, y in court_corners_ft()],
            "H_image_to_court": [[float(v) for v in row] for row in self.H_image_to_court],
            "note": (
                "H maps image pixels to court feet using homogeneous coordinates. "
                "Matching image dimensions do NOT prove this calibration fits another "
                "photo: the camera position, orientation, zoom and crop must match too."
            ),
        }

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        required = ("image_width", "image_height", "image_corners", "H_image_to_court")
        missing = [k for k in required if k not in d]
        if missing:
            raise CalibrationError(f"calibration file is missing key(s): {', '.join(missing)}")
        H = np.asarray(d["H_image_to_court"], dtype=np.float64)
        corners = np.asarray(d["image_corners"], dtype=np.float64)
        if H.shape != (3, 3):
            raise CalibrationError(f"H_image_to_court must be 3x3, got {H.shape}")
        if corners.shape != (4, 2):
            raise CalibrationError(f"image_corners must be 4x2, got {corners.shape}")
        if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            raise CalibrationError("stored homography is singular or contains non-finite values")
        return cls(
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            image_corners=corners,
            H_image_to_court=H,
            court_width_ft=float(d.get("court_width_ft", COURT_WIDTH_FT)),
            court_length_ft=float(d.get("court_length_ft", COURT_LENGTH_FT)),
            source_image=str(d.get("source_image", "")),
            created_utc=str(d.get("created_utc", "")),
        )

    @classmethod
    def load(cls, path, image_shape=None):
        """Load calibration JSON, optionally checking it against an image.

        The dimension check is a guard rail, not a guarantee: two photos of the
        same size taken from different camera positions will both pass it, and
        only one of them is actually calibrated.
        """
        if not os.path.exists(path):
            raise CalibrationError(f"calibration file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"calibration file is not valid JSON: {exc}") from exc
        calib = cls.from_dict(data)
        if image_shape is not None:
            h, w = image_shape[:2]
            if (int(w), int(h)) != (calib.image_width, calib.image_height):
                raise CalibrationError(
                    f"image is {w}x{h} but this calibration was made for "
                    f"{calib.image_width}x{calib.image_height}. Re-calibrate for this image."
                )
        return calib


# --------------------------------------------------------------------------
# Rendering: court feet -> output pixels
# --------------------------------------------------------------------------
def pixel_scale_matrix(ppf=DEFAULT_PPF, margin_ft=0.0):
    """3x3 matrix mapping COURT FEET -> OUTPUT PIXELS (scale + margin offset)."""
    return np.array(
        [[ppf, 0.0, margin_ft * ppf],
         [0.0, ppf, margin_ft * ppf],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def feet_to_output_px(points_ft, ppf=DEFAULT_PPF, margin_ft=0.0):
    p = np.asarray(points_ft, dtype=np.float64).reshape(-1, 2)
    return p * ppf + margin_ft * ppf


def output_px_to_feet(points_px, ppf=DEFAULT_PPF, margin_ft=0.0):
    p = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    return (p - margin_ft * ppf) / ppf


def rectified_size(ppf=DEFAULT_PPF):
    """(width, height) in pixels of the rectified image -- a true 20:44 ratio."""
    return (int(round(COURT_WIDTH_FT * ppf)), int(round(COURT_LENGTH_FT * ppf)))


def rectify(image, calib, ppf=DEFAULT_PPF):
    """Warp the whole photo into a top-down image with a true 20:44 aspect ratio.

    cv2.warpPerspective needs a matrix from image pixels to OUTPUT pixels, so we
    compose the calibrated image->feet homography with the feet->pixels scale.
    Matrix order matters: the scale is applied after H, so it goes on the left.
    """
    M = pixel_scale_matrix(ppf) @ calib.H_image_to_court
    return cv2.warpPerspective(image, M, rectified_size(ppf), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# A clean, synthetic court diagram (drawn, not warped from the photo)
# --------------------------------------------------------------------------
COLOR_SURFACE = (90, 120, 60)      # BGR: court paint
COLOR_OUTSIDE = (70, 80, 70)
COLOR_LINE = (255, 255, 255)
COLOR_NET = (40, 40, 40)
DIAGRAM_MARGIN_FT = 3.0


def draw_court_diagram(ppf=DEFAULT_PPF, margin_ft=DIAGRAM_MARGIN_FT, with_labels=True):
    """Render a clean top-down court diagram.  Returns (image, ppf, margin_ft).

    The margin means diagram pixels are NOT simply feet * ppf; use
    feet_to_output_px(pt, ppf, margin_ft) so the offset is applied consistently.
    """
    w = int(round((COURT_WIDTH_FT + 2 * margin_ft) * ppf))
    h = int(round((COURT_LENGTH_FT + 2 * margin_ft) * ppf))
    img = np.full((h, w, 3), COLOR_OUTSIDE, dtype=np.uint8)

    def px(x_ft, y_ft):
        p = feet_to_output_px([[x_ft, y_ft]], ppf, margin_ft)[0]
        return (int(round(p[0])), int(round(p[1])))

    line_w = max(1, int(round(0.2 * ppf)))   # painted lines are about 2 inches

    cv2.rectangle(img, px(0, 0), px(COURT_WIDTH_FT, COURT_LENGTH_FT), COLOR_SURFACE, -1)
    # outside boundary (sidelines + baselines)
    cv2.rectangle(img, px(0, 0), px(COURT_WIDTH_FT, COURT_LENGTH_FT), COLOR_LINE, line_w)
    # kitchen (non-volley zone) lines at y = 15 and y = 29
    for y in (KITCHEN_FAR_Y_FT, KITCHEN_NEAR_Y_FT):
        cv2.line(img, px(0, y), px(COURT_WIDTH_FT, y), COLOR_LINE, line_w)
    # service centerlines: only OUTSIDE the kitchen, never through it
    cv2.line(img, px(CENTERLINE_X_FT, 0), px(CENTERLINE_X_FT, KITCHEN_FAR_Y_FT), COLOR_LINE, line_w)
    cv2.line(img, px(CENTERLINE_X_FT, KITCHEN_NEAR_Y_FT),
             px(CENTERLINE_X_FT, COURT_LENGTH_FT), COLOR_LINE, line_w)
    # the net at y = 22, drawn a little past the sidelines like real posts
    cv2.line(img, px(-1.0, NET_Y_FT), px(COURT_WIDTH_FT + 1.0, NET_Y_FT), COLOR_NET, max(2, line_w))

    if with_labels:
        f, s, c = cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235)
        cv2.putText(img, "FAR baseline  y=0 ft", px(0.2, -1.2), f, s, c, 1, cv2.LINE_AA)
        cv2.putText(img, "NEAR baseline  y=44 ft", px(0.2, COURT_LENGTH_FT + 2.0), f, s, c, 1, cv2.LINE_AA)
        # line labels sit just inside the right sideline so they never clip
        cv2.putText(img, "kitchen y=15", px(COURT_WIDTH_FT - 5.6, KITCHEN_FAR_Y_FT - 0.5),
                    f, 0.4, c, 1, cv2.LINE_AA)
        cv2.putText(img, "kitchen y=29", px(COURT_WIDTH_FT - 5.6, KITCHEN_NEAR_Y_FT + 1.3),
                    f, 0.4, c, 1, cv2.LINE_AA)
        cv2.putText(img, "net y=22", px(COURT_WIDTH_FT - 4.0, NET_Y_FT - 0.6),
                    f, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "(0,0)", px(0.2, 1.3), f, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "x ->", px(COURT_WIDTH_FT / 2.0 - 1.0, -1.2), f, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return img, ppf, margin_ft


def put_label(img, text, org, color, scale=0.45):
    """Draw outlined text, nudged so it never runs off the edge of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = min(max(int(org[0]), 2), max(2, img.shape[1] - tw - 3))
    y = min(max(int(org[1]), th + 2), img.shape[0] - 3)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_markers_on_diagram(diagram, points_ft, labels, ppf=DEFAULT_PPF,
                            margin_ft=DIAGRAM_MARGIN_FT, color=(0, 220, 255),
                            oob_color=(0, 0, 255), show_coords=True):
    """Return a copy of the diagram with labelled markers at the given points."""
    img = diagram.copy()
    for pt, label in zip(np.asarray(points_ft, dtype=np.float64).reshape(-1, 2), labels):
        out = is_out_of_bounds(pt)
        c = oob_color if out else color
        p = feet_to_output_px([pt], ppf, margin_ft)[0]
        xy = (int(round(p[0])), int(round(p[1])))

        # A point far outside the court would fall off this canvas.  Only the
        # DRAWING is pulled back to the edge; the printed coordinate below is
        # always the true one, never clamped.
        h, w = img.shape[:2]
        clamped = (min(max(xy[0], 12), w - 13), min(max(xy[1], 12), h - 13))
        off_canvas = clamped != xy

        if off_canvas:
            cv2.drawMarker(img, clamped, c, cv2.MARKER_TRIANGLE_UP, 16, 2)
        else:
            cv2.drawMarker(img, clamped, c, cv2.MARKER_CROSS, 14, 2)
            cv2.circle(img, clamped, 8, c, 2)
        xy = clamped

        text = str(label)
        if show_coords:
            text += f" ({pt[0]:.1f}, {pt[1]:.1f})"
            if out:
                text += " OFF-DIAGRAM" if off_canvas else " OUT"
        put_label(img, text, (xy[0] + 11, xy[1] - 9), c)
    return img


# --------------------------------------------------------------------------
# Display scaling: big photo on a small screen, clicks back to original pixels
# --------------------------------------------------------------------------
class DisplayScaler:
    """Shrinks an image to fit a window and converts clicks back to original px.

    The window shows a scaled copy, so a click at display (x, y) corresponds to
    original pixel (x / scale, y / scale).  Forgetting this division is the
    single most common cause of a "slightly wrong" homography.
    """

    def __init__(self, image_shape, max_width=1280, max_height=800):
        h, w = image_shape[:2]
        self.original_size = (int(w), int(h))
        self.scale = float(min(1.0, max_width / float(w), max_height / float(h)))
        self.display_size = (max(1, int(round(w * self.scale))),
                             max(1, int(round(h * self.scale))))

    def resize(self, image):
        if self.scale >= 1.0:
            return image.copy()
        return cv2.resize(image, self.display_size, interpolation=cv2.INTER_AREA)

    def to_original(self, x, y):
        """Display pixel -> original image pixel (floats; sub-pixel is fine)."""
        return (float(x) / self.scale, float(y) / self.scale)

    def to_display(self, x, y):
        return (float(x) * self.scale, float(y) * self.scale)

    def to_display_int(self, point):
        p = self.to_display(point[0], point[1])
        return (int(round(p[0])), int(round(p[1])))


# --------------------------------------------------------------------------
# Accuracy statistics
# --------------------------------------------------------------------------
def position_errors(measured_ft, truth_ft):
    """Per-point Euclidean error in feet."""
    m = np.asarray(measured_ft, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(truth_ft, dtype=np.float64).reshape(-1, 2)
    if m.shape != t.shape:
        raise ValueError(f"shape mismatch: {m.shape} vs {t.shape}")
    return np.linalg.norm(m - t, axis=1)


def error_summary(errors):
    """Mean / RMSE / max / median of a list of errors, in feet."""
    e = np.asarray(errors, dtype=np.float64).ravel()
    if e.size == 0:
        return {"n": 0, "mean_ft": float("nan"), "rmse_ft": float("nan"),
                "max_ft": float("nan"), "median_ft": float("nan")}
    return {
        "n": int(e.size),
        "mean_ft": float(np.mean(e)),
        "rmse_ft": float(np.sqrt(np.mean(e ** 2))),
        "max_ft": float(np.max(e)),
        "median_ft": float(np.median(e)),
    }
