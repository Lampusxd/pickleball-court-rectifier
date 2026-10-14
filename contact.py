"""
contact.py -- pick one ground-contact pixel from the two heels.

This file is pure arithmetic.  It does not open video, load a model, or know
about the court.  track_video.py feeds it heel locations and then maps the
returned pixel into court feet with the existing homography.
"""

from __future__ import annotations

# MediaPipe Pose Landmarker landmark indices.
# https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
LEFT_HEEL = 29
RIGHT_HEEL = 30

DEFAULT_MIN_VISIBILITY = 0.5


def _usable(heel, min_visibility):
    """Return (x, y) when a heel observation is usable, else None.

    A heel is (x, y), (x, y, visibility), or None.  Missing visibility means
    the caller already decided the point is real.
    """
    if heel is None:
        return None
    if len(heel) >= 3 and heel[2] is not None and float(heel[2]) < min_visibility:
        return None
    return (float(heel[0]), float(heel[1]))


def contact_pixel(left_heel, right_heel, min_visibility=DEFAULT_MIN_VISIBILITY):
    """Return the ground-contact pixel, or None when neither heel is usable.

    Both heels usable: the midpoint, the same spot as a click between the feet.
    One heel usable: that heel.  Neither: None, and the frame is left blank.
    """
    left = _usable(left_heel, min_visibility)
    right = _usable(right_heel, min_visibility)
    if left is not None and right is not None:
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    if left is not None:
        return left
    return right


def landmark_pixel(landmark, width, height):
    """Convert one normalized pose landmark to an image-pixel heel sample.

    Returns (x, y, visibility) or None.  x and y on a landmark are fractions
    of the frame width and height.  Visibility is omitted from the sample when
    the model did not provide it.
    """
    if landmark is None:
        return None
    x = float(landmark.x) * float(width)
    y = float(landmark.y) * float(height)
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        return (x, y)
    return (x, y, float(visibility))


def contact_from_landmarks(landmarks, width, height, min_visibility=DEFAULT_MIN_VISIBILITY):
    """Contact pixel from one pose's landmark list, or None if the heels are missing."""
    if not landmarks or len(landmarks) <= RIGHT_HEEL:
        return None
    left = landmark_pixel(landmarks[LEFT_HEEL], width, height)
    right = landmark_pixel(landmarks[RIGHT_HEEL], width, height)
    return contact_pixel(left, right, min_visibility)
