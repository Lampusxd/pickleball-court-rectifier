"""Tests for choosing a foot-contact pixel from two heels.

No video, no model, and no display.  Run with the rest of the suite:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contact  # noqa: E402


class TestContactPixel(unittest.TestCase):
    def test_both_heels_use_the_midpoint(self):
        point = contact.contact_pixel((10.0, 20.0), (30.0, 40.0))
        self.assertEqual(point, (20.0, 30.0))

    def test_visibility_is_ignored_when_both_clear_the_threshold(self):
        point = contact.contact_pixel((0.0, 10.0, 0.9), (0.0, 30.0, 0.6))
        self.assertEqual(point, (0.0, 20.0))

    def test_a_hidden_heel_is_dropped(self):
        point = contact.contact_pixel((5.0, 6.0, 0.2), (15.0, 26.0, 0.8))
        self.assertEqual(point, (15.0, 26.0))

    def test_the_other_heel_is_used_when_only_one_is_present(self):
        self.assertEqual(contact.contact_pixel(None, (8.0, 9.0)), (8.0, 9.0))
        self.assertEqual(contact.contact_pixel((8.0, 9.0, 0.5), None), (8.0, 9.0))

    def test_no_usable_heel_returns_none(self):
        self.assertIsNone(contact.contact_pixel(None, None))
        self.assertIsNone(contact.contact_pixel((1.0, 2.0, 0.49), (3.0, 4.0, 0.0)))

    def test_threshold_boundary_is_kept(self):
        point = contact.contact_pixel((1.0, 2.0, 0.5), None)
        self.assertEqual(point, (1.0, 2.0))


class TestLandmarks(unittest.TestCase):
    def test_normalized_landmarks_become_image_pixels(self):
        landmarks = [None] * 33
        landmarks[contact.LEFT_HEEL] = SimpleNamespace(x=0.25, y=0.50, visibility=0.9)
        landmarks[contact.RIGHT_HEEL] = SimpleNamespace(x=0.75, y=0.50, visibility=0.9)
        point = contact.contact_from_landmarks(landmarks, width=200, height=100)
        self.assertEqual(point, (100.0, 50.0))

    def test_a_short_landmark_list_returns_none(self):
        self.assertIsNone(contact.contact_from_landmarks([], 10, 10))

    def test_missing_visibility_still_counts_as_a_heel(self):
        landmarks = [None] * 33
        landmarks[contact.LEFT_HEEL] = SimpleNamespace(x=0.5, y=0.5)
        landmarks[contact.RIGHT_HEEL] = None
        point = contact.contact_from_landmarks(landmarks, width=80, height=40)
        self.assertEqual(point, (40.0, 20.0))
