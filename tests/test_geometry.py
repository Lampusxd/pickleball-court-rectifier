"""
Tests for the geometry and calibration code.

These use only the Python standard library (unittest), so they run anywhere
OpenCV is installed -- no display and no extra packages required.

    python -m unittest discover -s tests -v

Each test targets one of the ideas the project is meant to teach:
known points map correctly, round trips are lossless, resized clicks convert
back, bad corner selections are rejected, and calibration survives a save and
reload unchanged.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import perspective as P  # noqa: E402

# A plausible angled view of a court: a trapezoid, far edge narrower.
ANGLED_CORNERS = np.array([[472.0, 232.0],
                           [846.0, 226.0],
                           [1128.0, 706.0],
                           [214.0, 724.0]], dtype=np.float64)
IMAGE_SHAPE = (800, 1280, 3)


class TestCourtConstants(unittest.TestCase):
    def test_court_dimensions(self):
        self.assertEqual(P.COURT_WIDTH_FT, 20.0)
        self.assertEqual(P.COURT_LENGTH_FT, 44.0)
        self.assertEqual(P.NET_Y_FT, 22.0)
        self.assertEqual(P.KITCHEN_FAR_Y_FT, 15.0)
        self.assertEqual(P.KITCHEN_NEAR_Y_FT, 29.0)
        # the kitchen lines are 7 ft from the net on both sides
        self.assertEqual(P.NET_Y_FT - P.KITCHEN_FAR_Y_FT, 7.0)
        self.assertEqual(P.KITCHEN_NEAR_Y_FT - P.NET_Y_FT, 7.0)

    def test_corner_order_matches_court_rectangle(self):
        """far-left is the origin; x grows right, y grows toward the near end."""
        c = P.court_corners_ft()
        np.testing.assert_allclose(c[0], [0.0, 0.0])                       # far-left
        np.testing.assert_allclose(c[1], [20.0, 0.0])                      # far-right
        np.testing.assert_allclose(c[2], [20.0, 44.0])                     # near-right
        np.testing.assert_allclose(c[3], [0.0, 44.0])                      # near-left
        self.assertEqual(P.CORNER_LABELS,
                         ("far-left", "far-right", "near-right", "near-left"))

    def test_validation_landmarks_exclude_the_calibration_corners(self):
        """Independent accuracy needs landmarks the homography was not fitted to."""
        corners = {tuple(c) for c in P.court_corners_ft()}
        for label, x, y in P.VALIDATION_LANDMARKS:
            self.assertNotIn((x, y), corners, f"{label} is a calibration corner")
        self.assertGreaterEqual(len(P.VALIDATION_LANDMARKS), 5)


class TestHomography(unittest.TestCase):
    def setUp(self):
        self.calib = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "test.png")

    def test_corners_map_to_the_court_rectangle(self):
        mapped = self.calib.image_to_court(ANGLED_CORNERS)
        np.testing.assert_allclose(mapped, P.court_corners_ft(), atol=1e-9)

    def test_known_point_under_a_synthetic_transformation(self):
        """Build a view from a KNOWN homography, then recover the known points.

        This is the core check: if we invent a perspective, project known court
        coordinates through it, and feed the resulting pixels back in, we must
        get the original court coordinates out.
        """
        H_court_to_image = np.linalg.inv(self.calib.H_image_to_court)
        known_ft = np.array([[10.0, 22.0],    # centre of the net
                             [0.0, 15.0],     # far kitchen / left sideline
                             [20.0, 29.0],    # near kitchen / right sideline
                             [10.0, 0.0],     # far baseline / centerline
                             [3.5, 40.0]])    # an arbitrary interior point
        pixels = P.apply_homography(H_court_to_image, known_ft)
        recovered = self.calib.image_to_court(pixels)
        np.testing.assert_allclose(recovered, known_ft, atol=1e-9)

    def test_center_of_court_is_between_the_kitchen_lines(self):
        """A sanity check that does not depend on the inverse being right."""
        centre_px = self.calib.court_to_image([[10.0, 22.0]])[0]
        centre_ft = self.calib.image_to_court([centre_px])[0]
        self.assertAlmostEqual(centre_ft[0], 10.0, places=6)
        self.assertAlmostEqual(centre_ft[1], 22.0, places=6)

    def test_round_trip_preserves_points(self):
        rng = np.random.default_rng(1234)
        pts = np.column_stack([rng.uniform(230, 1120, 500), rng.uniform(235, 715, 500)])
        back = self.calib.court_to_image(self.calib.image_to_court(pts))
        self.assertLess(float(np.max(np.linalg.norm(pts - back, axis=1))), 1e-7)

    def test_perspective_is_not_affine(self):
        """Equal steps in feet are NOT equal steps in pixels -- that is the point.

        If this ever passed with equal spacing, the transform would be affine
        and the whole perspective correction would be pointless.
        """
        ys = np.array([[10.0, 0.0], [10.0, 11.0], [10.0, 22.0], [10.0, 33.0], [10.0, 44.0]])
        px = self.calib.court_to_image(ys)
        gaps = np.diff(px[:, 1])
        self.assertGreater(gaps.max() / gaps.min(), 1.5)

    def test_out_of_court_points_are_preserved_not_clamped(self):
        outside_ft = np.array([[-3.0, 12.0], [25.0, 50.0], [10.0, -6.0]])
        px = self.calib.court_to_image(outside_ft)
        back = self.calib.image_to_court(px)
        np.testing.assert_allclose(back, outside_ft, atol=1e-8)
        for pt in back:
            self.assertTrue(P.is_out_of_bounds(pt))
        self.assertFalse(P.is_out_of_bounds([10.0, 22.0]))
        self.assertFalse(P.is_out_of_bounds([0.0, 0.0]))       # on the line = in
        self.assertTrue(P.is_out_of_bounds([20.01, 22.0]))

    def test_empty_point_list(self):
        self.assertEqual(self.calib.image_to_court(np.zeros((0, 2))).shape, (0, 2))


class TestCornerValidation(unittest.TestCase):
    def test_good_selection_accepted(self):
        self.assertEqual(P.validate_corners(ANGLED_CORNERS, IMAGE_SHAPE), [])

    def test_duplicate_points_rejected(self):
        quad = [[300, 200], [300, 200], [1100, 700], [200, 700]]
        problems = P.validate_corners(quad, IMAGE_SHAPE)
        self.assertTrue(any("duplicate" in p for p in problems), problems)

    def test_near_duplicate_points_rejected(self):
        quad = [[300, 200], [304, 203], [1100, 700], [200, 700]]
        self.assertTrue(P.validate_corners(quad, IMAGE_SHAPE))

    def test_self_intersecting_bowtie_rejected(self):
        quad = [[300, 200], [1100, 700], [1000, 200], [350, 760]]
        problems = P.validate_corners(quad, IMAGE_SHAPE)
        self.assertTrue(any("self-intersecting" in p or "non-convex" in p for p in problems),
                        problems)

    def test_collinear_points_rejected(self):
        quad = [[300, 200], [600, 200], [900, 200], [1200, 200]]
        self.assertTrue(P.validate_corners(quad, IMAGE_SHAPE))

    def test_tiny_area_rejected(self):
        quad = [[300, 200], [340, 200], [340, 240], [300, 240]]
        problems = P.validate_corners(quad, IMAGE_SHAPE)
        self.assertTrue(any("area" in p for p in problems), problems)

    def test_reversed_click_order_rejected(self):
        """Counter-clockwise order would mirror the court, so it is refused."""
        quad = ANGLED_CORNERS[::-1]
        problems = P.validate_corners(quad, IMAGE_SHAPE)
        self.assertTrue(any("reverse" in p for p in problems), problems)

    def test_wrong_number_of_points_rejected(self):
        self.assertTrue(P.validate_corners([[0, 0], [10, 0], [10, 10]], IMAGE_SHAPE))

    def test_calibration_constructor_raises_on_bad_corners(self):
        with self.assertRaises(P.CalibrationError):
            P.Calibration.from_corners([[0, 0], [0, 0], [10, 10], [0, 10]], IMAGE_SHAPE)

    def test_signed_area_sign_convention(self):
        """Clockwise on screen (y down) is positive with this shoelace."""
        self.assertGreater(P.signed_area([[0, 0], [10, 0], [10, 10], [0, 10]]), 0)
        self.assertLess(P.signed_area([[0, 0], [0, 10], [10, 10], [10, 0]]), 0)


class TestDisplayScaling(unittest.TestCase):
    def test_large_image_is_scaled_down(self):
        scaler = P.DisplayScaler((2160, 3840, 3), 1280, 800)
        self.assertLess(scaler.scale, 1.0)
        self.assertLessEqual(scaler.display_size[0], 1280)
        self.assertLessEqual(scaler.display_size[1], 800)

    def test_small_image_is_not_scaled_up(self):
        scaler = P.DisplayScaler((400, 600, 3), 1280, 800)
        self.assertEqual(scaler.scale, 1.0)
        self.assertEqual(scaler.display_size, (600, 400))

    def test_aspect_ratio_preserved(self):
        scaler = P.DisplayScaler((1080, 1920, 3), 1280, 800)
        original_ratio = 1920 / 1080
        shown_ratio = scaler.display_size[0] / scaler.display_size[1]
        self.assertAlmostEqual(original_ratio, shown_ratio, places=2)

    def test_display_click_maps_back_to_original_pixel(self):
        scaler = P.DisplayScaler((2160, 3840, 3), 1280, 800)
        for original in [(0.0, 0.0), (1920.0, 1080.0), (3839.0, 2159.0), (1234.0, 567.0)]:
            disp = scaler.to_display(*original)
            back = scaler.to_original(*disp)
            self.assertAlmostEqual(back[0], original[0], places=6)
            self.assertAlmostEqual(back[1], original[1], places=6)

    def test_scaled_click_gives_the_right_court_coordinate(self):
        """The bug this guards against: using display pixels as if they were
        original pixels, which silently shifts every mapped position."""
        calib = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "test.png")
        scaler = P.DisplayScaler(IMAGE_SHAPE, 640, 400)     # exactly half size
        self.assertAlmostEqual(scaler.scale, 0.5)

        true_court = np.array([12.0, 30.0])
        original_px = calib.court_to_image([true_court])[0]
        display_px = scaler.to_display(*original_px)

        correct = calib.image_to_court([scaler.to_original(*display_px)])[0]
        np.testing.assert_allclose(correct, true_court, atol=1e-8)

        wrong = calib.image_to_court([display_px])[0]       # forgot to convert
        self.assertGreater(float(np.linalg.norm(wrong - true_court)), 1.0)


class TestRenderScaling(unittest.TestCase):
    def test_feet_to_pixels_round_trip(self):
        pts = np.array([[0.0, 0.0], [20.0, 44.0], [10.0, 22.0], [-3.0, 50.0]])
        for ppf in (10.0, 20.0, 37.5):
            for margin in (0.0, 3.0):
                px = P.feet_to_output_px(pts, ppf, margin)
                back = P.output_px_to_feet(px, ppf, margin)
                np.testing.assert_allclose(back, pts, atol=1e-9)

    def test_pixels_per_foot_scale_is_consistent(self):
        px = P.feet_to_output_px([[0.0, 0.0], [1.0, 0.0]], 20.0)
        self.assertAlmostEqual(float(px[1][0] - px[0][0]), 20.0)

    def test_rectified_size_has_the_court_aspect_ratio(self):
        for ppf in (10.0, 20.0, 30.0):
            w, h = P.rectified_size(ppf)
            self.assertAlmostEqual(w / h, 20.0 / 44.0, places=6)

    def test_rectify_produces_the_expected_size(self):
        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        calib = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "t.png")
        rect = P.rectify(image, calib, 20.0)
        self.assertEqual((rect.shape[1], rect.shape[0]), P.rectified_size(20.0))

    def test_changing_ppf_does_not_change_court_coordinates(self):
        """Rendering resolution must not leak into measured feet."""
        calib = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "t.png")
        pt_px = [[700.0, 500.0]]
        a = calib.image_to_court(pt_px)
        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        P.rectify(image, calib, 8.0)
        P.rectify(image, calib, 40.0)
        np.testing.assert_allclose(calib.image_to_court(pt_px), a)

    def test_diagram_has_court_proportions_plus_margins(self):
        img, ppf, margin = P.draw_court_diagram(20.0, 3.0)
        self.assertEqual(img.shape[1], int((20.0 + 6.0) * 20.0))
        self.assertEqual(img.shape[0], int((44.0 + 6.0) * 20.0))


class TestCalibrationPersistence(unittest.TestCase):
    def setUp(self):
        self.calib = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "court.png")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "calibration.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_reload_preserves_the_mapping(self):
        self.calib.save(self.path)
        reloaded = P.Calibration.load(self.path, IMAGE_SHAPE)
        np.testing.assert_array_equal(reloaded.H_image_to_court, self.calib.H_image_to_court)
        np.testing.assert_array_equal(reloaded.image_corners, self.calib.image_corners)

        rng = np.random.default_rng(99)
        pts = np.column_stack([rng.uniform(230, 1120, 100), rng.uniform(235, 715, 100)])
        np.testing.assert_array_equal(reloaded.image_to_court(pts),
                                      self.calib.image_to_court(pts))

    def test_saved_file_records_the_dimensions_and_court_size(self):
        self.calib.save(self.path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["image_width"], 1280)
        self.assertEqual(data["image_height"], 800)
        self.assertEqual(data["court_width_ft"], 20.0)
        self.assertEqual(data["court_length_ft"], 44.0)
        self.assertEqual(len(data["image_corners"]), 4)
        self.assertEqual(np.array(data["H_image_to_court"]).shape, (3, 3))
        self.assertEqual(data["corner_order"], list(P.CORNER_LABELS))

    def test_mismatched_image_size_rejected(self):
        self.calib.save(self.path)
        with self.assertRaises(P.CalibrationError):
            P.Calibration.load(self.path, (600, 800, 3))

    def test_matching_size_loads_even_though_it_proves_nothing(self):
        """Same dimensions pass the check -- documented as a guard rail only."""
        self.calib.save(self.path)
        other_photo_same_size = (800, 1280, 3)
        loaded = P.Calibration.load(self.path, other_photo_same_size)
        self.assertIsInstance(loaded, P.Calibration)

    def test_missing_file_raises(self):
        with self.assertRaises(P.CalibrationError):
            P.Calibration.load(os.path.join(self.tmp.name, "nope.json"), IMAGE_SHAPE)

    def test_corrupt_json_raises(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(P.CalibrationError):
            P.Calibration.load(self.path, IMAGE_SHAPE)

    def test_missing_keys_raise(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"image_width": 1280, "image_height": 800}, f)
        with self.assertRaises(P.CalibrationError):
            P.Calibration.load(self.path, IMAGE_SHAPE)

    def test_singular_matrix_rejected(self):
        data = self.calib.to_dict()
        data["H_image_to_court"] = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]   # rank 1
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with self.assertRaises(P.CalibrationError):
            P.Calibration.load(self.path, IMAGE_SHAPE)


class TestErrorStatistics(unittest.TestCase):
    def test_position_errors_are_euclidean(self):
        measured = [[3.0, 4.0], [0.0, 0.0]]
        truth = [[0.0, 0.0], [0.0, 0.0]]
        np.testing.assert_allclose(P.position_errors(measured, truth), [5.0, 0.0])

    def test_summary_mean_and_rmse(self):
        s = P.error_summary([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean_ft"], 1.5)
        self.assertAlmostEqual(s["rmse_ft"], np.sqrt((0 + 1 + 4 + 9) / 4))
        self.assertAlmostEqual(s["max_ft"], 3.0)

    def test_rmse_is_at_least_the_mean(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            s = P.error_summary(np.abs(rng.normal(0.4, 0.3, 12)))
            self.assertGreaterEqual(s["rmse_ft"], s["mean_ft"] - 1e-12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            P.position_errors([[0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]])

    def test_empty_summary_is_safe(self):
        self.assertEqual(P.error_summary([])["n"], 0)


class TestSyntheticGroundTruth(unittest.TestCase):
    """End-to-end check against generate_demo.py's known warp."""

    def test_demo_landmarks_recover_their_true_coordinates(self):
        import generate_demo as G

        topdown = G.draw_topdown_court()
        _, H_feet_to_image = G.build_angled_view(topdown)
        truth = G.ground_truth(H_feet_to_image)

        calib = P.Calibration.from_corners(truth["corners_image"],
                                           (G.OUT_HEIGHT, G.OUT_WIDTH, 3), "demo")
        for lm in truth["landmarks"]:
            measured = calib.image_to_court([[lm["image_x"], lm["image_y"]]])[0]
            err = float(np.linalg.norm(measured - np.array([lm["x_ft"], lm["y_ft"]])))
            self.assertLess(err, 1e-6, f"{lm['label']} off by {err} ft")

        for pl in truth["players"]:
            measured = calib.image_to_court([[pl["image_x"], pl["image_y"]]])[0]
            err = float(np.linalg.norm(measured - np.array([pl["x_ft"], pl["y_ft"]])))
            self.assertLess(err, 1e-6, f"{pl['label']} off by {err} ft")

    def test_a_one_pixel_click_error_stays_small(self):
        """Shows how sensitive the result is to imprecise corner clicks.

        Not a promise of accuracy: it documents roughly how a small clicking
        mistake propagates, which is why validation mode exists.
        """
        calib_exact = P.Calibration.from_corners(ANGLED_CORNERS, IMAGE_SHAPE, "t")
        nudged = ANGLED_CORNERS + np.array([[1.0, 1.0], [0, 0], [0, 0], [0, 0]])
        calib_off = P.Calibration.from_corners(nudged, IMAGE_SHAPE, "t")
        probe = [[700.0, 480.0]]
        shift = float(np.linalg.norm(calib_exact.image_to_court(probe)
                                     - calib_off.image_to_court(probe)))
        self.assertGreater(shift, 0.0)
        self.assertLess(shift, 1.0)


class TestSyntheticViewsAndDistortion(unittest.TestCase):
    """The demo variants: extra camera angles and known lens distortion."""

    def test_every_view_is_a_valid_corner_selection(self):
        import generate_demo as G

        for name, quad in G.VIEWS.items():
            problems = P.validate_corners(quad, (G.OUT_HEIGHT, G.OUT_WIDTH, 3))
            self.assertEqual(problems, [], f"view '{name}' would be rejected: {problems}")

    def test_distortion_inverse_is_accurate(self):
        """The ideal->distorted map inverts to machine precision, both signs."""
        import generate_demo as G

        centre, radius = G.distortion_frame((800, 1280, 3))
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 1280, 500), rng.uniform(0, 800, 500)])
        for k1 in (0.3, 0.16, 0.0, -0.1):
            t_max = G.max_invertible_ideal_radius(k1)
            keep = np.linalg.norm(pts - centre, axis=1) / radius < t_max * 0.99
            sub = pts[keep]
            d = G.distorted_from_undistorted(sub, k1, 0.0, centre, radius)
            back = G.undistorted_from_distorted(d, k1, 0.0, centre, radius)
            self.assertLess(float(np.max(np.abs(back - sub))), 1e-9, f"k1={k1}")

    def test_barrel_distortion_has_an_invertible_limit(self):
        """Negative k1 folds past a critical radius; asking for it must raise.

        Guards a real trap: without the check, Newton silently diverges and the
        "ground truth" would be nonsense.
        """
        import generate_demo as G

        self.assertEqual(G.max_invertible_ideal_radius(0.16), float("inf"))
        self.assertEqual(G.max_invertible_ideal_radius(0.0), float("inf"))
        t_max = G.max_invertible_ideal_radius(-0.16)
        self.assertLess(t_max, 1.0)          # image corners (r ~ 1) are past it

        centre, radius = G.distortion_frame((800, 1280, 3))
        with self.assertRaises(ValueError):
            G.distorted_from_undistorted([[0.0, 0.0]], -0.16, 0.0, centre, radius)
        # a point comfortably inside the limit is still fine
        G.distorted_from_undistorted([[700.0, 420.0]], -0.16, 0.0, centre, radius)

    def test_zero_distortion_changes_nothing(self):
        import generate_demo as G

        centre, radius = G.distortion_frame((800, 1280, 3))
        pts = np.array([[0.0, 0.0], [640.0, 400.0], [1279.0, 799.0]])
        np.testing.assert_allclose(
            G.distorted_from_undistorted(pts, 0.0, 0.0, centre, radius), pts, atol=1e-12)

    def test_distortion_moves_points_more_near_the_edges(self):
        """Radial distortion is zero at the centre and grows with radius."""
        import generate_demo as G

        centre, radius = G.distortion_frame((800, 1280, 3))
        pts = np.array([[640.0, 400.0], [900.0, 400.0], [1270.0, 790.0]])
        moved = np.linalg.norm(
            G.distorted_from_undistorted(pts, 0.16, 0.0, centre, radius) - pts, axis=1)
        self.assertLess(moved[0], 1e-9)          # the centre never moves
        self.assertLess(moved[1], moved[2])      # farther out moves more

    def test_exact_demo_is_flagged_exact_and_fits_perfectly(self):
        import generate_demo as G

        topdown = G.draw_topdown_court()
        _, H = G.build_angled_view(topdown, corners=G.VIEWS["high"])
        truth = G.ground_truth(H, view="high", k1=0.0)
        self.assertTrue(truth["exact"])

        calib = P.Calibration.from_corners(truth["corners_image"],
                                           (G.OUT_HEIGHT, G.OUT_WIDTH, 3), "demo")
        errs = [np.linalg.norm(calib.image_to_court([[lm["image_x"], lm["image_y"]]])[0]
                               - np.array([lm["x_ft"], lm["y_ft"]]))
                for lm in truth["landmarks"]]
        self.assertLess(max(errs), 1e-6)

    def test_distorted_demo_is_flagged_and_produces_real_errors(self):
        """A bent image cannot be fitted by one homography -- that is the point.

        The band is deliberately wide: this asserts the demo is realistically
        imperfect, not that any particular accuracy is achieved.
        """
        import generate_demo as G

        topdown = G.draw_topdown_court()
        _, H = G.build_angled_view(topdown, corners=G.VIEWS["default"])
        truth = G.ground_truth(H, view="default", k1=0.16)
        self.assertFalse(truth["exact"])
        self.assertEqual(truth["distortion_k1"], 0.16)

        calib = P.Calibration.from_corners(truth["corners_image"],
                                           (G.OUT_HEIGHT, G.OUT_WIDTH, 3), "demo")
        errs = [np.linalg.norm(calib.image_to_court([[lm["image_x"], lm["image_y"]]])[0]
                               - np.array([lm["x_ft"], lm["y_ft"]]))
                for lm in truth["landmarks"]]
        summary = P.error_summary(errs)
        self.assertGreater(summary["mean_ft"], 0.01, "distortion produced no error at all")
        self.assertLess(summary["mean_ft"], 3.0, "errors are implausibly large")
        # and the corners still fit exactly, because the fit goes through them
        corner_err = np.max(np.abs(calib.image_to_court(truth["corners_image"])
                                   - P.court_corners_ft()))
        self.assertLess(float(corner_err), 1e-4)

    def test_demo_base_names(self):
        import generate_demo as G

        self.assertEqual(G.base_name("default", 0.0), "demo_angled")
        self.assertEqual(G.base_name("default", 0.16), "demo_angled_distorted")
        self.assertEqual(G.base_name("low", 0.0), "demo_low")
        self.assertEqual(G.base_name("low", 0.16), "demo_low_distorted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
