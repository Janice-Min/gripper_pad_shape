"""Meaningful geometry/integration tests; no fake Shap-E inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from padopt.common import read_config, save_json, source_signature, parse_gripper
from padopt.geometry import flat_pad, save_pad, grid_vertices, surface_mesh, from_shap_e, convex_parts
from padopt.xml_model import build_xml
from padopt.evaluation import compile_case, case_pose, single_trial, score_from_drop, max_score


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = read_config(Path(__file__).resolve().parents[1] / "config.json")
        self.candidate = Path(self.tmp.name) / "pad_flat"
        s, z, r, thickness = flat_pad(self.cfg)
        self.grid = (s, z, r, thickness)
        stats = save_pad(self.candidate, self.cfg, *self.grid)
        save_json(self.candidate / "metadata.json", {"source": "flat_baseline", "source_signature": source_signature(self.cfg), **stats})

    def tearDown(self):
        self.tmp.cleanup()

    def test_dimensions_mirror_watertight(self):
        s, z, r, h = self.grid
        self.assertAlmostEqual((s[-1] - s[0]) * 1000, 30.7695)
        upper = grid_vertices(s, z, r, h, .00015, -1)
        lower = grid_vertices(s, z, r, h, .00015, 1)
        for a, b in zip(upper, lower):
            np.testing.assert_allclose(a * [1, -1, 1], b)
        for side in (upper, lower):
            v, f = surface_mesh(*side)
            edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            self.assertTrue(np.all(counts == 2))
            tri = v[f]
            signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
            self.assertGreater(signed, 0)
            self.assertEqual(len(list(convex_parts(*side))), 96)

    def test_original_gripper_elements_unchanged_and_xml_compiles(self):
        import mujoco as mj
        path = build_xml(self.cfg, self.candidate)
        original = ET.parse(self.cfg["base_xml"]).getroot()
        modified = ET.parse(path).getroot()
        for e in original.iter():
            name = e.get("name")
            if name:
                match = modified.find(".//" + e.tag + "[@name='" + name + "']")
                self.assertIsNotNone(match)
                self.assertEqual(e.attrib, match.attrib)
        m0 = mj.MjModel.from_xml_path(self.cfg["base_xml"])
        m1 = mj.MjModel.from_xml_path(str(path))
        self.assertEqual(m0.nq, m1.nq)
        np.testing.assert_allclose(m0.jnt_range, m1.jnt_range)
        np.testing.assert_allclose(m0.jnt_stiffness, m1.jnt_stiffness)
        np.testing.assert_allclose(m0.actuator_gear, m1.actuator_gear)

    def test_evaluation_model_adds_only_support_and_lift_axis(self):
        import mujoco as mj
        case = self.cfg["evaluation"]["train"][0]
        m, d = compile_case(self.cfg, self.candidate, case)
        base = mj.MjModel.from_xml_path(self.cfg["base_xml"])
        self.assertEqual(m.nq, base.nq + 1)  # one vertical slide joint for lifting
        self.assertEqual(m.nu, base.nu + 1)  # its position actuator
        self.assertGreaterEqual(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "object_support"), 0)
        self.assertEqual(m.neq, base.neq)    # no weld or other force holding the object
        key_m, key_b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "open"), mj.mj_name2id(base, mj.mjtObj.mjOBJ_KEY, "open")
        for name in ("joint1", "joint2", "joint3", "joint4"):
            jm, jb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name), mj.mj_name2id(base, mj.mjtObj.mjOBJ_JOINT, name)
            self.assertAlmostEqual(m.key_qpos[key_m, m.jnt_qposadr[jm]], base.key_qpos[key_b, base.jnt_qposadr[jb]])
            self.assertAlmostEqual(m.jnt_stiffness[jm], base.jnt_stiffness[jb])
        np.testing.assert_allclose(m.actuator_gear[0], base.actuator_gear[0])

    def test_object_rests_on_support_under_gravity(self):
        import mujoco as mj
        case = self.cfg["evaluation"]["train"][0]
        m, d = compile_case(self.cfg, self.candidate, case)
        # Without tendon force nothing holds the object: it must go down with the lowered support.
        loose = copy.deepcopy(self.cfg)
        loose["evaluation"]["control_max"] = 0.0
        row = single_trial(loose, m, d, *case_pose(loose, 0, 0, "train"))
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["success"], 0)
        self.assertEqual(row["support_contact"], 1)
        self.assertLess(row["lift_achieved_mm"], -0.5 * 1000 * self.cfg["evaluation"]["support_drop_m"])
        self.assertLess(row["initial_penetration_mm"], self.cfg["evaluation"]["initial_penetration_tolerance_mm"])
        self.assertEqual((row["score"], row["held_after_lift"], row["disturbances_survived"]), (0, 0, 0))
        self.assertIsNotNone(row["drop_time_s"])

    def test_full_trial_runs(self):
        m, d = compile_case(self.cfg, self.candidate, self.cfg["evaluation"]["train"][2])
        row = single_trial(self.cfg, m, d, *case_pose(self.cfg, 2, 0, "train"))
        self.assertEqual(row["status"], "ok")
        self.assertTrue(np.isfinite(row["max_drift_mm"]))
        self.assertLessEqual(row["max_tendon_force_N"], 10.00001)
        self.assertIn(row["pad_success"], (0, 1))
        self.assertLessEqual(row["pad_success"], row["success"])
        self.assertEqual(row["score"], score_from_drop(self.cfg, row["drop_time_s"])[2])
        self.assertLessEqual(row["score"], row["max_score"])
        if row["success"]:
            self.assertEqual(row["score"], row["max_score"])  # a strict success is never dropped

    def test_score_rules(self):
        cfg = copy.deepcopy(self.cfg)
        e = cfg["evaluation"]
        e.update({"score_grasp": 5, "score_per_disturbance": 2, "close_s": 1.0, "lift_s": 0.5, "hold_s": 3.0})
        e["disturbances"] = [{"start_s": 0.3 + 0.5 * k, "duration_s": 0.3, "force_N": [1, 0, 0]} for k in range(5)]
        self.assertEqual(max_score(cfg), 15)
        # pulses start at 1.8, 2.3, 2.8, 3.3, 3.8 s; trial ends at 4.5 s
        for drop, expected in [(None, (1, 5, 15)), (1.2, (0, 0, 0)), (1.85, (1, 0, 5)), (2.31, (1, 1, 7)),
                               (3.9, (1, 4, 13)), (4.5, (1, 5, 15))]:
            self.assertEqual(score_from_drop(cfg, drop), expected, drop)

    def test_large_original_object_rejected_for_overlap(self):
        case = {"type": "cylinder", "size": [.055, .019]}
        m, d = compile_case(self.cfg, None, case)
        row = single_trial(self.cfg, m, d, np.array([-.075, 0, .16]), np.array([1.0, 0, 0, 0]))
        self.assertEqual(row["status"], "initial_overlap")
        self.assertEqual(row["success"], 0)
        self.assertGreater(row["initial_penetration_mm"], 7)

    def test_position_calibration_codes(self):
        from calibrate_positions import classify, classify_envelop, longest_run, choose
        self.assertEqual(classify(["upper_pad", "lower_pad"]), "B")
        self.assertEqual(classify(["upper_pad", "lower_pad", "link"]), "b")
        self.assertEqual(classify(["link"]), "L")
        links = ["upper_link1", "upper_link2", "lower_link1", "lower_link2"]
        self.assertEqual(classify_envelop(["upper_pad", "lower_pad", "link"], links), "E")
        self.assertEqual(classify_envelop(["upper_pad", "link"], links), "e")
        self.assertEqual(classify_envelop(["upper_pad", "lower_pad"], links[1::2]), "B")
        self.assertEqual(longest_run([1, 2, 3, 4, 5], list("BB-BB"), {"B"}), [1, 2])
        xs = [-.040, -.045, -.050, -.055]
        self.assertEqual(choose(xs, list("EEBB"), "envelop", "pad"), (-.040, "envelop"))
        self.assertEqual(choose(xs, list("-BB-"), "envelop", "pad"), (-.045, "pad"))
        self.assertEqual(choose(xs, list("-BB-"), "envelop", "keep"), (None, "keep"))

    def test_configured_objects_are_enveloped(self):
        # Every configured object must start free of overlap and be touched by all four links and both pads
        # when the fingers have closed. Contact sets can flip on a single pose because of tiny numerical
        # differences between machines, so 4 of 5 nearby poses must be enveloping.
        import math
        from calibrate_positions import classify_envelop
        poses = [(0, 0), (5e-4, 0), (-5e-4, 0), (0, 2), (0, -2)]  # (y offset m, yaw deg)
        for suite in ("train", "validation"):
            for index, case in enumerate(self.cfg["evaluation"][suite]):
                m, d = compile_case(self.cfg, self.candidate, case)
                x = case_pose(self.cfg, index, 0, suite)[0][0]
                codes = []
                for dy, yaw in poses:
                    q = np.array([math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)])
                    row = single_trial(self.cfg, m, d, np.array([x, dy, self.cfg["evaluation"]["position_z_m"]]), q, until="closing")
                    self.assertEqual(row["status"], "closed", case["name"] + ": initial overlap")
                    codes.append(classify_envelop(row["closing_contacts"], row["closing_bodies"]))
                self.assertGreaterEqual(codes.count("E"), 4, case["name"] + " is not enveloped at its configured position (codes "
                                        + "".join(codes) + "). Re-run: python calibrate_positions.py --write --also-write config_light.json")

    def test_common_poses_and_source_guard(self):
        for a, b in zip(case_pose(self.cfg, 1, 2, "train"), case_pose(self.cfg, 1, 2, "train")):
            np.testing.assert_array_equal(a, b)
        changed = copy.deepcopy(self.cfg)
        changed["pad"]["width_mm"] = 12
        with self.assertRaises(ValueError):
            build_xml(changed, self.candidate)

    def test_geometry_adapter_on_explicit_synthetic_mesh(self):
        # Unit-test fixture only. Never counted as a Shap-E pad or elite.
        from scipy.spatial import ConvexHull
        rng = np.random.default_rng(123)
        vertices = rng.normal(size=(100, 3)) * [2, 1, .6]
        faces = ConvexHull(vertices).simplices
        s, z, radius, h, coverage = from_shap_e(vertices, faces, self.cfg)
        self.assertGreaterEqual(coverage, .25)
        self.assertGreaterEqual(h.min(), .0015 - 1e-12)
        self.assertLessEqual(h.max(), .0055 + 1e-12)
        ds, dz = np.gradient(h, s, z)
        self.assertLessEqual(float(np.sqrt(ds * ds + dz * dz).max()), 1 + 1e-8)


if __name__ == "__main__":
    unittest.main()
