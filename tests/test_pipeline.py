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
from padopt.evaluation import compile_case, case_pose, single_trial


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

    def test_hold_fixture_is_released(self):
        m, d = compile_case(self.cfg, self.candidate, self.cfg["evaluation"]["train"][0])
        row = single_trial(self.cfg, m, d, *case_pose(self.cfg, 0, 0, "train"))
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["fixture_released"], 1)
        self.assertTrue(np.isfinite(row["max_drift_mm"]))
        self.assertLessEqual(row["max_tendon_force_N"], 10.00001)

    def test_large_original_object_rejected_for_overlap(self):
        case = {"type": "cylinder", "size": [.055, .019]}
        m, d = compile_case(self.cfg, None, case)
        row = single_trial(self.cfg, m, d, *case_pose(self.cfg, 0, 0, "train"))
        self.assertEqual(row["status"], "initial_overlap")
        self.assertEqual(row["success"], 0)
        self.assertGreater(row["initial_penetration_mm"], 7)

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
