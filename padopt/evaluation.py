"""Common closure / object-release / disturbance protocol for every design."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .xml_model import model_tree


def case_pose(cfg, case_index, repeat, suite):
    e = cfg["evaluation"]
    # Independent of candidate ID and evaluation order: common random numbers.
    rng = np.random.default_rng(e["pose_seed"] + (100000 if suite == "validation" else 0) + 1000 * case_index + repeat)
    dx, dy = (rng.uniform(-1, 1, 2) * np.array(e["position_jitter_mm"]) * .001) if repeat > 0 else (0, 0)
    yaw = math.radians(rng.uniform(-e["yaw_jitter_deg"], e["yaw_jitter_deg"])) if repeat > 0 else 0
    pos = np.array([e["position_x_m"] + dx, dy, e["position_z_m"]])
    quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    return pos, quat


def compile_case(cfg, candidate, case):
    import mujoco as mj
    root = model_tree(cfg, candidate, object_case=case)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mj.MjData(model)
    return model, data


def single_trial(cfg, model, data, pos, quat, visualize=False):
    import mujoco as mj
    e = cfg["evaluation"]
    named = lambda kind, name: mj.mj_name2id(model, kind, name)
    obj = named(mj.mjtObj.mjOBJ_BODY, "object_body")
    objgeom = named(mj.mjtObj.mjOBJ_GEOM, "object")
    floor = named(mj.mjtObj.mjOBJ_GEOM, "floor")
    joint = named(mj.mjtObj.mjOBJ_JOINT, "object_free")
    actuator = named(mj.mjtObj.mjOBJ_ACTUATOR, "pulley_pull")
    eq = named(mj.mjtObj.mjOBJ_EQUALITY, "temporary_object_hold")
    fixture = named(mj.mjtObj.mjOBJ_BODY, "object_fixture")
    if min(obj, objgeom, joint, actuator, eq, fixture) < 0:
        raise ValueError("The gripper adapter could not find a required XML name.")
    upper = {named(mj.mjtObj.mjOBJ_BODY, n) for n in ("upper_link1", "upper_link2")}
    lower = {named(mj.mjtObj.mjOBJ_BODY, n) for n in ("lower_link1", "lower_link2")}
    if model.actuator_ctrllimited[actuator] and e["control_max"] > model.actuator_ctrlrange[actuator, 1]:
        raise ValueError("control_max exceeds the original actuator ctrlrange")
    kid = named(mj.mjtObj.mjOBJ_KEY, "open")
    if kid < 0:
        raise ValueError("The base model must provide keyframe 'open'.")
    mj.mj_resetDataKeyframe(model, data, kid)
    qadr = model.jnt_qposadr[joint]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    mocap = model.body_mocapid[fixture]
    data.mocap_pos[mocap] = pos
    data.mocap_quat[mocap] = quat
    data.eq_active[eq] = 1
    data.ctrl[:] = 0
    mj.mj_forward(model, data)
    penetration = 0.0
    for c in data.contact:
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (c.geom1, c.geom2)]
        involves_object = objgeom in (c.geom1, c.geom2)
        involves_pad = any(n.startswith("pad_collision_") for n in names)
        if involves_object or involves_pad:
            penetration = max(penetration, -float(c.dist))
    base = {"initial_penetration_mm": penetration * 1000, "success": 0, "contact_fraction": 0.0,
            "pad_contact_fraction": 0.0, "max_drift_mm": 0.0, "max_rotation_deg": 0.0,
            "mean_normal_force_N": 0.0, "max_tendon_force_N": 0.0, "floor_contact": 0, "fixture_released": 0,
            "object_x_m": float(pos[0]), "object_y_m": float(pos[1]), "object_yaw_deg": math.degrees(2 * math.atan2(quat[3], quat[0]))}
    if penetration * 1000 > e["initial_penetration_tolerance_mm"]:
        return {**base, "status": "initial_overlap"}
    timestep = float(model.opt.timestep)
    total = e["close_s"] + e["hold_s"]
    stride = max(1, round(e["sample_interval_s"] / timestep))
    window_start = total - min(e["measure_last_s"], e["hold_s"])
    contact_samples, pad_samples, forces = [], [], []
    max_drift = max_rot = max_tendon = 0.0
    reference_matrix = np.empty(9)
    mj.mju_quat2Mat(reference_matrix, quat)
    reference_axis = reference_matrix.reshape(3, 3)[:, 2]
    floor_seen = False
    released = False
    viewer = None
    if visualize:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for step in range(int(math.ceil(total / timestep))):
            t = float(data.time)
            if t >= e["close_s"] and not released:
                data.eq_active[eq] = 0
                released = True
            data.ctrl[actuator] = e["control_max"] * min(1.0, t / max(e["ramp_s"], timestep))
            data.xfrc_applied[obj] = 0
            hold_t = t - e["close_s"]
            if released and e["disturb_start_s"] <= hold_t < e["disturb_start_s"] + e["disturb_duration_s"]:
                data.xfrc_applied[obj, :3] = e["disturb_force_N"]
            mj.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or sum(int(w.number) for w in data.warning):
                return {**base, "status": "simulation_warning", "fixture_released": int(released)}
            if viewer and step % 10 == 0:
                viewer.sync()
            if not released:
                continue
            drift = float(np.linalg.norm(data.xpos[obj] - pos))
            dot = float(np.clip(abs(data.xquat[obj] @ quat), 0, 1))
            rotation = math.degrees(2 * math.acos(dot))
            if model.geom_type[objgeom] == mj.mjtGeom.mjGEOM_SPHERE:
                rotation = 0.0  # Sphere orientation is physically indistinguishable.
            elif model.geom_type[objgeom] in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
                axis = data.xmat[obj].reshape(3, 3)[:, 2]
                rotation = math.degrees(math.acos(float(np.clip(abs(axis @ reference_axis), 0, 1))))
            max_drift, max_rot = max(max_drift, drift), max(max_rot, rotation)
            max_tendon = max(max_tendon, abs(float(data.actuator_force[actuator] * model.actuator_gear[actuator, 0])))
            for c in data.contact:
                if objgeom in (c.geom1, c.geom2) and floor in (c.geom1, c.geom2) and c.dist <= 0:
                    floor_seen = True
            if t < window_start or step % stride:
                continue
            left_force = right_force = 0.0
            pad_force = 0.0
            for ci, c in enumerate(data.contact):
                if objgeom not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == objgeom else c.geom1
                wrench = np.zeros(6)
                mj.mj_contactForce(model, data, ci, wrench)
                normal = max(0.0, float(wrench[0]))
                body = int(model.geom_bodyid[other])
                left_force += normal if body in upper else 0.0
                right_force += normal if body in lower else 0.0
                if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(other)) or "").startswith("pad_collision_"):
                    pad_force += normal
            contact_samples.append(left_force >= e["contact_force_min_N"] and right_force >= e["contact_force_min_N"])
            pad_samples.append(pad_force >= e["contact_force_min_N"])
            forces.append(left_force + right_force)
        fraction = float(np.mean(contact_samples)) if contact_samples else 0.0
        success = (released and fraction >= e["required_contact_fraction"] and max_drift * 1000 <= e["translation_limit_mm"]
                   and max_rot <= e["rotation_limit_deg"] and not floor_seen)
        return {**base, "status": "ok", "success": int(success), "contact_fraction": fraction,
                "pad_contact_fraction": float(np.mean(pad_samples)) if pad_samples else 0.0,
                "max_drift_mm": max_drift * 1000, "max_rotation_deg": max_rot,
                "mean_normal_force_N": float(np.mean(forces)) if forces else 0.0,
                "max_tendon_force_N": max_tendon, "floor_contact": int(floor_seen), "fixture_released": int(released)}
    finally:
        if viewer is not None:
            viewer.close()


def summarize(rows):
    # Invalid initial geometry is a failure, never removed from the denominator.
    groups = {}
    for r in rows:
        groups.setdefault(r["object_name"], []).append(r["success"])
    return {"trials": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "worst_object_success": min(float(np.mean(v)) for v in groups.values()),
            "contact_fraction": float(np.mean([r["contact_fraction"] for r in rows])),
            "mean_drift_mm": float(np.mean([r["max_drift_mm"] for r in rows])),
            "invalid_trials": sum(r["status"] != "ok" for r in rows)}
