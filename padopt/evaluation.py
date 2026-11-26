"""Common grasp protocol for every design.

1. closing  : the object stands FREE on a support plate (gravity on, nothing holds it)
               while the tendon closes the fingers.
2. lifting  : the gripper base rises (smooth profile) and the support is lowered away.
3. holding  : the gripper holds the object in the air; short force pulses push it in
               several directions; contacts are measured in the last window.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .xml_model import model_tree, add_lift_axis, object_half_height


def case_pose(cfg, case_index, repeat, suite):
    e = cfg["evaluation"]
    case = e[suite][case_index]
    # Independent of candidate ID and evaluation order: common random numbers.
    rng = np.random.default_rng(e["pose_seed"] + (100000 if suite == "validation" else 0) + 1000 * case_index + repeat)
    dx, dy = (rng.uniform(-1, 1, 2) * np.array(e["position_jitter_mm"]) * .001) if repeat > 0 else (0, 0)
    yaw = math.radians(rng.uniform(-e["yaw_jitter_deg"], e["yaw_jitter_deg"])) if repeat > 0 else 0
    pos = np.array([case.get("position_x_m", e["position_x_m"]) + dx, dy, e["position_z_m"]])
    quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    return pos, quat


def compile_case(cfg, candidate, case):
    import mujoco as mj
    root = add_lift_axis(model_tree(cfg, candidate, object_case=case), cfg)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mj.MjData(model)
    return model, data


def protocol_times(cfg):
    """(end of closing, end of lifting, end of trial) in seconds."""
    e = cfg["evaluation"]
    return e["close_s"], e["close_s"] + e["lift_s"], e["close_s"] + e["lift_s"] + e["hold_s"]


def lift_fraction(cfg, t):
    """Smooth 0 -> 1 profile (zero velocity at both ends) during the lifting phase."""
    close, lift_end, _ = protocol_times(cfg)
    if t <= close:
        return 0.0
    if t >= lift_end:
        return 1.0
    return 0.5 * (1 - math.cos(math.pi * (t - close) / (lift_end - close)))


def disturbance_at(cfg, t):
    """Total disturbance force [N] on the object and the index of the active pulse (or None)."""
    e = cfg["evaluation"]
    _, lift_end, _ = protocol_times(cfg)
    hold_t = t - lift_end
    force, active = np.zeros(3), None
    for k, pulse in enumerate(e["disturbances"]):
        if pulse["start_s"] <= hold_t < pulse["start_s"] + pulse["duration_s"]:
            force += np.asarray(pulse["force_N"], dtype=float)
            active = k
    return force, active


def phase_at(cfg, t):
    """Human-readable protocol phase at time t (used by the viewer)."""
    e = cfg["evaluation"]
    close, lift_end, end = protocol_times(cfg)
    if t < close:
        return "closing (object free on support)"
    if t < lift_end:
        return "lifting (support lowered)"
    force, active = disturbance_at(cfg, t)
    if active is not None:
        return "disturbance %d/%d  F = [%s] N" % (active + 1, len(e["disturbances"]), ", ".join(format(v, "g") for v in force))
    if t >= end - e["measure_last_s"]:
        return "measuring contacts"
    return "holding"


def _object_half_height(model, geom):
    import mujoco as mj
    names = {mj.mjtGeom.mjGEOM_SPHERE: "sphere", mj.mjtGeom.mjGEOM_CYLINDER: "cylinder", mj.mjtGeom.mjGEOM_CAPSULE: "capsule",
             mj.mjtGeom.mjGEOM_BOX: "box", mj.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid"}
    return object_half_height(names.get(int(model.geom_type[geom])), model.geom_size[geom])


def single_trial(cfg, model, data, pos, quat, visualize=False, until=None):
    """Run one trial. ``until="closing"`` stops when the fingers have closed and reports
    which finger parts touch the object (used by calibrate_positions.py)."""
    import mujoco as mj
    e = cfg["evaluation"]
    named = lambda kind, name: mj.mj_name2id(model, kind, name)
    obj = named(mj.mjtObj.mjOBJ_BODY, "object_body")
    objgeom = named(mj.mjtObj.mjOBJ_GEOM, "object")
    floor = named(mj.mjtObj.mjOBJ_GEOM, "floor")
    joint = named(mj.mjtObj.mjOBJ_JOINT, "object_free")
    actuator = named(mj.mjtObj.mjOBJ_ACTUATOR, "pulley_pull")
    lift_actuator = named(mj.mjtObj.mjOBJ_ACTUATOR, "lift_actuator")
    lift_joint = named(mj.mjtObj.mjOBJ_JOINT, "lift_slide")
    support = named(mj.mjtObj.mjOBJ_BODY, "object_support")
    support_geom = named(mj.mjtObj.mjOBJ_GEOM, "support")
    if min(obj, objgeom, joint, actuator, lift_actuator, lift_joint, support, support_geom) < 0:
        raise ValueError("The gripper adapter could not find a required XML name. Build models with compile_case().")
    base = int(model.jnt_bodyid[lift_joint])
    upper = {named(mj.mjtObj.mjOBJ_BODY, n) for n in ("upper_link1", "upper_link2")}
    lower = {named(mj.mjtObj.mjOBJ_BODY, n) for n in ("lower_link1", "lower_link2")}
    geom_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    upper_pad = {g for g, n in enumerate(geom_names) if n.startswith("pad_collision_upper_")}
    lower_pad = {g for g, n in enumerate(geom_names) if n.startswith("pad_collision_lower_")}
    if model.actuator_ctrllimited[actuator] and e["control_max"] > model.actuator_ctrlrange[actuator, 1]:
        raise ValueError("control_max exceeds the original actuator ctrlrange")
    kid = named(mj.mjtObj.mjOBJ_KEY, "open")
    if kid < 0:
        raise ValueError("The base model must provide keyframe 'open'.")

    mj.mj_resetDataKeyframe(model, data, kid)
    qadr = model.jnt_qposadr[joint]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    # The object stands on the support: plate top = lowest point of the upright object.
    mocap = model.body_mocapid[support]
    support_start = np.array([pos[0], pos[1], pos[2] - _object_half_height(model, objgeom) - model.geom_size[support_geom][2]])
    data.mocap_pos[mocap] = support_start
    data.mocap_quat[mocap] = [1, 0, 0, 0]
    data.ctrl[:] = 0
    mj.mj_forward(model, data)

    def is_pair(c, a, b):
        return (c.geom1 == a and c.geom2 == b) or (c.geom1 == b and c.geom2 == a)

    penetration = 0.0
    for c in data.contact:
        if is_pair(c, objgeom, support_geom):
            continue  # resting contact with the support is intended
        if objgeom in (c.geom1, c.geom2) or c.geom1 in upper_pad | lower_pad or c.geom2 in upper_pad | lower_pad:
            penetration = max(penetration, -float(c.dist))
    result = {"initial_penetration_mm": penetration * 1000, "success": 0, "pad_success": 0, "contact_fraction": 0.0,
              "pad_contact_fraction": 0.0, "max_drift_mm": 0.0, "max_rotation_deg": 0.0, "closing_shift_mm": 0.0,
              "lift_achieved_mm": 0.0, "mean_normal_force_N": 0.0, "max_tendon_force_N": 0.0, "floor_contact": 0,
              "support_contact": 0, "object_x_m": float(pos[0]), "object_y_m": float(pos[1]),
              "object_yaw_deg": math.degrees(2 * math.atan2(quat[3], quat[0]))}
    if penetration * 1000 > e["initial_penetration_tolerance_mm"]:
        return {**result, "status": "initial_overlap"}

    def object_in_base():
        rot = data.xmat[base].reshape(3, 3)
        return rot.T @ (data.xpos[obj] - data.xpos[base]), rot.T @ data.xmat[obj].reshape(3, 3)

    timestep = float(model.opt.timestep)
    close, lift_end, total = protocol_times(cfg)
    stride = max(1, round(e["sample_interval_s"] / timestep))
    window_start = total - min(e["measure_last_s"], e["hold_s"])
    start_rel, _ = object_in_base()
    reference = None
    contact_samples, pad_samples, forces = [], [], []
    max_drift = max_rot = max_tendon = 0.0
    floor_seen = support_seen = False
    lift_start_z = None
    viewer = None
    if visualize:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for step in range(int(math.ceil(total / timestep))):
            t = float(data.time)
            s = lift_fraction(cfg, t)
            data.ctrl[actuator] = e["control_max"] * min(1.0, t / max(e["ramp_s"], timestep))
            data.ctrl[lift_actuator] = e["lift_height_m"] * s
            data.mocap_pos[mocap] = support_start - [0, 0, e["support_drop_m"] * s]
            data.xfrc_applied[obj] = 0
            data.xfrc_applied[obj, :3] = disturbance_at(cfg, t)[0]
            mj.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or sum(int(w.number) for w in data.warning):
                return {**result, "status": "simulation_warning"}
            if viewer and step % 10 == 0:
                viewer.sync()
            if t < close:
                continue
            rel, rot = object_in_base()
            if reference is None:
                # Grasp as formed by closing: later slip is measured relative to the gripper base.
                reference = (rel.copy(), rot.copy())
                lift_start_z = float(data.xpos[obj][2])
                result["closing_shift_mm"] = float(np.linalg.norm(rel - start_rel) * 1000)
                if until == "closing":
                    touching = set()
                    for c in data.contact:
                        if c.dist <= 0 and objgeom in (c.geom1, c.geom2):
                            other = c.geom2 if c.geom1 == objgeom else c.geom1
                            body = int(model.geom_bodyid[other])
                            if other in upper_pad:
                                touching.add("upper_pad")
                            elif other in lower_pad:
                                touching.add("lower_pad")
                            elif body in upper | lower or body == base:
                                touching.add("link")
                    return {**result, "status": "closed", "closing_contacts": sorted(touching)}
            drift = float(np.linalg.norm(rel - reference[0]))
            if model.geom_type[objgeom] == mj.mjtGeom.mjGEOM_SPHERE:
                rotation = 0.0  # Sphere orientation is physically indistinguishable.
            elif model.geom_type[objgeom] in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
                rotation = math.degrees(math.acos(float(np.clip(abs(rot[:, 2] @ reference[1][:, 2]), 0, 1))))
            else:
                relative = reference[1].T @ rot
                rotation = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            max_drift, max_rot = max(max_drift, drift), max(max_rot, rotation)
            max_tendon = max(max_tendon, abs(float(data.actuator_force[actuator] * model.actuator_gear[actuator, 0])))
            for c in data.contact:
                if c.dist > 0:
                    continue
                if is_pair(c, objgeom, floor):
                    floor_seen = True
                if t >= lift_end and is_pair(c, objgeom, support_geom):
                    support_seen = True
            if t < window_start or step % stride:
                continue
            left_force = right_force = upper_pad_force = lower_pad_force = 0.0
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
                upper_pad_force += normal if other in upper_pad else 0.0
                lower_pad_force += normal if other in lower_pad else 0.0
            minimum = e["contact_force_min_N"]
            contact_samples.append(left_force >= minimum and right_force >= minimum)
            pad_samples.append(upper_pad_force >= minimum and lower_pad_force >= minimum)
            forces.append(left_force + right_force)
        fraction = float(np.mean(contact_samples)) if contact_samples else 0.0
        pad_fraction = float(np.mean(pad_samples)) if pad_samples else 0.0
        success = (fraction >= e["required_contact_fraction"] and max_drift * 1000 <= e["translation_limit_mm"]
                   and max_rot <= e["rotation_limit_deg"] and not floor_seen and not support_seen)
        pad_success = success and pad_fraction >= e["required_pad_contact_fraction"]
        if e.get("success_requires_pad_contact", False):
            success = pad_success
        return {**result, "status": "ok", "success": int(success), "pad_success": int(pad_success),
                "contact_fraction": fraction, "pad_contact_fraction": pad_fraction,
                "max_drift_mm": max_drift * 1000, "max_rotation_deg": max_rot,
                "lift_achieved_mm": (float(data.xpos[obj][2]) - lift_start_z) * 1000 if lift_start_z is not None else 0.0,
                "mean_normal_force_N": float(np.mean(forces)) if forces else 0.0,
                "max_tendon_force_N": max_tendon, "floor_contact": int(floor_seen), "support_contact": int(support_seen)}
    finally:
        if viewer is not None:
            viewer.close()


def summarize(rows):
    # Invalid initial geometry is a failure, never removed from the denominator.
    groups = {}
    for r in rows:
        groups.setdefault(r["object_name"], []).append(r["success"])
    return {"trials": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "pad_success_rate": float(np.mean([r["pad_success"] for r in rows])),
            "worst_object_success": min(float(np.mean(v)) for v in groups.values()),
            "contact_fraction": float(np.mean([r["contact_fraction"] for r in rows])),
            "pad_contact_fraction": float(np.mean([r["pad_contact_fraction"] for r in rows])),
            "mean_drift_mm": float(np.mean([r["max_drift_mm"] for r in rows])),
            "invalid_trials": sum(r["status"] != "ok" for r in rows)}
