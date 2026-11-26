"""Contact-driven FR3 Duo cup service using the repository's canonical model."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "plugins" / "robotics" / "skills"
sys.path[:0] = [str(SKILLS / name / "scripts") for name in (
    "fr3-duo-control", "mujoco-playground", "rgb-object-localization",
)]
from detect_with_openai import EFFORT, MODEL, plan, require_motion
from fr3_mjcf import (
    DEFAULT_REST_POSE_RAD, FR3_IK_MARGINS_RAD, FR3_JOINT_LIMITS_RAD,
    arm_joint_qpos_and_dof_addrs, build_fr3_duo_spec, gripper_actuator_name,
    gripper_pinch_site_name,
)
from playground_sim2real import (
    camera_intrinsics_from_model, camera_world_pose_opencv, solve_site_ik,
)

SIZE = 704
CUP_HEIGHT = 0.066
SOURCE = (0.535, 0.244)
BREWER = (0.572, 0.0)
TRAY = (0.556, -0.272)
BREWER_TOP = 0.021
TRAY_TOP = 0.018
GRIP_HEIGHT = 0.047
DOWN = np.diag([1.0, -1.0, -1.0])
OPEN = 0.0
PAD_TARGET_N = 8.0
MIN_ACQUIRED_N = 2.0
COLLISION_STOP_N = 0.25
ARM_NAMES = ("left", "right")
DROP_COUNT = 64
DROP_RADIUS = 0.0048
DROP_PERIOD = 0.0315
DROP_MASS = 4 * np.pi * DROP_RADIUS**3 / 3 * 1000
POUR_LABEL = "Coffee machine | Pour espresso"


def geom(body, name, kind, position, size, color, **options):
    return body.add_geom(name=name, type=getattr(mj.mjtGeom, f"mjGEOM_{kind}"),
                         pos=position, size=size, rgba=color, **options)


def camera(spec, name, eye, focus, fovy):
    back = np.asarray(eye) - focus
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(name=name, pos=eye, fovy=fovy,
                             xyaxes=[*right, *np.cross(back, right)])


def make_model(*, table_offset_m=0.0):
    spec = build_fr3_duo_spec(menagerie_dir=ROOT / ".menagerie", add_gravcomp=True)
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002
    spec.option.iterations = 16
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    spec.visual.headlight.ambient = [0.42, 0.42, 0.42]
    world = spec.worldbody
    spec.geom("table").rgba = [0.38, 0.28, 0.20, 1]
    spec.geom("table").pos[2] += table_offset_m
    spec.geom("workspace_center_marker").rgba = [0, 0, 0, 0]
    geom(world, "floor", "PLANE", [0, 0, -0.75], [3, 3, 0.1], [0.16, 0.18, 0.2, 1])
    geom(world, "counter_front", "BOX", [0.90, 0, -0.30], [0.125, 0.60, 0.26],
         [0.18, 0.20, 0.23, 1])
    cup = world.add_body(name="cup", pos=[*SOURCE, 0.0002])
    cup.add_freejoint(name="cup_free")
    cyan = [0.04, 0.71, 0.78, 1]
    friction = [1.6, 0.02, 0.002]
    geom(cup, "cup_base", "CYLINDER", [0, 0, 0.004], [0.030, 0.004, 0], cyan,
         mass=0.047, friction=friction)
    # Overlapping wall segments leave the cup open and participate in contact.
    for i in range(16):
        angle = 2 * np.pi * i / 16
        geom(cup, f"cup_wall_{i:02d}", "BOX",
             [0.029 * np.cos(angle), 0.029 * np.sin(angle), 0.035],
             [0.003, 0.006, 0.031], cyan, euler=[0, 0, angle], mass=0.002,
             friction=friction)

    dark = [0.13, 0.16, 0.19, 1]
    steel = [0.69, 0.73, 0.75, 1]
    geom(world, "drip_platform", "BOX", [*BREWER, BREWER_TOP / 2],
         [0.068, 0.071, BREWER_TOP / 2], steel)
    for i in range(-3, 4):
        geom(world, f"drip_slot_{i}", "BOX", [BREWER[0]+i*0.014, 0, BREWER_TOP+0.0002],
             [0.002, 0.059, 0.0003], dark, contype=0, conaffinity=0)
    geom(world, "machine_body", "BOX", [0.800, 0, 0.154], [0.090, 0.119, 0.154], dark)
    geom(world, "machine_top", "BOX", [0.807, 0, 0.315], [0.075, 0.125, 0.019], steel)
    geom(world, "display", "BOX", [0.707, 0, 0.272], [0.005, 0.045, 0.019],
         [0.17, 0.55, 0.76, 1], contype=0, conaffinity=0)
    spout = world.add_body(name="swing_spout", pos=[0.708, 0, 0.159])
    spout.add_joint(name="spout_hinge", axis=[0, 0, 1], range=[-1.7, 0.1], damping=0.2)
    geom(spout, "spout_arm", "BOX", [-0.068, 0, 0], [0.069, 0.010, 0.010], steel, mass=0.05)
    geom(spout, "spout_tip", "CYLINDER", [-0.136, 0, -0.022], [0.009, 0.015, 0],
         dark, mass=0.02)
    actuator = spec.add_actuator(name="spout", target="spout_hinge", trntype=mj.mjtTrn.mjTRN_JOINT,
                                 ctrlrange=[-1.6, 0], ctrllimited=True)
    actuator.set_to_position(20, 0.8, -1)

    tray = [0.44, 0.19, 0.075, 1]
    geom(world, "tray_floor", "BOX", [*TRAY, TRAY_TOP / 2],
         [0.105, 0.10, TRAY_TOP / 2], tray)
    for i in (-1, 1):
        geom(world, f"tray_x_{i}", "BOX", [TRAY[0]+i*0.101, TRAY[1], 0.023],
             [0.004, 0.10, 0.007], tray)
        geom(world, f"tray_y_{i}", "BOX", [TRAY[0], TRAY[1]+i*0.096, 0.023],
             [0.10, 0.004, 0.007], tray)
    geom(world, "cup_mat", "CYLINDER", [*SOURCE, 0.0003], [0.052, 0.0003, 0], dark,
         contype=0, conaffinity=0)

    # Reserve the drops outside the workcell. The nozzle releases each as
    # a free body, after which only the physics solver moves it.
    espresso = ([0.28, 0.105, 0.040, 1], [0.38, 0.17, 0.068, 1],
                [0.49, 0.25, 0.108, 1])
    for i in range(DROP_COUNT):
        drop = world.add_body(name=f"coffee_drop_{i:03d}",
                              pos=[4 + 0.020 * (i % 12), 3 + 0.020 * (i // 12),
                                   -0.60])
        drop.add_freejoint(name=f"coffee_drop_{i:03d}")
        geom(drop, f"coffee_drop_{i:03d}", "SPHERE", [0, 0, 0], [DROP_RADIUS, 0, 0],
             espresso[i % len(espresso)], mass=DROP_MASS, priority=2, condim=1,
             friction=[0.08, 0.001, 0.0001], solref=[0.007, 1.3],
             solimp=[0.9, 0.96, 0.0008, 0.5, 2])

    spec.worldbody.add_camera(name="overhead", pos=[0.51, 0, 1.42], fovy=51,
                              xyaxes=[0, 1, 0, -1, 0, 0])
    camera(spec, "overview", [0.60, -1.70, 1.11], [0.34, 0, 0.22], 43)
    camera(spec, "counter", [0.22, 0.92, 0.64], [0.54, -0.04, 0.15], 49)
    camera(spec, "pour_detail", [0.34, -0.23, 0.265], [0.59, 0, 0.086], 43)
    return spec.compile()


def reset(model, data):
    mj.mj_resetData(model, data)
    for side, pose in DEFAULT_REST_POSE_RAD.items():
        qpos, _ = arm_joint_qpos_and_dof_addrs(model, side)
        data.qpos[qpos] = pose
        for i, value in enumerate(pose, start=1):
            data.ctrl[model.actuator(f"{side}/fr3_joint{i}").id] = value
    data.qpos[model.joint("spout_hinge").qposadr.item()] = -1.57
    data.ctrl[model.actuator("spout").id] = -1.57
    data.xfrc_applied[drop_bodies(model), :3] = -DROP_MASS * model.opt.gravity
    mj.mj_step(model, data, nstep=220)
    data.time = 0


def drop_bodies(model):
    return [model.body(f"coffee_drop_{i:03d}").id for i in range(DROP_COUNT)]


def observation(model, data, target_path):
    with mj.Renderer(model, height=SIZE, width=SIZE) as renderer:
        renderer.update_scene(data, camera="overhead")
        image = renderer.render().copy()
    Image.fromarray(image).save(target_path)
    return image


def locate(path, stage):
    task = ("Pick up the empty cup with the left arm and set it on the coffee platform."
            if stage == "source" else
            "Lift the filled cup from the coffee platform and keep it upright on the way to the tray.")
    return plan(
        path, task, {"source": "the single turquoise ceramic cup with a circular turquoise rim"},
        {"standard_transfer": "Use the normal upright transfer when cup access is clear.",
         "gentle_transfer": "Use a slower lift and transfer for a filled cup or a tight approach."},
    )


def project(model, data, image, detection, top_z):
    y0, x0, y1, x1 = np.round(np.array(detection["box_2d"]) * (SIZE - 1) / 1000).astype(int)
    color = image[y0:y1+1, x0:x1+1].astype(float)
    r, g, b = color.transpose(2, 0, 1)
    v, u = np.where((g > 62) & (b > 65) & (g > 1.45 * r) & (b > 1.4 * r))
    if u.size < 70:
        raise RuntimeError("Too little of the turquoise cup is visible in the model's box.")
    pixel = [float(u.mean() + x0), float(v.mean() + y0)]
    idx = model.camera("overhead").id
    intrinsics, *_ = camera_intrinsics_from_model(model, idx, SIZE, SIZE)
    pose = camera_world_pose_opencv(data, idx)
    direction = pose[:3, :3] @ np.linalg.solve(intrinsics, [*pixel, 1])
    origin = pose[:3, 3]
    return pixel, (origin + direction * ((top_z - origin[2]) / direction[2]))[:2]


def transfer(model, data, side, xy, z0, to_xy, z1, *, strategy="standard_transfer"):
    qpos, dofs = arm_joint_qpos_and_dof_addrs(model, side)
    site = model.site(gripper_pinch_site_name(side)).id
    actuators = [model.actuator(f"{side}/fr3_joint{i}").id for i in range(1, 8)]
    grip = model.actuator(gripper_actuator_name(side)).id
    home = np.array(DEFAULT_REST_POSE_RAD[side])
    lo, hi = FR3_JOINT_LIMITS_RAD.T
    lo, hi = lo + FR3_IK_MARGINS_RAD, hi - FR3_IK_MARGINS_RAD
    ctrl = data.ctrl.copy()
    q = data.qpos[qpos].copy()
    high = 0.217

    def joint_pose(point):
        return solve_site_ik(model, site, qpos, dofs, lo, hi, point,
                             q_seed=q, q_posture=home, context_qpos=data.qpos,
                             target_rot_world=DOWN, max_iters=150)

    def segment(label, target, seconds):
        nonlocal ctrl, q
        goal = ctrl.copy()
        goal[actuators] = target
        goal[grip] = OPEN
        n = round(seconds / model.opt.timestep)
        x = np.arange(1, n + 1)[:, None] / n
        blend = x * x * (3 - 2 * x)
        controls = ctrl + blend * (goal - ctrl)
        q, ctrl = target, goal
        return label, controls

    pickup = np.array([*xy, z0 + GRIP_HEIGHT])
    drop = np.array([*to_xy, z1 + GRIP_HEIGHT + 0.0015])
    prefix = side.title()
    lift_time, travel_time, set_down_time = (1.60, 2.3, 1.6) if side == "right" else (1.05, 1.35, 1.05)
    if strategy == "gentle_transfer":
        lift_time *= 1.2
        travel_time *= 1.2
    yield segment(f"{prefix} | Approach cup", joint_pose([*xy, high]), 1.2)
    yield segment(f"{prefix} | Lower", joint_pose(pickup), 1.05)
    yield segment(f"{prefix} | Close gripper", q, 1.35)
    yield segment(f"{prefix} | Lift", joint_pose([*xy, high]), lift_time)
    yield segment(f"{prefix} | Transfer", joint_pose([*to_xy, high]), travel_time)
    yield segment(f"{prefix} | Set cup down", joint_pose(drop), set_down_time)
    yield segment(f"{prefix} | Release", q, 0.65)
    yield segment(f"{prefix} | Retract", joint_pose([*to_xy, high]), 1.0)
    yield segment(f"{prefix} | Return home", home, 1.2)


def brewer_controls(model, data):
    ctrl = data.ctrl.copy()
    spout = model.actuator("spout").id
    dt = model.opt.timestep
    for text, angle, seconds in (
        ("Coffee machine | Swing spout over cup", 0.0, 0.75),
        ("Coffee machine | Align spout", 0.0, 0.45),
        (POUR_LABEL, 0.0, DROP_COUNT * DROP_PERIOD),
        ("Coffee machine | Let espresso settle", 0.0, 0.85),
        ("Coffee machine | Park spout", -1.57, 0.75),
    ):
        goal = ctrl.copy()
        goal[spout] = angle
        yield text, np.linspace(ctrl, goal, round(seconds / dt) + 1)[1:]
        ctrl = goal


def cup_supported(model, data, surface):
    support = model.geom(surface).id
    cup = model.body("cup").id
    return any(
        support in contact.geom and any(model.geom_bodyid[i] == cup for i in contact.geom)
        and contact.dist <= 0.0005
        for contact in data.contact
    )


def cup_position(data):
    return data.body("cup").xpos.copy()


def check_stop(model, data, surface, xy, height):
    pos = cup_position(data)
    upright = data.body("cup").xmat[8] > 0.94
    velocity_address = model.joint("cup_free").dofadr.item()
    motion = np.linalg.norm(data.qvel[velocity_address:velocity_address + 3])
    return bool(np.linalg.norm(pos[:2]-xy) < 0.035 and abs(pos[2]-height) < 0.008
                and cup_supported(model, data, surface) and upright and motion < 0.04)


@dataclass(frozen=True)
class VisionRequest:
    name: str
    top_z: float


class SafetyStop(RuntimeError):
    """A sim contact has latched the motion stop."""


class ContactFeedback:
    """Read pad and collision forces and regulate each gripper in sim."""

    def __init__(self, model, data, cup_geoms):
        self.model, self.data = model, data
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        self.arm_ids = np.array([0 if n.startswith("left/") else 1 if n.startswith("right/") else -1
                                 for n in self.names])
        self.fixtures = np.array([n.startswith(("table", "counter_", "machine_", "drip_platform",
                                                "tray_", "spout_")) for n in self.names])
        self.cup = np.isin(np.arange(model.ngeom), list(cup_geoms))
        self.pad_ids = np.array([
            2 * arm + int("/right_pad" in name)
            if arm >= 0 and ("/left_pad" in name or "/right_pad" in name) else -1
            for arm, name in zip(self.arm_ids, self.names)
        ])
        self.grippers = [model.actuator(gripper_actuator_name(arm)).id for arm in ARM_NAMES]
        self.pad_n = np.zeros((2, 2))
        self.filtered_n = np.zeros_like(self.pad_n)
        self.peaks_n = np.zeros_like(self.pad_n)
        self.hold_sum = np.zeros_like(self.pad_n)
        self.hold_count = np.zeros(2, dtype=int)
        self.command = np.zeros(2)
        self.command_range = np.array([[np.inf, 0.0], [np.inf, 0.0]])
        self.contact_sides = set()
        self.phase = ""
        self.phase_started = data.time
        self.stop = None
        self.samples = []
        self.next_sample = 0.0
        self.wrench = np.zeros(6)

    def apply(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.phase_started = self.data.time
            self.release_start = self.command.copy()
        side, _, action = phase.partition(" | ")
        if side not in ("Left", "Right"):
            return
        arm = int(side == "Right")
        dt = self.model.opt.timestep
        if action in ("Close gripper", "Lift", "Transfer", "Set cup down"):
            sample = self.filtered_n[arm]
            if action == "Lift" and self.data.time == self.phase_started and sample.min() < MIN_ACQUIRED_N:
                raise RuntimeError(f"{side} gripper did not establish force on both pads before lift.")
            # Integrate force error into position demand, with bounded jaw speed and travel.
            rate = np.clip(37 * (PAD_TARGET_N - sample.mean()), -310, 155)
            self.command[arm] = np.clip(self.command[arm] + dt * rate, OPEN, 215)
            low, high = self.command_range[arm]
            self.command_range[arm] = [min(low, self.command[arm]), max(high, self.command[arm])]
        elif action == "Release":
            elapsed = (self.data.time - self.phase_started + dt) / 0.65
            t = np.clip(elapsed, 0, 1)
            self.command[arm] = self.release_start[arm] * (1 - t * t * (3 - 2 * t))
        else:
            self.command[arm] = OPEN
        self.data.ctrl[self.grippers[arm]] = self.command[arm]

    def read(self):
        data, model = self.data, self.model
        contacts = data.contact
        geom = contacts.geom
        first, second = geom.T
        arm1, arm2 = self.arm_ids[first], self.arm_ids[second]
        connected = contacts.efc_address >= 0
        inter_arm = (arm1 >= 0) & (arm2 >= 0) & (arm1 != arm2)
        arm_fixture = ((arm1 >= 0) & self.fixtures[second]) | ((arm2 >= 0) & self.fixtures[first])
        hits = connected & (contacts.dist <= 0.0002) & (inter_arm | arm_fixture)
        max_collision = 0.0
        hazardous = None
        for i in np.flatnonzero(hits):
            mj.mj_contactForce(model, data, int(i), self.wrench)
            normal = float(self.wrench[0])
            max_collision = max(max_collision, normal)
            if normal >= COLLISION_STOP_N and (hazardous is None or normal > hazardous[0]):
                hazardous = (normal, i)

        pads = np.where(self.cup[first], self.pad_ids[second],
                        np.where(self.cup[second], self.pad_ids[first], -1))
        self.pad_n.fill(0)
        forces = self.pad_n.ravel()
        for i in np.flatnonzero(connected & (pads >= 0)):
            mj.mj_contactForce(model, data, int(i), self.wrench)
            forces[pads[i]] += max(0.0, float(self.wrench[0]))
        self.peaks_n = np.maximum(self.peaks_n, self.pad_n)
        dt = model.opt.timestep
        self.filtered_n += dt / (0.018 + dt) * (self.pad_n - self.filtered_n)
        for arm in range(2):
            if np.all(self.pad_n[arm] >= MIN_ACQUIRED_N):
                self.contact_sides.add(ARM_NAMES[arm])
        side, _, action = self.phase.partition(" | ")
        if side in ("Left", "Right") and action in ("Lift", "Transfer", "Set cup down"):
            arm = int(side == "Right")
            self.hold_sum[arm] += self.pad_n[arm]
            self.hold_count[arm] += 1

        if data.time >= self.next_sample - 1e-9 or hazardous is not None:
            self.next_sample = data.time + 0.01
            self.samples.append((round(data.time, 5), self.phase,
                                 *self.pad_n.ravel().round(4), *self.command.round(4),
                                 round(max_collision, 4)))
        if hazardous is not None:
            force, i = hazardous
            names = [self.names[first[i]], self.names[second[i]]]
            kind = "arm-arm" if inter_arm[i] else "arm-fixture"
            message = f"{kind} contact {force:.2f} N: {names[0]} / {names[1]}"
            self.stop = {"time_s": round(data.time, 6), "phase": self.phase,
                         "kind": kind, "geoms": names, "normal_force_n": round(force, 4),
                         "message": message}
            raise SafetyStop(message)

    def save(self, path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "phase", "left_arm_left_pad_n", "left_arm_right_pad_n",
                             "right_arm_left_pad_n", "right_arm_right_pad_n",
                             "left_gripper_command", "right_gripper_command", "collision_peak_normal_n"))
            writer.writerows(self.samples)
        return {
            "pad_target_n": PAD_TARGET_N,
            "collision_threshold_n": COLLISION_STOP_N,
            "stop": self.stop,
            "arms": {side: {
                "pad_peak_n": self.peaks_n[i].round(3).tolist(),
                "pad_mean_while_carrying_n": ((self.hold_sum[i] / self.hold_count[i]).round(3).tolist()
                                              if self.hold_count[i] else None),
                "feedback_command_range": (self.command_range[i].round(3).tolist()
                                            if np.isfinite(self.command_range[i, 0]) else None),
            } for i, side in enumerate(ARM_NAMES)},
        }


class CoffeeDrops:
    """Track the free-body drops in the cup's moving coordinate frame."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.bodies = drop_bodies(model)
        joints = [model.joint(f"coffee_drop_{i:03d}").id for i in range(DROP_COUNT)]
        self.positions = model.jnt_qposadr[joints, None] + np.arange(3)
        self.velocities = model.jnt_dofadr[joints, None] + np.arange(6)
        self.emitted = 0
        self.pour_started = None
        self.after_pour = False
        self.carrying = False
        self.spilled_during_carry = np.zeros(DROP_COUNT, dtype=bool)
        self.max_tilt_deg = 0.0

    def release_due(self):
        if self.pour_started is None:
            self.pour_started = self.data.time
        due = min(DROP_COUNT, int((self.data.time - self.pour_started + 1e-8) / DROP_PERIOD) + 1)
        outlet = self.data.geom("spout_tip").xpos + [0, 0, -0.015 - DROP_RADIUS - 0.001]
        while self.emitted < due:
            i = self.emitted
            # A small repeatable sideways offset prevents a column of coincident impacts.
            angle = i * 2.399963
            offset = [0.0018 * np.cos(angle), 0.0018 * np.sin(angle),
                      -(due - i - 1) * 2.2 * DROP_RADIUS]
            self.data.qpos[self.positions[i]] = outlet + offset
            self.data.qvel[self.velocities[i]] = [0, 0, -0.65, 0, 0, 0]
            self.data.xfrc_applied[self.bodies[i], :3] = 0
            self.emitted += 1

    def inside(self):
        cup = self.data.body("cup")
        local = (self.data.qpos[self.positions[:self.emitted]] - cup.xpos) @ cup.xmat.reshape(3, 3)
        radial = np.linalg.norm(local[:, :2], axis=1)
        return (radial <= 0.035) & (local[:, 2] >= -0.002) & (local[:, 2] <= CUP_HEIGHT + 0.003)

    def observe(self):
        if not self.carrying:
            return
        self.spilled_during_carry |= self.in_cup_after_pour & ~self.inside()
        tilt = np.degrees(np.arccos(np.clip(self.data.body("cup").xmat[8], -1, 1)))
        self.max_tilt_deg = max(self.max_tilt_deg, float(tilt))

    def finish_pour(self):
        if self.emitted != DROP_COUNT:
            raise RuntimeError(f"Expected {DROP_COUNT} drops from the nozzle, got {self.emitted}.")
        self.in_cup_after_pour = self.inside()
        self.after_pour = True
        if not np.any(self.in_cup_after_pour):
            raise RuntimeError("The cup is empty after pouring.")

    def summary(self):
        in_cup = int(np.count_nonzero(self.inside()))
        return {
            "model": "free-body rigid particles",
            "emitted": self.emitted,
            "nominal_volume_ml": round(float(self.emitted * DROP_MASS) * 1000, 2),
            "retained_after_pour": int(self.in_cup_after_pour.sum()) if self.after_pour else None,
            "left_cup_during_right_arm_transfer": int(self.spilled_during_carry.sum()),
            "in_cup_at_end": in_cup,
            "outside_cup_at_end": self.emitted - in_cup,
            "max_cup_tilt_during_right_arm_deg": round(self.max_tilt_deg, 2),
        }


class CoffeeCycle:
    """Share stage checks and commands between fast validation and Studio."""

    def __init__(self, model, data, output):
        self.model, self.data, self.output = model, data, output
        output.mkdir(parents=True, exist_ok=True)
        self.program = self.stages()
        self.result = {
            "mujoco": mj.__version__, "model": MODEL, "reasoning_effort": EFFORT,
            "vision": {}, "checkpoints": {}, "two_pad_contact": {},
        }
        self.liquid = CoffeeDrops(model, data)
        cup_geoms = np.flatnonzero(model.geom_bodyid == model.body("cup").id)
        self.feedback = ContactFeedback(model, data, cup_geoms)

    def target(self, request, image, decision):
        self.result["vision"][request.name] = {"plan": decision}
        require_motion(decision)
        detection = decision["targets"]["source"]
        pixel, xy = project(self.model, self.data, image, detection, request.top_z)
        self.result["vision"][request.name].update({
            "detection": detection, "pixel_uv": pixel, "xy_m": xy.tolist(),
        })
        print(f"{request.name}: GPT chose {decision['strategy']} at {xy.round(4).tolist()}", flush=True)
        return xy

    def step(self, phase, control):
        if self.feedback.stop is not None:
            raise SafetyStop(self.feedback.stop["message"])
        self.data.ctrl[:] = control
        self.feedback.apply(phase)
        if phase == POUR_LABEL:
            self.liquid.release_due()
        mj.mj_step(self.model, self.data)
        self.feedback.read()
        self.liquid.observe()

    def checkpoint(self, surface, xy, height):
        passed = check_stop(self.model, self.data, surface, xy, height)
        self.result["checkpoints"][surface] = {"passed": passed,
                                              "cup_xyz_m": cup_position(self.data).tolist()}
        if not passed:
            raise RuntimeError(f"Cup did not settle upright on {surface}.")
        print(f"Cup supported by {surface}.", flush=True)

    def stages(self):
        model, data = self.model, self.data
        source = VisionRequest("source", CUP_HEIGHT)
        image, decision = yield source
        source_xy = self.target(source, image, decision)
        yield from transfer(model, data, "left", source_xy, 0, BREWER, BREWER_TOP,
                            strategy=decision["strategy"])
        self.checkpoint("drip_platform", BREWER, BREWER_TOP)
        for label, controls in brewer_controls(model, data):
            yield label, controls
            if label.endswith("Align spout"):
                offset = np.linalg.norm(data.geom("spout_tip").xpos[:2] - cup_position(data)[:2])
                self.result["spout_to_cup_xy_mm"] = round(float(offset) * 1000, 2)
                if offset > 0.026:
                    raise RuntimeError("Cup did not align under the espresso spout.")
            if label.endswith("Let espresso settle"):
                self.liquid.finish_pour()
        station = VisionRequest("at_coffee_machine", BREWER_TOP + CUP_HEIGHT)
        image, decision = yield station
        station_xy = self.target(station, image, decision)
        self.liquid.carrying = True
        yield from transfer(model, data, "right", station_xy, BREWER_TOP, TRAY, TRAY_TOP,
                            strategy=decision["strategy"])
        self.liquid.carrying = False
        self.checkpoint("tray_floor", TRAY, TRAY_TOP)
        if self.feedback.contact_sides != {"left", "right"}:
            raise RuntimeError("Two-pad cup contact was not observed for both arms.")

    def save(self, error=None):
        self.result.update(
            success=error is None, simulation_seconds=round(self.data.time, 3),
            two_pad_contact={arm: arm in self.feedback.contact_sides for arm in ARM_NAMES},
            unintended_contacts=([self.feedback.stop["message"]] if self.feedback.stop else []),
            coffee=self.liquid.summary(),
            force_feedback=self.feedback.save(self.output / "forces.csv"),
        )
        if error is not None:
            self.result["error"] = str(error)
        (self.output / "result.json").write_text(json.dumps(self.result, indent=2), encoding="utf-8")
