"""YAM Pro workcell for a two-class, contact-driven cookie-sorting batch."""

import json
from pathlib import Path

import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DESIGN = Path(__file__).with_name("scene-plan.json")
ARM = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER = "gripper"
OPEN = 0.033
BELT_X, BELT_Z = 0.430, 0.039
PICK_Y, LEAD, PITCH = 0.045, 0.044, 0.075
COOKIE_HALF_Z = 0.010
BIN_FLOOR = 0.014
COUNT = 6
KINDS = ("orange", "dark", "dark", "orange", "dark", "orange")
COLORS = {"orange": [0.96, 0.22, 0.09, 1], "dark": [0.25, 0.12, 0.065, 1]}
BIN_COLORS = {"orange": [0.14, 0.37, 0.72, 1], "dark": [0.92, 0.75, 0.24, 1]}
CAMERA = np.array([0.32, -0.055, 1.04])
FOV = 56
PARK = np.array([-0.60, 1.25, 1.45, -1.40, 0.0, 1.1])


def design():
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def bin_centers(layout):
    # Both layouts keep enough clearance for the 4310 fingers and camera bracket.
    x = 0.300 if layout == "side_by_side" else 0.553
    return {"orange": (x, 0.237), "dark": (x, -0.237)}


def box(body, name, pos, half, rgba, **kwargs):
    return body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX,
                         pos=pos, size=half, rgba=rgba, **kwargs)


def camera(world, name, eye, target, fov):
    back = np.asarray(eye) - target
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    world.add_camera(name=name, pos=eye, fovy=fov, xyaxes=[*right, *np.cross(back, right)])


def make_model(kinds=KINDS, rng=None):
    asset = ROOT / ".menagerie" / "i2rt_yam" / "yam_linear_4310.xml"
    if not asset.is_file():
        raise FileNotFoundError("Run 'git -C .menagerie sparse-checkout add i2rt_yam' to obtain the YAM model.")
    spec = mj.MjSpec.from_file(str(asset))
    spec.option.timestep = 0.004
    spec.option.iterations = 35
    spec.option.impratio = 8
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 760
    spec.visual.headlight.ambient = [0.38] * 3
    world = spec.worldbody
    world.add_light(name="workcell_light", pos=[0.32, -0.32, 0.90], dir=[-0.10, 0.18, -1])
    box(world, "worktop", [0.22, 0, -0.030], [0.58, 0.65, 0.030], [0.76, 0.68, 0.52, 1])

    belt = world.add_body(name="belt", pos=[BELT_X, -0.225, BELT_Z - 0.008])
    belt.add_joint(name="belt_travel", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                   limited=True, range=[-0.002, 0.47], damping=8)
    box(belt, "moving_tread", [0, 0, 0], [0.058, 0.443, 0.008], [0.10, 0.39, 0.31, 1],
        mass=9.0, friction=[1.05, 0.008, 0.001])
    drive = spec.add_actuator(name="belt_drive", target="belt_travel", trntype=mj.mjtTrn.mjTRN_JOINT,
                              ctrllimited=True, ctrlrange=[0, 0.46])
    drive.set_to_position(3000, 220, -1)
    for i in range(35):
        box(belt, f"tread_strip_{i}", [0, (i-17)*0.025, 0.0081], [0.057, 0.0014, 0.0002],
            [0.19, 0.48, 0.41, 1], contype=0, conaffinity=0, mass=0)
    for side in (-1, 1):
        x = BELT_X + side * 0.063
        box(world, f"rail_{side}", [x, -0.013, 0.022], [0.004, 0.657, 0.017],
            [0.50, 0.53, 0.49, 1])
        for y in (PICK_Y-0.030, PICK_Y+0.030):
            box(world, f"pickup_mark_{side}_{y:.3f}", [x,y,BELT_Z+0.001], [0.005,0.003,0.001],
                [0.98,0.82,0.18,1], contype=0, conaffinity=0)

    for kind, (x,y) in bin_centers(design()["layout"]).items():
        tray = world.add_body(name=f"{kind}_bin", pos=[x, y, 0])
        color = BIN_COLORS[kind]
        box(tray, f"{kind}_bin_floor", [0,0,BIN_FLOOR/2], [0.061,0.101,BIN_FLOOR/2], color)
        for side in (-1,1):
            box(tray, f"{kind}_bin_long_{side}", [side*0.059,0,0.023],
                [0.003,0.101,0.013], color)
            box(tray, f"{kind}_bin_end_{side}", [0,side*0.098,0.023],
                [0.058,0.003,0.013], color)

    for i, kind in enumerate(kinds):
        name = f"cookie_{i}"
        dx, dy = rng.uniform(-0.003, 0.003, size=2) if rng is not None else (0, 0)
        item = world.add_body(name=name, pos=[BELT_X+dx, PICK_Y-LEAD-PITCH*i+dy, BELT_Z+COOKIE_HALF_Z+0.0003])
        item.add_freejoint(name=f"{name}_joint")
        box(item, f"{name}_wrapper", [0,0,0], [0.018,0.022,COOKIE_HALF_Z], COLORS[kind],
            mass=0.036, friction=[0.95,0.012,0.002])
        box(item, f"{name}_label", [0,0.001,COOKIE_HALF_Z+0.0002],
            [0.014 if kind == "orange" else 0.010,0.013,0.0003],
            [0.98,0.93,0.80,1], mass=0, contype=0, conaffinity=0)
        for j in range(6):
            box(item, f"{name}_label_line_{j}", [(j-2.5)*0.003,0.001,COOKIE_HALF_Z+0.0006],
                [0.00045,0.005,0.0001], COLORS[kind], mass=0, contype=0, conaffinity=0)

    world.add_camera(name="overhead", pos=CAMERA, fovy=FOV)
    camera(world,"overview", [1.05,-0.88,0.66], [0.32,-0.04,0.11], 49)
    camera(world,"pickup", [0.72,0.30,0.32], [0.40,0.025,0.07], 45)
    return spec.compile()


def addresses(model):
    joints = [model.joint(name).id for name in ARM]
    return model.jnt_qposadr[joints].tolist(), model.jnt_dofadr[joints].tolist()


def reset(model, data, pose):
    mj.mj_resetData(model, data)
    qpos, _ = addresses(model)
    data.qpos[qpos] = pose
    for name in ("left_finger", "right_finger"):
        data.qpos[model.joint(name).qposadr.item()] = OPEN
    data.ctrl[[model.actuator(name).id for name in ARM]] = pose
    data.ctrl[model.actuator(GRIPPER).id] = OPEN
    mj.mj_forward(model,data)
