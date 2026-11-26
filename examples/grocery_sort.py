"""Randomized SO-101 grocery sorting using live GPT plans and guarded placement."""

import argparse
import json
from pathlib import Path
import secrets

import mujoco as mj
import numpy as np
from PIL import Image

from gauntlet_windows.controller import Arm, RetryGrasp, StopEpisode, image_points, locate_block
from gauntlet_windows.native import (
    COLLISION, DEPTH, ENV_FORCE, FRAME, HOME, INSERT_FORCE, JOINTS, OBSTACLE_FORCE,
    OPEN, PAD_FORCE, READ, SIZE, WRITE, scene,
)
from detect_with_openai import plan, pixel_box, require_motion
from so101_grasp.feedback import ContactFeedback


ITEMS = {
    "carton": ("blue rectangular carton", "purple tray", (0.014, 0.015, 0.020), (0.08, 0.33, 0.96, 1)),
    "produce": ("red rounded produce", "green tray", (0.017, 0.017, 0.014), (0.91, 0.08, 0.10, 1)),
    "egg": ("white oval egg", "orange padded tray", (0.013, 0.018, 0.012), (0.97, 0.91, 0.81, 1)),
}
SLOTS = np.array([[0.155, -0.108], [0.244, -0.108], [0.222, -0.019]])
TRAYS = ((0.146, 0.098), (0.225, 0.105), (0.280, 0.045))
TRAY_COLORS = ((0.53, 0.23, 0.83, 1), (0.13, 0.64, 0.25, 1), (0.91, 0.40, 0.08, 1))
FLOOR = 0.007
RIM = 0.016
EGG_GRIP_N = 27.0
EGG_IMPACT_N = 12.0


def make_model(seed):
    spec = scene("cube_in_bowl", nominal=True)
    rng = np.random.default_rng(seed)
    slots = SLOTS[rng.permutation(len(SLOTS))] + rng.uniform(-0.004, 0.004, size=SLOTS.shape)
    for (kind, (_, _, size, color)), xy in zip(ITEMS.items(), slots):
        body = spec.worldbody.add_body(name=kind, pos=[*xy, size[2] + .0002])
        body.add_freejoint(name=kind + "_joint")
        contact = dict(name=kind + "_contact", type=mj.mjtGeom.mjGEOM_BOX,
                       size=size, mass=.042 if kind == "egg" else .063,
                       solref=[.009, 1], friction=[1.75, .015, .004])
        if kind == "carton":
            body.add_geom(rgba=color, **contact)
            body.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[.010,.009,.00025],
                          pos=[0,0,size[2]+.0001], rgba=[.70,.89,.98,1], mass=0,
                          contype=0, conaffinity=0)
        else:
            # Rounded groceries have flatter contact faces through their midsections.
            body.add_geom(rgba=[*color[:3], 0], **contact)
            body.add_geom(type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=size, rgba=color,
                          mass=0, contype=0, conaffinity=0)
    for kind, xy, color in zip(ITEMS, TRAYS, TRAY_COLORS):
        tray = spec.worldbody.add_body(name=kind+"_tray", pos=[*xy, 0])
        tray.add_geom(name=kind+"_floor", type=mj.mjtGeom.mjGEOM_BOX, pos=[0,0,FLOOR/2],
                      size=[.029,.029,FLOOR/2], rgba=color, solref=[.024,1] if kind=="egg" else [.012,1])
        for dim in (0, 1):
            for side in (-1, 1):
                center, half = [0, 0, (RIM+FLOOR)/2], [.030, .030, (RIM-FLOOR)/2]
                center[dim], half[dim] = side*.028, .002
                tray.add_geom(name=f"{kind}_rim_{dim}_{side}", type=mj.mjtGeom.mjGEOM_BOX,
                              pos=center, size=half, rgba=color)
    witness = spec.worldbody.add_body(name="witness", pos=[.319, -.048, .0142])
    witness.add_freejoint(name="witness_joint")
    witness.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[.012,.012,.014],
                     rgba=[.95,.76,.08,1], mass=.05)
    model = spec.compile()
    jaw = model.actuator("robot/gripper")
    jaw.gainprm[0] = 115.0
    jaw.biasprm[1] = -115.0
    return model


def prepare(seed=42):
    model = make_model(seed)
    data = mj.MjData(model)
    for name, value in zip(JOINTS, HOME):
        data.qpos[model.joint("robot/"+name).qposadr.item()] = value
        data.ctrl[model.actuator("robot/"+name).id] = value
    mj.mj_forward(model, data)
    return model, data


class GroceryLayer:
    def __init__(self, model, data, observe):
        self.model, self.data, self.observe = model, data, observe
        self.ctrl = [model.actuator("robot/"+name).id for name in JOINTS]
        self.jaw = model.actuator("robot/gripper")
        self.joints = [model.joint("robot/"+name).qposadr.item() for name in JOINTS]
        self.feedback = ContactFeedback(model, "carton", prefix="robot/")
        self.bodies = {model.body(name).id for name in ITEMS}
        self.witness = model.body("witness").id
        self.egg = model.body("egg").id
        self.protect = self.bodies | {self.witness}
        self.active = None
        self.renderer = mj.Renderer(model, SIZE, SIZE)
        self.tick = 0
        self.pads = np.zeros(2)
        self.peak = np.zeros(2)
        self.egg_peak = np.zeros(2)
        self.egg_impact = 0.0
        self.unexpected = 0.0
        self.contact = 0.0
        self.stop = None
        self.wrench = np.zeros(6)

    def select(self, name):
        self.active = self.model.body(name).id if name is not None else None
        self.feedback.targets = {self.active} if self.active is not None else set()
        torque_limit = 1.28 if name == "egg" else 2.94
        self.jaw.forcerange[:] = [-torque_limit, torque_limit]

    def begin_stepping(self):
        return self.measure()

    def measure(self):
        if self.tick % 2 == 0:
            self.renderer.update_scene(self.data, camera=FRAME)
            image = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            try:
                depth = self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering()
            self.images = image, depth
        if self.observe:
            self.observe(self.model, self.data, self.tick, self.pads, self.contact, (0, 0))
        return {**dict(zip(READ, self.data.qpos[self.joints])),
                FRAME: self.images[0], DEPTH: self.images[1], PAD_FORCE: self.pads.copy(),
                COLLISION: 0.0, INSERT_FORCE: (0, 0),
                ENV_FORCE: self.contact, OBSTACLE_FORCE: self.feedback.obstacle_n,
                f"{FRAME}_timestamp": self.tick * .020}

    def protect_contact(self):
        feedback = self.feedback
        for i, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            a, b = map(int, self.model.geom_bodyid[contact.geom])
            if self.egg not in (a, b) and not (
                    (a in self.protect and (feedback.robot[b] or b in self.protect)) or
                    (b in self.protect and feedback.robot[a])):
                continue
            mj.mj_contactForce(self.model, self.data, i, self.wrench)
            force = max(0.0, float(self.wrench[0]))
            egg_other = b if a == self.egg else a
            if self.egg in (a, b) and not (self.active == self.egg and egg_other in feedback.pads):
                self.egg_impact = max(self.egg_impact, force)
                if force >= EGG_IMPACT_N:
                    self.stop = f"Egg impact limit ({force:.1f} N)"
            robot_touch = feedback.robot[a] != feedback.robot[b]
            other = b if feedback.robot[a] else a
            if ((robot_touch and other in self.protect and other != self.active)
                or (a in self.protect and b in self.protect)):
                self.unexpected = max(self.unexpected, force)
                if force >= 2.5:
                    self.stop = f"Contact with unsorted or neighboring grocery ({force:.1f} N)"

    def step(self, commands):
        self.data.ctrl[self.ctrl] = [commands[name] for name in WRITE]
        for _ in range(10):
            mj.mj_step(self.model, self.data)
            self.feedback.read(self.data)
            self.pads[:] = self.feedback.pad_n
            self.peak = np.maximum(self.peak, self.pads)
            self.contact = self.feedback.obstacle_n
            if self.active == self.egg:
                self.egg_peak = np.maximum(self.egg_peak, self.pads)
                if self.pads.max() >= EGG_GRIP_N:
                    self.stop = f"Egg finger force limit ({self.pads.max():.1f} N)"
            self.protect_contact()
            if self.feedback.stop:
                self.stop = f"Collision: {self.feedback.stop['kind']}"
            if self.stop:
                self.tick += 1
                self.measure()
                raise StopEpisode(self.stop)
        self.tick += 1
        return self.measure()


def region(shape, detection):
    y0, x0, y1, x1 = pixel_box(detection, shape, padding=8)
    out = np.zeros(shape, bool)
    out[y0:y1, x0:x1] = True
    return out


def locate_grocery(measurement, decision):
    image, x, y, z = image_points(measurement)
    r, g, b = image.transpose(2, 0, 1)
    kind = decision["strategy"]
    found = decision["targets"]
    source, destination = found["source"], found["destination"]
    if kind == "carton":
        sx, sy, top, angle = locate_block(measurement, "blue", source)
    else:
        color = (r > 1.55*g) & (r > 1.5*b) if kind == "produce" else (
            (image.min(axis=2) > 135) & (r < 1.38*b)
        )
        mask = region(z.shape, source) & color & (z > .009) & (z < .042) & (y < .002)
        if mask.sum() < 22:
            raise StopEpisode(f"RGB-D could not confirm {kind} in GPT box")
        sx, sy = (float((a[mask].min()+a[mask].max()) / 2) for a in (x,y))
        top, angle = float(np.quantile(z[mask], .83)), 0.0
    if not (0.125 < sx < .279 and -.134 < sy < -.002):
        raise StopEpisode("Model pickup is outside the calibrated pickup area")
    colors = {"carton": (b > 1.30*g) & (r > 1.15*g),
              "produce": (g > 1.26*r) & (g > 1.22*b),
              "egg": (r > 1.15*g) & (g > 1.48*b)}
    rim = region(z.shape, destination) & colors[kind] & (abs(z-RIM) < .0026) & (y > .012)
    if rim.sum() < 40:
        raise StopEpisode(f"RGB-D could not confirm matching {ITEMS[kind][1]} in GPT box")
    dx, dy = (float((a[rim].min()+a[rim].max())/2) for a in (x,y))
    if not all(.047 < np.ptp(a[rim]) < .064 for a in (x,y)):
        raise StopEpisode("Destination perimeter is incomplete")
    return (sx, sy, top, angle), (dx, dy)


def perform(arm, layer, decision, status):
    kind = decision["strategy"]
    (x,y,top,angle), (dx,dy) = locate_grocery(arm.meas, decision)
    layer.select(kind)
    fragile = kind == "egg"
    arm.grip_target = 6.5 if fragile else 15.0
    grip_z = top - {"egg": .002, "produce": .009, "carton": .015}[kind]
    high = .078 if x > .23 else .089
    above = arm.approach([x,y,high], [x,y,grip_z], angle + np.pi/2)
    arm.move(above, OPEN, 39)
    arm.calibrate_tool()
    arm.target([x,y,grip_z], .74, 46)
    status(f"GPT chose {kind} | {'Gentle force feedback' if fragile else 'Force feedback'}")
    minimum = 3.0 if fragile else 4.0
    for jaw in np.linspace(.61, -.03, 144 if fragile else 122):
        arm.act(arm.q, jaw)
        if min(arm.pads) >= minimum:
            arm.hold_jaw = float(jaw)
            arm.wait(None, 9)
            if min(arm.pads) >= minimum:
                break
            arm.hold_jaw = None
    else:
        raise StopEpisode(f"{kind} did not establish bilateral grip")
    arm.guard = True
    status(f"{kind} | Lifting and transferring")
    arm.target([x,y,grip_z+.018], None, 23)
    arm.move(above, None, 32)
    transit = .077 if dx > .26 else .091
    for t in (.33, .66, 1.0):
        arm.target([x+t*(dx-x),y+t*(dy-y),high+t*(transit-high)], None, 28)
    release = grip_z+FLOOR+(.006 if fragile else .010)
    arm.target([dx,dy,release+.012], None, 38)
    arm.guard = False
    status(f"{kind} | Controlled placement")
    arm.target([dx,dy,release], None, 36 if fragile else 28)
    for jaw in np.linspace(float(arm.meas[READ[-1]]), .76, 25):
        arm.act(arm.q, float(jaw))
    arm.wait(.76, 12)
    arm.target([dx,dy,transit], .76, 31)
    arm.move(HOME[:5], OPEN, 48)
    arm.hold_jaw, arm.missing_pads, arm.yaw = None, 0, None
    arm.offset[:] = 0
    layer.select(None)


def run(output, *, seed=None, observe=None, status=print, planner=None):
    output.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbits(32) if seed is None else seed
    model, data = prepare(seed)
    layer = GroceryLayer(model, data, observe)
    initial = {k: data.body(k).xpos.copy() for k in (*ITEMS, "witness")}
    decisions = []
    remaining = dict(ITEMS)
    stop = None
    try:
        arm = Arm(layer)
        arm.wait(OPEN, 9)
        for count in range(1, len(ITEMS)+1):
            status(f"Item {count}/{len(ITEMS)} | GPT-6.2 Sol High is reading the overhead image")
            frame = output / f"{count:02d}-overhead.png"
            Image.fromarray(arm.meas[FRAME]).resize((768,768), Image.Resampling.BICUBIC).save(frame)
            instruction = (
                "Sort exactly ONE of the loose groceries. Pick the remaining item most accessible to the "
                "robot without disturbing others. The three low square trays are at the top-right and the "
                "groceries are scattered below them. Ignore the yellow cube at far right. The white egg "
                "is fragile; use its padded orange tray and gentle handling. Choose only a remaining "
                "category and the matching tray. Localize on the current image."
            )
            strategies = {name: f"Pick the loose {item[0]}; place into the {item[1]}."
                          for name, item in remaining.items()}
            targets = {"source": "Tight outer box of the chosen loose grocery, including its complete outline.",
                       "destination": "Tight outer box of the chosen matching square tray, with point at its center."}
            decision = (planner or plan)(frame, instruction, targets, strategies)
            (output / f"{count:02d}-plan.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
            decisions.append(decision)
            require_motion(decision)
            perform(arm, layer, decision, status)
            del remaining[decision["strategy"]]
        arm.wait(OPEN, 35)
    except (StopEpisode, RetryGrasp, RuntimeError, ValueError) as error:
        stop = str(error)
    finally:
        layer.renderer.close()
    results = {}
    for name in ITEMS:
        p = data.body(name).xpos
        center = data.body(name+"_tray").xpos
        dof = model.joint(name+"_joint").dofadr.item()
        settled = np.linalg.norm(data.qvel[dof:dof+6]) < .035
        results[name] = {"initial_position_m": initial[name].tolist(), "final_position_m": p.tolist(),
                         "correct_and_settled": bool(np.max(abs(p[:2]-center[:2])) < .016
                                                     and p[2] > FLOOR+.006 and settled)}
    drift = float(np.linalg.norm(data.body("witness").xpos[:2]-initial["witness"][:2]))
    correct = sum(item["correct_and_settled"] for item in results.values())
    result = {"success": correct == len(ITEMS) and stop is None and drift < .003,
              "correct": correct, "total": len(ITEMS), "seed": seed, "stop": stop,
              "model_plans": len(decisions), "model": decisions[0].get("model") if decisions else None,
              "pad_force_peak_n": layer.peak.tolist(), "egg_pad_force_peak_n": layer.egg_peak.tolist(),
              "egg_contact_peak_n": layer.egg_impact, "egg_limits_n": [EGG_GRIP_N, EGG_IMPACT_N],
              "neighbor_contact_peak_n": layer.unexpected, "witness_displacement_m": drift,
              "collision_peak_n": layer.unexpected, "simulation_seconds": data.time, "items": results}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = run(args.output, seed=args.seed)
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["success"])
