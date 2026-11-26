"""Run a YAM Pro sorting batch in MuJoCo with live OpenAI RGB plans."""

import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "yam_cookie"

from . import scene

SKILLS = scene.ROOT / "plugins" / "robotics" / "skills"
sys.path[:0] = [str(SKILLS / name / "scripts")
                for name in ("mujoco-playground", "rgb-object-localization")]
from detect_with_openai import plan, pixel_box, require_motion
from playground_sim2real import camera_intrinsics_from_model, solve_site_ik


SIZE = 704
FORCE_TARGET = 7.0
COLLISION_LIMIT = 8.0
HOVER = 0.128
TASK = (
    "Operate the YAM Pro two-class cookie sorter. The belt is stationary at the index. "
    "Inspect the overhead image; the green belt runs vertically, to the right of both bins. "
    "Select ONLY the cookie package between the upper and lower pairs of yellow pickup ticks, "
    "approximately halfway down the image. Ignore the queued packages further down the belt. "
    "A pale cream label with a bright orange-red sleeve goes into the blue bin; "
    "a dark-brown wrapper with a smaller cream label goes into the yellow bin. "
    "Choose stop if the pickup point, type or matching bin is unclear or blocked."
)
TARGETS = {"source": "The complete outer silhouette of the single wrapped cookie between the "
                     "pickup ticks. The box should be tight, including the colored wrapper.",
           "destination": "The complete outer rectangle of the corresponding blue or yellow "
                          "receiving bin. Put the point at the center of its opening."}
STRATEGIES = {"orange": "Pick the pale-label orange-red package; place in the blue bin.",
              "dark": "Pick the dark-brown package; place in the yellow bin."}


class SortStop(RuntimeError):
    pass


class Sorter:
    def __init__(self, output, seed, observe, status):
        self.output, self.observe, self.status = output, observe, status
        rng = np.random.default_rng(seed)
        self.kinds = rng.permutation(scene.KINDS).tolist()
        self.model = model = scene.make_model(self.kinds, rng)
        self.data = mj.MjData(model)
        self.ik_data = mj.MjData(model)
        scene.reset(model, self.data, scene.PARK)
        self.qpos, self.dofs = scene.addresses(model)
        self.bounds = model.jnt_range[[model.joint(name).id for name in scene.ARM]].T
        self.arm_ctrl = [model.actuator(name).id for name in scene.ARM]
        self.site = model.site("grasp_site").id
        self.jaw_ctrl = model.actuator(scene.GRIPPER).id
        self.belt_ctrl = model.actuator("belt_drive").id
        self.belt_pos = model.joint("belt_travel").qposadr.item()
        self.pads = {model.geom(f"{side}_{part}").id: i
                     for i, side in enumerate(("left", "right"))
                     for part in ("pad", "pad_edge_a", "pad_edge_b")}
        self.robot = model.body_rootid[model.geom_bodyid] == model.body("arm").id
        self.cookies = {model.geom(f"cookie_{i}_wrapper").id for i in range(scene.COUNT)}
        self.tick = 0
        self.payload = None
        self.force_peak = np.zeros(2)
        self.contact_peak = 0.0
        self.contact_pair = None
        self.force = np.zeros(6)
        self.detached_ticks = 0
        self.grasping = False
        self.carrying = False
        self.placed = dict.fromkeys(scene.COLORS, 0)
        self.requests = []
        self.rows, self.cols = np.mgrid[:SIZE, :SIZE]
        self.speed = scene.design()["belt_speed_m_s"]
        self.centers = scene.bin_centers(scene.design()["layout"])

    def capture(self, pads=(0, 0), environment=0.0):
        if self.observe:
            self.observe(self.model, self.data, self.tick, pads, environment, (0, 0))

    def step(self):
        model, data = self.model, self.data
        mj.mj_step(model, data)
        self.tick += 1
        pads = np.zeros(2)
        environment = 0.0
        pair = None
        for index, contact in enumerate(data.contact):
            a, b = contact.geom
            if not (self.robot[a] or self.robot[b]) or contact.efc_address < 0:
                continue
            mj.mj_contactForce(model, data, index, self.force)
            magnitude = max(0.0, self.force[0])
            if not magnitude:
                continue
            expected_grip = False
            for finger, item in ((a, b), (b, a)):
                if finger in self.pads and item in self.cookies:
                    if self.grasping and self.payload is None:
                        self.payload = item
                    if item == self.payload:
                        expected_grip = True
                        pads[self.pads[finger]] += magnitude
            if not expected_grip and magnitude > environment:
                environment = magnitude
                pair = f"{model.geom(a).name} / {model.geom(b).name}"
        self.force_peak = np.maximum(self.force_peak, pads)
        if environment > self.contact_peak:
            self.contact_peak, self.contact_pair = environment, pair
        if self.tick % 5 == 0 or environment >= COLLISION_LIMIT:
            self.capture(pads, environment)
        if environment >= COLLISION_LIMIT:
            raise SortStop(f"Collision {pair}: {environment:.1f} N")
        if pads.max() > 24:
            raise SortStop(f"Gripper load exceeded 24 N ({pads.max():.1f} N)")
        if self.carrying:
            self.detached_ticks = self.detached_ticks + 1 if min(pads) < 0.8 else 0
            if self.detached_ticks * model.opt.timestep >= 0.10:
                raise SortStop("Grip lost while carrying")
            data.ctrl[self.jaw_ctrl] = np.clip(
                data.ctrl[self.jaw_ctrl] + model.opt.timestep * np.clip(
                    0.003 * (min(pads) - FORCE_TARGET), -0.018, 0.025),
                *model.actuator_ctrlrange[self.jaw_ctrl])
        return pads

    def hold(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            self.step()

    def index(self, index):
        target = scene.LEAD + scene.PITCH * index
        ctrl = self.data.ctrl
        delta = target - ctrl[self.belt_ctrl]
        # Cosine ramp limits starts and stops without moving the arm during camera capture.
        steps = int(np.ceil(delta * np.pi / (2 * self.speed * self.model.opt.timestep)))
        start = ctrl[self.belt_ctrl]
        for i in range(1, steps + 1):
            ctrl[self.belt_ctrl] = start + delta * (1 - np.cos(np.pi * i / steps)) / 2
            self.step()
        self.hold(0.30)
        belt_dof = self.model.joint("belt_travel").dofadr.item()
        if abs(self.data.qpos[self.belt_pos] - target) > 0.004 or abs(self.data.qvel[belt_dof]) > 0.004:
            raise SortStop("Belt did not settle at the pickup index")

    def joint_move(self, pose):
        start = self.data.qpos[self.qpos].copy()
        duration = max(0.40, np.max(np.abs(pose - start)) / 0.86)
        for t in np.linspace(0, 1, round(duration / self.model.opt.timestep) + 1)[1:]:
            self.data.ctrl[self.arm_ctrl] = start + (pose - start) * (3*t*t - 2*t*t*t)
            self.step()
        self.hold(0.30)
        if np.max(np.abs(self.data.qpos[self.qpos] - pose)) > 0.045:
            raise SortStop("Arm did not reach the commanded posture")

    def move(self, xyz):
        model, data = self.model, self.data
        # Keep the wrist roll away from its limit as the base swings between bins.
        yaw = np.arctan2(xyz[1], xyz[0]) + 1.20
        c, s = np.cos(yaw), np.sin(yaw)
        down = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])
        pose = solve_site_ik(model, self.site, self.qpos, self.dofs, *self.bounds, xyz,
                             q_seed=data.qpos[self.qpos], context_qpos=data.qpos,
                             target_rot_world=down)
        self.ik_data.qpos[:] = data.qpos
        self.ik_data.qpos[self.qpos] = pose
        mj.mj_kinematics(model, self.ik_data)
        actual = self.ik_data.site_xmat[self.site].reshape(3, 3)
        if (np.linalg.norm(self.ik_data.site_xpos[self.site] - xyz) > 0.003
            or np.trace(actual.T @ down) < 2.985):
            raise SortStop(f"Target outside validated tool pose range ({np.round(xyz, 3)})")
        self.joint_move(pose)

    def descend(self, xy, start, end):
        for z in np.linspace(start, end, 4)[1:]:
            self.move((*xy, z))

    def close(self):
        ctrl = self.data.ctrl
        self.grasping = True
        for _ in range(295):
            ctrl[self.jaw_ctrl] -= 0.028 * self.model.opt.timestep
            if ctrl[self.jaw_ctrl] < 0.001:
                break
            pads = self.step()
            if min(pads) >= 3:
                self.grasping = False
                self.carrying = True
                self.hold(0.22)
                return
        raise SortStop("Failed to establish bilateral finger contact")

    def open(self):
        self.carrying = False
        self.detached_ticks = 0
        ctrl = self.data.ctrl
        for jaw in np.linspace(ctrl[self.jaw_ctrl], scene.OPEN, 75)[1:]:
            ctrl[self.jaw_ctrl] = jaw
            self.step()
        self.payload = None

    def observe_scene(self, renderer, index):
        image = self.output / f"{index+1:02d}-overhead.png"
        renderer.update_scene(self.data, camera="overhead")
        self.hold(0.06)
        renderer.update_scene(self.data, camera="overhead")
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(image)
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        self.status(f"Cookie {index+1}/{scene.COUNT} | GPT-6.2 Sol High: reading overhead frame")
        result = plan(image, TASK, TARGETS, STRATEGIES)
        (self.output / f"{index+1:02d}-plan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.requests.append(result)
        require_motion(result)
        return result, rgb, depth

    def world_points(self, depth):
        camera = self.model.camera("overhead").id
        _, fx, fy, cx, cy = camera_intrinsics_from_model(self.model, camera, SIZE, SIZE)
        x = scene.CAMERA[0] + (self.cols + 0.5 - cx) * depth / fx
        y = scene.CAMERA[1] - (self.rows + 0.5 - cy) * depth / fy
        return x, y, scene.CAMERA[2] - depth

    @staticmethod
    def crop(shape, detection, pad):
        crop = np.zeros(shape, dtype=bool)
        y0, x0, y1, x1 = pixel_box(detection, shape, padding=pad)
        crop[y0:y1, x0:x1] = True
        return crop

    def localize(self, result, rgb, depth):
        x, y, z = self.world_points(depth)
        targets = result["targets"]
        pickup = self.crop(z.shape, targets["source"], 11)
        pickup &= ((abs(y - scene.PICK_Y) < 0.033) & (abs(x - scene.BELT_X) < 0.033)
                   & (z > scene.BELT_Z + 0.017) & (z < scene.BELT_Z + 0.025))
        if np.count_nonzero(pickup) < 220:
            raise SortStop("Model's pickup box has insufficient depth evidence")
        px, py, pz = (float(np.median(a[pickup])) for a in (x, y, z))

        kind = result["strategy"]
        destination = self.crop(z.shape, targets["destination"], 14)
        r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
        color = (b > r + 32) if kind == "orange" else ((r > b + 45) & (g > b + 25))
        destination &= (z > 0.034) & (z < 0.038) & color
        if np.count_nonzero(destination) < 250:
            raise SortStop("Model's selected bin has insufficient matching wall depth and color")
        limits = [(float(a[destination].min()), float(a[destination].max())) for a in (x, y)]
        center = [sum(interval)/2 for interval in limits]
        if (any(abs(actual - expected) > 0.018 for actual, expected in zip(center, self.centers[kind]))
            or not 0.104 < limits[0][1]-limits[0][0] < 0.137
            or not 0.181 < limits[1][1]-limits[1][0] < 0.215):
            raise SortStop("Bin geometry is outside the calibrated safe placement region")
        if self.placed[kind] >= 3:
            raise SortStop("Matching bin is full")
        center[1] += (-0.052, 0.0, 0.052)[self.placed[kind]]
        return (px, py, pz + 0.001), tuple(center)

    def sort_one(self, renderer, index):
        count = f"Cookie {index+1}/{scene.COUNT}"
        self.status(f"{count} | Indexing conveyor")
        self.index(index)
        result, rgb, depth = self.observe_scene(renderer, index)
        source, destination = self.localize(result, rgb, depth)
        kind = result["strategy"]
        bin_name = "blue" if kind == "orange" else "yellow"
        self.status(f"{count} | GPT chose {bin_name} bin. Picking with force feedback.")
        self.move((*source[:2], HOVER))
        self.descend(source[:2], HOVER, source[2])
        self.close()
        self.descend(source[:2], source[2], HOVER)
        self.move((*destination, HOVER))
        self.descend(destination, HOVER, 0.040)
        self.open()
        self.descend(destination, 0.040, HOVER)
        self.joint_move(scene.PARK)
        self.placed[kind] += 1

    def grade(self):
        outcomes = []
        for i, kind in enumerate(self.kinds):
            body = self.data.body(f"cookie_{i}")
            x, y, z = body.xpos
            cx, cy = self.centers[kind]
            joint = self.model.joint(f"cookie_{i}_joint").dofadr.item()
            landed = (abs(x - cx) < 0.045 and abs(y - cy) < 0.080 and
                      abs(z - scene.BIN_FLOOR - scene.COOKIE_HALF_Z) < 0.003 and
                      np.linalg.norm(self.data.qvel[joint:joint+6]) < 0.03)
            outcomes.append({"cookie": i+1, "class": kind, "correct_bin_and_settled": bool(landed),
                             "position_m": body.xpos.tolist()})
        return outcomes


def run(output, *, seed=None, observe=None, status=print):
    output.mkdir(parents=True, exist_ok=True)
    seed = int(np.random.SeedSequence().entropy) if seed is None else seed
    batch = Sorter(output, seed, observe, status)
    stop = None
    batch.capture()
    try:
        with mj.Renderer(batch.model, SIZE, SIZE) as renderer:
            batch.hold(0.20)
            for index in range(scene.COUNT):
                batch.sort_one(renderer, index)
            batch.hold(0.50)
    except (RuntimeError, ValueError) as error:
        stop = str(error)
    outcomes = batch.grade()
    correct = sum(item["correct_bin_and_settled"] for item in outcomes)
    result = {"success": correct == scene.COUNT and stop is None, "seed": seed,
              "correct": correct, "total": scene.COUNT, "stop": stop, "model_plans": len(batch.requests),
              "pad_force_peak_n": batch.force_peak.tolist(),
              "collision_peak_n": batch.contact_peak, "collision_pair": batch.contact_pair,
              "simulation_seconds": batch.data.time, "cookies": outcomes}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = run(args.output, seed=args.seed)
    print(json.dumps(result, indent=2))
    sys.exit(not result["success"])
