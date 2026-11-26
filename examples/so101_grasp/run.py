"""Run a force-controlled SO-101 grasp in MuJoCo 4 and record the result."""

import argparse
import csv
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_grasp.feedback import ContactFeedback, PAD_TARGET_N, regulate

SKILLS = ROOT / "plugins" / "robotics" / "skills"
sys.path[:0] = [
    str(SKILLS / name / "scripts")
    for name in ("mujoco-playground", "rgb-object-localization")
]
from detect_with_openai import EFFORT, MODEL, plan, require_motion
from playground_sim2real import camera_intrinsics_from_model, camera_world_pose_opencv, solve_site_ik


CUBE_HALF = 0.014
SIZE = 640
OPEN = 0.72
CLOSE = 0.04
HOME = np.array([0.55, -0.55, 0.65, 1.1, 1.57])


def make_model():
    source = ROOT / ".menagerie" / "robotstudio_so101" / "so101.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = SIZE
    spec.visual.headlight.ambient = [0.42, 0.42, 0.42]
    spec.worldbody.add_light(pos=[0.12, -0.22, 0.72], dir=[0.1, 0.3, -1])
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.12, 0, -0.022], size=[0.42, 0.33, 0.022],
        rgba=[0.16, 0.22, 0.29, 1],
    )
    for name, xy, color in [
        ("red_cube", (0.205, -0.035), [0.95, 0.10, 0.08, 1]),
        ("blue_cube", (0.19, 0.085), [0.09, 0.48, 0.94, 1]),
    ]:
        body = spec.worldbody.add_body(name=name, pos=[*xy, CUBE_HALF + 0.0002])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(
            name=f"{name}_surface", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF] * 3, rgba=color, mass=0.02, friction=[1, 0.01, 0.001],
        )
    spec.worldbody.add_camera(name="overhead", pos=[0.19, 0, 0.54], fovy=37)
    eye = np.array([0.70, -0.55, 0.39])
    back = eye - np.array([0.15, 0.0, 0.11])
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(
        name="overview", pos=eye, xyaxes=[*right, *np.cross(back, right)], fovy=34,
    )
    return spec.compile()


def photo(renderer, data, camera):
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def localize(rgb, detection, model, data):
    scale = (SIZE - 1) / 1000
    y0, x0, y1, x1 = np.round(np.array(detection["box_2d"]) * scale).astype(int)
    # The model selects the object. A red mask inside its box refines the pixel centroid.
    crop = rgb[y0:y1 + 1, x0:x1 + 1].astype(float)
    ys, xs = np.where((crop[:, :, 0] > 85) & (crop[:, :, 0] > 1.8 * crop[:, :, 1])
                      & (crop[:, :, 0] > 1.8 * crop[:, :, 2]))
    if not len(xs):
        raise RuntimeError("No red pixels inside the model's target box.")
    u, v = xs.mean() + x0, ys.mean() + y0
    camera = model.camera("overhead").id
    intrinsics, *_ = camera_intrinsics_from_model(model, camera, SIZE, SIZE)
    transform = camera_world_pose_opencv(data, camera)
    origin = transform[:3, 3]
    ray = transform[:3, :3] @ np.linalg.solve(intrinsics, [u, v, 1])
    point = origin + ray * ((2 * CUBE_HALF - origin[2]) / ray[2])
    return (float(u), float(v)), point


def reset_scene(model, data, red_xy=None):
    mujoco.mj_resetData(model, data)
    qpos = model.jnt_qposadr[model.actuator_trnid[:5, 0]]
    data.qpos[qpos] = HOME
    data.qpos[model.joint("gripper").qposadr] = OPEN
    if red_xy is not None:
        address = model.joint("red_cube_free").qposadr.item()
        data.qpos[address:address + 2] = red_xy
    data.ctrl[:] = [*HOME, OPEN]
    mujoco.mj_step(model, data, nstep=100)
    data.time = 0


def overhead_observation(model, data, destination):
    with mujoco.Renderer(model, height=SIZE, width=SIZE) as overhead:
        rgb = photo(overhead, data, "overhead")
    Image.fromarray(rgb).save(destination)
    return rgb


def plan_grasp(observation):
    return plan(
        observation, "Pick up the red cube and hold it clear of the table.",
        {"source": "red cube on the dark table, not the yellow robot or blue cube"},
        {"direct_lift": "Centered top-down pickup when the red cube is isolated.",
         "slow_lift": "Slower final approach and lift for a tight or partially obstructed pickup."},
    )


def grasp_phases(model, xy, *, grip_height=CUBE_HALF, strategy="direct_lift"):
    site = model.site("gripperframe").id
    arm_joints = model.actuator_trnid[:5, 0]
    qpos = model.jnt_qposadr[arm_joints]
    dofs = model.jnt_dofadr[arm_joints]
    lo, hi = model.actuator_ctrlrange[:5].T
    seed = np.array([0.1, 0.0, 0.47, 1.1, 1.57])

    def ik(point, start):
        return solve_site_ik(
            model, site, qpos, dofs, lo, hi, point, q_seed=start,
            q_posture=seed, approach_axis_ee=(1, 0, 0),
        )

    pick = np.array([*xy, grip_height])
    above = ik(pick + [0, 0, 0.070], HOME)
    grasp = ik(pick, above)
    start = np.array([*HOME, OPEN])
    for label, end, grip, seconds in (
        ("SO-101 | Observe", HOME, OPEN, 0.55),
        ("01  Approach", above, OPEN, 0.8),
        ("02  Lower", grasp, OPEN, 1.25),
        ("03  Close gripper", grasp, CLOSE, 0.75),
        ("04  Lift", above, CLOSE, 1.2),
        ("05  Hold", above, CLOSE, 1.2),
    ):
        if strategy == "slow_lift" and label in ("02  Lower", "04  Lift"):
            seconds *= 1.3
        count = round(seconds / model.opt.timestep)
        t = np.arange(1, count + 1)[:, None] / count
        mix = t * t * (3 - 2 * t)
        goal = np.array([*end, grip])
        yield label, (1 - mix) * start + mix * goal
        start = goal


class GraspFeedback:
    """Apply the planned arm trajectory and regulate the jaw each physics step."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.sensor = ContactFeedback(model, "red_cube")
        self.jaw = None
        self.low_force_ticks = 0
        self.peak = np.zeros(2)
        self.samples = []
        self.next_sample = 0.0

    def step(self, phase, command):
        if self.sensor.stop is not None:
            raise RuntimeError(self.stop_message())
        if phase in ("03  Close gripper", "04  Lift", "05  Hold"):
            if self.jaw is None and min(self.sensor.pad_n) >= 2.0:
                self.jaw = float(command[-1])
            if phase != "03  Close gripper":
                if self.jaw is None:
                    raise RuntimeError("No stable two-finger grip before lift")
                self.low_force_ticks = self.low_force_ticks + 1 if min(self.sensor.pad_n) < 2.0 else 0
                if self.low_force_ticks * self.model.opt.timestep >= 0.12:
                    raise RuntimeError("Object slipped during lift")
        self.data.ctrl[:] = command
        if self.jaw is not None:
            self.jaw = regulate(self.jaw, self.sensor.pad_n, self.model.opt.timestep)
            self.data.ctrl[-1] = self.jaw
        mujoco.mj_step(self.model, self.data)
        self.sensor.read(self.data)
        self.peak = np.maximum(self.peak, self.sensor.pad_n)
        if self.data.time >= self.next_sample - 1e-9 or self.sensor.stop is not None:
            self.next_sample = self.data.time + 0.02
            self.samples.append((self.data.time, phase, *self.sensor.pad_n,
                                 self.data.ctrl[-1], self.sensor.obstacle_n))
        if self.sensor.stop is not None:
            raise RuntimeError(self.stop_message())

    def stop_message(self):
        stop = self.sensor.stop
        return f"{stop['kind']} {stop['normal_force_n']:.2f} N | {' / '.join(stop['geoms'])}"

    def save(self, path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "phase", "fixed_n", "moving_n", "jaw_command_rad", "fixture_n"))
            writer.writerows(self.samples)
        return {"target_per_pad_n": PAD_TARGET_N, "pad_peak_n": self.peak.tolist(),
                "two_finger_grip": self.jaw is not None, "stop": self.sensor.stop}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    model = make_model()
    data = mujoco.MjData(model)
    reset_scene(model, data)
    site = model.site("gripperframe").id
    feedback = GraspFeedback(model, data)
    cube = data.body("red_cube")
    start_z = float(cube.xpos[2])

    observation = output / "observation.png"
    rgb = overhead_observation(model, data, observation)
    print(f"Requesting RGB detection with {MODEL} / {EFFORT}", flush=True)
    decision = plan_grasp(observation)
    (output / "vision-plan.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    require_motion(decision)
    detection = decision["targets"]["source"]
    uv, point = localize(rgb, detection, model, data)
    print(f"Detected {detection}, projected XY: {point[:2]}", flush=True)

    peak = start_z
    tick = 0
    font = ImageFont.load_default(size=19)
    stopped = None
    with (mujoco.Renderer(model, height=540, width=960) as camera,
          imageio.get_writer(output / "so101-grasp.mp4", fps=25,
                             codec="libx264", macro_block_size=2, quality=8) as video):
        for label, controls in grasp_phases(model, point[:2], strategy=decision["strategy"]):
            for control in controls:
                try:
                    feedback.step(label, control)
                except RuntimeError as error:
                    stopped = str(error)
                    break
                peak = max(peak, float(cube.xpos[2]))
                if tick % 8 == 0:
                    frame = Image.fromarray(photo(camera, data, "overview"))
                    draw = ImageDraw.Draw(frame)
                    draw.rounded_rectangle((18, 18, 245, 60), radius=8, fill="#162232")
                    draw.text((31, 29), label, fill="white", font=font)
                    video.append_data(np.asarray(frame))
                tick += 1
            print(f"{label}: TCP {np.round(data.site_xpos[site], 4)}, cube z={cube.xpos[2]:.4f}", flush=True)
            if stopped:
                break

        Image.fromarray(photo(camera, data, "overview")).save(output / "lift.png")

    final_lift = float(cube.xpos[2]) - start_z
    result = {
        "success": stopped is None and final_lift >= 0.05 and feedback.jaw is not None,
        "mujoco": mujoco.__version__,
        "model": MODEL, "reasoning_effort": EFFORT,
        "detection": detection, "refined_pixel_uv": uv,
        "vision_plan": decision,
        "estimated_target_xy_m": point[:2].tolist(),
        "peak_lift_mm": round(1000 * (peak - start_z), 2),
        "held_lift_mm": round(1000 * final_lift, 2),
        "hold_seconds": 1.2,
        "observed_contact_on_both_jaws": feedback.jaw is not None,
        "force_feedback": feedback.save(output / "forces.csv"),
        "error": stopped,
        "object_constraint_count": model.neq,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["success"]:
        raise RuntimeError("The cube did not remain at least 50 mm above its start height with a two-jaw grasp.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "so101-grasp")
    run(parser.parse_args().output)
