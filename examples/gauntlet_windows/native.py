"""Sim-only SO-101 DeviceLayer for the repository's MuJoCo Gauntlet."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

from dm_env import specs
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "plugins" / "robotics" / "skills"
GAUNTLET = SKILLS / "sim2real-gauntlet"
sys.path[:0] = [str(GAUNTLET / "src"), str(GAUNTLET / "examples"),
                str(SKILLS / "mujoco-playground" / "scripts"),
                str(SKILLS / "rgb-object-localization" / "scripts")]
from robosim_gauntlet import perturb
from robosim_gauntlet.task import SafetyLimits
from so101_grasp.feedback import ContactFeedback
from task_props import _bowl, _box, _in_socket, _inside_bowl, _socket, _stacked
from . import extra_tasks

TASKS = {
    "cube_in_bowl": ("target_cube", "bowl", "red", "blue"),
    "peg_in_plate": ("peg", "plate_socket", "green", "purple"),
    "two_block_stack": ("block_top", "block_base", "orange", "blue"),
    **extra_tasks.EXTRA_TASKS,
}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
READ = [f"joint_position/{j}" for j in JOINTS]
WRITE = [f"goal_position/{j}" for j in JOINTS]
FRAME, DEPTH = "overhead_cam", "overhead_cam_depth"
PAD_FORCE, ENV_FORCE = "finger_normal_n", "environment_contact_n"
OBSTACLE_FORCE = "obstacle_normal_n"
INSERT_FORCE, COLLISION = "insertion_axial_lateral_n", "collision_stop_n"
INSERT_DIRECTION = "insertion_force_xy_n"
MARK_RADIUS = 0.007
SIZE = 192
OPEN = 1.32
HOME = np.array([1.43, -0.55, 0.65, 1.1, 1.57, OPEN])
SOURCE, GOAL = (0.195, -0.068), (0.214, 0.071)
CAM_X, CAM_Z, FOV = 0.20, 0.58, 42


def scene(task_id, nominal=False):
    robot = mj.MjSpec.from_file(str(ROOT / ".menagerie" / "robotstudio_so101" / "so101.xml"))
    spec = mj.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 800, 600
    spec.visual.headlight.ambient = [0.45] * 3
    robot.option.timestep = spec.option.timestep
    robot.option.iterations = spec.option.iterations
    robot.option.ls_iterations = spec.option.ls_iterations
    spec.attach(robot, prefix="robot/", frame=spec.worldbody.add_frame())
    # A visual reference on the fixed finger frame for hand-to-camera alignment.
    tint = spec.add_material(name="tool_marker_color", rgba=[0.04, 0.85, 0.85, 1],
                             emission=0.6, specular=0.0)
    for suffix, position in (("a", [0.035, -0.050, -0.070]), ("b", [0, 0.070, 0])):
        spec.body("robot/gripper").add_geom(
            name=f"tool_marker_{suffix}", type=mj.mjtGeom.mjGEOM_SPHERE, pos=position,
            size=[MARK_RADIUS, 0, 0], material=tint.name, mass=0, contype=0, conaffinity=0,
        )
    spec.worldbody.add_geom(name="table", type=mj.mjtGeom.mjGEOM_BOX,
                            pos=[0.17, 0, -0.024], size=[0.37, 0.33, 0.024],
                            rgba=[0.22, 0.27, 0.30, 1])
    spec.worldbody.add_light(pos=[0.12, -0.20, 0.72], dir=[0.10, 0.25, -1])
    spec.worldbody.add_camera(name=FRAME, pos=[CAM_X, 0, CAM_Z], fovy=FOV)
    eye = np.array([0.69, -0.46, 0.43])
    back = eye - [0.155, 0, 0.075]
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(name="overview", pos=eye, fovy=39,
                              xyaxes=[*right, *np.cross(back, right)])
    if nominal:
        return spec
    if task_id == "cube_in_bowl":
        _box(spec, "target_cube", (*SOURCE, 0.015), [0.015] * 3, [0.95, 0.08, 0.08, 1], 0.045)
        _bowl(spec, (*GOAL, 0), 0.047)
    elif task_id == "peg_in_plate":
        _box(spec, "peg", (*SOURCE, 0.031), [0.012, 0.031, 0], [0.12, 0.85, 0.23, 1],
             0.055, geom_type=mj.mjtGeom.mjGEOM_CYLINDER)
        _socket(spec, (*GOAL, 0), 0.020)
        for geom in spec.body("plate_socket").geoms:
            geom.rgba = [0.61, 0.29, 0.91, 1]
    elif task_id == "two_block_stack":
        _box(spec, "block_top", (*SOURCE, 0.016), [0.016] * 3, [0.94, 0.36, 0.08, 1], 0.048)
        _box(spec, "block_base", (*GOAL, 0.018), [0.028, 0.028, 0.018],
             [0.10, 0.39, 0.94, 1], 0.13)
    elif task_id in extra_tasks.EXTRA_TASKS:
        goal = (0.214, 0.088) if task_id == "usb_in_port" else GOAL
        source = (SOURCE[0], -0.116) if task_id == "banana_in_bowl" else SOURCE
        extra_tasks.add_scene(spec, task_id, source, goal)
    else:
        raise ValueError(task_id)
    _box(spec, "bystander_cube", (0.316, -0.033, 0.016), [0.016] * 3,
         [0.92, 0.81, 0.14, 1], 0.062)
    return spec


@dataclass(frozen=True)
class Task:
    task_id: str
    video: Path | None = None
    observe_tick: Callable | None = None
    native_mujoco: bool = True
    control_dt_s: float = 0.02
    bystander_bodies: tuple = ("bystander_cube",)
    robot_pattern: str = r"^robot/"
    pad_pattern: str = r"^robot/(gripper|moving_jaw_so101_v1)$"

    @property
    def target_body(self):
        return TASKS[self.task_id][0]

    @property
    def support_body(self):
        return TASKS[self.task_id][1]

    @property
    def prompt(self):
        return self.task_id

    @property
    def episode_s(self):
        return 34.0 if self.task_id == "stack3" else 15.0

    @property
    def safety(self):
        setter = (extra_tasks.goal_setter(self.task_id)
                  if self.task_id in ("usb_in_port", "cap_fixed_marker", "stack3") else None)
        return SafetyLimits(max_grasp_force_n=150.0, goal_setter=setter)

    @property
    def additional_grasp_bodies(self):
        return ("stack_green",) if self.task_id == "stack3" else ()

    def success(self, poses, _):
        if self.task_id == "cube_in_bowl":
            return _inside_bowl(0.032)(poses) and poses[self.target_body][2] < 0.039
        if self.task_id == "peg_in_plate":
            return _in_socket(0.0085)(poses) and 0.034 < poses["peg"][2] < 0.046
        if self.task_id in extra_tasks.EXTRA_TASKS:
            return extra_tasks.success(self.task_id, poses)
        return _stacked(poses)

    def deployment_keys(self):
        sensors = (PAD_FORCE, ENV_FORCE, OBSTACLE_FORCE, INSERT_FORCE, INSERT_DIRECTION, COLLISION)
        return frozenset([*READ, FRAME, DEPTH, *sensors]), frozenset(WRITE)

    def build_device_layer(self, *, mode, extra_devices):
        if mode != "sim":
            raise ValueError("This DeviceLayer supports only local simulation.")
        layer = NativeLayer(self, extra_devices)
        return layer, layer.reset, layer.close


class NativeLayer:
    """Expose joint readings, RGB-D and contact sensors to the controller."""

    def __init__(self, task, devices):
        self.task, self.devices = task, devices
        self.mj_spec = scene(task.task_id)
        self.mj_model = self.model = self.mj_spec.compile()
        self.mj_data = mj.MjData(self.mj_model)
        self.sim = self
        self.qpos = [self.model.joint("robot/" + name).qposadr.item() for name in JOINTS]
        self.actuators = [self.model.actuator("robot/" + name).id for name in JOINTS]
        insertion = task.task_id in ("peg_in_plate", "cap_fixed_marker", "usb_in_port")
        self.feedback = ContactFeedback(
            self.model, task.target_body, prefix="robot/",
            support=task.support_body if insertion else None,
            extra_targets=task.additional_grasp_bodies,
            axis=1 if task.task_id == "usb_in_port" else 2,
        )
        self.monitor = devices[perturb.MONITOR]
        self.disturber = devices.get(perturb.DISTURBER)
        self.camera = mj.Renderer(self.model, width=SIZE, height=SIZE)
        self.video = None
        self.video_camera = None
        if task.video is not None:
            import imageio.v2 as imageio
            self.video = imageio.get_writer(task.video, fps=25, codec="libx264", macro_block_size=2)
            self.video_camera = mj.Renderer(self.model, width=720, height=480)
        for device in self.devices.values():
            device.initialize(self)

    def get_device(self, name):
        return self.devices[name]

    def set_const(self):
        q, v, ctrl = self.mj_data.qpos.copy(), self.mj_data.qvel.copy(), self.mj_data.ctrl.copy()
        mj.mj_setConst(self.model, self.mj_data)
        self.mj_data.qpos[:], self.mj_data.qvel[:], self.mj_data.ctrl[:] = q, v, ctrl

    def forward(self):
        mj.mj_forward(self.model, self.mj_data)

    def reset(self):
        mj.mj_resetData(self.model, self.mj_data)
        self.mj_data.qpos[self.qpos] = HOME
        self.mj_data.ctrl[self.actuators] = HOME
        self.forward()
        self.tick = 0
        self.feedback.stop = None

    def commands_spec(self):
        return {key: specs.BoundedArray((), np.float32, *self.model.actuator_ctrlrange[i])
                for key, i in zip(WRITE, self.actuators)}

    def measurements_spec(self):
        return {
            **{key: specs.Array((), np.float32) for key in READ},
            FRAME: specs.Array((SIZE, SIZE, 3), np.uint8),
            DEPTH: specs.Array((SIZE, SIZE), np.float32),
            PAD_FORCE: specs.Array((2,), np.float32),
            ENV_FORCE: specs.Array((), np.float32),
            OBSTACLE_FORCE: specs.Array((), np.float32),
            INSERT_FORCE: specs.Array((2,), np.float32),
            INSERT_DIRECTION: specs.Array((2,), np.float32),
            COLLISION: specs.Array((), np.float32),
            **self.monitor.substep_measurements_spec(),
        }

    def _frames(self):
        self.camera.update_scene(self.mj_data, camera=FRAME)
        rgb = self.camera.render().copy()
        self.camera.enable_depth_rendering()
        depth = self.camera.render().copy()
        self.camera.disable_depth_rendering()
        return rgb, depth

    def _measure(self):
        if self.tick == 0 or self.tick % 2 == 0:
            self.images = self._frames()
        if self.video is not None and self.tick % 2 == 0:
            self.video_camera.update_scene(self.mj_data, camera="overview")
            self.video.append_data(self.video_camera.render())
        readings = self.monitor.read_numpy(self)
        feedback = self.feedback
        pads = feedback.pad_n.copy()
        env = readings["gauntlet/forces"][1]
        if self.task.observe_tick is not None:
            self.task.observe_tick(self.model, self.mj_data, self.tick, pads, env, feedback.insertion_n)
        return {
            **dict(zip(READ, self.mj_data.qpos[self.qpos].astype(np.float32))),
            FRAME: self.images[0], DEPTH: self.images[1],
            PAD_FORCE: pads, ENV_FORCE: env, OBSTACLE_FORCE: np.float32(feedback.obstacle_n),
            INSERT_FORCE: feedback.insertion_n.copy(),
            INSERT_DIRECTION: feedback.insertion_xy.copy(),
            COLLISION: np.float32(feedback.stop["normal_force_n"] if feedback.stop else 0.0),
            **readings,
        }

    def begin_stepping(self):
        self.monitor.on_begin_stepping(self)
        self.feedback.read(self.mj_data)
        return self._measure()

    def end_stepping(self):
        pass

    def step(self, commands):
        if self.feedback.stop is not None:
            return self._measure()
        for k, a in zip(WRITE, self.actuators):
            if k in commands:
                self.mj_data.ctrl[a] = commands[k]
        for _ in range(round(self.task.control_dt_s / self.model.opt.timestep)):
            if self.disturber is not None:
                self.disturber.on_substep(self)
            mj.mj_step(self.model, self.mj_data)
            self.monitor.on_substep(self)
            self.feedback.read(self.mj_data)
            if self.feedback.stop is not None:
                break
        self.monitor.on_step(self)
        self.tick += 1
        return self._measure()

    def close(self):
        self.camera.close()
        if self.video is not None:
            self.video_camera.close()
            self.video.close()
