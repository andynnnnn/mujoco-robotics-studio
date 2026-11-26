"""Image- and joint-driven manipulation with camera and gripper monitors."""

import mujoco as mj
import numpy as np

from .native import (
    CAM_X, CAM_Z, COLLISION, DEPTH, ENV_FORCE, FOV, FRAME, HOME, INSERT_DIRECTION, INSERT_FORCE,
    JOINTS, MARK_RADIUS, OBSTACLE_FORCE, OPEN, PAD_FORCE, READ, SIZE, TASKS, WRITE, scene,
)
from .extra_tasks import USB_FACE
from playground_sim2real import solve_site_ik
from so101_grasp.feedback import PAD_TARGET_N, regulate
from detect_with_openai import pixel_box


class StopEpisode(Exception):
    """A confirmed device fault, or an observation that prevents an action."""


class VisionStop(Exception):
    """The current image has no valid model plan."""


class RetryGrasp(Exception):
    """A grasp attempt that can be retried after moving to a clear view."""


class PinSeated(Exception):
    """The socket is resisting axial insertion."""


class PinJammed(Exception):
    """The peg is loading the side of the socket."""


def image_points(measurement):
    image = measurement[FRAME].astype(float)
    depth = measurement[DEPTH]
    rows, cols = np.indices(depth.shape)
    f = (SIZE / 2) / np.tan(np.deg2rad(FOV / 2))
    x = CAM_X + (cols - (SIZE - 1) / 2) * depth / f
    y = - (rows - (SIZE - 1) / 2) * depth / f
    return image, x, y, CAM_Z - depth


def locate(measurement, color, *, bowl=False, detection=None):
    image, x, y, z = image_points(measurement)
    r, g, b = image.transpose(2, 0, 1)
    choices = {
        "red": (r > 1.7 * g) & (r > 1.7 * b),
        "orange": (r > 1.07 * g) & (g > 1.30 * b) & (b > 0.28 * g) & (r > 1.9 * b),
        "green": (g > 1.38 * r) & (g > 1.30 * b),
        "blue": (b > 1.52 * r) & (b > 1.10 * g),
        "purple": (b > 1.26 * g) & (r > 1.22 * g),
        "yellow": (r > 1.38 * b) & (g > 1.35 * b) & (r > 0.95 * g),
    }
    mask = (choices[color] & (image.max(axis=2) > 75) & (z > 0.002) & (z < 0.092)
            & (x > 0.11) & (x < 0.34) & (np.abs(y) < 0.16))
    region = np.ones(z.shape, dtype=bool)
    if detection is not None:
        y0, x0, y1, x1 = pixel_box(detection, image.shape, padding=12)
        region.fill(False)
        region[y0:y1, x0:x1] = True
    mask &= region
    if color in ("red", "orange", "yellow"):
        mask &= x < 0.29  # The yellow bystander sits beyond the pickup area.
    if bowl:
        # Fit the ring in the depth image, including blue pixels shifted by hue noise.
        mask = region & (b > 1.18*r) & (b > 0.90*g) & (z > 0.038) & (z < 0.050) & (y > 0)
        if mask.sum() >= 40:
            a, b = x[mask], y[mask]
            center = np.linalg.lstsq(np.column_stack([2*a, 2*b, np.ones_like(a)]),
                                     a*a + b*b, rcond=None)[0][:2]
            return *center, float(np.median(z[mask]))
    if mask.sum() < 16:
        raise StopEpisode(f"cannot_localize_{color}")
    top = np.quantile(z[mask], 0.85)
    surface = mask & (z > top - 0.006)
    return np.median(x[surface]), np.median(y[surface]), top


def locate_block(measurement, color, detection=None):
    x, y, top = locate(measurement, color, detection=detection)
    _, xx, yy, zz = image_points(measurement)
    surface = (np.hypot(xx - x, yy - y) < 0.032) & (abs(zz - top) < 0.008)
    points = np.column_stack([xx[surface], yy[surface]])
    if len(points) < 20:
        raise StopEpisode(f"cannot_fit_{color}")
    # Fit the top face's center and in-plane angle from the overhead depth pixels.
    def fit(angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        bounds = np.quantile(points @ rotation, [0.015, 0.985], axis=0)
        return np.prod(bounds[1] - bounds[0]), bounds.mean(axis=0) @ rotation.T
    angle = min(np.linspace(-np.pi/4, np.pi/4, 25), key=lambda a: fit(a)[0])
    _, center = fit(angle)
    return *center, float(np.median(zz[surface])), float(angle)


def locate_banana(measurement, detection):
    image, x, y, z = image_points(measurement)
    r, g, b = image.transpose(2, 0, 1)
    mask = (r > 1.38*b) & (g > 1.35*b) & (r > 0.95*g) & (r > 75)
    mask &= (z > 0.007) & (z < 0.033) & (x > 0.11) & (x < 0.29) & (y < 0)
    if detection is not None:
        y0, x0, y1, x1 = pixel_box(detection, image.shape, padding=6)
        region = np.zeros(z.shape, dtype=bool)
        region[y0:y1, x0:x1] = True
        mask &= region
    if mask.sum() < 30:
        raise StopEpisode("cannot_localize_banana")
    points = np.column_stack((x[mask], y[mask]))
    _, _, axes = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
    axis = axes[0] if axes[0, 1] >= 0 else -axes[0]
    distance = points @ axis
    middle = mask.copy()
    middle[mask] = abs(distance - np.median(distance)) < 0.009
    if middle.sum() < 12:
        raise StopEpisode("cannot_fit_banana_midpoint")
    return (float(np.median(x[middle])), float(np.median(y[middle])),
            float(np.quantile(z[middle], 0.85)), float(np.arctan2(-axis[0], axis[1])))


class Arm:
    def __init__(self, device, planner=None):
        self.device = device
        self.model = scene("cube_in_bowl", nominal=True).compile()
        joints = [self.model.joint("robot/" + j).id for j in JOINTS[:5]]
        self.qpos = self.model.jnt_qposadr[joints].tolist()
        self.dofs = self.model.jnt_dofadr[joints].tolist()
        self.lo, self.hi = self.model.jnt_range[joints].T
        self.site = self.model.site("robot/gripperframe").id
        self.meas = device.begin_stepping()
        self.q = np.array([self.meas[k] for k in READ[:5]])
        self.tick = 0
        self.last_frame = None
        self.offset = np.zeros(3)
        self.hold_jaw = None
        self.grip_target = PAD_TARGET_N
        self.guard = False
        self.missing_pads = 0
        self.events = []
        self.yaw = None
        self.max_delta = 0.026
        self.inserting = False
        self.acquiring_pin = False
        self.planner = planner
        self.decision = None

    def plan(self, stage, **kwargs):
        if self.planner is not None:
            try:
                self.decision = self.planner(self.meas, stage, **kwargs)
            except Exception as error:
                raise VisionStop(str(error)) from error
            self.events.append(f"GPT {self.decision['strategy']}")

    def detection(self, role):
        return self.decision["targets"][role] if self.decision is not None else None

    def chooses(self, strategy):
        return self.decision is not None and self.decision["strategy"] == strategy

    @property
    def pads(self):
        return self.meas[PAD_FORCE]

    def calibrate_tool(self):
        self.wait(OPEN, 7)
        image, x, y, z = image_points(self.meas)
        r, g, b = image.transpose(2, 0, 1)
        local = mj.MjData(self.model)
        local.qpos[self.qpos] = [self.meas[k] for k in READ[:5]]
        mj.mj_kinematics(self.model, local)
        cyan = (g > 1.6*r) & (b > 1.6*r) & (b > 0.76*g) & (g > 0.66*b)
        for name in ("tool_marker_a", "tool_marker_b"):
            expected = local.geom(name).xpos
            mask = (cyan & (np.hypot(x - expected[0], y - expected[1]) < 0.022)
                    & (abs(z - expected[2]) < 0.035))
            if mask.sum() >= 12:
                center = np.array([np.median(x[mask]), np.median(y[mask]),
                                   np.quantile(z[mask], 0.90) - MARK_RADIUS])
                self.offset = center - expected
                return
        raise RetryGrasp("tool marker hidden")

    def act(self, arm, jaw):
        if jaw is None:
            self.hold_jaw = regulate(self.hold_jaw, self.pads, target=self.grip_target)
            jaw = self.hold_jaw
        self.meas = self.device.step(dict(zip(WRITE, np.asarray([*arm, jaw], np.float32))))
        self.tick += 1
        if float(self.meas[COLLISION]) > 0:
            raise StopEpisode("collision_stop")
        if self.inserting:
            axial, lateral = self.meas[INSERT_FORCE]
            if lateral >= 3.0:
                raise PinJammed
            if axial >= 1.8:
                raise PinSeated
        if self.tick % 8 == 0:
            timestamp = float(self.meas[f"{FRAME}_timestamp"])
            image = self.meas[FRAME]
            measured = np.array([self.meas[k] for k in READ[:5]])
            if self.last_frame:
                ts, previous, previous_joints = self.last_frame
                moved = np.max(np.abs(measured - previous_joints)) > 0.045
                if timestamp <= ts or (moved and np.array_equal(image, previous)):
                    raise StopEpisode("camera_freeze")
            self.last_frame = timestamp, image.copy(), measured
        if self.guard and (float(self.meas[OBSTACLE_FORCE]) > 10.0 or float(self.meas[ENV_FORCE]) > 43.0):
            raise RetryGrasp("unexpected contact")
        if self.guard and self.hold_jaw is not None and not self.acquiring_pin:
            self.missing_pads = self.missing_pads + 1 if min(self.pads) < 3.0 else 0
            if self.missing_pads >= 8:
                raise RetryGrasp("object slipped")

    def wait(self, jaw, ticks):
        for _ in range(ticks):
            self.act(self.q, jaw)

    def move(self, goal, jaw, ticks):
        start = self.q.copy()
        count = max(ticks, int(np.ceil(np.max(np.abs(goal - start)) / self.max_delta)))
        for t in np.linspace(0, 1, count + 1)[1:]:
            self.q = start + t*t*(3 - 2*t) * (goal - start)
            self.act(self.q, jaw)
        self.wait(jaw, 5)

    def solve(self, point, start):
        seed = np.array([0.1, 0, 0.47, 1.1, 1.57])
        return solve_site_ik(self.model, self.site, self.qpos, self.dofs, self.lo, self.hi,
                             np.asarray(point) - self.offset, q_seed=start,
                             q_posture=seed, approach_axis_ee=(1, 0, 0), jaw_yaw_rad=self.yaw)

    def target(self, point, jaw, ticks):
        try:
            goal = self.solve(point, self.q)
        except RuntimeError as exc:
            raise StopEpisode(str(exc)) from exc
        self.move(goal, jaw, ticks)

    def approach(self, hover, pick, angle):
        model = self.model
        data = mj.MjData(model)
        robot = np.array([model.body(i).name.startswith("robot/") for i in range(model.nbody)])
        data.qpos[model.joint("robot/gripper").qposadr.item()] = 0.74
        fallback = None
        for self.yaw in (angle + np.pi/2, angle):
            for roll in (0.0, 2.55, -1.4):
                start = self.q.copy()
                start[-1] = roll
                try:
                    above = self.solve(hover, start)
                    below = self.solve(pick, above)
                except RuntimeError:
                    continue
                clear = True
                for blend in np.linspace(0, 1, 9):
                    data.qpos[self.qpos] = above + blend * (below - above)
                    mj.mj_fwdPosition(model, data)
                    if any(c.dist < -0.0005 and robot[model.geom_bodyid[c.geom1]]
                           and robot[model.geom_bodyid[c.geom2]] for c in data.contact):
                        clear = False
                        break
                if clear:
                    if np.max(np.abs(above - self.q)) < 2.7:
                        return above
                    if fallback is None:
                        fallback = self.yaw, above
        if fallback is not None:
            self.yaw, above = fallback
            return above
        self.yaw = None
        raise RetryGrasp("no clear approach")

    def check_jaw(self):
        start = float(self.meas[READ[-1]])
        self.wait(0.72, 18)
        if start - float(self.meas[READ[-1]]) < 0.18:
            raise StopEpisode("dead_gripper")
        self.wait(OPEN, 18)
        if OPEN - float(self.meas[READ[-1]]) > 0.18:
            raise StopEpisode("dead_gripper")

    def grasp(self, pin=False):
        if pin:
            self.wait(0.54, 11)
        ramp = np.linspace(0.52, -0.05, 72) if pin else np.linspace(0.73, -0.05, 92)
        self.acquiring_pin = pin
        try:
            for desired in ramp:
                self.act(self.q, desired)
                if min(self.pads) > 4.0:
                    self.hold_jaw = float(desired)
                    self.wait(None, 12)
                    if min(self.pads) >= 6.0:
                        return
                    if not pin:
                        break
                    self.hold_jaw = None
        finally:
            self.acquiring_pin = False
        raise RetryGrasp("no stable two-finger grip")

    def insert(self, xy, depth, hover):
        xy = np.asarray(xy, float)
        for attempt in range(3):
            self.inserting = True
            try:
                self.target([*xy, depth], None, 48 if self.chooses("slow_insert") else 40)
                return xy
            except PinSeated:
                self.q = np.array([self.meas[k] for k in READ[:5]])
                self.events.append("axial stop at socket")
                return xy
            except PinJammed:
                direction = self.meas[INSERT_DIRECTION]
                strength = np.linalg.norm(direction)
                if strength < 0.7 or attempt == 2:
                    raise StopEpisode("insertion_jam") from None
                self.q = np.array([self.meas[k] for k in READ[:5]])
                self.events.append("socket side load | retract and realign")
            finally:
                self.inserting = False
            self.target([*xy, hover], None, 24)
            xy = xy + 0.0035 * direction / strength
            self.target([*xy, hover], None, 16)
        raise StopEpisode("insertion_jam")


def pick_and_place_block(arm, task, *, layer=None):
    _, _, source_color, goal_color = TASKS[task.task_id]
    stack = task.task_id in ("two_block_stack", "stack3")
    banana = task.task_id == "banana_in_bowl"
    if layer == "middle":
        source_color, goal_color = "green", "red"
    elif layer == "top":
        source_color, goal_color = "blue", "green"
    retry_heights = [0.001, -0.003, 0.006]
    for attempt in range(1, 4):
        height_adjust = retry_heights.pop(0)
        if attempt > 1:
            arm.plan(f"Grasp attempt {attempt}, after {arm.events[-1]}. Re-localize before retrying",
                     **({"layer": layer} if layer else {}))
        arm.events.append(f"attempt {attempt} RGB-D localization")
        if banana:
            x, y, top, angle = locate_banana(arm.meas, arm.detection("source"))
        else:
            x, y, top, angle = locate_block(arm.meas, source_color, arm.detection("source"))
        dx, dy, dest_top = locate(arm.meas, goal_color, bowl=not stack,
                                  detection=arm.detection("destination"))
        grip_z = float(top - (0.011 if banana else 0.010) + height_adjust)
        if layer == "middle":
            angle += np.pi / 2  # Keep the open jaw away from the loose blue block.
        high = max(0.085, dest_top + 0.044) if layer else (0.105 if banana else 0.085)
        hover = [x, y, 0.067 if x < 0.16 else 0.078]
        above = None
        at_pickup = True
        try:
            pick = [x, y, grip_z]
            above = arm.approach(hover, pick, angle)
            arm.move(above, OPEN, 32)
            arm.calibrate_tool()
            arm.guard = True
            arm.target(pick, 0.74, 37)
            arm.grasp()
            arm.target([x, y, grip_z + 0.018], None, 14)
            arm.wait(None, 6)
            # Reuse the collision-checked joint configuration on the way back up.
            arm.move(above, None, 22)
            at_pickup = False
            if x < 0.16:
                arm.target([0.19, y, high], None, 36)
            slow = arm.chooses("slow_place")
            arm.target([dx, dy, high], None, 56 if slow else 46)
            release_height = (0.025 if layer == "top" else 0.028) if stack else (0.012 if banana else 0.015)
            arm.target([dx, dy, dest_top + release_height], None, 32 if slow else 25)
            arm.guard = False
            before_release = float(arm.meas[READ[-1]])
            for jaw in np.linspace(before_release, 0.83, 18):
                arm.act(arm.q, jaw)
            arm.wait(0.83, 8)
            if float(arm.meas[READ[-1]]) - before_release < 0.15:
                raise StopEpisode("dead_gripper")
            arm.target([dx, dy, high], 0.83, 22)
            arm.move(HOME[:5], OPEN, 42)
            arm.hold_jaw, arm.missing_pads = None, 0
            arm.offset[:] = 0
            arm.yaw = None
            arm.events.append("placed")
            return
        except RetryGrasp as error:
            arm.events.append(str(error))
            if str(error) == "unexpected contact":
                retry_heights.sort(reverse=True)
            arm.guard, arm.hold_jaw, arm.missing_pads = False, None, 0
            arm.max_delta = 0.034
            arm.q = np.array([arm.meas[k] for k in READ[:5]])
            if above is not None and at_pickup:
                arm.move(above, OPEN, 24)
            elif not at_pickup:
                data = mj.MjData(arm.model)
                data.qpos[arm.qpos] = arm.q
                mj.mj_kinematics(arm.model, data)
                safe = data.site_xpos[arm.site] + arm.offset
                safe[2] = max(safe[2], high)
                arm.target(safe, OPEN, 24)
            observation_pose = np.r_[HOME[:4], arm.q[-1]]
            arm.move(observation_pose, OPEN, 30)
            arm.yaw = None
    raise StopEpisode("grasp retries exhausted")


def fit_part(arm, task):
    _, _, source_color, goal_color = TASKS[task.task_id]
    x, y, top = locate(arm.meas, source_color, detection=arm.detection("source"))
    dx, dy, goal_top = locate(arm.meas, goal_color, detection=arm.detection("destination"))
    usb = task.task_id == "usb_in_port"
    grip_z = float(top - (0.004 if usb else 0.009))
    high = 0.091
    arm.yaw = np.pi/2 if usb else None
    arm.target([x, y, high], OPEN, 36)
    arm.calibrate_tool()
    arm.guard = True
    arm.target([x, y, grip_z], 0.74, 38)
    arm.grasp(pin=not usb)
    arm.target([x, y, high], None, 34)
    if usb:
        front = dy - USB_FACE
        insertion_z = goal_top - 0.007
        arm.target([dx, front - 0.060, high], None, 30)
        arm.target([dx, front - 0.060, insertion_z], None, 36)
        arm.inserting = True
        try:
            arm.target([dx, front - 0.020, insertion_z], None,
                       65 if arm.chooses("slow_insert") else 52)
        except PinJammed:
            raise StopEpisode("insertion_jam") from None
        except PinSeated:
            arm.events.append("horizontal axial stop at port")
        finally:
            arm.inserting = False
        leave = [dx, front - 0.020, high]
    else:
        arm.target([dx, dy, high], None, 42)
        dx, dy = arm.insert([dx, dy], goal_top - 0.010, high)
        arm.target([dx, dy, goal_top + 0.003], None, 20)
        leave = [dx, dy, high]
    arm.guard = False
    before = float(arm.meas[READ[-1]])
    for jaw in np.linspace(before, 0.69, 20):
        arm.act(arm.q, jaw)
    arm.wait(0.69, 8)
    if float(arm.meas[READ[-1]]) - before < 0.12:
        raise StopEpisode("dead_gripper")
    arm.target(leave, 0.69, 28)
    arm.move(HOME[:5], OPEN, 42)
    arm.events.append("released after insertion")


def controller(device, task, planner=None):
    arm = Arm(device, planner)
    try:
        arm.plan("Initial scene before motion",
                 **({"layer": "middle"} if task.task_id == "stack3" else {}))
        arm.check_jaw()
        arm.wait(OPEN, 9)
        if task.task_id == "stack3":
            pick_and_place_block(arm, task, layer="middle")
            arm.plan("Second stage after placing the green middle layer", layer="top")
            pick_and_place_block(arm, task, layer="top")
            return {"status": "done", "fault": "", "detail": " | ".join(arm.events)}
        if task.task_id in ("usb_in_port", "cap_fixed_marker"):
            fit_part(arm, task)
            return {"status": "done", "fault": "", "detail": " | ".join(arm.events)}
        if task.task_id != "peg_in_plate":
            pick_and_place_block(arm, task)
            return {"status": "done", "fault": "", "detail": " | ".join(arm.events)}
        _, _, source_color, goal_color = TASKS[task.task_id]
        x, y, top = locate(arm.meas, source_color, detection=arm.detection("source"))
        dx, dy, _ = locate(arm.meas, goal_color, detection=arm.detection("destination"))
        grip_z = float(top * 0.52)
        high = 0.085
        arm.target([x, y, high], OPEN, 34)
        arm.target([x, y, grip_z], OPEN, 28)
        arm.guard = True
        arm.grasp(pin=True)
        arm.target([x, y, high], None, 29)
        arm.target([dx, dy, high], None, 40)
        dx, dy = arm.insert([dx, dy], grip_z + 0.007, high)
        # Back off the seating load while the pin is still guided by the socket.
        arm.target([dx, dy, grip_z + 0.014], None, 20)
        arm.guard = False
        before_release = float(arm.meas[READ[-1]])
        # Unload the pin inside the fixture before opening the jaw fully in free space.
        for jaw in np.linspace(before_release, 0.45, 18):
            arm.act(arm.q, jaw)
        arm.wait(0.45, 8)
        if float(arm.meas[READ[-1]]) - before_release < 0.12:
            raise StopEpisode("dead_gripper")
        arm.target([dx, dy, high], 0.45, 22)
        arm.move(HOME[:5], OPEN, 42)
        return {"status": "done", "fault": "", "detail": " | ".join(arm.events)}
    except VisionStop as error:
        return {"status": "abort", "fault": "vision_stop", "detail": str(error)}
    except RetryGrasp as error:
        return {"status": "abort", "fault": "grasp_stop", "detail": str(error)}
    except StopEpisode as error:
        fault = str(error)
        if fault in ("camera_freeze", "dead_gripper"):
            return {"status": "abort", "fault": fault}
        if fault in ("collision_stop", "insertion_jam"):
            return {"status": "abort", "fault": fault, "detail": " | ".join([*arm.events, fault])}
        return {"status": "done", "fault": "", "detail": " | ".join([*arm.events, fault])}
