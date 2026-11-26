"""Contact-driven props and goal predicates for four additional SO-101 scenes."""

import math

import mujoco as mj
import numpy as np

from task_props import _bowl, _box, _tilt_deg


EXTRA_TASKS = {
    "usb_in_port": ("usb_drive", "usb_port", "orange", "purple"),
    "cap_fixed_marker": ("marker_cap", "fixed_marker", "orange", "purple"),
    "stack3": ("stack_blue", "stack_red", "blue", "red"),
    "banana_in_bowl": ("banana", "bowl", "yellow", "blue"),
}
ORANGE = [0.94, 0.34, 0.08, 1]
PURPLE = [0.62, 0.27, 0.88, 1]
USB_Z = 0.049
USB_FACE = 0.020
USB_BODY_REACH = 0.018
CAP_HALF_Z = 0.018
MARKER_TOP = 0.036
BANANA_RADIUS = 0.065
BANANA_POINTS = np.array([
    [0.017, -0.045, 0.003], [0.004, -0.033, 0], [-0.004, -0.018, 0],
    [-0.008, 0, 0], [-0.004, 0.018, 0], [0.004, 0.033, 0], [0.017, 0.045, 0.003],
])


def _geom(body, name, kind, pos, size, color, **kwargs):
    return body.add_geom(name=name, type=getattr(mj.mjtGeom, f"mjGEOM_{kind}"),
                         pos=pos, size=size, rgba=color, **kwargs)


def add_scene(spec, task, source, goal):
    if task == "usb_in_port":
        drive = spec.worldbody.add_body(name="usb_drive", pos=[*source, 0.010])
        drive.add_freejoint(name="usb_drive_free")
        _geom(drive, "usb_shell", "BOX", [0, 0, 0], [0.013, USB_BODY_REACH, 0.010], ORANGE,
              mass=0.039, friction=[1.4, 0.03, 0.005])
        _geom(drive, "usb_plug", "BOX", [0, 0.026, 0], [0.007, 0.012, 0.0037],
              [0.68, 0.73, 0.78, 1], mass=0.008)
        port = spec.worldbody.add_body(name="usb_port", pos=[*goal, USB_Z])
        for x in (-0.021, 0.021):
            _geom(port, f"port_side_{x}", "BOX", [x, 0, 0], [0.006, USB_FACE, 0.014], PURPLE)
        _geom(port, "port_roof", "BOX", [0, 0, 0.014], [0.027, USB_FACE, 0.004], PURPLE)
        _geom(port, "port_bottom", "BOX", [0, 0, -0.011], [0.027, USB_FACE, 0.004], PURPLE)
        _geom(port, "port_back", "BOX", [0, 0.018, 0], [0.027, 0.002, 0.014], PURPLE)
        _geom(port, "port_support", "BOX", [0, 0.006, -0.033], [0.023, 0.014, 0.016],
              [0.32, 0.28, 0.39, 1])
        _geom(port, "usb_rest", "BOX", [0, -0.039, -0.015], [0.010, 0.020, 0.005],
              [0.35, 0.32, 0.40, 1])
    elif task == "cap_fixed_marker":
        marker = spec.worldbody.add_body(name="fixed_marker", pos=[*goal, 0])
        _geom(marker, "marker_body", "CYLINDER", [0, 0, MARKER_TOP/2],
              [0.008, MARKER_TOP/2, 0], PURPLE)
        _geom(marker, "marker_foot", "CYLINDER", [0, 0, 0.002], [0.019, 0.002, 0],
              [0.32, 0.31, 0.37, 1])
        cap = spec.worldbody.add_body(name="marker_cap", pos=[*source, CAP_HALF_Z])
        cap.add_freejoint(name="marker_cap_free")
        _geom(cap, "cap_roof", "CYLINDER", [0, 0, 0.015], [0.019, 0.003, 0], ORANGE, mass=0.015)
        for i in range(12):
            angle = 2*math.pi*i/12
            _geom(cap, f"cap_wall_{i}", "BOX", [0.0175*math.cos(angle), 0.0175*math.sin(angle), -0.003],
                  [0.0025, 0.005, 0.015], ORANGE, mass=0.002,
                  quat=[math.cos(angle/2), 0, 0, math.sin(angle/2)], friction=[1.2, 0.02, 0.004])
    elif task == "stack3":
        _box(spec, "stack_red", [*goal, 0.014], [0.027, 0.027, 0.014], [.95,.08,.08,1], 0.115)
        _box(spec, "stack_green", [*source, 0.015], [0.018, 0.018, 0.015], [.10,.80,.23,1], 0.053)
        _box(spec, "stack_blue", [0.258, -0.061, 0.014], [0.014]*3, [.11,.38,.94,1], 0.039)
    elif task == "banana_in_bowl":
        banana = spec.worldbody.add_body(name="banana", pos=[*source, 0.011])
        banana.add_freejoint(name="banana_free")
        for i, (a,b) in enumerate(zip(BANANA_POINTS, BANANA_POINTS[1:])):
            collision = 0 if i in (2, 3) else 1
            banana.add_geom(name=f"banana_curve_{i}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                            fromto=[*a,*b], size=[0.0095 if i in (0,5) else 0.011,0,0],
                            rgba=[1, .76, .04, 1], mass=0.011, friction=[1.4,.02,.004],
                            solref=[0.008, 1], contype=collision, conaffinity=collision)
        # Approximate the two flatter faces of the banana's center in contact.
        _geom(banana, "banana_midsection", "BOX", [-0.007, 0, 0], [0.0105, 0.019, 0.0105],
              [1, .76, .04, 0], mass=0, friction=[1.8, .025, .004], solref=[0.008, 1])
        _bowl(spec, [*goal, 0], BANANA_RADIUS)
        spec.geom("bowl_floor").solref = [0.008, 1]
    else:
        raise ValueError(task)


def success(task, poses):
    target, support, *_ = EXTRA_TASKS[task]
    p, goal = poses[target], poses[support]
    d = p[:3] - goal[:3]
    if task == "usb_in_port":
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, np.asarray(p[3:], dtype=float))
        axis = mat.reshape(3, 3)[:, 1]
        tip = d + 0.038 * axis
        return bool(axis[1] > 0.92 and abs(d[0]) < 0.013 and abs(d[2]) < 0.012
                    and 0.009 < tip[1] + USB_FACE < 0.035)
    if task == "cap_fixed_marker":
        return bool(np.linalg.norm(d[:2]) < 0.0085 and 0.022 < d[2] < 0.030
                    and _tilt_deg(p) < 11)
    if task == "stack3":
        middle = poses["stack_green"]
        return bool(np.linalg.norm(middle[:2] - goal[:2]) < 0.014
                    and np.linalg.norm(p[:2] - middle[:2]) < 0.014
                    and 0.025 < middle[2] - goal[2] < 0.033 and 0.025 < p[2] - middle[2] < 0.033
                    and max(_tilt_deg(each) for each in (p,middle,goal)) < 10)
    mat = np.zeros(9)
    mj.mju_quat2Mat(mat, np.asarray(p[3:], dtype=float))
    contour = BANANA_POINTS @ mat.reshape(3,3).T + d
    return bool(np.max(np.linalg.norm(contour[:,:2], axis=1)) + 0.010 < BANANA_RADIUS
                and np.max(contour[:,2]) < 0.036 and d[2] > 0)


def goal_setter(task):
    def place(sim, poses):
        target, support, *_ = EXTRA_TASKS[task]
        goal = poses[support][:3]
        placements = {
            "usb_in_port": [(target, goal + [0, -0.040, 0])],
            "cap_fixed_marker": [(target, goal + [0, 0, 0.044])],
            "stack3": [("stack_green", goal + [0, 0, 0.029]), (target, goal + [0, 0, 0.058])],
        }
        for body, position in placements[task]:
            joint = sim.mj_model.joint(f"{body}_free")
            start = joint.qposadr.item()
            sim.mj_data.qpos[start:start+7] = [*position, 1,0,0,0]
    return place
