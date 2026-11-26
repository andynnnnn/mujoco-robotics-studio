"""Exercise local collision stops with real MuJoCo contacts and save the evidence."""

import json

import mujoco as mj
import numpy as np

from fr3_coffee import coffee
from gauntlet_windows.controller import Arm, StopEpisode
from gauntlet_windows.native import OPEN, ROOT, TASKS, Task
from robosim_gauntlet import episode
from so101_grasp import run as grasp


def verify_latch(stop, data, retry):
    assert stop is not None, "Expected a physical collision"
    qpos, time = data.qpos.copy(), data.time
    try:
        retry()
    except RuntimeError:
        pass
    assert data.time == time and np.array_equal(data.qpos, qpos)
    assert np.isclose(time, stop["time_s"], rtol=0, atol=1e-7)
    return {**stop, "hold_verified": True}


def grasp_table():
    model = grasp.make_model()
    data = mj.MjData(model)
    grasp.reset_scene(model, data)
    feedback = grasp.GraspFeedback(model, data)
    try:
        for name, commands in grasp.grasp_phases(model, (0.213, -0.020), grip_height=-0.055):
            for command in commands:
                feedback.step(name, command)
    except RuntimeError:
        stop = feedback.sensor.stop
        assert stop and stop["kind"] == "arm_fixture" and "table" in stop["geoms"]
        return verify_latch(stop, data, lambda: feedback.step(name, command))
    raise AssertionError("SO-101 did not stop")


def gauntlet_collision(task_id, *, self_contact=False):
    def probe(device):
        arm = Arm(device)
        try:
            if self_contact:
                arm.move(np.array([0.326, -0.867, 0.907, 1.532, -0.743]), 0.74, 70)
                arm.move(np.array([0.472, -0.671, 0.583, 1.658, 2.529]), 0.74, 94)
            else:
                arm.target([0.213, -0.020, 0.060], OPEN, 42)
                arm.target([0.213, -0.020, -0.047], 0.74, 155)
        except StopEpisode as error:
            return {"status": "abort", "fault": str(error)}
        return {"status": "done"}

    layers = []

    class ObservedTask(Task):
        def build_device_layer(self, **kwargs):
            built = super().build_device_layer(**kwargs)
            layers.append(built[0])
            return built

    result = episode.run_episode(ObservedTask(task_id), probe, base_seed=42)
    layer = layers[0]
    stop = layer.feedback.stop
    assert result.outcome["fault"] == "collision_stop" and not result.grade.safe_pass
    assert stop and stop["kind"] == ("self_contact" if self_contact else "arm_fixture")
    if not self_contact:
        assert "table" in stop["geoms"]
    # Gauntlet has already attempted to settle the stopped scene.
    assert layer.mj_data.time == stop["time_s"]
    return {**stop, "hold_verified": True, "controller_outcome": result.outcome["fault"]}


def coffee_targets(model, data, goals):
    command = data.ctrl.copy()
    lo, hi = coffee.FR3_JOINT_LIMITS_RAD.T
    for side, point in goals.items():
        qpos, dofs = coffee.arm_joint_qpos_and_dof_addrs(model, side)
        site = model.site(coffee.gripper_pinch_site_name(side)).id
        q = coffee.solve_site_ik(
            model, site, qpos, dofs, lo + coffee.FR3_IK_MARGINS_RAD,
            hi - coffee.FR3_IK_MARGINS_RAD, point, q_seed=data.qpos[qpos],
            q_posture=coffee.DEFAULT_REST_POSE_RAD[side], context_qpos=data.qpos,
            target_rot_world=coffee.DOWN, max_iters=150,
        )
        command[[model.actuator(f"{side}/fr3_joint{i}").id for i in range(1, 8)]] = q
    return command


def coffee_collision(output, *, both_arms):
    model = coffee.make_model(table_offset_m=0 if both_arms else 0.095)
    data = mj.MjData(model)
    coffee.reset(model, data)
    cycle = coffee.CoffeeCycle(model, data, output)
    if both_arms:
        waypoints = (
            {"left": [0.52, 0.16, 0.24], "right": [0.52, -0.16, 0.24]},
            {"left": [0.52, 0.0, 0.24], "right": [0.52, 0.0, 0.24]},
        )
    else:
        waypoints = ({"left": [0.535, 0.145, 0.18]}, {"left": [0.535, 0.145, 0.043]})
    try:
        for goals in waypoints:
            start = data.ctrl.copy()
            target = coffee_targets(model, data, goals)
            for t in np.linspace(0, 1, round(2.5 / model.opt.timestep)):
                command = start + t*t*(3 - 2*t) * (target - start)
                cycle.step("Protection validation", command)
    except coffee.SafetyStop as error:
        stop = cycle.feedback.stop
        assert stop and stop["kind"] == ("arm-arm" if both_arms else "arm-fixture")
        if not both_arms:
            assert "table" in stop["geoms"]
        result = verify_latch(stop, data, lambda: cycle.step("Protection validation", command))
        cycle.save(error)
        return result
    raise AssertionError("FR3 did not stop")


def main():
    out = ROOT / "runs" / "collision-stop-checks"
    out.mkdir(parents=True, exist_ok=True)
    results = {"so101_grasp_table": grasp_table()}
    for task in TASKS:
        results[f"{task}_table"] = gauntlet_collision(task)
    results["so101_self_contact"] = gauntlet_collision("cube_in_bowl", self_contact=True)
    results["fr3_table"] = coffee_collision(out / "fr3-table", both_arms=False)
    results["fr3_inter_arm"] = coffee_collision(out / "fr3-arms", both_arms=True)
    summary = json.dumps(results, indent=2)
    (out / "collision-stops.json").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
