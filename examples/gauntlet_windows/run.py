"""Run the local SO-101 scenes through the repository's Gauntlet."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauntlet_windows.controller import controller
from gauntlet_windows.extra_tasks import EXTRA_TASKS
from gauntlet_windows.native import COLLISION, INSERT_FORCE, PAD_FORCE, READ, ROOT, TASKS, WRITE, Task
from gauntlet_windows.vision import Planner
from robosim_gauntlet import episode, grade, perturb, preflight, scorecard


def cases(seed, task_id):
    if task_id in EXTRA_TASKS:
        yield "T0", 0
        return
    yield from (("T0", 0), ("T1", 0), ("T1", 1), ("T2", 0), ("T2", 1))
    # Select coverage by injected fault and trigger before running the controller.
    found = set()
    for idx in range(5000):
        rng = perturb.streams(seed, task_id, "T4", idx).spec
        p = perturb.sample("T4", idx, rng)
        variant = (p.fault, p.fault_restamp)
        camera = p.fault == "camera_freeze" and p.fault_trigger == "arm_motion" and p.fault_delay <= 20
        jaw = p.fault == "dead_gripper" and p.fault_trigger == "gripper_close" and p.fault_delay <= 8
        if (camera or jaw) and variant not in found:
            yield "T4", idx
            found.add(variant)
            if len(found) == 3:
                return
    raise RuntimeError(f"Could not sample all fault variants for {task_id}")


def save_trace(path, task, result):
    fields = ("grasp_force_n", "contact_force_n", "impact_force_n", "self_force_n",
              "target_robot_contacts", "max_penetration_m")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tick", "sim_time_s", "settling", "target_x_m", "target_y_m",
                         "target_z_m", "bystander_displacement_m", *fields,
                         "fixed_finger_normal_n", "moving_finger_normal_n",
                         "insertion_axial_n", "insertion_lateral_n", "collision_latch_n", *READ, *WRITE])
        initial = grade.poses_at(result.log[0], result.body_names)["bystander_cube"][:3]
        for idx, tick in enumerate(result.log):
            poses = grade.poses_at(tick, result.body_names)
            drift = np.linalg.norm(poses["bystander_cube"][:3] - initial)
            writer.writerow([idx, round(idx * task.control_dt_s, 6), idx >= result.settle_start,
                             *poses[task.target_body][:3], drift,
                             *tick.meas[grade.FORCES], *tick.meas[PAD_FORCE],
                             *tick.meas[INSERT_FORCE], float(tick.meas[COLLISION]),
                             *(float(tick.meas[k]) for k in READ),
                             *(float(tick.cmd[k]) if k in tick.cmd else "" for k in WRITE)])
    return initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "gpt-plans" / "gauntlet")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--tier", choices=("T0", "T1", "T2", "T4"))
    parser.add_argument("--idx", type=int)
    parser.add_argument("--local-baseline", action="store_true",
                        help="Use the previous RGB-D controller for offline comparison.")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [args.task] if args.task else list(TASKS)
    audits = {name: preflight.run(Task(name)) for name in names}
    (args.output / "preflight.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    if any(audits.values()):
        raise RuntimeError(f"Gauntlet preflight failed: {audits}")

    results, details = [], []
    for name in names:
        for tier, idx in cases(args.seed, name):
            if args.tier and tier != args.tier:
                continue
            if args.idx is not None and idx != args.idx:
                continue
            stem = f"{name}-{tier}-{idx}"
            video = args.output / f"{stem}.mp4" if tier in ("T0", "T2") and not args.no_video else None
            task = Task(name, video=video)
            vision = args.output / "vision" / stem
            planner = None if args.local_baseline else Planner(name, vision, lambda msg: print(msg, flush=True))
            result = episode.run_episode(task, lambda device: controller(device, task, planner),
                                         tier=tier, idx=idx, base_seed=args.seed, mode="sim")
            trace = args.output / f"{stem}.csv"
            initial = save_trace(trace, task, result)
            final = grade.poses_at(result.log[-1], result.body_names)
            drift = float(np.linalg.norm(final["bystander_cube"][:3] - initial))
            stopped = result.outcome["status"] == "abort"
            onset = result.onset_tick
            delay = (result.outcome["ticks"] - onset) * task.control_dt_s if stopped and onset is not None else None
            results.append(result)
            details.append({
                "task_id": name, "tier": tier, "idx": idx, "seed": args.seed,
                "perturbation": asdict(result.pspec), "outcome": dict(result.outcome),
                "success": result.grade.success, "safe": result.grade.safe,
                "safe_pass": result.grade.safe_pass, "aborted_ok": result.grade.aborted_ok,
                "violations": list(result.grade.violations), "peak": dict(result.grade.metrics),
                "insertion_peak_axial_lateral_n": np.max([t.meas[INSERT_FORCE] for t in result.log], axis=0).tolist(),
                "collision_latch_n": max(float(t.meas[COLLISION]) for t in result.log),
                "fault_onset_tick": onset, "controller_stop_delay_s": delay,
                "settle_start_tick": result.settle_start, "control_dt_s": task.control_dt_s,
                "bystander_displacement_m": drift,
                "final_target_pose": final[task.target_body].tolist(),
                "final_support_pose": final[task.support_body].tolist(),
                "trace": trace.name, "video": video.name if video else None,
                "model_requests": planner.requests if planner else 0,
                "vision_artifacts": vision.relative_to(args.output).as_posix() if planner else None,
            })
            print(stem, "success=", result.grade.success, "safe=", result.grade.safe,
                  "abort=", result.grade.aborted_ok, "delay=", delay,
                  "grasp/contact=", *(round(result.grade.metrics[f], 2)
                                      for f in ("grasp_force_n", "contact_force_n")),
                  result.grade.violations, result.outcome.get("detail", ""), flush=True)
            scorecard.save_json(args.output / "scorecard.json", results)
            (args.output / "episodes.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
