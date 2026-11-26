"""Run the original strong-perturbation cases and show recovery in MuJoCo Studio."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
import sys
from tempfile import mkdtemp
from threading import Event
from time import monotonic, sleep

import mujoco
import numpy as np
from mujoco.studio import app, launch_native, messages, plugin, sim, step_control, ux
from mujoco.studio.app.plugins import imgui
from mujoco.studio.viewer_protocol import ViewerConfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauntlet_windows.controller import controller
from gauntlet_windows.native import FRAME, HOME, JOINTS, ROOT, Task, scene
from gauntlet_windows.vision import Planner
from robosim_gauntlet import episode, scorecard


def run_local_experiment(task_id, output, capture, status):
    if task_id == "yam_cookie":
        from yam_cookie import sorter
        return sorter.run(output, observe=capture, status=status)
    if task_id == "grocery_sort":
        from grocery_sort import run
    else:
        from pelican_ride import run
    return run(output, observe=capture, status=status)


LOCAL_EXPERIMENTS = ("yam_cookie", "grocery_sort", "pelican_ride")
SEED = 42
CASES = (
    ("Cube in bowl / T2-1", "cube_in_bowl", "T2", 1),
    ("Block stack / T2-0", "two_block_stack", "T2", 0),
    ("Block stack / T2-1", "two_block_stack", "T2", 1),
    ("Block stack / T1-1 (retry)", "two_block_stack", "T1", 1),
)
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class RunCase(messages.Event):
    index: int


@dataclass(frozen=True)
class Progress(messages.Snapshot):
    busy: bool = False
    pads: tuple = (0.0, 0.0)
    contact: float = 0.0
    text: str = "Choose a case and click Run / rerun."
    insertion: tuple = (0.0, 0.0)


class GauntletSimulation(plugin.SimPlugin):
    def __init__(self, worker, closed, stepper, output):
        self.worker = worker
        self.closed = closed
        self.stepper = stepper
        self.frames = Queue(maxsize=2)
        self.notices = SimpleQueue()
        self.output = output
        self.round = 0
        self.pending = None
        self.next_frame = 0.0

    def pause(self, value):
        state = sim.PauseState.NORMAL_PAUSED if value else sim.PauseState.UNPAUSED
        self.handle.dispatch(messages.RequestPauseEvent(pause_state=state))

    def capture(self, model, data, tick, pads, contact, insertion):
        state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, state, STATE)
        frame = (copy(model) if tick == 0 else None, state, tuple(map(float, pads)),
                 float(contact), tuple(map(float, insertion)))
        while not self.closed.is_set():
            try:
                self.frames.put(frame, timeout=0.05)
                return
            except Full:
                continue
        raise InterruptedError("Studio closed")

    def run(self, task_id, tier, idx, output):
        if task_id in LOCAL_EXPERIMENTS:
            return run_local_experiment(task_id, output, self.capture, self.notices.put)
        task = Task(task_id, observe_tick=self.capture)
        planner = Planner(task_id, output, self.notices.put)
        result = episode.run_episode(task, lambda device: controller(device, task, planner),
                                     tier=tier, idx=idx, base_seed=SEED, mode="sim")
        summary = scorecard.to_json_list([result])[0]
        summary.update(model_requests=planner.requests, outcome=dict(result.outcome))
        (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return result

    @messages.handler
    def start(self, event: RunCase):
        self.start_case(*CASES[event.index])

    def start_case(self, label, task_id, tier, idx):
        if self.pending is not None:
            return
        self.round += 1
        output = self.output / f"{self.round:03d}-{task_id}-{tier}-{idx}"
        self.phase = f"Starting {label}"
        self.handle.send_to_viewer(Progress(busy=True, text=self.phase))
        self.pending = self.worker.submit(self.run, task_id, tier, idx, output)
        self.next_frame = 0.0
        self.pause(False)
        print(f"Started {label}" if task_id in LOCAL_EXPERIMENTS else f"Started {label}, seed {SEED}", flush=True)

    @messages.handler
    def step(self, _: messages.StepEvent) -> bool:
        if self.pending is None:
            sleep(0.004)
            return True
        while not self.notices.empty():
            self.phase = self.notices.get_nowait()
            self.handle.send_to_viewer(Progress(busy=True, text=self.phase))
            print(self.phase, flush=True)
        if self.frames.empty() and self.pending.done():
            try:
                result = self.pending.result()
                if isinstance(result, dict) and "distance_m" in result:
                    verdict = "PASS" if result["success"] else "FAIL"
                    text = (f"{verdict} | {result['distance_m']:.2f} m traveled | "
                            f"{result['crank_turns']:.2f} crank revolutions\n"
                            f"Wheel contact {result['wheel_contact_fraction']:.0%} | "
                            f"Rear slip {result['mean_rear_slip_m_s']:.3f} m/s")
                elif isinstance(result, dict):
                    left, right = result["pad_force_peak_n"]
                    verdict = "PASS" if result["success"] else "Stopped" if result["stop"] else "FAIL"
                    text = (f"{verdict} | {result['correct']}/{result['total']} sorted | "
                            f"{result['model_plans']} GPT plans\n"
                            f"Peak finger {left:.1f} / {right:.1f} N | Collision {result['collision_peak_n']:.1f} N")
                    if result["stop"]:
                        text += f"\n{result['stop']}"
                    elif "egg_contact_peak_n" in result:
                        egg = max(result["egg_pad_force_peak_n"])
                        text += f"\nEgg peak finger {egg:.1f} N | Impact {result['egg_contact_peak_n']:.1f} N"
                else:
                    fault = result.outcome.get("fault")
                    verdict = ("PASS" if result.grade.safe_pass else
                               "Stopped" if fault in ("collision_stop", "insertion_jam", "grasp_stop", "vision_stop") else "FAIL")
                    detail = result.outcome.get("detail") or result.outcome.get("fault") or ""
                    text = f"{verdict} | {detail}" if detail else verdict
            except Exception as error:
                text = f"Stopped | {error}"
            self.pending = None
            self.pause(True)
            self.handle.send_to_viewer(Progress(text=text))
            print(text, flush=True)
            return True
        pacing = self.stepper.step_control
        single = pacing.consume_single_step_request()
        running = pacing.get_pause_state() == sim.PauseState.UNPAUSED
        now = monotonic()
        if not single and (not running or now < self.next_frame):
            sleep(0.002)
            return True
        try:
            model, state, pads, contact, insertion = self.frames.get_nowait()
        except Empty:
            sleep(0.002)
            return True
        if model is not None:
            model.vis.global_.cameraid = model.camera("overview").id
            self.handle.set_model(model)
        model, data = self.handle.model, self.handle.data
        mujoco.mj_setState(model, data, state, STATE)
        mujoco.mj_forward(model, data)
        speed = max(pacing.get_speed(), 1)
        self.next_frame = now + 0.02 * 100 / speed
        self.handle.send_to_viewer(Progress(True, pads, contact, self.phase, insertion))
        return True


class GauntletPanel(plugin.ViewerPlugin):
    def __init__(self, studio_app):
        self.app = studio_app
        self.index = 0
        self.progress = Progress()

    @messages.handler
    def status(self, event: Progress):
        self.progress = event

    @messages.handler
    def draw(self, _: messages.BuildGuiEvent):
        imgui.SetNextWindowPos(imgui.Vec2(194, 46), imgui.Cond.Once)
        imgui.SetNextWindowSize(imgui.Vec2(430, 250), imgui.Cond.Once)
        visible = imgui.Begin("SO-101 Force Feedback")
        try:
            if not visible:
                return
            imgui.BeginDisabled(self.progress.busy)
            try:
                for idx, (label, *_case) in enumerate(CASES):
                    if imgui.RadioButton(label, self.index == idx):
                        self.index = idx
                if imgui.Button("Run / rerun"):
                    self.viewer.send_to_sim(RunCase(self.index))
            finally:
                imgui.EndDisabled()
            imgui.SameLine()
            imgui.TextUnformatted(f"Seed {SEED} | {self.viewer.data.time:.2f} s")
            imgui.TextWrapped(self.progress.text)
            left, right = self.progress.pads
            imgui.TextUnformatted(f"Finger N {left:.1f} / {right:.1f} | Contact {self.progress.contact:.1f} N")
            model = self.viewer.model
            for i, (label, name) in enumerate((("Overview", "overview"), ("Top", FRAME), ("Free", None))):
                if i:
                    imgui.SameLine()
                if imgui.Button(label):
                    camera = model.camera(name).id if name else ux.TUMBLE_CAMERA_IDX
                    self.app.ux_state.camera_index = ux.set_camera(model, self.viewer.camera, camera)
                    print(f"Camera: {label}", flush=True)
        finally:
            imgui.End()


def main():
    model = scene(CASES[0][1]).compile()
    model.vis.global_.cameraid = model.camera("overview").id
    data = mujoco.MjData(model)
    addresses = [model.joint("robot/" + name).qposadr.item() for name in JOINTS]
    data.qpos[addresses] = HOME
    studio_app = app.App()
    stepper = step_control.StepControl()
    closed = Event()
    parent = ROOT / "runs" / "gpt-plans"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="gauntlet-studio-", dir=parent))
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            launch_native.run(
                ViewerConfig(title="MuJoCo Studio - SO-101 Gauntlet", width=1240, height=790,
                             gfx="opengl_software"),
                model=model, data=data,
                plugins=[studio_app, stepper, GauntletSimulation(worker, closed, stepper, output),
                         GauntletPanel(studio_app)],
            )
        finally:
            closed.set()


if __name__ == "__main__":
    main()
