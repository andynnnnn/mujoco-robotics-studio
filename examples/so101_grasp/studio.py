"""Run randomized RGB-guided SO-101 grasps in native MuJoCo Studio."""

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from time import monotonic, sleep

import mujoco
import numpy as np
from mujoco.studio import app, launch_native, messages, plugin, sim, sim_history, step_control, ux
from mujoco.studio.app.plugins import imgui
from mujoco.studio.viewer_protocol import ViewerConfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_grasp.run import (
    EFFORT, GraspFeedback, MODEL, ROOT, grasp_phases, localize, plan_grasp, require_motion,
    make_model, overhead_observation, reset_scene,
)


class Rerun(messages.Event):
    """Reset the scene and run the grasp from the beginning."""


@dataclass(frozen=True)
class GraspStatus(messages.Snapshot):
    text: str = "Ready"
    waiting: bool = False
    labels: tuple = ()
    boundaries: tuple = ()
    initial_height: float = 0.014
    pads: tuple = (0.0, 0.0)


class GraspSimulation(plugin.SimPlugin):
    def __init__(self, executor, output, stepper):
        self.executor = executor
        self.output = output
        self.stepper = stepper
        self.rng = np.random.default_rng()
        self.previous_xy = np.array([0.205, -0.035])
        self.pending = None
        self.active = False
        self.initialized = False
        self.runs = 0

    def pause(self, paused):
        state = sim.PauseState.NORMAL_PAUSED if paused else sim.PauseState.UNPAUSED
        self.handle.dispatch(messages.RequestPauseEvent(pause_state=state))

    @messages.handler
    def on_rerun(self, event: Rerun):
        if self.pending is not None:
            return
        self.active = False
        self.pause(True)
        self.handle.dispatch(messages.ResetEvent())
        model, data = self.handle.model, self.handle.data
        xy = self.previous_xy
        while np.linalg.norm(xy - self.previous_xy) < 0.018:
            xy = self.rng.uniform([0.177, -0.038], [0.257, 0.025])
        self.previous_xy = xy
        reset_scene(model, data, xy)
        self.initial_height = float(data.body("red_cube").xpos[2])
        self.runs += 1
        self.folder = self.output / f"round-{self.runs:03d}"
        self.folder.mkdir()
        self.result = {
            "round": self.runs, "model": MODEL, "reasoning_effort": EFFORT,
            "spawn_xy_m": xy.tolist(),
        }
        self.handle.send_to_viewer(GraspStatus(
            f"Round {self.runs} | GPT-6.2 Sol High is locating",
            waiting=True, initial_height=self.initial_height,
        ))
        try:
            observation = self.folder / "overhead.png"
            self.rgb = overhead_observation(model, data, observation)
            self.vision_started = monotonic()
            self.pending = self.executor.submit(plan_grasp, observation)
            print(f"Round {self.runs}: new cube XY {xy.round(4).tolist()}, sent fresh overhead frame", flush=True)
        except Exception as error:
            self.vision_failed(error)

    def save_result(self):
        (self.folder / "result.json").write_text(
            json.dumps(self.result, indent=2), encoding="utf-8",
        )

    def vision_failed(self, error):
        self.result.update(success=False, error=str(error))
        self.save_result()
        self.handle.send_to_viewer(GraspStatus(
            f"Round {self.runs} | Stopped | {str(error).splitlines()[0][:160]}",
            initial_height=self.initial_height,
        ))
        print(f"Round {self.runs}: {error}", flush=True)

    @messages.handler
    def before_step(self, event: messages.PreStepEvent):
        if not self.initialized:
            self.pause(True)
            self.initialized = True
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            try:
                decision = future.result()
                self.result["vision_plan"] = decision
                require_motion(decision)
                detection = decision["targets"]["source"]
                pixel, point = localize(self.rgb, detection, event.model, event.data)
                labels, phases = zip(*grasp_phases(event.model, point[:2], strategy=decision["strategy"]))
            except Exception as error:
                self.vision_failed(error)
                return
            self.controls = np.concatenate(phases)
            self.labels = labels
            self.boundaries = tuple(int(n) for n in np.cumsum([len(p) for p in phases]))
            self.feedback = GraspFeedback(event.model, event.data)
            self.index = 0
            self.next_step_at = None
            self.result.update(
                detection=detection, refined_pixel_uv=pixel,
                estimated_target_xy_m=point[:2].tolist(),
                vision_seconds=round(monotonic() - self.vision_started, 2),
            )
            self.handle.send_to_viewer(GraspStatus(
                f"Round {self.runs} | GPT chose {decision['strategy']}", labels=labels,
                boundaries=self.boundaries,
                initial_height=self.initial_height,
            ))
            self.active = True
            self.pause(False)
            print(f"Round {self.runs}: GPT-6.2 Sol High estimated XY {point[:2].round(4).tolist()}", flush=True)
    @messages.handler
    def step(self, event: messages.StepEvent) -> bool:
        if self.pending is not None:
            sleep(0.003)
            return True
        if not self.active:
            return False
        pacing = self.stepper.step_control
        running = pacing.get_pause_state() == sim.PauseState.UNPAUSED
        single = pacing.consume_single_step_request()
        if not (running or single):
            self.next_step_at = None
            sleep(0.002)
            return True
        now = monotonic()
        if running and self.next_step_at is not None and now < self.next_step_at:
            sleep(min(0.001, self.next_step_at - now))
            return True
        interval = event.model.opt.timestep * 100 / max(pacing.get_speed(), 1)
        self.next_step_at = max(now - 0.012, self.next_step_at or now) + interval
        phase = self.labels[bisect_right(self.boundaries, self.index)]
        try:
            self.feedback.step(phase, self.controls[self.index])
        except RuntimeError as error:
            self.finish(event.data, str(error))
            return True
        self.index += 1
        if self.index % 20 == 0:
            self.handle.send_to_viewer(GraspStatus(
                phase, initial_height=self.initial_height, pads=tuple(self.feedback.sensor.pad_n),
            ))
        return True

    @messages.handler
    def after_step(self, event: messages.PostStepEvent):
        if self.active and self.index >= len(self.controls):
            self.finish(event.data)

    def finish(self, data, error=None):
        self.active = False
        self.pause(True)
        lift = (float(data.body("red_cube").xpos[2]) - self.initial_height) * 1000
        passed = error is None and lift >= 50
        self.result.update(success=passed, held_lift_mm=round(lift, 2),
                           simulation_seconds=round(data.time, 2), error=error,
                           force_feedback=self.feedback.save(self.folder / "forces.csv"))
        self.save_result()
        text = "Grasp complete" if passed else error or "Grasp missed"
        self.handle.send_to_viewer(GraspStatus(
            f"Round {self.runs} | {text}", initial_height=self.initial_height,
            pads=tuple(self.feedback.sensor.pad_n),
        ))
        print(f"Round {self.runs}: {text}, lift={lift:.1f} mm", flush=True)


class GraspPanel(plugin.ViewerPlugin):
    def __init__(self, studio_app):
        self.app = studio_app
        self.status = GraspStatus()

    @messages.handler
    def on_status(self, status: GraspStatus):
        self.status = status

    @messages.handler
    def draw(self, event: messages.BuildGuiEvent):
        model = self.viewer.model
        if not model.ncam:
            return
        imgui.SetNextWindowPos(imgui.Vec2(194, 46), imgui.Cond.Once)
        imgui.SetNextWindowSize(imgui.Vec2(360, 151), imgui.Cond.Once)
        visible = imgui.Begin("SO-101 Grasp")
        try:
            if not visible:
                return
            status = self.status
            imgui.BeginDisabled(status.waiting)
            try:
                if imgui.Button("Run grasp again"):
                    self.viewer.send_to_sim(Rerun())
            finally:
                imgui.EndDisabled()
            data = self.viewer.data
            tick = round(data.time / model.opt.timestep)
            phase = status.text
            if status.boundaries and 0 < tick < status.boundaries[-1]:
                phase = status.labels[bisect_right(status.boundaries, tick)]
            lift = (data.body("red_cube").xpos[2] - status.initial_height) * 1000
            imgui.TextUnformatted(phase)
            imgui.TextUnformatted(f"Simulation {data.time:.2f} s   |   Lift {lift:.1f} mm")
            imgui.TextUnformatted(f"Finger {status.pads[0]:.1f} / {status.pads[1]:.1f} N")
            for index, (label, name) in enumerate((
                ("Overview", "overview"), ("Top", "overhead"),
                ("Wrist", "wrist_cam"), ("Free", None),
            )):
                if index:
                    imgui.SameLine()
                if imgui.Button(label):
                    camera_index = model.camera(name).id if name else ux.TUMBLE_CAMERA_IDX
                    self.app.ux_state.camera_index = ux.set_camera(model, self.viewer.camera, camera_index)
                    print(f"Camera: {label}", flush=True)
        finally:
            imgui.End()


def main():
    model = make_model()
    model.vis.global_.cameraid = model.camera("overview").id
    data = mujoco.MjData(model)
    reset_scene(model, data)
    parent = ROOT / "runs" / "so101-grasp" / "studio"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="session-", dir=parent))
    print(f"Session output: {output}", flush=True)
    studio_app = app.App()
    stepper = step_control.StepControl()
    with ThreadPoolExecutor(max_workers=1) as executor:
        launch_native.run(
            ViewerConfig(title="MuJoCo Studio - SO-101 Grasp", width=1180, height=760,
                         gfx="opengl_software"),
            model=model, data=data,
            plugins=[studio_app, stepper, sim_history.SimHistory(), GraspSimulation(executor, output, stepper),
                     GraspPanel(studio_app)],
        )


if __name__ == "__main__":
    main()
