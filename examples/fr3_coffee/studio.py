"""Run the coffee service interactively in native MuJoCo Studio."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from time import monotonic, sleep

import mujoco
from mujoco.studio import app, launch_native, messages, plugin, sim, sim_history, step_control, ux
from mujoco.studio.app.plugins import imgui
from mujoco.studio.viewer_protocol import ViewerConfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr3_coffee.coffee import (
    CoffeeCycle, DROP_COUNT, PAD_TARGET_N, ROOT, VisionRequest, locate, make_model,
    observation, reset,
)
from fr3_mjcf import resolve_mjcf_camera_name


class StartCoffee(messages.Event):
    """Reset the workcell and start a fresh cup service."""


@dataclass(frozen=True)
class CoffeeStatus(messages.Snapshot):
    headline: str = "Ready"
    detail: str = "Click Run coffee service to start."
    busy: bool = False
    forces: str = ""


class CoffeeSimulation(plugin.SimPlugin):
    def __init__(self, worker, output, stepper):
        self.worker = worker
        self.output = output
        self.stepper = stepper
        self.pending = None
        self.controls = None
        self.cycle = None
        self.round = 0
        self.initialized = False
        self.stop_latched = False
        self.next_step_at = None

    def pause(self, value):
        state = sim.PauseState.NORMAL_PAUSED if value else sim.PauseState.UNPAUSED
        self.handle.dispatch(messages.RequestPauseEvent(pause_state=state))

    def status(self, headline, detail="", busy=True):
        self.handle.send_to_viewer(CoffeeStatus(headline, detail, busy, self.force_line()))
        print(f"Round {self.round}: {headline} {detail}", flush=True)

    def force_line(self):
        if self.cycle is None:
            return ""
        left, right = self.cycle.feedback.pad_n
        return (f"Pad N   L {left[0]:.1f} / {left[1]:.1f}    R {right[0]:.1f} / {right[1]:.1f}"
                f"    Target {PAD_TARGET_N:g} per pad")

    @messages.handler
    def start(self, _: StartCoffee):
        if self.cycle is not None:
            return
        self.round += 1
        self.stop_latched = False
        self.next_step_at = None
        self.pause(True)
        self.handle.dispatch(messages.ResetEvent())
        model, data = self.handle.model, self.handle.data
        reset(model, data)
        self.cycle = CoffeeCycle(model, data, self.output / f"round-{self.round:03d}")
        self.advance()

    def finish(self, error=None):
        self.pause(True)
        self.cycle.save(error)
        coffee = self.cycle.result["coffee"]
        self.stop_latched = self.cycle.feedback.stop is not None
        self.cycle = self.controls = None
        title = ("Safety stop latched" if self.stop_latched else
                 "Service complete" if error is None else "Service stopped")
        self.status(title,
                    (f"Coffee on tray: {coffee['in_cup_at_end']}/{coffee['emitted']} retained; "
                     f"{coffee['outside_cup_at_end']} outside cup."
                     if error is None else str(error)), busy=False)

    def advance(self, response=None):
        model, data = self.handle.model, self.handle.data
        try:
            program = self.cycle.program
            step = next(program) if response is None else program.send(response)
            if isinstance(step, VisionRequest):
                self.pause(True)
                self.controls = None
                self.next_step_at = None
                self.request = step
                self.status("GPT-6.2 Sol High is locating",
                            "Empty cup on counter" if step.name == "source" else "Cup at espresso machine")
                path = self.cycle.output / f"{step.name}-overhead.png"
                self.rgb = observation(model, data, path)
                self.pending = self.worker.submit(locate, path, step.name)
            else:
                self.phase, self.controls = step
                self.index = 0
                self.last_progress_update = data.time
                self.status(self.phase)
                if response is not None:
                    self.pause(False)
        except StopIteration:
            self.finish()
        except Exception as error:
            self.finish(error)

    @messages.handler
    def pre_step(self, event: messages.PreStepEvent):
        if not self.initialized:
            self.pause(True)
            self.initialized = True
        if self.pending is not None and self.pending.done():
            pending, self.pending = self.pending, None
            try:
                detection = pending.result()
            except Exception as error:
                self.finish(error)
                return
            self.advance((self.rgb, detection))
    @messages.handler
    def step(self, event: messages.StepEvent) -> bool:
        pacing = self.stepper.step_control
        single = pacing.consume_single_step_request()
        running = pacing.get_pause_state() == sim.PauseState.UNPAUSED
        if self.controls is None:
            if running and (self.stop_latched or self.pending is not None):
                self.pause(True)
            sleep(0.002)
            return True
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
        try:
            self.cycle.step(self.phase, self.controls[self.index])
        except Exception as error:
            self.finish(error)
            return True
        self.index += 1
        liquid = self.cycle.liquid
        if event.data.time - self.last_progress_update >= 0.10:
            self.last_progress_update = event.data.time
            detail = ""
            if liquid.emitted:
                stats = liquid.summary()
                detail = (f"Dispensed {liquid.emitted}/{DROP_COUNT} drops" if not liquid.after_pour else
                          f"In cup {stats['in_cup_at_end']}/{liquid.emitted}   |   "
                          f"Left cup during transfer {stats['left_cup_during_right_arm_transfer']}")
            self.handle.send_to_viewer(CoffeeStatus(self.phase, detail, True, self.force_line()))
        if self.index >= len(self.controls):
            self.advance()
        return True


class CoffeePanel(plugin.ViewerPlugin):
    def __init__(self, studio_app):
        self.app = studio_app
        self.status = CoffeeStatus()

    @messages.handler
    def update_status(self, status: CoffeeStatus):
        self.status = status

    @messages.handler
    def draw(self, _: messages.BuildGuiEvent):
        model = self.viewer.model
        if not model.ncam:
            return
        imgui.SetNextWindowPos(imgui.Vec2(194, 46), imgui.Cond.Once)
        imgui.SetNextWindowSize(imgui.Vec2(485, 153), imgui.Cond.Once)
        visible = imgui.Begin("FR3 Duo Coffee Service")
        try:
            if not visible:
                return
            imgui.BeginDisabled(self.status.busy)
            try:
                if imgui.Button("Run coffee service"):
                    self.viewer.send_to_sim(StartCoffee())
            finally:
                imgui.EndDisabled()
            imgui.TextUnformatted(f"{self.status.headline}   |   {self.viewer.data.time:.2f} s")
            if self.status.detail:
                imgui.TextWrapped(self.status.detail)
            if self.status.forces:
                imgui.TextUnformatted(self.status.forces)
            for i, (label, camera) in enumerate((
                ("Overview", "overview"), ("Counter", "counter"), ("Pour", "pour_detail"), ("Top", "overhead"),
                ("Left wrist", "wrist_camera_left"), ("Right wrist", "wrist_camera_right"),
                ("Free", None),
            )):
                if i:
                    imgui.SameLine()
                if imgui.Button(label):
                    idx = (model.camera(resolve_mjcf_camera_name(model, camera)).id
                           if camera else ux.TUMBLE_CAMERA_IDX)
                    self.app.ux_state.camera_index = ux.set_camera(model, self.viewer.camera, idx)
                    print(f"Camera: {label}", flush=True)
        finally:
            imgui.End()


def main():
    model = make_model()
    data = mujoco.MjData(model)
    reset(model, data)
    model.vis.global_.cameraid = model.camera("overview").id
    parent = ROOT / "runs" / "fr3-coffee-force"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="studio-", dir=parent))
    print(f"Studio session: {output}", flush=True)
    studio_app = app.App()
    stepper = step_control.StepControl()
    with ThreadPoolExecutor(max_workers=1) as worker:
        launch_native.run(
            ViewerConfig(title="MuJoCo Studio - FR3 Duo Coffee Force", width=1280,
                         height=800, gfx="opengl_software"),
            model=model, data=data,
            plugins=[studio_app, stepper, sim_history.SimHistory(),
                     CoffeeSimulation(worker, output, stepper),
                     CoffeePanel(studio_app)],
        )


if __name__ == "__main__":
    main()
