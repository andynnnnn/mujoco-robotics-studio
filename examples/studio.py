"""Open the local robotics experiments in one native MuJoCo Studio window."""

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from threading import Event

import mujoco
from mujoco.studio import app, launch_native, messages, plugin, sim, step_control, ux
from mujoco.studio.app.plugins import imgui
from mujoco.studio.viewer_protocol import ViewerConfig

from fr3_coffee import studio as coffee
import grocery_sort
import pelican_ride
from gauntlet_windows import studio as gauntlet
from gauntlet_windows.native import ROOT, TASKS
from gauntlet_windows.run import cases
from robosim_gauntlet import perturb
from so101_grasp import studio as grasp
from yam_cookie import scene as cookie


EXPERIMENTS = {
    "grasp": "SO-101 | Randomized red cube grasp",
    "coffee": "FR3 Duo | Coffee service with force feedback",
    "yam_cookie": "YAM Pro | Cookie factory conveyor sorting",
    "grocery_sort": "SO-101 | Grocery sorting with fragile handling",
    "pelican_ride": "Pelican | Coastal bicycle ride",
    "cube_in_bowl": "Gauntlet | Cube in bowl",
    "peg_in_plate": "Gauntlet | Pin insertion",
    "two_block_stack": "Gauntlet | Block stacking",
    "usb_in_port": "SO-101 | Horizontal USB insertion",
    "cap_fixed_marker": "SO-101 | Cap fixed marker",
    "stack3": "SO-101 | Three-block stack",
    "banana_in_bowl": "SO-101 | Banana in bowl",
}
COMMON_CAMERAS = (("Overview", "overview"),)
EXTRA_CAMERAS = {
    "grasp": (("Top", "overhead"), ("Wrist", "wrist_cam")),
    "coffee": (("Top", "overhead"), ("Counter", "counter"), ("Pour", "pour_detail"),
               ("Left wrist", "wrist_camera_left"), ("Right wrist", "wrist_camera_right")),
    "yam_cookie": (("Top", "overhead"), ("Pickup", "pickup"), ("Wrist", "wrist_cam")),
    "grocery_sort": (("Top", gauntlet.FRAME), ("Wrist", "robot/wrist_cam")),
    "pelican_ride": (("Side", "side"), ("Pedals", "pedals")),
    "gauntlet": (("Top", gauntlet.FRAME),),
}


def case_options(task_id):
    for tier, idx in cases(gauntlet.SEED, task_id):
        label = f"{tier}-{idx}"
        if tier == "T4":
            rng = perturb.streams(gauntlet.SEED, task_id, tier, idx).spec
            case = perturb.sample(tier, idx, rng)
            if case.fault == "camera_freeze":
                condition = "fresh timestamps" if case.fault_restamp else "stale timestamps"
                label += f" | Camera freeze, {condition}"
            else:
                label += " | Gripper failure"
        elif (task_id, tier, idx) == ("two_block_stack", "T1", 1):
            label += " | Grasp retry"
        yield label, tier, idx


OPTIONS = {task: tuple(case_options(task)) for task in TASKS}


def prepare(experiment):
    if experiment == "grasp":
        model = grasp.make_model()
        data = mujoco.MjData(model)
        grasp.reset_scene(model, data)
    elif experiment == "coffee":
        model = coffee.make_model()
        data = mujoco.MjData(model)
        coffee.reset(model, data)
    elif experiment == "yam_cookie":
        model = cookie.make_model()
        data = mujoco.MjData(model)
        cookie.reset(model, data, cookie.PARK)
    elif experiment == "grocery_sort":
        model, data = grocery_sort.prepare()
    elif experiment == "pelican_ride":
        model, data = pelican_ride.prepare()
    else:
        model = gauntlet.scene(experiment).compile()
        data = mujoco.MjData(model)
        joints = [model.joint("robot/" + name) for name in gauntlet.JOINTS]
        data.qpos[[joint.qposadr.item() for joint in joints]] = gauntlet.HOME
        data.ctrl[[model.actuator(joint.name).id for joint in joints]] = gauntlet.HOME
    model.vis.global_.cameraid = model.camera("overview").id
    mujoco.mj_forward(model, data)
    return model, data


@dataclass(frozen=True)
class SelectExperiment(messages.Event):
    name: str


@dataclass(frozen=True)
class RunExperiment(messages.Event):
    name: str
    case: int = 0


@dataclass(frozen=True)
class ExperimentState(messages.Snapshot):
    name: str = "grasp"
    busy: bool = False
    error: str = ""


class Experiments(plugin.SimPlugin):
    def __init__(self, worker, closed, stepper, output):
        self.name = "grasp"
        self.state = None
        self.error = ""
        self.grasp = grasp.GraspSimulation(worker, output / "grasp", stepper)
        self.coffee = coffee.CoffeeSimulation(worker, output / "coffee", stepper)
        self.gauntlet = gauntlet.GauntletSimulation(worker, closed, stepper, output / "gauntlet")
        for folder in ("grasp", "coffee"):
            (output / folder).mkdir()

    @messages.handler
    def connect(self, event: plugin.SimInitEvent):
        for adapter in (self.grasp, self.coffee, self.gauntlet):
            adapter.handle = event.handle

    def busy(self):
        if self.name == "grasp":
            return self.grasp.pending is not None or self.grasp.active
        if self.name == "coffee":
            return self.coffee.cycle is not None
        return self.gauntlet.pending is not None

    def publish(self):
        state = ExperimentState(self.name, self.busy(), self.error)
        if state != self.state:
            self.state = state
            self.handle.send_to_viewer(state)

    @messages.handler
    def select(self, event: SelectExperiment):
        if event.name == self.name or event.name not in EXPERIMENTS or self.busy():
            return
        try:
            model, data = prepare(event.name)
            self.handle.set_model(model, data=data)
        except Exception as error:
            self.error = f"Could not open {EXPERIMENTS[event.name]}: {error}"
            self.publish()
            return
        self.handle.dispatch(messages.RequestPauseEvent(pause_state=sim.PauseState.NORMAL_PAUSED))
        self.name = event.name
        self.error = ""
        self.handle.send_to_viewer(grasp.GraspStatus())
        self.handle.send_to_viewer(coffee.CoffeeStatus(detail=""))
        self.handle.send_to_viewer(gauntlet.Progress())
        self.publish()
        print(f"Selected {EXPERIMENTS[self.name]}", flush=True)

    @messages.handler
    def run(self, event: RunExperiment):
        if event.name != self.name or self.busy():
            return
        self.error = ""
        if self.name == "grasp":
            self.grasp.on_rerun(grasp.Rerun())
        elif self.name == "coffee":
            self.coffee.start(coffee.StartCoffee())
        elif self.name in gauntlet.LOCAL_EXPERIMENTS:
            self.gauntlet.start_case(EXPERIMENTS[self.name], self.name, "batch", 0)
        else:
            label, tier, idx = OPTIONS[self.name][event.case]
            self.gauntlet.start_case(f"{EXPERIMENTS[self.name]} | {label}", self.name, tier, idx)
        self.publish()

    @messages.handler
    def pre_step(self, event: messages.PreStepEvent):
        if self.name == "grasp":
            self.grasp.before_step(event)
        elif self.name == "coffee":
            self.coffee.pre_step(event)
        self.publish()

    @messages.handler
    def step(self, event: messages.StepEvent) -> bool:
        if self.name == "grasp":
            return self.grasp.step(event)
        if self.name == "coffee":
            return self.coffee.step(event)
        return self.gauntlet.step(event)

    @messages.handler
    def post_step(self, event: messages.PostStepEvent):
        if self.name == "grasp":
            self.grasp.after_step(event)
        self.publish()


class ExperimentPanel(plugin.ViewerPlugin):
    def __init__(self, studio_app):
        self.app = studio_app
        self.state = ExperimentState()
        self.case = 0
        self.grasp = grasp.GraspStatus()
        self.coffee = coffee.CoffeeStatus(detail="")
        self.gauntlet = gauntlet.Progress()
        self.reset_camera = False

    @messages.handler
    def selection(self, state: ExperimentState):
        if state.name != self.state.name:
            self.case = 0
            self.reset_camera = True
        self.state = state

    @messages.handler
    def grasp_status(self, status: grasp.GraspStatus):
        self.grasp = status

    @messages.handler
    def coffee_status(self, status: coffee.CoffeeStatus):
        self.coffee = status

    @messages.handler
    def gauntlet_status(self, status: gauntlet.Progress):
        self.gauntlet = status

    def camera(self, name):
        model = self.viewer.model
        index = ux.TUMBLE_CAMERA_IDX if not name else model.camera(name).id
        self.app.ux_state.camera_index = ux.set_camera(model, self.viewer.camera, index)

    def progress(self):
        model, data = self.viewer.model, self.viewer.data
        if self.state.name == "grasp":
            status = self.grasp
            tick = round(data.time / model.opt.timestep)
            text = status.text
            if status.boundaries and 0 < tick < status.boundaries[-1]:
                text = status.labels[bisect_right(status.boundaries, tick)]
            imgui.TextWrapped(text)
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube") >= 0:
                lift = (data.body("red_cube").xpos[2] - status.initial_height) * 1000
                imgui.TextUnformatted(f"Cube lift {lift:.1f} mm")
            imgui.TextUnformatted(f"Finger {status.pads[0]:.1f} / {status.pads[1]:.1f} N")
        elif self.state.name == "coffee":
            imgui.TextWrapped(self.coffee.headline)
            if self.coffee.detail:
                imgui.TextWrapped(self.coffee.detail)
            if self.coffee.forces:
                imgui.TextWrapped(self.coffee.forces)
        else:
            imgui.TextWrapped(self.gauntlet.text)
            fixed, moving = self.gauntlet.pads
            if self.state.name != "pelican_ride" and (self.gauntlet.busy or self.state.name not in gauntlet.LOCAL_EXPERIMENTS):
                imgui.TextUnformatted(f"Finger {fixed:.1f} / {moving:.1f} N | Contact {self.gauntlet.contact:.1f} N")
            if self.state.name in ("peg_in_plate", "usb_in_port", "cap_fixed_marker"):
                axial, lateral = self.gauntlet.insertion
                imgui.TextUnformatted(f"Socket load | Axial {axial:.1f} N | Lateral {lateral:.1f} N")

    @messages.handler
    def draw(self, _: messages.BuildGuiEvent):
        imgui.SetNextWindowPos(imgui.Vec2(194, 43), imgui.Cond.Once)
        imgui.SetNextWindowSize(imgui.Vec2(505, 480), imgui.Cond.Once)
        visible = imgui.Begin("Robotics Experiments")
        try:
            if not visible or not self.viewer.model.ncam:
                return
            if self.reset_camera:
                self.camera("overview")
                self.reset_camera = False
            name = self.state.name
            imgui.BeginDisabled(self.state.busy)
            try:
                for key, label in EXPERIMENTS.items():
                    if imgui.RadioButton(label, key == name):
                        self.viewer.send_to_sim(SelectExperiment(key))
                if name in OPTIONS:
                    changed, self.case = imgui.Combo("Case", self.case, [x[0] for x in OPTIONS[name]])
                    if changed:
                        self.gauntlet = gauntlet.Progress(text="Click Run / rerun to load this case.")
                if imgui.Button("Run / rerun"):
                    self.viewer.send_to_sim(RunExperiment(name, self.case))
            finally:
                imgui.EndDisabled()
            imgui.SameLine()
            seed = f"Seed {gauntlet.SEED} | " if name in OPTIONS else ""
            imgui.TextUnformatted(f"{seed}{self.viewer.data.time:.2f} s")
            if self.state.error:
                imgui.TextWrapped(self.state.error)
            else:
                self.progress()
            model = self.viewer.model
            cameras = COMMON_CAMERAS + EXTRA_CAMERAS.get(name, EXTRA_CAMERAS["gauntlet"])
            for index, (label, camera_name) in enumerate((*cameras, ("Free", None))):
                if index:
                    imgui.SameLine()
                if imgui.Button(label):
                    if name == "coffee" and camera_name and mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                        camera_name = coffee.resolve_mjcf_camera_name(model, camera_name)
                    self.camera(camera_name)
                    print(f"Camera: {EXPERIMENTS[name]} | {label}", flush=True)
        finally:
            imgui.End()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="grasp", choices=EXPERIMENTS)
    initial = parser.parse_args().experiment
    model, data = prepare(initial)
    studio_app, stepper, closed = app.App(), step_control.StepControl(), Event()
    runs = ROOT / "runs" / "studio-experiments"
    runs.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="session-", dir=runs))
    print(f"Studio session: {output}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as worker:
        experiments = Experiments(worker, closed, stepper, output)
        experiments.name = initial
        panel = ExperimentPanel(studio_app)
        panel.state = ExperimentState(initial)
        try:
            launch_native.run(
                ViewerConfig(title="MuJoCo Studio - Robotics Experiments", width=1280,
                             height=810, gfx="opengl_software"),
                model=model, data=data,
                plugins=[studio_app, stepper, experiments, panel],
            )
        finally:
            closed.set()


if __name__ == "__main__":
    main()
