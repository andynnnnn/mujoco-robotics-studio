"""Request and record decisions from the simulated overhead camera."""

import json

from PIL import Image

from .native import FRAME
from detect_with_openai import plan, require_motion


SCENES = {
    "cube_in_bowl": (
        "Pick up the red block and place it inside the blue bowl. Leave the yellow bystander in place.",
        "small red block", "blue circular bowl with an open center",
    ),
    "peg_in_plate": (
        "Pick up the green cylindrical peg and insert it upright into the purple socket.",
        "green cylindrical peg seen from directly overhead", "purple socket with a central hole",
    ),
    "two_block_stack": (
        "Place the small orange block on top of the larger blue block.",
        "small orange block, not the bright yellow bystander or the yellow robot",
        "larger blue block",
    ),
    "usb_in_port": (
        "Pick up the USB drive and slide its silver connector horizontally into the front side of the purple port.",
        "orange USB drive including its silver tip, not the yellow block", "purple socket housing",
    ),
    "cap_fixed_marker": (
        "Lift the orange hollow cap, align it upright and put it over the fixed upright purple marker.",
        "orange circular marker cap", "slender upright purple marker, including its visible top",
    ),
    "stack3": (
        "Place the loose green block directly onto the red base for the middle layer.",
        "loose green block", "large red base block",
    ),
    "banana_in_bowl": (
        "Pick up the long curved yellow banana near its middle and place the whole fruit inside the blue bowl.",
        "curved yellow banana, not the small yellow cube or robot", "wide blue bowl including its rim",
    ),
}


class Planner:
    def __init__(self, task_id, output, status=print):
        self.task_id = task_id
        self.output = output
        self.status = status
        self.requests = 0

    def __call__(self, observation, stage, *, layer=None):
        self.requests += 1
        self.output.mkdir(parents=True, exist_ok=True)
        stem = self.output / f"plan-{self.requests:02d}"
        frame = stem.with_suffix(".png")
        Image.fromarray(observation[FRAME]).resize((768, 768), Image.Resampling.BICUBIC).save(frame)
        instruction, source, goal = SCENES[self.task_id]
        if layer == "top":
            instruction = "Place the loose blue block on the green middle layer already on the red base."
            source, goal = "loose blue block", "green middle block already on top of the red base"
        if self.task_id in ("peg_in_plate", "cap_fixed_marker", "usb_in_port"):
            strategies = {
                "force_guided_insert": "Use guarded insertion when both parts are clear.",
                "slow_insert": "Approach and insert more slowly when clearance is tight or alignment is difficult.",
            }
        else:
            strategies = {
                "standard_place": "Transfer at the normal speed when the pickup and placement are clear.",
                "slow_place": "Use slower transfer and lowering near a rim, tight clearance or a bystander.",
            }
        self.status(f"GPT-6.2 Sol High | {stage}")
        result = plan(frame, f"{instruction} Stage: {stage}.",
                      {"source": source, "destination": goal}, strategies)
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.status(f"GPT chose {result['strategy']}")
        return require_motion(result)
