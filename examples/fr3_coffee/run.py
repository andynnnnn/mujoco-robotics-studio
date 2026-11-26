"""Validate the two-arm coffee service and record its actual MuJoCo frames."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coffee import CoffeeCycle, POUR_LABEL, ROOT, VisionRequest, locate, make_model, observation, reset


def run(output):
    model = make_model()
    data = mujoco.MjData(model)
    reset(model, data)
    cycle = CoffeeCycle(model, data, output)
    font = ImageFont.load_default(size=20)
    tick = 0
    snapshots = {
        "Left | Close gripper": "cup-pickup.png",
        "Coffee machine | Let espresso settle": "under-coffee-machine.png",
        "Right | Return home": "cup-on-tray.png",
    }
    try:
        with (mujoco.Renderer(model, height=596, width=1060) as renderer,
              imageio.get_writer(output / "fr3-coffee.mp4", fps=25, codec="libx264",
                                 macro_block_size=2, quality=8) as writer):
            step = next(cycle.program)
            while True:
                if isinstance(step, VisionRequest):
                    print(f"{step.name}: taking a fresh overhead image", flush=True)
                    path = output / f"{step.name}-overhead.png"
                    rgb = observation(model, data, path)
                    step = cycle.program.send((rgb, locate(path, step.name)))
                label, controls = step
                detail = label in (POUR_LABEL, "Coffee machine | Let espresso settle")
                for index, control in enumerate(controls):
                    cycle.step(label, control)
                    if label == POUR_LABEL and index == len(controls) // 2:
                        renderer.update_scene(data, camera="pour_detail")
                        Image.fromarray(renderer.render()).save(output / "coffee-pouring.png")
                    if tick % 20 == 0:
                        renderer.update_scene(data, camera="pour_detail" if detail else "overview")
                        frame = Image.fromarray(renderer.render())
                        panel = ImageDraw.Draw(frame)
                        width = int(font.getlength(label)) + 35
                        panel.rounded_rectangle((20, 20, 20 + width, 63), radius=8, fill="#182029")
                        panel.text((36, 30), label, font=font, fill="white")
                        writer.append_data(np.asarray(frame))
                    tick += 1
                if label in snapshots:
                    renderer.update_scene(data, camera="pour_detail" if detail else "counter")
                    Image.fromarray(renderer.render()).save(output / snapshots[label])
                step = next(cycle.program)
    except StopIteration:
        cycle.save()
    except Exception as error:
        cycle.save(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "fr3-coffee-force" / "recording")
    run(parser.parse_args().output)
