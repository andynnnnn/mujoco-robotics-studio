# YAM Pro Cookie Factory

Local MuJoCo reproduction of [DeepMind's Cookie Factory demonstration](https://github.com/google-deepmind/robotics-antigravity-plugin#cookie-factory). The robot and 4310 gripper use [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam). If needed, fetch the asset into the existing Menagerie checkout with `git -C .menagerie sparse-checkout add i2rt_yam`.

From the repository directory:

```powershell
.venv/Scripts/python.exe examples/studio.py --experiment yam_cookie
```

Select **YAM Pro | Cookie factory conveyor sorting** in **Robotics Experiments**. Click **Run / rerun** to generate a new batch of three cookies of each class, with randomized order and small position offsets. The left-side Studio **Pause / Run** controls simulation pacing. The experiment panel switches among **Overview**, **Top**, **Pickup**, **Wrist**, and **Free**.

The belt carries the cookies by contact friction. It runs at up to 0.05 m/s, indexes one item into the pickup window, and stops while that item is inspected and picked. The finite-length belt returns to its starting position at the next run. The layout and belt speed came from GPT-6.2 Sol High's analysis of the public demonstration frames, recorded in `scene-plan.json`.

Each pickup sends the current RGB frame to GPT-6.2 Sol High through the OpenAI Responses API. Set `OPENAI_API_KEY` in the process environment before starting Studio. The model selects the cookie and bin and returns their image regions. The local controller back-projects MuJoCo camera depth within these regions, validates the geometry, and drives inverse kinematics and finger force control. Simulation and motion run on the PC.

The local controller regulates bilateral finger contact toward 7 N on the weaker side. It stops on unexpected robot contact at 8 N per contact, excessive grip force, grip loss, or unusable model output. MuJoCo scores each item from its final bin and settled pose. Run folders under `runs/studio-experiments` contain each overhead PNG, the corresponding model decision, and `result.json`.

Run without the viewer, with a reproducible seed:

```powershell
.venv/Scripts/python.exe examples/yam_cookie/sorter.py --seed 16 --output runs/yam-cookie/batch-16
```
