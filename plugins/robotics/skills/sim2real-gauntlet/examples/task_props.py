# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MuJoCo task props and geometry-only goal predicates."""

from collections.abc import Callable, Mapping, Sequence
import math

import mujoco
import numpy as np


def _box(
    spec: mujoco.MjSpec,
    name: str,
    pos: Sequence[float],
    size: Sequence[float],
    rgba: Sequence[float],
    mass: float,
    friction: Sequence[float] = (1.4, 0.03, 0.005),
    geom_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX,
) -> None:
  body = spec.worldbody.add_body(name=name, pos=list(pos))
  body.add_freejoint(name=f"{name}_free")
  body.add_geom(
      name=f"{name}_geom",
      type=geom_type,
      size=list(size),
      mass=mass,
      friction=list(friction),
      solref=[0.012, 1.0],
      rgba=list(rgba),
  )


def _bowl(spec: mujoco.MjSpec, pos: Sequence[float], inner_r: float) -> None:
  """A round bowl: a floor disc and 16 rim boxes."""
  rgba, thick, half_h, n = [0.16, 0.42, 0.88, 1.0], 0.008, 0.022, 16
  bowl = spec.worldbody.add_body(name="bowl", pos=list(pos))
  bowl.add_geom(
      name="bowl_floor",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=[inner_r + 2 * thick, 0.004, 0],
      pos=[0, 0, 0.004],
      rgba=rgba,
  )
  ring = inner_r + thick
  half_len = ring * math.tan(math.pi / n) + 0.001
  for i in range(n):
    th = 2 * math.pi * i / n
    bowl.add_geom(
        name=f"bowl_rim_{i:02d}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick, half_len, half_h],
        pos=[ring * math.cos(th), ring * math.sin(th), half_h],
        quat=[math.cos(th / 2), 0, 0, math.sin(th / 2)],
        rgba=rgba,
    )


def _socket(spec: mujoco.MjSpec, pos: Sequence[float], inner_r: float) -> None:
  """An octagonal socket plate."""
  rgba = [0.32, 0.36, 0.42, 1.0]
  sock = spec.worldbody.add_body(name="plate_socket", pos=list(pos))
  sock.add_geom(
      name="socket_base",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=[max(0.052, inner_r + 0.028), 0.004, 0],
      pos=[0, 0, 0.004],
      friction=[0.7, 0.01, 0.002],
      rgba=rgba,
  )
  r = inner_r + 0.010
  for k in range(8):
    a = k * math.pi / 4
    sock.add_geom(
        name=f"socket_wall_{k}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, r * math.tan(math.pi / 8) + 0.0015, 0.012],
        pos=[r * math.cos(a), r * math.sin(a), 0.020],
        quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
        friction=[0.6, 0.01, 0.002],
        rgba=rgba,
    )


def _tilt_deg(pose: np.ndarray) -> float:
  """Angle between the body z axis and world z, from `[pos, wxyz]`."""
  qx, qy = pose[4], pose[5]
  return math.degrees(math.acos(float(np.clip(1 - 2 * (qx**2 + qy**2), -1, 1))))


def _inside_bowl(inner_r: float) -> Callable[..., bool]:
  def ok(p: Mapping[str, np.ndarray]) -> bool:
    d = p["target_cube"][:3] - p["bowl"][:3]
    return bool(np.hypot(d[0], d[1]) < inner_r and 0.0 < d[2] < 0.064)

  return ok


def _in_socket(tol: float) -> Callable[..., bool]:
  def ok(p: Mapping[str, np.ndarray]) -> bool:
    d = p["peg"][:3] - p["plate_socket"][:3]
    return bool(
        np.hypot(d[0], d[1]) < tol and d[2] < 0.057 and _tilt_deg(p["peg"]) < 8
    )

  return ok


def _stacked(p: Mapping[str, np.ndarray]) -> bool:
  d = p["block_top"][:3] - p["block_base"][:3]
  return bool(
      np.hypot(d[0], d[1]) < 0.022
      and 0.03 < d[2] < 0.05
      and _tilt_deg(p["block_top"]) < 10
      and _tilt_deg(p["block_base"]) < 10
  )
