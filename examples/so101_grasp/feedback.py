"""SO-101 force sensing and limits shared by the local simulation experiments."""

import mujoco as mj
import numpy as np


PAD_TARGET_N = 15.0
FIXTURE_STOP_N = 35.0
SELF_STOP_N = 20.0


def regulate(jaw, pads, dt=0.02, target=PAD_TARGET_N):
    error = float(np.mean(pads)) - target
    change = np.clip(0.0015 * error, -0.014, 0.018) * dt / 0.02
    return float(np.clip(jaw + change, -0.07, 0.75))


class ContactFeedback:
    """Resolve normal forces by contacting bodies and latch hard collisions."""

    def __init__(self, model, target, *, prefix="", support=None, extra_targets=(), axis=2):
        self.model = model
        self.target = model.body(target).id
        self.support = model.body(support).id if support else None
        self.targets = {self.target, *(model.body(name).id for name in extra_targets)}
        self.axis = axis
        self.transverse = [i for i in range(3) if i != axis]
        self.robot = model.body_rootid == model.body(prefix + "base").id
        self.pads = {
            model.body(prefix + name).id: i
            for i, name in enumerate(("gripper", "moving_jaw_so101_v1"))
        }
        self.wrench = np.zeros(6)
        self.pad_n = np.zeros(2, np.float32)
        self.insertion_n = np.zeros(2, np.float32)
        self.insertion_xy = np.zeros(2, np.float32)
        self.obstacle_n = 0.0
        self.stop = None

    def read(self, data):
        self.pad_n.fill(0)
        self.insertion_n.fill(0)
        self.insertion_xy.fill(0)
        self.obstacle_n = 0.0
        fixture = {}
        for i, c in enumerate(data.contact):
            if c.efc_address < 0:
                continue
            a, b = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if self.robot[a] and self.robot[b]:
                if (self.model.body_weldid[a] != self.model.body_weldid[b]
                        and not (a in self.pads and b in self.pads)):
                    self._collision(data, i, c, "self_contact", SELF_STOP_N)
            elif self.robot[a] != self.robot[b]:
                arm, other = (a, b) if self.robot[a] else (b, a)
                if other in self.targets and arm in self.pads:
                    mj.mj_contactForce(self.model, data, i, self.wrench)
                    self.pad_n[self.pads[arm]] += max(0.0, self.wrench[0])
                elif other not in self.targets:
                    peak = self._collision(data, i, c, "arm_fixture", FIXTURE_STOP_N)
                    self.obstacle_n = max(self.obstacle_n, peak)
                    vector = peak * c.frame[:3] * (1 if arm == a else -1)
                    key = (other, c.geom2 if arm == a else c.geom1)
                    accumulated, largest, pair = fixture.get(key, (np.zeros(3), -1, c.geom))
                    fixture[key] = (accumulated + vector, max(largest, peak),
                                    c.geom if peak > largest else pair)
            elif self.support is not None and {a, b} == {self.target, self.support}:
                mj.mj_contactForce(self.model, data, i, self.wrench)
                normal = self.wrench[0] * c.frame[:3]
                self.insertion_n += (abs(normal[self.axis]), np.linalg.norm(normal[self.transverse]))
                self.insertion_xy += normal[:2] if b == self.target else -normal[:2]
        for vector, _, pair in fixture.values():
            self._latch(data, pair, "arm_fixture", float(np.linalg.norm(vector)), FIXTURE_STOP_N)

    def _collision(self, data, index, contact, kind, limit):
        mj.mj_contactForce(self.model, data, index, self.wrench)
        force = float(self.wrench[0])
        self._latch(data, contact.geom, kind, force, limit)
        return force

    def _latch(self, data, pair, kind, force, limit):
        if force >= limit and self.stop is None:
            geom_names = [
                self.model.geom(g).name or f"{self.model.body(int(self.model.geom_bodyid[g])).name}#{g}"
                for g in pair
            ]
            self.stop = {"kind": kind, "time_s": data.time, "normal_force_n": force,
                         "geoms": geom_names}
