"""A crank-driven, laterally guided pelican bicycle on a coastal path."""

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np


RADIUS = .305
CRANK = np.array([-.105, 0, .432])
HIP = np.array([-.28, 0, .872])
CRANK_LENGTH = .108
SECONDS = 9


def xyz(values):
    return " ".join(f"{v:.7g}" for v in values)


def capsule(start, end, radius, color):
    return f'<geom type="capsule" fromto="{xyz([*start,*end])}" size="{radius}" rgba="{color}"/>'


def ellipsoid(pos, size, color, euler=(0,0,0)):
    return f'<geom type="ellipsoid" pos="{xyz(pos)}" size="{xyz(size)}" rgba="{color}" euler="{xyz(euler)}"/>'


def camera(name, eye, target, fovy):
    back = np.asarray(eye) - target
    right = np.cross([0,0,1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back / np.linalg.norm(back), right)
    return f'<camera name="{name}" pos="{xyz(eye)}" xyaxes="{xyz([*right,*up])}" fovy="{fovy}"/>'


def wheel(name, position):
    sand, dark = ".79 .82 .71 1", ".075 .094 .11 1"
    parts = [
        f'<body name="{name}" pos="{xyz(position)}"><joint name="{name}_axle" type="hinge" axis="0 1 0" damping=".012"/>',
        f'<geom name="{name}_tire" class="physics" type="cylinder" size="{RADIUS} .023" euler="90 0 0" mass=".48" rgba="0 0 0 0"/>',
        ellipsoid((0,0,0),(.039,.034,.039),".32 .43 .47 1"),
    ]
    for i in range(32):
        a, b = np.arange(i,i+2) * (2*np.pi/32)
        for radius, thickness, color in ((.282,.023,dark),(.258,.006,sand)):
            start, end = ([radius*np.cos(t),0,radius*np.sin(t)] for t in (a,b))
            parts.append(capsule(start,end,thickness,color))
        if i % 4 == 0:
            parts.append(capsule([0,.009,0],[.258*np.cos(a),0,.258*np.sin(a)],.004,sand))
    return "".join(parts)+"</body>"


def make_model():
    teal = ".08 .52 .55 1"
    cream = ".96 .92 .82 1"
    gray = ".77 .78 .75 1"
    orange = ".95 .46 .16 1"
    coral = ".88 .31 .20 1"
    rear, front = (-.54,0,RADIUS), (.53,0,RADIUS)
    seat, bar, stem = (-.27,0,.735), (.38,0,.775), (.44,0,.630)
    frame = "".join(capsule(a,b,.018,teal) for a,b in
                    ((rear,seat),(rear,CRANK),(seat,CRANK),(seat,stem),(CRANK,stem),(front,stem)))
    frame += capsule(stem,bar,.013,teal)
    frame += capsule([bar[0],-.15,bar[2]],[bar[0],.15,bar[2]],.014,".18 .23 .25 1")
    frame += ellipsoid(seat,(.114,.084,.025),".18 .23 .25 1")
    rider = "".join((
        ellipsoid((-.25,0,.91),(.265,.134,.163),cream,euler=(0,-12,0)),
        ellipsoid((-.48,0,.895),(.138,.082,.050),gray,euler=(0,-14,0)),
        ellipsoid((-.08,0,1.08),(.102,.089,.203),cream,euler=(0,-20,0)),
        ellipsoid((.002,0,1.275),(.109,.088,.098),cream),
        ellipsoid((.164,0,1.248),(.229,.056,.027),orange,euler=(0,6,0)),
        ellipsoid((.133,0,1.213),(.191,.068,.049),".94 .63 .28 1",euler=(0,7,0)),
    ))
    for sign in (-1, 1):
        rider += ellipsoid((-.265,sign*.109,.918),(.193,.048,.111),gray,euler=(0,-13,0))
        rider += ellipsoid((.031,sign*.080,1.297),(.021,.010,.018),".11 .14 .15 1")
        rider += ellipsoid((.036,sign*.088,1.303),(.006,.003,.006),"1 1 .98 1")
        rider += capsule([-.11,sign*.095,1.00],[.335,sign*.124,.780],.028,cream)
    pedals = "".join(capsule([0,0,0],[sign*CRANK_LENGTH,sign*.107,0],.009,gray)
                     + capsule([sign*CRANK_LENGTH,sign*.081,0],
                               [sign*CRANK_LENGTH,sign*.145,0],.013,coral)
                     for sign in (-1,1))
    limbs = ""
    for suffix in ("l", "r"):
        for limb, length in (("thigh",.287),("shin",.284)):
            limbs += f'<body name="{limb}_{suffix}" mocap="true">' + capsule(
                (0,0,-length/2),(0,0,length/2),.025 if limb=="thigh" else .016, orange) + "</body>"
        limbs += f'<body name="foot_{suffix}" mocap="true">' + ellipsoid(
            (.026,0,0),(.064,.032,.018),orange) + "</body>"
    road = "".join(f'<geom type="box" pos="{x} -1.34 .001" size=".24 .024 .001" rgba=".93 .83 .60 1"/>'
                   for x in range(-3, 35))
    road += "".join(f'<geom type="box" pos="{x} 1.61 .13" size=".038 .055 .13" rgba=".76 .64 .42 1"/>'
                    for x in range(-4, 38, 2))
    road += '<geom type="box" pos="16 1.61 .24" size="21 .027 .026" rgba=".88 .78 .55 1"/>'
    views = (camera("overview",(2.8,-4.0,1.92),(.05,0,.72),34)
             + camera("side",(.06,-3.75,1.21),(.03,0,.70),35)
             + camera("pedals",(1.02,-1.7,.77),(-.10,0,.48),38))
    xml = f'''<mujoco model="Coastal pelican">
      <compiler angle="degree"/>
      <option timestep=".004" integrator="implicitfast"/>
      <visual>
        <global offwidth="960" offheight="620"/>
        <headlight ambient=".48 .48 .48" diffuse=".60 .60 .60" specular=".12 .12 .12"/>
      </visual>
      <asset><texture type="skybox" builtin="gradient" rgb1=".52 .78 .90" rgb2=".84 .91 .91" width="256" height="256"/></asset>
      <default>
        <geom contype="0" conaffinity="0" mass="0"/>
        <default class="physics"><geom contype="1" conaffinity="1" friction="1.05 .015 .006"/></default>
      </default>
      <worldbody>
        <light directional="true" pos="0 -3 6" dir=".24 .42 -1" diffuse=".85 .80 .69"/>
        <geom name="road" class="physics" type="box" pos="16 0 -.08" size="22 1.5 .08" rgba=".48 .54 .54 1"/>
        <geom type="box" pos="16 -4 -.093" size="22 2.5 .09" rgba=".80 .72 .49 1"/>
        <geom type="box" pos="16 2.50 -.103" size="22 1 .10" rgba=".56 .66 .37 1"/>
        <geom type="box" pos="16 15 -.16" size="40 12 .12" rgba=".24 .61 .68 1"/>
        {road}
        <body name="bike">
          <joint name="travel" type="slide" axis="1 0 0" damping=".32"/>
          <joint name="heave" type="slide" axis="0 0 1" damping=".30" range="-.075 .075" limited="true"/>
          <inertial pos="-.10 0 .60" mass="5.2" diaginertia=".7 1.1 1.3"/>
          {views}{frame}{rider}
          {wheel("rear",rear)}{wheel("front",front)}
          <body pos="{xyz(CRANK)}">
            <joint name="crank" type="hinge" axis="0 1 0" damping=".035" armature=".035"/>
            <inertial pos="0 0 0" mass=".2" diaginertia=".002 .003 .002"/>
            {pedals}
          </body>
        </body>
        {limbs}
      </worldbody>
      <equality><joint joint1="rear_axle" joint2="crank" polycoef="0 3 0 0 0" solref=".010 1"/></equality>
      <actuator><velocity name="pedal_drive" joint="crank" kv="1.7" ctrllimited="true" ctrlrange="0 2.6"
                          forcelimited="true" forcerange="-3 3"/></actuator>
    </mujoco>'''
    return mj.MjModel.from_xml_string(xml)


def leg_pose(model, data):
    phase = data.joint("crank").qpos.item()
    travel = data.joint("travel").qpos.item()
    origin = np.array([travel, 0, data.joint("heave").qpos.item()])
    for suffix, sign in (("l",-1),("r",1)):
        ankle = CRANK + origin + [sign*CRANK_LENGTH*np.cos(phase), sign*.112,
                                   -sign*CRANK_LENGTH*np.sin(phase)]
        hip = HIP + origin + [0, sign*.089, 0]
        direction = ankle-hip
        length = np.linalg.norm(direction)
        direction /= length
        a, b = .287, .284
        reach = (a*a-b*b+length*length)/(2*length)
        bend = np.array([-direction[2],0,direction[0]])
        bend /= np.linalg.norm(bend)
        knee = hip + direction*reach + bend*np.sqrt(max(0,a*a-reach*reach))
        for name, start, end in (("thigh",hip,knee),("shin",knee,ankle)):
            ident = model.body(name+"_"+suffix).mocapid.item()
            axis = (end-start) / np.linalg.norm(end-start)
            quat = np.array([1+axis[2], -axis[1], axis[0], 0])
            if np.linalg.norm(quat) < 1e-9:
                quat = np.array([0,1,0,0])
            data.mocap_pos[ident] = (start+end)/2
            data.mocap_quat[ident] = quat/np.linalg.norm(quat)
        foot = model.body("foot_"+suffix).mocapid.item()
        data.mocap_pos[foot] = ankle + [0,0,.019]


def prepare():
    model = make_model()
    data = mj.MjData(model)
    leg_pose(model,data)
    mj.mj_forward(model,data)
    return model,data


def run(output, *, observe=None, status=print):
    output.mkdir(parents=True, exist_ok=True)
    model,data = prepare()
    if observe:
        observe(model,data,0,(0,0),0,(0,0))
    status("Pedaling | rear-wheel traction and measured travel")
    crank = model.actuator("pedal_drive").id
    rear = model.geom("rear_tire").id
    front = model.geom("front_tire").id
    ground = model.geom("road").id
    samples, rolling_error = 0, []
    steps = round(SECONDS/model.opt.timestep)
    for tick in range(1,steps+1):
        data.ctrl[crank] = min(2.25, data.time*1.8)
        mj.mj_step(model,data)
        if tick % 5 == 0:
            leg_pose(model,data)
            mj.mj_forward(model,data)
            contacts = {tuple(sorted(c.geom)) for c in data.contact if c.efc_address >= 0}
            if all(tuple(sorted((ground,wheel))) in contacts for wheel in (rear,front)):
                samples += 1
            speed = data.joint("travel").qvel.item()
            if data.time >= 2:
                rolling_error.append(abs(speed - RADIUS*data.joint("rear_axle").qvel.item()))
            if observe:
                observe(model,data,tick,(0,0),0,(0,0))
    distance = data.joint("travel").qpos.item()
    turns = data.joint("crank").qpos.item()/(2*np.pi)
    contact_ratio = samples/(steps//5)
    slip = float(np.mean(rolling_error))
    result = {"success": bool(distance > 7 and turns > 1.4 and contact_ratio > .98 and slip < .25),
              "distance_m": distance, "crank_turns": turns, "wheel_contact_fraction": contact_ratio,
              "mean_rear_slip_m_s": slip, "simulation_seconds": data.time}
    (output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["success"])
