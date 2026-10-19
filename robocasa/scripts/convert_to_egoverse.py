"""Batch-convert MimicDroid TaskDemos -> EgoVerse-style GR1 h5.

Per demo: replay on GR1 (mink IK), capture a FIXED head frame (base->egoview at
that demo's frame 0, EgoVerse/OpenCV +z-forward convention), render the original
4 agentview views, store actions + eef + gripper + obs_head_pose.

Output mirrors TaskDemos structure under .../MimicDroidDataset/GR1/<Task>/003/.
No drop/smoothing filter is applied to replay data (by request).

Usage:
  python convert_to_egoverse.py --batch
  python convert_to_egoverse.py <in_im128.hdf5> <out.h5>
"""
import os, sys, json, glob, traceback
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
import robosuite, robocasa
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

IMG_VIEWS = ["robot0_agentview_center", "robot0_agentview_center_top",
             "robot0_agentview_sideleft", "robot0_agentview_sideright"]
IMG_HW = 128
MJ2CV = np.diag([1.0, -1.0, -1.0])  # mujoco cam(-z fwd,+y up) -> OpenCV(+z fwd,+y down)

def pose4(p, m):
    T = np.eye(4); T[:3,:3] = m; T[:3,3] = p; return T

def mat_to_xyzwxyz(T):
    q = R.from_matrix(T[:3,:3]).as_quat()  # xyzw
    return np.array([T[0,3],T[1,3],T[2,3], q[3],q[0],q[1],q[2]])

def convert_file(in_path, out_path):
    f = h5py.File(in_path, "r")
    env_args = f["data"].attrs["env_args"]
    ea = EnvArgs(robots="GR1TwoHand", task_name=None, render=False, control_freq=20,
                 controller="WHOLE_BODY_MINK_IK", use_camera_obs=False, reset_mode=None)
    env, _ = make_env(file_name=in_path, env_args=ea)
    sim = env.sim
    bid = sim.model.body_name2id("robot0_base")
    egid = sim.model.camera_name2id("robot0_egoview")
    def base_T(): return pose4(sim.data.body_xpos[bid].copy(), sim.data.body_xmat[bid].reshape(3,3).copy())
    def ego_T(): return pose4(sim.data.cam_xpos[egid].copy(), sim.data.cam_xmat[egid].reshape(3,3).copy() @ MJ2CV)

    demos = sorted(f["data"].keys(), key=lambda x:int(x.split("_")[-1]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fo = h5py.File(out_path, "w"); dg = fo.create_group("data")
    n_ok = 0
    for di, dm in enumerate(demos):
        try:
            g = f["data"][dm]
            states = g["states"][:]; actions = g["actions"][:]
            ep = g.attrs.get("ep_meta"); ep = json.loads(ep) if isinstance(ep,(bytes,str)) else ep
            reset_to(env, {"states":states[0], "ep_meta":ep, "model":g.attrs["model_file"],
                           "non_robot_qpos_idx":g.attrs["non_robot_qpos_idx"],
                           "qpos_state":g.attrs["initial_qpos"]},
                     replace_robot_joints=True, change_to_gr1=True)
            sim.model.site_rgba[:, 3] = 0.0; sim.forward()  # hide reset_region viz overlay
            T_base_ego0 = np.linalg.inv(base_T()) @ ego_T()
            imgs={v:[] for v in IMG_VIEWS}; head=[]; ree=[]; lee=[]; rg=[]; lg=[]
            for a in actions:
                obs,_,_,_ = env.step(a)
                head.append(mat_to_xyzwxyz(base_T() @ T_base_ego0))
                rp,rq=obs["robot0_right_eef_pos"],obs["robot0_right_eef_quat"]
                lp,lq=obs["robot0_left_eef_pos"],obs["robot0_left_eef_quat"]
                ree.append([*rp,rq[3],rq[0],rq[1],rq[2]]); lee.append([*lp,lq[3],lq[0],lq[1],lq[2]])
                rg.append(obs.get("robot0_right_gripper_qpos",np.zeros(1))[:1])
                lg.append(obs.get("robot0_left_gripper_qpos",np.zeros(1))[:1])
                for v in IMG_VIEWS: imgs[v].append(sim.render(width=IMG_HW,height=IMG_HW,camera_name=v)[::-1])
            # success filter: only keep demos the GR1 replay actually completes
            success = bool(env._check_success())
            if not success:
                print(f"    [{di+1}/{len(demos)}] {dm}: FAIL (task not success) -> skipped", flush=True)
                continue
            eg = dg.create_group(dm); ob = eg.create_group("obs")
            for v in IMG_VIEWS: ob.create_dataset(v+"_image", data=np.asarray(imgs[v],np.uint8), compression="gzip")
            eg.create_dataset("actions", data=actions)
            eg.create_dataset("obs_head_pose", data=np.asarray(head))
            eg.create_dataset("right.obs_ee_pose", data=np.asarray(ree))
            eg.create_dataset("left.obs_ee_pose", data=np.asarray(lee))
            eg.create_dataset("right.obs_gripper", data=np.asarray(rg))
            eg.create_dataset("left.obs_gripper", data=np.asarray(lg))
            eg.attrs["num_samples"]=len(actions)
            n_ok += 1
            print(f"    [{di+1}/{len(demos)}] {dm}: {len(actions)} frames SUCCESS", flush=True)
        except Exception as e:
            print(f"    [{di+1}/{len(demos)}] {dm}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    dg.attrs["env_args"]=env_args; dg.attrs["robot"]="GR1TwoHand"
    dg.attrs["head_frame"]="fixed base->egoview@frame0, EgoVerse +z-forward convention"
    fo.close(); f.close()
    env.close()
    return n_ok, len(demos)

def main():
    ROOT="/mnt/ddn/dataset/MimicDroidDataset"
    if sys.argv[1]=="--batch":
        files=sorted(glob.glob(f"{ROOT}/TaskDemos/*/*/demo_im128_notp.hdf5"))
        if len(sys.argv) > 2:  # optional limit: --batch N
            files = files[:int(sys.argv[2])]
        print(f"converting {len(files)} task files -> {ROOT}/GR1/")
        tot_ok=tot=0
        for fp in files:
            rel=os.path.relpath(fp, f"{ROOT}/TaskDemos")          # <Task>/003/demo_im128_notp.hdf5
            out=os.path.join(ROOT,"GR1",rel)
            print(f"[FILE] {rel}", flush=True)
            try:
                ok,n=convert_file(fp,out); tot_ok+=ok; tot+=n
            except Exception as e:
                print(f"  FILE FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
        print(f"\nDONE: {tot_ok}/{tot} demos converted -> {ROOT}/GR1/")
    else:
        ok,n=convert_file(sys.argv[1], sys.argv[2]); print(f"{ok}/{n} demos")

if __name__=="__main__":
    main()
