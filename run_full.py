import math
import argparse
import datetime
import os
import time
from collections import deque
from bullet.objects import VCylinder

import torch
import numpy as np

from env.utils import make_env, make_vec_envs
from com.logger import CSVLogger

now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

DEG2RAD = np.pi / 180


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["train", "play", "test", "plan"]
    )
    parser.add_argument("--env", type=str, default="env:Gibbon2DCustomEnv-v0")
    parser.add_argument("--dir", type=str, default=os.path.join("exp_f", now_str))
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--frames", type=int, default=2.5e7)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--use_curriculum", type=int, default=1)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def train(args):
    from algo.ppo import PPO, PPOReplayBuffer, SoftsignActor, Policy

    num_frames = args.frames
    episode_steps = 40000
    num_processes = 125 if os.name != "nt" else torch.multiprocessing.cpu_count()
    num_steps = episode_steps // num_processes
    mini_batch_size = 2000
    num_mini_batch = episode_steps // mini_batch_size
    save_every = int(num_frames // 5)
    save_name = "-".join(args.env.split(":"))

    if args.tag is not None:
        args.dir = f"{args.dir}_{args.tag}"

    ppo_params = {
        "use_clipped_value_loss": False,
        "num_mini_batch": num_mini_batch,
        "entropy_coef": 0.0,
        "value_loss_coef": 1.0,
        "ppo_epoch": 10,
        "clip_param": 0.2,
        "lr": 3e-4,
        "eps": 1e-5,
        "max_grad_norm": 2.0,
    }

    final_lr = 3e-5
    envs = make_vec_envs(args.env, args.seed, num_processes)

    obs_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]

    dummy_env = make_env(args.env)

    if args.net is None:
        policy = Policy(SoftsignActor(dummy_env)).to(device)
    else:
        policy = torch.load(args.net, map_location=device)
        ppo_params["lr"] = 3e-5

    agent = PPO(policy, **ppo_params)

    mirror_indices = dummy_env.unwrapped.get_mirror_indices()
    rollouts = PPOReplayBuffer(
        num_steps,
        num_processes,
        obs_dim,
        action_dim,
        device=device,
        mirror=mirror_indices,
    )

    ep_rewards = deque(maxlen=num_processes)
    curriculum_metrics = deque(maxlen=num_processes)
    num_updates = int(num_frames) // num_steps // num_processes

    # don't divide by 0
    ep_rewards.append(0)
    curriculum_metrics.append(0)

    # This has to be done before reset
    if args.use_curriculum:
        current_curriculum = dummy_env.unwrapped.curriculum
        max_curriculum = dummy_env.unwrapped.max_curriculum
        advance_threshold = dummy_env.unwrapped.advance_threshold
        envs.set_env_params({"curriculum": current_curriculum})
        del dummy_env
    else:
        current_curriculum = -1
        max_curriculum = -1
        advance_threshold = -1

    obs = envs.reset()
    rollouts.observations[0].copy_(torch.from_numpy(obs))

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    logger = CSVLogger(log_dir=args.dir)
    save_checkpoint = save_every
    best_reward_so_far = float("-inf")
    start_time = time.time()

    for iteration in range(num_updates):

        scheduled_lr = max(ppo_params["lr"] * (0.99**iteration), final_lr)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = scheduled_lr

        # Disable gradient for data collection
        torch.set_grad_enabled(False)
        policy.train(mode=False)

        for (
            env_obs,
            obs_buf,
            act_buf,
            act_log_prob_buf,
            value_buf,
            reward_buf,
            mask_buf,
            bad_mask_buf,
        ) in zip(
            rollouts.observations[:-1],
            rollouts.observations[1:],
            rollouts.actions,
            rollouts.action_log_probs,
            rollouts.value_preds,
            rollouts.rewards,
            rollouts.masks[1:],
            rollouts.bad_masks[1:],
        ):
            value, action, action_log_prob = policy.act(env_obs)
            cpu_actions = action.cpu().numpy()

            obs, reward, done, info = envs.step(cpu_actions)

            mask = torch.from_numpy((~done).astype(np.float32)[:, None])
            reward = torch.from_numpy(reward.astype(np.float32)[:, None])
            bad_mask = torch.tensor(
                [0.0 if "bad_transition" in d else 1.0 for d in info]
            ).view(-1, 1)
            ep_rewards.extend([d["episode"]["r"] for d in info if "episode" in d])
            curriculum_metrics.extend(
                [d["curriculum_metric"] for d in info if "curriculum_metric" in d]
            )

            obs_buf.copy_(torch.from_numpy(obs))
            act_buf.copy_(action)
            act_log_prob_buf.copy_(action_log_prob)
            value_buf.copy_(value)
            reward_buf.copy_(reward)
            mask_buf.copy_(mask)
            bad_mask_buf.copy_(bad_mask)

        next_value = policy.get_value(rollouts.observations[-1]).detach()

        # Update curriculum after roll-out
        mean_curriculum_metric = sum(curriculum_metrics) / len(curriculum_metrics)
        if (
            args.use_curriculum
            and mean_curriculum_metric > advance_threshold
            and current_curriculum < max_curriculum
        ):
            current_curriculum += 1
            envs.set_env_params({"curriculum": current_curriculum})
            curriculum_metrics.clear()
            curriculum_metrics.append(0)  # append 0 to make sure we don't divide by 0
            ep_rewards.clear()
            ep_rewards.append(0)
            obs = envs.reset()
            rollouts.observations[0].copy_(torch.from_numpy(obs))

        # Enable gradients for training
        torch.set_grad_enabled(True)
        policy.train(mode=True)

        rollouts.compute_returns(next_value)
        _, _, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        model_name = f"{save_name}_latest.pt"
        torch.save(policy, os.path.join(args.dir, model_name))

        frame_count = (iteration + 1) * num_steps * num_processes
        if frame_count >= save_checkpoint or iteration == num_updates - 1:
            model_name = f"{save_name}_{int(save_checkpoint)}.pt"
            save_checkpoint += save_every
            torch.save(policy, os.path.join(args.dir, model_name))

        mean_ep_reward = sum(ep_rewards) / len(ep_rewards)
        if len(ep_rewards) > 1 and mean_ep_reward > best_reward_so_far:
            best_reward_so_far = mean_ep_reward
            model_name = f"{save_name}_best.pt"
            torch.save(policy, os.path.join(args.dir, model_name))

        if len(ep_rewards) > 1:
            elapsed_time = time.time() - start_time
            fps = int(frame_count / elapsed_time)
            print(
                f"Steps: {frame_count:d} | FPS: {fps:d} |",
                f"Mean: {mean_ep_reward:.1f} | Max: {max(ep_rewards):.1f} |",
                f"Cur: {current_curriculum:2d} | CurM: {mean_curriculum_metric:.1f}",
                flush=True,
            )
            logger.log_epoch(
                {
                    "iter": iteration + 1,
                    "total_num_steps": frame_count,
                    "fps": fps,
                    "entropy": dist_entropy,
                    "curriculum": current_curriculum,
                    "curriculum_metric": mean_curriculum_metric,
                    "stats": {"rew": ep_rewards},
                }
            )

    envs.close()


def play(args):
    policy = torch.load(args.net, map_location="cpu")
    controller = policy.actor

    render = args.render == 1
    env = make_env(args.env, render=render)
    env.unwrapped.curriculum = args.use_curriculum

    obs = env.reset()
    env.camera.lookat(env.robot.body_xyz)

    # Set global no_grad
    torch.set_grad_enabled(False)
    policy.train(mode=False)

    ep_reward = 0

    env.camera.tracking = True
    MAX_STEPS = float("inf")

    if args.save:
        import tempfile
        from imageio import imwrite

        MAX_STEPS = 1000
        tmpdir = tempfile.mkdtemp()
        print(tmpdir)

    dump_pos = []
    dump_quat = []
    dump_grab = []
    dump_current_handhold_index = []

    all_links = [4, 5, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    pos = np.zeros((len(all_links), 3), dtype="f4")
    quat = np.zeros((len(all_links), 4), dtype="f4")
    from scipy.spatial.transform import Rotation as R

    while env.timestep < MAX_STEPS:
        if not render or not env.camera.env_should_wait:
            obs = torch.from_numpy(obs).float().unsqueeze(0)

            action = controller(obs)
            cpu_actions = action.squeeze().cpu().numpy()

            if args.save:
                path = os.path.join(tmpdir, f"{env.timestep}.png")
                imwrite(path, env.camera.dump_rgb_array())

            obs, reward, done, _ = env.step(cpu_actions)
            ep_reward += reward

            ref_in_swing = (
                hasattr(env.unwrapped, "ref_swing")
                and env.unwrapped.ref_swing[env.timestep]
            )
            colour = (1, 0, 0) if ref_in_swing else (0, 0, 0)
            env.unwrapped._p.addUserDebugLine(
                env.ref_xyz[env.timestep - 1],
                env.ref_xyz[env.timestep],
                lifeTime=0,
                lineWidth=0.5,
                lineColorRGB=colour,
            )

            env.unwrapped._p.getLinkStates2(
                env.robot.id,
                all_links,
                outPositions=pos,
                outOrientations=quat,
                computeLinkVelocity=0,
            )

            dump_pos.append(pos.copy())
            dump_quat.append(quat.copy())
            dump_grab.append(list((env.grab_constraint_ids >= 0).astype(float)))
            dump_current_handhold_index.append(int(env.next_step_index))

            if done:
                print("--- Episode reward:", ep_reward)

                import json

                all_pos = np.array(dump_pos)
                all_quat = np.array(dump_quat)
                all_pitch = (
                    R.from_quat(all_quat.reshape(-1, 4))
                    .as_euler("yxz")
                    .astype("f4")[:, 0]
                )
                with open(f"{now_str}_{env.current_traj_id}.json", "w") as file:
                    json.dump(
                        {
                            "xys": all_pos[:, :, [0, 2]].tolist(),
                            "rots": all_pitch.reshape(-1, len(all_links)).tolist(),
                            "handholds": env.handholds.tolist(),
                            "reference": env.ref_xyz.tolist(),
                            "reference_is_grabbing": env.ref_swing.tolist(),
                            "is_grabbing": dump_grab,
                            "handhold_idx": dump_current_handhold_index,
                        },
                        file,
                    )

                dump_pos = []
                dump_quat = []
                dump_grab = []
                dump_current_handhold_index = []

                if args.save:
                    break

                ep_reward = 0
                obs = env.reset()
                env.unwrapped._p.removeAllUserDebugItems()

        if render:
            env.camera.wait()
            camera_xyz = (
                *env.robot.body_xyz[0:2],
                env.handholds[env.next_step_index][2],
            )
            env.camera.track(camera_xyz)
            env.unwrapped._handle_keyboard()

    if args.save:
        import subprocess

        subprocess.call(
            [
                "ffmpeg",
                "-f",
                "image2",
                "-r",
                "60",
                "-i",
                f"{tmpdir}/%d.png",
                "-vcodec",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "test.mp4",
            ]
        )


def plan(args):
    class Terrain:
        def __init__(self, env, device=device):
            h = 60
            self.w = torch.zeros(h).to(device)
            self.b = torch.zeros(h).to(device)

            self.bc = env.unwrapped._p
            self.render = env.is_rendered
            self.z0 = env.robot.feet_xyz[1, 2]

            self.segments = []

        def reset(self):
            self.w.uniform_(0, 1).round_().add_(-0.5).mul_(2)
            cumw = 0
            b = 15 * DEG2RAD
            for i in range(len(self.w)):
                a = self.w[i].uniform_(-3 * DEG2RAD, 3 * DEG2RAD).mul_(5)
                if (cumw + a).abs() > b:
                    a = self.w[i].uniform_(-b, b)
                self.w[i] = a
                cumw += a

            self.b.uniform_(0.5, 1.5).cumsum_(dim=0)

        def sample(self, x):
            w = self.w[None, None]
            b = self.b[None, None]
            v = torch.tensor([0.0], device=b.device)
            y = torch.relu(x.unsqueeze(-1) - b).mul(w).sum(-1)
            y = y * 0
            y[(x > self.b[2]) * (x < self.b[3])] += 1
            y[(x > self.b[6]) * (x < self.b[7])] += 1
            y[(x > self.b[8]) * (x < self.b[9])] += 1
            return y + self.z0

        def get_terrain_data(self):
            a = [self.b[0] - 10]
            for i in self.b:
                a.append(i - 0.01)
                a.append(i + 0.01)
            a.append(self.b[-1] + 10)

            x = torch.tensor(a, device=self.b.device)[None]
            y = self.sample(x)
            return torch.cat((x, y), dim=0).T

        def plot(self):
            if not self.render:
                return

            data = self.get_terrain_data()
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
            for p0, p1 in zip(data[:-1], data[1:]):
                length = (p0 - p1).norm()
                ns = VCylinder(self.bc, 0.04, length)

                x, z = (p0 + p1).mul(0.5)
                angle = torch.atan2(*(p0 - p1))
                quat = self.bc.getQuaternionFromEuler((0, angle, 0))
                ns.set_position((x, 0, z), quat)
                self.segments.append(ns)
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

    policy = torch.load(args.net, map_location="cpu")
    controller = policy.actor
    simple_policy_path = "data/best_simple.pt"
    simple_policy = torch.load(simple_policy_path, map_location=device)

    policy = policy.to(device)
    simple_policy = simple_policy.to(device)

    render = args.render == 1
    M = 100000
    full_env = make_env(args.env, render=render)
    full_env.unwrapped.curriculum = args.use_curriculum
    simple_env = make_env(
        "env:Gibbon2DPointMassEnv-v0", num_parallel=M, device=device, render=False
    )

    dump_pos = []
    dump_quat = []

    all_links = [4, 5, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    pos = np.zeros((len(all_links), 3), dtype="f4")
    quat = np.zeros((len(all_links), 4), dtype="f4")
    from scipy.spatial.transform import Rotation

    def plan_next_step(terrain, H=10):
        print(f"Planning next step {full_env.next_step_index}")
        current_pos_x, _, current_pos_z = full_env.robot.body_xyz
        current_pos = np.fromiter([current_pos_x, current_pos_z], dtype="f4")
        factor = 0 if full_env.timestep == 1 else 1
        vx, _, vz = full_env.robot.body_world_vel
        current_vel = np.fromiter([vx, vz], dtype="f4") * factor

        dx = torch.ones(M, H, 1).uniform_(1, 2)
        x_coord = (
            dx.cumsum(dim=1)
            .add_(full_env.handholds[full_env.next_step_index - 1, 0])
            .to(device)
        )
        y_coord = terrain.sample(x_coord)
        hh = torch.cat([x_coord, y_coord], dim=-1)

        curr_full_hh_idx = full_env.next_step_index
        simple_env.reset()
        simple_env.body_positions[:] = torch.from_numpy(current_pos)
        simple_env.body_velocities[:] = torch.from_numpy(current_vel)
        simple_env.handholds[:, 0] = torch.from_numpy(
            full_env.handholds[curr_full_hh_idx - 1, [0, 2]]
        )
        simple_env.handholds[:, 1 : H + 1] = hh

        tot_r = torch.zeros(M, 1).to(device)
        completed_envs = torch.zeros(M, 1).bool().to(device)
        s = torch.cat(simple_env.get_observation_components(), dim=-1)

        trajectories = torch.zeros(M, simple_env.max_timesteps, 2).to(device)
        # compute reward for hh sequence based on simplified model
        for ts, _ in enumerate(range(240)):
            a = simple_policy.actor(s)
            s, r, d, i = simple_env.step(a)
            tot_r = tot_r + (~completed_envs) * i["just_grabbed"]
            completed_envs += d
            trajectories[:, ts] = simple_env.body_positions.clone()

        values = 0
        # compute v based on full model and first 2 hh
        all_y = (
            -0.16
            * 2
            * (torch.arange(full_env.num_steps + full_env.lookahead + H) % 2 - 0.5)
        ).to(device)
        window = slice(full_env.next_step_index, full_env.next_step_index + H)
        y = all_y[window].repeat((M, 1)).unsqueeze(-1)
        xyz_targets = torch.cat((x_coord, y, y_coord), dim=-1)
        R = full_env.robot.observation_space.shape[0] + 3  # +3 includes current step
        robot_states = torch.from_numpy(obs[:R]).repeat((M, 1)).to(device)
        robot_body_xyz = torch.from_numpy(full_env.robot.body_xyz).to(device)
        target_delta = (xyz_targets - robot_body_xyz)[:, :2]
        window = slice(1, 31, 5)
        trajectories = torch.nn.functional.pad(
            trajectories, (0, 1, 0, 0), mode="constant", value=0
        )
        trajectories = torch.index_select(
            trajectories, 2, torch.tensor([0, 2, 1]).to(device)
        )
        ref_delta = trajectories[:, window] - robot_body_xyz
        ref_delta = ref_delta.to(device)

        pitch = full_env.robot.body_rpy[1]
        cos_ = math.cos(-pitch)
        sin_ = math.sin(-pitch)

        target_delta[:, :, 0] = (
            target_delta[:, :, 0] * cos_ - target_delta[:, :, 2] * sin_
        )
        target_delta[:, :, 2] = (
            target_delta[:, :, 0] * sin_ + target_delta[:, :, 2] * cos_
        )

        ref_delta[:, :, 0] = ref_delta[:, :, 0] * cos_ - ref_delta[:, :, 2] * sin_
        ref_delta[:, :, 2] = ref_delta[:, :, 0] * sin_ + ref_delta[:, :, 2] * cos_

        states = torch.cat(
            (robot_states, target_delta.flatten(1, 2), ref_delta.flatten(1, 2)), dim=-1
        )
        values = policy.critic(states.float())

        best_score, idx = (tot_r + 0.01 * values).max(dim=0)
        best = hh[idx].squeeze(0).cpu().numpy()
        window = slice(curr_full_hh_idx, curr_full_hh_idx + H)
        remain = full_env.handholds[window, [0, 2]].shape[0]
        full_env.handholds[window, [0, 2]] = best[:remain]

        window = slice(full_env.ref_timestep, full_env.ref_timestep + 240)
        remain = full_env.ref_xyz[window].shape[0]
        full_env.ref_xyz[window] = trajectories[idx, :remain].cpu().numpy()

        if full_env.is_rendered:
            pc = full_env.unwrapped._p
            pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 0)
            for h, pos in zip(full_env.handhold_markers, full_env.handholds):
                h.set_position((pos[0], -0.1, pos[2]))
            pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 1)

    obs = full_env.reset()
    full_env.camera.lookat(full_env.robot.body_xyz)

    terrain = Terrain(full_env, device=device)

    # Set global no_grad
    torch.set_grad_enabled(False)
    policy.train(mode=False)
    simple_policy.train(mode=False)

    terrain.reset()
    terrain.plot()
    plan_next_step(terrain=terrain)

    dump_grab = []
    dump_current_handhold_index = []

    ep_reward = 0
    com_xyzs = [full_env.robot.body_xyz]
    while True:
        if not render or not full_env.camera.env_should_wait:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            action = controller(obs)
            cpu_actions = action.squeeze().cpu().numpy()

            obs, reward, done, info = full_env.step(cpu_actions)

            full_env.unwrapped._p.getLinkStates2(
                full_env.robot.id,
                all_links,
                outPositions=pos,
                outOrientations=quat,
                computeLinkVelocity=0,
            )

            dump_pos.append(pos.copy())
            dump_quat.append(quat.copy())
            dump_grab.append(list((full_env.grab_constraint_ids >= 0).astype(float)))
            dump_current_handhold_index.append(int(full_env.next_step_index))

            ep_reward += reward

            if (key := "just_grabbed") in info:
                if info[key]:
                    print("just grabbed, planning")
                    plan_next_step(terrain=terrain)

            com_xyzs.append(full_env.robot.body_xyz)
            in_swing = full_env.robot.feet_contact.any()
            colour = (1, 0, 0) if in_swing else (0, 0, 1)
            full_env.unwrapped._p.addUserDebugLine(
                com_xyzs[-2],
                com_xyzs[-1],
                lifeTime=0,
                lineWidth=0.5,
                lineColorRGB=colour,
            )

            if done:

                import json

                all_pos = np.array(dump_pos)
                all_quat = np.array(dump_quat)
                all_pitch = (
                    Rotation.from_quat(all_quat.reshape(-1, 4))
                    .as_euler("yxz")
                    .astype("f4")[:, 0]
                )
                with open(f"{now_str}_{full_env.current_traj_id}.json", "w") as file:
                    json.dump(
                        {
                            "xys": all_pos[:, :, [0, 2]].tolist(),
                            "rots": all_pitch.reshape(-1, len(all_links)).tolist(),
                            "handholds": full_env.handholds.tolist(),
                            "terrain": terrain.get_terrain_data()
                            .cpu()
                            .numpy()
                            .tolist(),
                            "reference": full_env.ref_xyz.tolist(),
                            "is_grabbing": dump_grab,
                            "handhold_idx": dump_current_handhold_index,
                        },
                        file,
                    )

                print("--- Episode reward:", ep_reward)
                for s in terrain.segments:
                    terrain.bc.removeBody(s.id)
                terrain.segments = []
                obs = full_env.reset()
                full_env.camera.lookat(full_env.robot.body_xyz)
                full_env.unwrapped._p.removeAllUserDebugItems()

                dump_pos = []
                dump_quat = []
                dump_grab = []
                dump_current_handhold_index = []

                terrain.reset()
                terrain.plot()
                print("DONE! Resetting terrain and planning again")
                plan_next_step(terrain=terrain)

                ep_reward = 0

        if render:
            full_env.camera.wait()
            camera_xyz = (
                *full_env.robot.body_xyz[0:2],
                full_env.handholds[full_env.next_step_index][2],
            )
            full_env.camera.track(camera_xyz)
            full_env.unwrapped._handle_keyboard()


def test(args):
    render = args.render == 1
    env = make_env(args.env, render=render)

    bc = env.unwrapped._p
    robot_id = env.unwrapped.robot.id
    num_joints = bc.getNumJoints(robot_id)

    # Links
    max_z = float("-inf")
    min_z = float("inf")
    for i in range(num_joints):
        link_state = bc.getLinkState(robot_id, i)
        _, _, z = link_state[4]
        max_z = z if z > max_z else max_z
        min_z = z if z < min_z else min_z
    print(f"\nHeight: {(max_z - min_z):.2f} meters")

    # Dynamics
    print("\nWeights:")
    total_mass = 0
    for pid in range(num_joints):
        dynamics_info = bc.getDynamicsInfo(robot_id, pid)
        joint_info = bc.getJointInfo(robot_id, pid)
        mass = dynamics_info[0]
        total_mass += mass
        if mass != 0:
            print(f"{joint_info[12].decode('utf-8'):25} {mass:.4f}")
    print("-" * 32)
    print(f"Total Mass: {total_mass:.4f} kg\n")

    env.reset()
    while True:
        if not render or not env.camera.env_should_wait:
            action = env.action_space.sample() * 0
            obs, rew, done, info = env.step(action)

            ref_in_swing = (
                hasattr(env.unwrapped, "ref_swing")
                and env.unwrapped.ref_swing[env.timestep]
            )
            colour = (1, 0, 0) if ref_in_swing else (0, 0, 0)
            env.unwrapped._p.addUserDebugLine(
                env.ref_xyz[env.timestep - 1],
                env.ref_xyz[env.timestep],
                lifeTime=0,
                lineWidth=0.5,
                lineColorRGB=colour,
            )

            if done:
                env.reset()
                env.unwrapped._p.removeAllUserDebugItems()

        if render:
            env.camera.wait()
            env.camera.track(env.ref_xyz[env.timestep])
            env.unwrapped._handle_keyboard()


if __name__ == "__main__":
    args = arg_parser().parse_args()
    (globals().get(args.mode))(args)
