import argparse
import datetime
import os
import time
from collections import deque
import pickle

import torch

from env.utils import make_env
from com.logger import CSVLogger

now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["train", "play", "test", "dump"]
    )
    parser.add_argument("--env", type=str, default="env:Gibbon2DPointMassEnv-v0")
    parser.add_argument("--dir", type=str, default=os.path.join("exp_s", now_str))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--frames", type=int, default=2.5e7)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--use_curriculum", type=int, default=1)
    parser.add_argument("--min_grab_duration", type=int, default=15)
    parser.add_argument("--max_grab_duration", type=int, default=240)
    parser.add_argument("--early_termination", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=2)
    return parser


def train(args):
    from algo.ppo import (
        PPO,
        PPOReplayBuffer,
        SoftsignActorSM,
        PolicySM,
    )

    num_frames = args.frames
    episode_steps = 80_000
    num_processes = 8_000 if os.name != "nt" else torch.multiprocessing.cpu_count()

    if args.tag is not None:
        args.dir = f"{args.dir}_{args.tag}"

    print(f"Launching new run at {args.dir}")

    num_steps = episode_steps // num_processes
    mini_batch_size = 2_000
    num_mini_batch = episode_steps // mini_batch_size
    save_every = int(num_frames // 5)
    save_name = "-".join(args.env.split(":"))

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
    env_options = {
        "num_parallel": num_processes,
        "device": device,
        "lookahead": args.lookahead,
        "render": args.render,
        "curriculum": args.use_curriculum,
        "min_grab_duration": args.min_grab_duration,
        "max_grab_duration": args.max_grab_duration,
    }
    envs = make_env(args.env, **env_options)
    envs.seed(args.seed)

    obs_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]

    if args.net is None:
        policy = PolicySM(SoftsignActorSM(envs)).to(device)
    else:
        policy = torch.load(args.net, map_location=device)
        ppo_params["lr"] = 3e-5
        final_lr = 1e-5

    agent = PPO(policy, **ppo_params)
    rollouts = PPOReplayBuffer(
        num_steps,
        num_processes,
        obs_dim,
        action_dim,
        device=device,
    )

    ep_rewards = deque(maxlen=100)
    curriculum_metrics = deque(maxlen=100)
    num_updates = int(num_frames) // num_steps // num_processes

    # don't divide by 0
    ep_rewards.append(0)
    curriculum_metrics.append(0)

    # This has to be done before reset
    if args.use_curriculum:
        current_curriculum = envs.curriculum
        max_curriculum = envs.max_curriculum
        advance_threshold = envs.advance_threshold
        envs.curriculum = current_curriculum
    else:
        current_curriculum = -1
        max_curriculum = -1
        advance_threshold = -1

    obs = envs.reset()
    rollouts.observations[0].copy_(obs)

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    logger = CSVLogger(log_dir=args.dir)
    save_checkpoint = save_every
    best_reward_so_far = float("-inf")
    start_time = time.time()

    for iteration in range(num_updates):

        scheduled_lr = max(ppo_params["lr"] * (0.99 ** iteration), final_lr)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = scheduled_lr

        # Disable gradient for data collection
        torch.set_grad_enabled(False)
        policy.train(mode=False)

        for (
            cur_obs,
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
            value, action, action_log_prob = policy.act(cur_obs)
            obs, reward, done, info = envs.step(action)

            mask = (~done).float()
            bad_mask = info["bad_mask"]

            if (key := "episode_rewards") in info:
                ep_rewards.extend(info[key])

            if (key := "curriculum_metric") in info:
                curriculum_metrics.append(info[key])

            obs_buf.copy_(obs)
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
            envs.curriculum = current_curriculum
            curriculum_metrics.clear()
            curriculum_metrics.append(0)  # append 0 to make sure we don't divide by 0
            ep_rewards.clear()
            ep_rewards.append(0)
            obs = envs.reset()
            rollouts.observations[0].copy_(obs)

        # Enable gradients for training
        torch.set_grad_enabled(True)
        policy.train(mode=True)

        rollouts.compute_returns(next_value, gamma=0.995)
        _, _, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        save_path = os.path.join(args.dir, f"{save_name}_latest")
        torch.save(policy, f"{save_path}.pt")

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


def play(args):
    render = args.render == 1
    device = "cpu"  # cpu should be faster

    env_options = {
        "num_parallel": 1,
        "device": device,
        "render": render,
        "lookahead": args.lookahead,
        "curriculum": args.use_curriculum,
        "min_grab_duration": args.min_grab_duration,
        "max_grab_duration": args.max_grab_duration,
    }
    env = make_env(args.env, **env_options)

    policy = torch.load(args.net, map_location=device)
    controller = policy.actor

    torch.set_grad_enabled(False)
    policy.train(mode=False)
    avg_speed = []

    obs = env.reset()
    while True:
        if not render or not env.camera.env_should_wait:
            action = controller(obs)
            obs, rew, done, info = env.step(action)
            avg_speed.append(env.body_velocities.norm(dim=-1))

            if (key := "curriculum_metric") in info:
                steps_completed = info[key]
                print(f"--- Episode steps: {steps_completed:.2f}")
                print(f"Avg speed: {torch.cat(avg_speed).mean():.2f}")

            if done[0] and render:
                env._p.removeAllUserDebugItems()

        if render:
            env.camera.wait()
            x, z = env.body_positions[0]
            env.camera.track((x, 0, z))
            env.unwrapped._handle_keyboard()


def dump(args):
    render = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    episodes_to_collect = 2000
    N = 2000
    max_episode_length = 1000

    env_options = {
        "num_parallel": N,
        "device": device,
        "render": 0,
        "lookahead": args.lookahead,
        "min_grab_duration": args.min_grab_duration,
        "max_grab_duration": args.max_grab_duration,
    }
    env = make_env(args.env, **env_options)

    policy = torch.load(args.net, map_location=device)
    controller = policy.actor

    torch.set_grad_enabled(False)
    policy.train(mode=False)

    for current_curriculum in range(10):
        all_trajs = []
        env.curriculum = current_curriculum
        collected = 0
        while collected < episodes_to_collect:
            obs = env.reset()
            com_xyzs = torch.zeros(N, max_episode_length, 2).to(device)
            grab_flags = torch.zeros(N, max_episode_length, 1).to(device)
            dones = torch.zeros(N, max_episode_length, 1).to(device)
            hh = env.handholds.clone()

            com_xyzs[:, 0, :] = env.body_positions.clone()
            grab_flags[:, 0, :] = env.grab_flags.clone()
            dones[:, 0, :] = env.done_flags.clone()

            for i in range(max_episode_length):
                action = controller(obs)
                obs, rew, done, info = env.step(action)
                com_xyzs[:, i, :] = env.body_positions.clone()
                grab_flags[:, i, :] = env.grab_flags.clone()
                dones[:, i, :] = done.clone()

            episode_lengths = dones.argmax(dim=1).squeeze().cpu().numpy()
            com_xyzs = com_xyzs.cpu().numpy()
            grab_flags = grab_flags.cpu().numpy()
            hh = hh.cpu().numpy()
            for a, b, c, d in zip(episode_lengths, com_xyzs, grab_flags, hh):
                if a > 700:
                    all_trajs.append((b[:a], c[:a], d))

            collected = len(all_trajs)
            print(
                f"Curr {current_curriculum} "
                f"min: {episode_lengths.min()} max {episode_lengths.max()} "
                f"Collected {collected}"
            )

        path = f"data/trajectories/simple_trajs_{current_curriculum}.pickle"
        with open(path, "wb") as pickle_file:
            pickle.dump(all_trajs, pickle_file)


def test(args):
    render = args.render == 1
    device = "cpu"  # cpu should be faster

    N = 1
    env_options = {
        "num_parallel": N,
        "device": device,
        "render": render,
        "lookahead": args.lookahead,
        "curriculum": args.use_curriculum,
        "min_grab_duration": args.min_grab_duration,
        "max_grab_duration": args.max_grab_duration,
    }
    env = make_env(args.env, **env_options)

    A = env.action_space.shape[0]
    action = torch.zeros((N, A), device=device)

    env.reset()
    while True:
        if not render or not env.camera.env_should_wait:
            action.uniform_(-1, 1)
            action[:, 0] = 1  # always grab
            obs, rew, done, info = env.step(action)

            if done:
                env.reset()

        if render:
            env.camera.wait()
            x, z = env.body_positions[0]
            env.camera.track((x, 0, z))
            env.unwrapped._handle_keyboard()


if __name__ == "__main__":
    args = arg_parser().parse_args()
    (globals().get(args.mode))(args)
