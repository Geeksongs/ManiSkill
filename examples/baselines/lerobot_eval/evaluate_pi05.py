"""
Evaluate Pi0.5 policy on ManiSkill environments.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs
from maniskill_env_wrapper import ManiSkillVectorEnvWrapper

# LeRobot imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import preprocess_observation, add_envs_task


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 policy on ManiSkill")
    parser.add_argument("--policy-path", type=str, default="pythonsong/pi05_franka_pickcube")
    parser.add_argument("--dataset-repo", type=str, default="pythonsong/franka_maniskill_pickcube_200")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--task-description", type=str, default="Pick up the cube and place it at the goal position")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    return parser.parse_args()


def make_eval_env(args):
    """Create evaluation environment."""
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "num_envs": args.num_envs,
    }
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **env_kwargs)
    env = ManiSkillVectorEnvWrapper(env, task_description=args.task_description)
    return env


def evaluate_policy(policy, preprocessor, postprocessor, env, num_episodes, device):
    """Evaluate policy on environment."""
    policy.eval()
    metrics = defaultdict(list)
    num_envs = env.num_envs
    episodes_completed = 0
    step_count = 0

    obs, info = env.reset()
    print(f"\nEvaluating {num_episodes} episodes across {num_envs} parallel environments...")

    with torch.no_grad():
        while episodes_completed < num_episodes:
            # Preprocess observation
            obs_processed = preprocess_observation(obs)
            obs_processed = add_envs_task(env, obs_processed)
            obs_batch = preprocessor(obs_processed)

            # Get action from policy
            action = policy.select_action(obs_batch)
            action_processed = postprocessor(action)
            action_numpy = action_processed.cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_numpy)
            step_count += 1

            # Collect metrics when episodes finish
            if terminated.any() or truncated.any():
                done = terminated | truncated
                for i, is_done in enumerate(done):
                    if is_done and episodes_completed < num_episodes:
                        episodes_completed += 1

                        # Get success from info
                        if "success" in info:
                            success = bool(info["success"][i]) if hasattr(info["success"], '__getitem__') else bool(info["success"])
                            metrics["success"].append(success)

                        metrics["length"].append(step_count)
                        print(f"Episode {episodes_completed}/{num_episodes} done at step {step_count}, success={metrics['success'][-1] if 'success' in metrics else 'N/A'}")
                        step_count = 0  # Reset for next episode

    return metrics


def main():
    args = parse_args()

    print("="*60)
    print("Pi0.5 ManiSkill Evaluation")
    print("="*60)
    print(f"Policy: {args.policy_path}")
    print(f"Environment: {args.env_id}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Num episodes: {args.num_episodes}")

    # Load policy
    print("\nLoading policy...")
    policy = PI05Policy.from_pretrained(args.policy_path, device=args.device)
    print("Policy loaded.")

    # Load dataset stats
    print("Loading dataset stats...")
    dataset = LeRobotDataset(args.dataset_repo, download_videos=False)

    # Create preprocessor/postprocessor
    print("Creating preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats
    )

    # Create environment
    print(f"Creating environment...")
    env = make_eval_env(args)
    print(f"Environment created with {args.num_envs} parallel instances")

    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    metrics = evaluate_policy(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env=env,
        num_episodes=args.num_episodes,
        device=args.device
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if "success" in metrics and len(metrics["success"]) > 0:
        success_rate = np.mean(metrics["success"]) * 100
        print(f"Success Rate: {success_rate:.2f}% ({sum(metrics['success'])}/{len(metrics['success'])})")

    if "length" in metrics and len(metrics["length"]) > 0:
        avg_length = np.mean(metrics["length"])
        print(f"Average Episode Length: {avg_length:.1f} steps")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
