"""
Evaluate Pi0.5 policy on ManiSkill environments.

This script evaluates a trained Pi0.5 model from HuggingFace on ManiSkill tasks
and computes success metrics.

Usage:
    python evaluate_pi05.py \
        --policy-path pythonsong/pi05_franka_pickcube \
        --env-id PickCube-v1 \
        --num-episodes 100 \
        --num-envs 10 \
        --device cuda
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode
from maniskill_env_wrapper import ManiSkillVectorEnvWrapper

# LeRobot imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import preprocess_observation, add_envs_task


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 policy on ManiSkill")

    # Model arguments
    parser.add_argument(
        "--policy-path",
        type=str,
        default="pythonsong/pi05_franka_pickcube",
        help="Path to pretrained Pi0.5 model on HuggingFace or local path"
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="pythonsong/franka_maniskill_pickcube_200",
        help="Dataset repo for loading stats (needed for normalization)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (cuda/cpu)"
    )

    # Environment arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickCube-v1",
        help="ManiSkill environment ID"
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb",
        help="Observation mode (rgb/rgbd/state)"
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="auto",
        help="Simulation backend (auto/cpu/gpu)"
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="Pick up the cube and place it at the goal position",
        help="Task description for VLA model"
    )

    # Evaluation arguments
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (None = use env default)"
    )

    # Output arguments
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Whether to save evaluation videos"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )

    return parser.parse_args()


def make_eval_env(args):
    """Create evaluation environment with LeRobot wrapper."""

    # Create base ManiSkill environment
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "num_envs": args.num_envs,
    }

    # Add render_mode only when video recording is requested
    if args.save_video:
        env_kwargs["render_mode"] = "rgb_array"

    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **env_kwargs)

    # Add video recording wrapper if requested (before LeRobot wrapper)
    if args.save_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            save_video=True,
            info_on_video=True,
            source_type="pi05",
            source_desc="Pi0.5 evaluation rollout",
            max_steps_per_video=args.max_episode_steps if args.max_episode_steps else 500,
        )
        print(f"✓ Video recording enabled, saving to: {video_dir}")

    # Wrap with LeRobot wrapper to add task description
    env = ManiSkillVectorEnvWrapper(env, task_description=args.task_description)

    return env


def evaluate_policy(policy, preprocessor, postprocessor, env, num_episodes, device):
    """
    Evaluate policy on environment.

    Args:
        policy: Pi0.5 policy
        preprocessor: LeRobot preprocessor
        postprocessor: LeRobot postprocessor
        env: ManiSkill environment
        num_episodes: Number of episodes to evaluate
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    policy.eval()

    metrics = defaultdict(list)
    num_envs = env.num_envs
    episodes_completed = 0

    obs, info = env.reset()

    print(f"\nEvaluating {num_episodes} episodes across {num_envs} parallel environments...")

    with torch.no_grad():
        while episodes_completed < num_episodes:
            # Convert ManiSkill obs to LeRobot format (already done by wrapper)
            # obs is now: {'pixels': (num_envs, H, W, 3), 'agent_pos': (num_envs, 8)}

            # Preprocess observation (converts to tensors, normalizes, etc.)
            obs_processed = preprocess_observation(obs)

            # Add task description
            obs_processed = add_envs_task(env, obs_processed)

            # Apply policy preprocessor (adds batch dim, tokenizes, etc.)
            obs_batch = preprocessor(obs_processed)

            # Get action from policy
            action = policy.select_action(obs_batch)

            # Postprocess action (unnormalize, etc.)
            action_processed = postprocessor(action)

            # Convert to numpy for environment
            action_numpy = action_processed.cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_numpy)

            # Collect metrics when episodes finish
            if terminated.any() or truncated.any():
                done = terminated | truncated

                # Check for final_info (contains success metrics)
                if "final_info" in info:
                    final_info = info["final_info"]

                    # Count episodes that just finished
                    for i, is_done in enumerate(done):
                        if is_done and episodes_completed < num_episodes:
                            episodes_completed += 1

                            # Extract success metric
                            if "is_success" in final_info:
                                success = final_info["is_success"][i]
                                metrics["success"].append(success)

                            # Extract episode return
                            if "episode" in final_info:
                                ep_info = final_info["episode"]
                                if "r" in ep_info:
                                    metrics["return"].append(ep_info["r"][i])
                                if "l" in ep_info:
                                    metrics["length"].append(ep_info["l"][i])

                    print(f"Episodes completed: {episodes_completed}/{num_episodes}")

    return metrics


def main():
    args = parse_args()

    print("="*80)
    print("Pi0.5 ManiSkill Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Policy: {args.policy_path}")
    print(f"  Environment: {args.env_id}")
    print(f"  Task: {args.task_description}")
    print(f"  Device: {args.device}")
    print(f"  Num episodes: {args.num_episodes}")
    print(f"  Num envs: {args.num_envs}")
    print()

    # 1. Load policy
    print("Loading policy...")
    try:
        policy = PI05Policy.from_pretrained(args.policy_path, device=args.device)
        print(f"✓ Policy loaded: {args.policy_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load policy from {args.policy_path}")
        print(f"Error: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Try again after network is restored")
        print("3. Download the model manually and use local path with --policy-path")
        return

    # 2. Load dataset for stats (needed for normalization)
    print("\nLoading dataset stats...")
    dataset = LeRobotDataset(args.dataset_repo, download_videos=False)
    print(f"✓ Dataset stats loaded from: {args.dataset_repo}")

    # 3. Create preprocessor and postprocessor
    print("\nCreating preprocessors...")
    try:
        # Try to load from HuggingFace
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=args.policy_path,
            dataset_stats=dataset.meta.stats
        )
        print("✓ Preprocessors loaded from HuggingFace")
    except (FileNotFoundError, ConnectionError) as e:
        # If network fails or files not found, create from scratch using dataset stats
        print(f"⚠ Could not load preprocessors from HuggingFace: {e}")
        print("Creating preprocessors from scratch using dataset stats...")
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=None,  # Don't try to download
            dataset_stats=dataset.meta.stats
        )
        print("✓ Preprocessors created from dataset stats")

    # 4. Create evaluation environment
    print(f"\nCreating environment: {args.env_id}")
    env = make_eval_env(args)
    print(f"✓ Environment created with {args.num_envs} parallel instances")

    # 5. Evaluate policy
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)

    metrics = evaluate_policy(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env=env,
        num_episodes=args.num_episodes,
        device=args.device
    )

    # 6. Compute and print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    if "success" in metrics:
        success_rate = np.mean(metrics["success"])
        print(f"\n✓ Success Rate: {success_rate*100:.2f}% ({np.sum(metrics['success'])}/{len(metrics['success'])})")

    if "return" in metrics:
        avg_return = np.mean(metrics["return"])
        std_return = np.std(metrics["return"])
        print(f"✓ Average Return: {avg_return:.2f} ± {std_return:.2f}")

    if "length" in metrics:
        avg_length = np.mean(metrics["length"])
        print(f"✓ Average Episode Length: {avg_length:.1f} steps")

    # 7. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("Pi0.5 ManiSkill Evaluation Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Policy: {args.policy_path}\n")
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Task: {args.task_description}\n")
        f.write(f"Num episodes: {args.num_episodes}\n\n")

        if "success" in metrics:
            f.write(f"Success Rate: {success_rate*100:.2f}%\n")
        if "return" in metrics:
            f.write(f"Average Return: {avg_return:.2f} ± {std_return:.2f}\n")
        if "length" in metrics:
            f.write(f"Average Episode Length: {avg_length:.1f}\n")

    print(f"\n✓ Results saved to: {results_file}")

    # Cleanup
    env.close()
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
