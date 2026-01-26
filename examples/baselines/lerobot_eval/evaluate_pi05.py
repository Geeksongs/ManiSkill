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
import os

import mani_skill.envs
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

    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **env_kwargs)

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
    step_count = 0  # Global step counter for overall progress
    MAX_STEPS_PER_EPISODE = 500  # Safety limit: maximum steps per episode

    # Per-environment step tracking for accurate timeout detection
    episode_steps = [0] * num_envs
    episode_completing = [False] * num_envs  # Track which environments are finishing an episode
    last_progress_percent = -10  # Track last progress percentage displayed

    obs, info = env.reset()

    print(f"\nEvaluating {num_episodes} episodes across {num_envs} parallel environments...")
    print(f"Safety: Maximum {MAX_STEPS_PER_EPISODE} steps per episode allowed\n")

    with torch.no_grad():
        while episodes_completed < num_episodes:
            # Progress display logic
            total_episodes = num_episodes
            progress_percent = int((episodes_completed / total_episodes) * 100)

            # Show detailed debug info for first 20 steps, then every 10 steps or at progress milestones
            show_detail = step_count < 20 or step_count % 10 == 0 or progress_percent != last_progress_percent
            if show_detail:
                print(f"  [Step {step_count}] Progress: {episodes_completed}/{total_episodes} episodes ({progress_percent}%)")
                if step_count < 20 or progress_percent != last_progress_percent:
                    print(f"    Episode steps per env: {episode_steps}")
                last_progress_percent = progress_percent

            # Check for individual environment timeouts
            for i in range(num_envs):
                if episode_steps[i] >= MAX_STEPS_PER_EPISODE and not episode_completing[i]:
                    print(f"    ⚠️  Env {i} reached max steps ({MAX_STEPS_PER_EPISODE}) - forcing episode end")
                    episode_completing[i] = True

            # Convert ManiSkill obs to LeRobot format (already done by wrapper)
            # obs is now: {'pixels': (num_envs, H, W, 3), 'agent_pos': (num_envs, 8)}

            # Preprocess observation (converts to tensors, normalizes, etc.)
            if step_count < 5:
                print(f"    → preprocess_observation()...")
            obs_processed = preprocess_observation(obs)
            if step_count < 5:
                print(f"    ← preprocess_observation() returned")

            # Add task description
            if step_count < 5:
                print(f"    → add_envs_task()...")
            obs_processed = add_envs_task(env, obs_processed)
            if step_count < 5:
                print(f"    ← add_envs_task() returned")

            # Apply policy preprocessor (adds batch dim, tokenizes, etc.)
            if step_count < 5:
                print(f"    → preprocessor()...")
            obs_batch = preprocessor(obs_processed)
            if step_count < 5:
                print(f"    ← preprocessor() returned")

            # Get action from policy (this is where it might hang)
            if step_count < 20:
                print(f"    → Calling policy.select_action()...")
            action = policy.select_action(obs_batch)
            if step_count < 20:
                print(f"    ← policy.select_action() returned")

            # Postprocess action (unnormalize, etc.)
            action_processed = postprocessor(action)

            # Convert to numpy for environment
            action_numpy = action_processed.cpu().numpy()

            # Step environment
            if step_count < 20:
                print(f"    → env.step()...")
            obs, reward, terminated, truncated, info = env.step(action_numpy)

            # FIX: Increment steps for each environment
            for i in range(num_envs):
                episode_steps[i] += 1

            if step_count < 20:
                print(f"    ← env.step() returned")
                print(f"      terminated: {terminated}, truncated: {truncated}")
                print(f"      episode steps: {episode_steps}")
                print(f"      info keys: {info.keys() if info else 'None'}")

            # Collect metrics when episodes finish - Multiple detection methods
            done = None
            done_detection_method = ""

            # Method 1: Check environment flags
            if terminated.any() or truncated.any():
                done = terminated | truncated
                done_detection_method = "env_flags"
            # Method 2: Check final_info presence (backup method)
            elif "final_info" in info and info["final_info"] is not None:
                # Some episodes might complete without terminal signal
                final_info = info["final_info"]
                if isinstance(final_info, dict) and any(k in final_info for k in ["is_success", "episode"]):
                    # Create done mask based on final_info content
                    if "is_success" in final_info:
                        success_array = final_info["is_success"]
                        if hasattr(success_array, '__len__') and len(success_array) == num_envs:
                            # Check which environments have completed
                            completed_envs = [i for i, s in enumerate(success_array) if s is not None]
                            if completed_envs:
                                done = np.zeros(num_envs, dtype=bool)
                                done[completed_envs] = True
                                done_detection_method = "final_info_success"

            # Only process if we detected any completed episodes
            if done is not None and done.any():

                # Enhanced episode completion processing with detailed debug
                completed_count = 0
                for i, is_done in enumerate(done):
                    if is_done and episodes_completed < num_episodes:
                        episodes_completed += 1
                        completed_count += 1

                        # Extract metrics with multiple fallback methods
                        success = None
                        episode_return = None
                        episode_length = None

                        # Method 1: Use final_info if available
                        if "final_info" in info and info["final_info"] is not None:
                            final_info = info["final_info"]

                            # Success metric
                            if isinstance(final_info, dict) and "is_success" in final_info:
                                if hasattr(final_info["is_success"], '__getitem__') and len(final_info["is_success"]) > i:
                                    success = final_info["is_success"][i]

                            # Episode stats
                            if isinstance(final_info, dict) and "episode" in final_info:
                                ep_info = final_info["episode"]
                                if isinstance(ep_info, dict):
                                    if "r" in ep_info and hasattr(ep_info["r"], '__getitem__') and len(ep_info["r"]) > i:
                                        episode_return = ep_info["r"][i]
                                    if "l" in ep_info and hasattr(ep_info["l"], '__getitem__') and len(ep_info["l"]) > i:
                                        episode_length = ep_info["l"][i]

                        # Store metrics
                        if success is not None:
                            metrics["success"].append(success)
                        if episode_return is not None:
                            metrics["return"].append(episode_return)
                        if episode_length is not None:
                            metrics["length"].append(episode_length)

                        # Reset environment tracking
                        episode_steps[i] = 0
                        episode_completing[i] = False

                # Print completion summary
                if completed_count > 0:
                    summary = f"  ✓ Completed {completed_count} episodes via {done_detection_method} method"
                    if done_detection_method == "env_flags":
                        term_count = terminated.sum() if hasattr(terminated, 'sum') else sum(terminated)
                        trunc_count = truncated.sum() if hasattr(truncated, 'sum') else sum(truncated)
                        summary += f" (term: {term_count}, trunc: {trunc_count})"
                    print(summary)

                # Detailed debug for early steps
                if step_count < 20:
                    print(f"    DEBUG: Detection method={done_detection_method}")
                    print(f"    DEBUG: Episodes completed: {episodes_completed}/{num_episodes}")
                    print(f"    DEBUG: Done mask: {done.tolist() if hasattr(done, 'tolist') else done}")

            # Check for overall timeout (emergency exit)
            # Each environment has max steps, but we also need overall safety timeout
            if step_count >= MAX_STEPS_PER_EPISODE * num_episodes * 2:  # Very conservative limit
                print(f"\n🔴 EMERGENCY: Reached overall maximum step limit ({MAX_STEPS_PER_EPISODE * num_episodes * 2} steps)")
                print(f"    Episodes completed: {episodes_completed}/{num_episodes}")
                print(f"    This should not happen in normal operation!")
                break

            # Increment global step counter
            step_count += 1

        # Safety check: Did we complete all episodes?
        if episodes_completed < num_episodes:
            print(f"\n⚠️  WARNING: Evaluation ended before completing all episodes")
            print(f"   Completed: {episodes_completed}/{num_episodes} episodes")
            print(f"   Total steps: {step_count}")
            print(f"   Episode steps per env: {episode_steps}")
            print(f"   This usually means the policy or environment had unexpected behavior.")
            print(f"   Possible causes:")
            print(f"   - Policy output is causing environment hangs")
            print(f"   - Environment is not properly resetting after completion")
            print(f"   - Something is blocking episode termination")

        # Print final summary
        print(f"\n🏁 Evaluation Summary:")
        print(f"   Episodes completed: {episodes_completed}/{num_episodes}")
        print(f"   Total global steps: {step_count}")
        print(f"   Average steps per episode: {step_count / max(episodes_completed, 1):.1f}")

        # Check if success metrics were collected
        if "success" not in metrics or len(metrics["success"]) == 0:
            print(f"\n❌ No success metrics were collected!")
            print(f"   This indicates episodes were detected as 'done' but no final_info was available")
            print(f"   Check that the environment is properly configured to return success signals")

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
    print(f"DEBUG: Policy path = {args.policy_path}")
    print(f"DEBUG: Is HuggingFace path = {not args.policy_path.startswith('/') and not args.policy_path.startswith('.') and not os.path.exists(args.policy_path)}")
    print(f"DEBUG: Current working directory = {os.getcwd()}")
    print(f"DEBUG: Absolute policy path = {os.path.abspath(args.policy_path)}")
    print(f"DEBUG: HF_HOME env var = {os.environ.get('HF_HOME', 'Not set')}")
    print(f"DEBUG: HF_HUB_CACHE env var = {os.environ.get('HF_HUB_CACHE', 'Not set')}")
    try:
        policy = PI05Policy.from_pretrained(args.policy_path, device=args.device)
        print(f"✓ Policy loaded: {args.policy_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load policy from {args.policy_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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
