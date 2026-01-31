#!/usr/bin/env python3
"""
Evaluate LeRobot policies on ManiSkill environments.

Usage:
    python eval_maniskill.py \
        --policy-path pythonsong/pi05_franka_pickcube \
        --task "PickCube-v1::Pick up the cube" \
        --n-episodes 100 \
        --batch-size 10
"""

import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

# Import our ManiSkill env wrapper
from env import make_env, ManiSkillEnvConfig

# LeRobot imports
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation, add_envs_task


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LeRobot policy on ManiSkill")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="HuggingFace model path or local path")
    parser.add_argument("--task", type=str, default="PickCube-v1::Pick up the red cube and place it at the target",
                        help="Task in format 'EnvID::task description'")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--save-video", action="store_true", help="Save evaluation videos")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS")
    return parser.parse_args()


def load_policy(policy_path: str, device: str):
    """Load policy from HuggingFace or local path."""
    print(f"Loading policy from: {policy_path}")

    # Load config to get policy type
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = device

    # Get policy class and load
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(policy_path, config=config)
    policy.to(device)
    policy.eval()

    print(f"  Policy type: {config.type}")
    print(f"  Device: {device}")

    return policy, config


def evaluate(
    policy,
    config,
    preprocessor,
    postprocessor,
    env,
    n_episodes: int,
    max_steps: int,
    device: str,
    use_amp: bool = False,
    save_video: bool = False,
    output_dir: Path = None,
    video_fps: int = 30,
):
    """Run evaluation loop."""
    import cv2

    successes = []
    rewards_sum = []
    episodes_done = 0
    step_count = 0

    # Video recording setup
    video_frames = []
    video_dir = output_dir / "videos" if output_dir else Path("./eval_results/videos")
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Videos will be saved to: {video_dir}")

    obs, info = env.reset()

    pbar = tqdm(total=n_episodes, desc="Evaluating")

    amp_context = torch.autocast(device_type=device.split(":")[0]) if use_amp else nullcontext()

    with torch.no_grad(), amp_context:
        while episodes_done < n_episodes:
            # Record frame for video
            if save_video:
                frame = env.render()
                if frame is not None:
                    if hasattr(frame, 'cpu'):
                        frame = frame.cpu().numpy()
                    # Handle batched rendering (take first env)
                    if frame.ndim == 4:
                        frame = frame[0]
                    video_frames.append(frame)

            # Convert observation to LeRobot format
            obs_processed = preprocess_observation(obs)

            # Add task description
            obs_processed = add_envs_task(env, obs_processed)

            # Preprocess (normalize, tokenize, etc.)
            obs_batch = preprocessor(obs_processed)

            # Get action from policy
            action = policy.select_action(obs_batch)

            # Postprocess (unnormalize)
            action = postprocessor(action)

            # Convert to numpy
            action_np = action.cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            step_count += 1

            # Check for done episodes
            done = terminated | truncated

            if done.any():
                for i, is_done in enumerate(done):
                    if is_done and episodes_done < n_episodes:
                        # Get success
                        if 'is_success' in info:
                            success = info['is_success']
                            if hasattr(success, '__getitem__'):
                                success = bool(success[i])
                            else:
                                success = bool(success)
                        elif 'success' in info:
                            success = info['success']
                            if hasattr(success, '__getitem__'):
                                success = bool(success[i])
                            else:
                                success = bool(success)
                        else:
                            success = False

                        successes.append(success)

                        # Save video for this episode
                        if save_video and video_frames:
                            status = "success" if success else "fail"
                            video_path = video_dir / f"episode_{episodes_done:03d}_{status}.mp4"
                            save_video_frames(video_frames, video_path, video_fps)
                            video_frames = []  # Reset for next episode

                        episodes_done += 1
                        pbar.update(1)
                        pbar.set_postfix({"success_rate": f"{np.mean(successes)*100:.1f}%"})

    pbar.close()

    return {
        "n_episodes": len(successes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "successes": successes,
    }


def save_video_frames(frames, output_path, fps=30):
    """Save frames as MP4 video."""
    import cv2

    if not frames:
        return

    # Get frame dimensions
    h, w = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

    out.release()
    print(f"  Saved video: {output_path}")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("ManiSkill LeRobot Evaluation")
    print("=" * 60)

    # Parse task
    if "::" in args.task:
        env_id, task_desc = args.task.split("::", 1)
    else:
        env_id = args.task
        task_desc = "Complete the task"

    print(f"Environment: {env_id}")
    print(f"Task: {task_desc}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save video: {args.save_video}")
    print()

    # Load policy
    policy, config = load_policy(args.policy_path, args.device)

    # Create preprocessor and postprocessor
    print("Creating preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
    )
    print("  Preprocessors loaded")

    # Create environment
    print(f"Creating environment: {env_id}")

    # Update env config
    env_config = ManiSkillEnvConfig(
        task=env_id,
        task_description=task_desc,
        max_episode_steps=args.max_steps,
    )

    # Manually create env (bypass make_env's config handling)
    import gymnasium as gym
    import mani_skill.envs
    from env import ManiSkillVectorEnvWrapper
    from mani_skill.envs.distraction_set import DistractionSet

    env_kwargs = {
        "obs_mode": env_config.obs_mode,
        "control_mode": env_config.control_mode,
        "render_mode": env_config.render_mode,
        "sim_backend": env_config.sim_backend,
        "num_envs": args.batch_size,
        "max_episode_steps": args.max_steps,
    }

    # Some Colosseum v2 tasks require distraction_set parameter
    # Try to create env, and if it fails with KeyError for distraction_set, add it
    try:
        base_env = gym.make(env_id, **env_kwargs)
    except KeyError as e:
        if "distraction_set" in str(e):
            print("  Adding distraction_set for Colosseum v2 task...")
            env_kwargs["distraction_set"] = DistractionSet()  # Empty distraction set
            base_env = gym.make(env_id, **env_kwargs)
        else:
            raise
    env = ManiSkillVectorEnvWrapper(base_env, env_config)

    print(f"  Created {args.batch_size} parallel environments")
    print()

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print("=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    results = evaluate(
        policy=policy,
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env=env,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=args.device,
        use_amp=args.use_amp,
        save_video=args.save_video,
        output_dir=output_dir,
        video_fps=args.video_fps,
    )

    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Episodes: {results['n_episodes']}")
    print(f"Success Rate: {results['success_rate']*100:.2f}%")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "policy_path": args.policy_path,
            "task": args.task,
            "n_episodes": results["n_episodes"],
            "success_rate": results["success_rate"],
            "config": {
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "device": args.device,
                "use_amp": args.use_amp,
                "seed": args.seed,
            }
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    if args.save_video:
        print(f"Videos saved to: {output_dir / 'videos'}")

    # Cleanup
    env.unwrapped.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
