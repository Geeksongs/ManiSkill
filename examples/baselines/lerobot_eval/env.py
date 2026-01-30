"""
ManiSkill environment wrapper for LeRobot evaluation.

This module provides a make_env function that creates ManiSkill environments
compatible with LeRobot's evaluation pipeline.

Usage with LeRobot:
    lerobot-eval \
        --policy.path=<your_model> \
        --env.type=hub \
        --env.hub_path="/path/to/this/folder" \
        --trust-remote-code \
        --eval.n_episodes=10

Or programmatically:
    from env import make_env
    envs = make_env(n_envs=4, use_async_envs=False)
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Import ManiSkill to register environments
import mani_skill.envs


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ManiSkillEnvConfig:
    """Configuration for ManiSkill environment."""
    task: str = "PickCube-v1"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"
    sim_backend: str = "auto"
    max_episode_steps: int = 200
    camera_name: str = "base_camera"
    state_dim: int = 8  # First 8 dims of qpos (7 joints + 1 gripper)
    task_description: str = "Pick up the red cube and place it at the green target position."


# Default configuration
DEFAULT_CONFIG = ManiSkillEnvConfig()


# =============================================================================
# Environment Wrapper
# =============================================================================

class ManiSkillLeRobotWrapper(gym.Wrapper):
    """
    Wrapper that adapts ManiSkill environments to LeRobot format.

    This wrapper:
    1. Converts ManiSkill observations to LeRobot's expected format
    2. Provides task_description() method for VLA models
    3. Ensures info contains 'is_success' key
    4. Provides _max_episode_steps for rollout limits
    """

    def __init__(
        self,
        env: gym.Env,
        config: ManiSkillEnvConfig,
    ):
        super().__init__(env)
        self.config = config
        self._task_description = config.task_description
        self._max_episode_steps_val = config.max_episode_steps

        # Store metadata for LeRobot
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata['render_fps'] = 30  # Default FPS for video saving

    def _max_episode_steps(self) -> int:
        """Return maximum episode steps (required by LeRobot rollout)."""
        return self._max_episode_steps_val

    def task_description(self) -> str:
        """Return task description for VLA models."""
        return self._task_description

    def task(self) -> str:
        """Alternative method for task description."""
        return self._task_description

    def reset(self, **kwargs):
        """Reset and convert observation to LeRobot format."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        """Step and convert observation to LeRobot format."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure info has 'is_success' key (required by LeRobot)
        if 'is_success' not in info:
            info['is_success'] = info.get('success', False)

        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Render for video recording."""
        return self.env.render()

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert ManiSkill observation to LeRobot format.

        ManiSkill format:
            {
                'sensor_data': {'base_camera': {'rgb': (H, W, 3)}},
                'agent': {'qpos': (state_dim,)}
            }

        LeRobot format:
            {
                'pixels': {'image': (H, W, 3) uint8},
                'agent_pos': (state_dim,) float32
            }
        """
        # Extract RGB image
        camera_name = self.config.camera_name
        rgb = obs['sensor_data'][camera_name]['rgb']

        # Convert to numpy if tensor
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()

        # Ensure uint8
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Extract robot state (first state_dim dimensions of qpos)
        qpos = obs['agent']['qpos']
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()

        state_dim = self.config.state_dim
        agent_pos = qpos[..., :state_dim].astype(np.float32)

        return {
            'pixels': {'image': rgb},
            'agent_pos': agent_pos,
        }


# =============================================================================
# Vectorized Environment Wrapper
# =============================================================================

class ManiSkillVectorEnvWrapper(gym.Wrapper):
    """
    Wrapper for vectorized ManiSkill environments.

    Handles batch observations and provides the interface expected by
    LeRobot's vectorized evaluation.
    """

    def __init__(
        self,
        env: gym.Env,
        config: ManiSkillEnvConfig,
    ):
        super().__init__(env)
        self.config = config
        self._task_description = config.task_description
        self._max_episode_steps_val = config.max_episode_steps

        # Store metadata
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata['render_fps'] = 30

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        if hasattr(self.unwrapped, 'num_envs'):
            return self.unwrapped.num_envs
        return 1

    @property
    def envs(self):
        """
        List-like access to environments.
        LeRobot expects env.envs[i] to access individual environments.
        """
        return [self] * self.num_envs

    def _max_episode_steps(self) -> int:
        """Return maximum episode steps."""
        return self._max_episode_steps_val

    def task_description(self) -> str:
        """Return task description."""
        return self._task_description

    def task(self) -> str:
        """Alternative method for task description."""
        return self._task_description

    def call(self, method_name: str, *args, **kwargs):
        """
        Support env.call() interface for vectorized environments.
        LeRobot uses this to get task descriptions from all envs.
        """
        if method_name in ("task_description", "task"):
            return [self._task_description] * self.num_envs
        elif method_name == "_max_episode_steps":
            return [self._max_episode_steps_val] * self.num_envs
        elif hasattr(self.unwrapped, 'call'):
            return self.unwrapped.call(method_name, *args, **kwargs)
        else:
            method = getattr(self, method_name, None) or getattr(self.unwrapped, method_name)
            return [method(*args, **kwargs)] * self.num_envs

    def reset(self, **kwargs):
        """Reset and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        """Step and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure info has 'is_success' for each environment
        if 'is_success' not in info:
            if 'success' in info:
                info['is_success'] = info['success']
            else:
                info['is_success'] = np.zeros(self.num_envs, dtype=bool)

        # Convert tensors to numpy if needed
        if hasattr(info.get('is_success'), 'cpu'):
            info['is_success'] = info['is_success'].cpu().numpy()

        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Render all environments for video."""
        return self.env.render()

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert batched ManiSkill observation to LeRobot format.

        ManiSkill format (vectorized):
            {
                'sensor_data': {'base_camera': {'rgb': (B, H, W, 3)}},
                'agent': {'qpos': (B, state_dim)}
            }

        LeRobot format:
            {
                'pixels': {'image': (B, H, W, 3) uint8},
                'agent_pos': (B, state_dim) float32
            }
        """
        camera_name = self.config.camera_name
        rgb = obs['sensor_data'][camera_name]['rgb']

        # Convert to numpy
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Extract robot state
        qpos = obs['agent']['qpos']
        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()

        state_dim = self.config.state_dim
        agent_pos = qpos[..., :state_dim].astype(np.float32)

        return {
            'pixels': {'image': rgb},
            'agent_pos': agent_pos,
        }


# =============================================================================
# Factory Function (Required by LeRobot)
# =============================================================================

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Optional[Any] = None,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """
    Create ManiSkill environments for LeRobot evaluation.

    This is the entry point called by LeRobot's make_env factory.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use async vectorized envs (not supported)
        cfg: Optional configuration object (HubEnvConfig from LeRobot)

    Returns:
        Dictionary mapping suite name to task environments:
        {"maniskill": {0: vec_env}}
    """
    # Build configuration
    config = ManiSkillEnvConfig()

    # Override from cfg if provided
    if cfg is not None:
        # task 参数可以是 "任务名" 或 "任务名::prompt描述" 格式
        # 例如: --env.task="PickCube-v1::Pick up the red cube"
        if hasattr(cfg, 'task') and cfg.task:
            if '::' in cfg.task:
                # 格式: "TaskName::Task description prompt"
                task_name, task_desc = cfg.task.split('::', 1)
                config.task = task_name.strip()
                config.task_description = task_desc.strip()
            else:
                config.task = cfg.task
        if hasattr(cfg, 'episode_length') and cfg.episode_length:
            config.max_episode_steps = cfg.episode_length

    # Create the ManiSkill environment
    env_kwargs = {
        "obs_mode": config.obs_mode,
        "control_mode": config.control_mode,
        "render_mode": config.render_mode,
        "sim_backend": config.sim_backend,
        "num_envs": n_envs,
        "max_episode_steps": config.max_episode_steps,
    }

    print(f"Creating ManiSkill environment: {config.task}")
    print(f"  n_envs: {n_envs}")
    print(f"  control_mode: {config.control_mode}")
    print(f"  obs_mode: {config.obs_mode}")
    print(f"  max_episode_steps: {config.max_episode_steps}")

    # Create base environment
    # Note: ManiSkill envs are registered without namespace prefix
    env = gym.make(config.task, **env_kwargs)

    # Wrap with LeRobot adapter
    wrapped_env = ManiSkillVectorEnvWrapper(env, config)

    # Return in LeRobot's expected format: {suite: {task_id: vec_env}}
    suite_name = "maniskill"
    return {suite_name: {0: wrapped_env}}


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    """Test the environment wrapper standalone."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=1, help="Number of envs (use 1 for CPU testing)")
    args = parser.parse_args()

    print("Testing ManiSkill LeRobot environment wrapper...")

    # Create environment
    envs = make_env(n_envs=args.n_envs)

    # Get the wrapped environment
    env = envs["maniskill"][0]

    print(f"\nEnvironment info:")
    print(f"  num_envs: {env.num_envs}")
    print(f"  task_description: {env.task_description()}")
    print(f"  max_episode_steps: {env._max_episode_steps()}")

    # Test reset
    obs, info = env.reset()
    print(f"\nObservation after reset:")
    print(f"  pixels/image shape: {obs['pixels']['image'].shape}")
    print(f"  pixels/image dtype: {obs['pixels']['image'].dtype}")
    print(f"  agent_pos shape: {obs['agent_pos'].shape}")
    print(f"  agent_pos dtype: {obs['agent_pos'].dtype}")

    # Test step with random action
    action_dim = env.unwrapped.action_space.shape[-1]
    action = np.random.uniform(-1, 1, (env.num_envs, action_dim)).astype(np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nAfter step:")
    print(f"  reward shape: {reward.shape if hasattr(reward, 'shape') else type(reward)}")
    print(f"  is_success in info: {'is_success' in info}")

    # Test call interface
    tasks = env.call("task_description")
    print(f"\nenv.call('task_description'): {tasks}")

    env.unwrapped.close()
    print("\nTest passed!")
