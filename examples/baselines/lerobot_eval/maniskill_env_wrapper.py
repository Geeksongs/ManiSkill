"""
ManiSkill environment wrapper for LeRobot evaluation.

This wrapper adapts ManiSkill environments to work with LeRobot's evaluation pipeline.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict


class ManiSkillLeRobotWrapper(gym.Wrapper):
    """
    Wrapper that adapts ManiSkill environment observations to LeRobot format.

    Converts ManiSkill's observation format to the format expected by LeRobot:
    - Extracts RGB images from sensor_data
    - Extracts robot state from agent.qpos (first 8 dimensions)
    - Adds task description for VLA models
    """

    def __init__(self, env: gym.Env, task_description: str = "Pick up the cube and place it at the goal position"):
        """
        Args:
            env: ManiSkill environment
            task_description: Task description string for VLA models
        """
        super().__init__(env)
        self.task_description_str = task_description

    def task_description(self):
        """Returns task description for LeRobot's add_envs_task() function."""
        return self.task_description_str

    def task(self):
        """Alternative method name for task description."""
        return self.task_description_str

    def reset(self, **kwargs):
        """Reset environment and convert observation to LeRobot format."""
        obs, info = self.env.reset(**kwargs)
        obs_lerobot = self._convert_observation(obs)
        return obs_lerobot, info

    def step(self, action):
        """Step environment and convert observation to LeRobot format."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_lerobot = self._convert_observation(obs)
        return obs_lerobot, reward, terminated, truncated, info

    def _convert_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert ManiSkill observation to LeRobot format.

        ManiSkill format:
            obs = {
                'sensor_data': {
                    'base_camera': {'rgb': torch.Tensor (1, H, W, 3)}
                },
                'agent': {
                    'qpos': torch.Tensor (1, 9)
                }
            }

        LeRobot format (for preprocess_observation):
            {
                'pixels': np.ndarray (batch, H, W, 3), uint8
                'agent_pos': np.ndarray (batch, 8), float32
            }

        Args:
            obs: ManiSkill observation dictionary

        Returns:
            LeRobot-formatted observation dictionary
        """
        # Extract RGB image from sensor_data
        # ManiSkill: (1, 128, 128, 3) torch.uint8
        rgb_tensor = obs['sensor_data']['base_camera']['rgb']

        # Convert to numpy and keep batch dimension
        # Shape: (1, 128, 128, 3)
        pixels = rgb_tensor.cpu().numpy() if hasattr(rgb_tensor, 'cpu') else rgb_tensor

        # Extract robot state from agent.qpos
        # ManiSkill: (1, 9) - we take first 8 dimensions (7 joints + 1 gripper)
        qpos = obs['agent']['qpos']
        agent_pos = qpos[:, :8].cpu().numpy() if hasattr(qpos, 'cpu') else qpos[:, :8]

        # Return pixels as dict so preprocess_observation maps to observation.images.image
        # (not observation.image)
        return {
            'pixels': {
                'image': pixels  # This will become observation.images.image
            },
            'agent_pos': agent_pos.astype(np.float32)
        }


class ManiSkillVectorEnvWrapper(gym.Wrapper):
    """
    Wrapper for vectorized ManiSkill environments.

    Similar to ManiSkillLeRobotWrapper but handles vectorized environments
    where observations already have batch dimension.
    """

    def __init__(self, env: gym.vector.VectorEnv, task_description: str = "Pick up the cube and place it at the goal position"):
        """
        Args:
            env: Vectorized ManiSkill environment
            task_description: Task description string for VLA models
        """
        super().__init__(env)
        self.task_description_str = task_description

    @property
    def num_envs(self):
        """Return number of parallel environments."""
        # ManiSkill stores num_envs in the unwrapped environment
        return self.unwrapped.num_envs if hasattr(self.unwrapped, 'num_envs') else 1

    @property
    def envs(self):
        """Return list-like access to environments for LeRobot compatibility."""
        # LeRobot's add_envs_task expects env.envs[0] to access the first environment
        # Return a list with self repeated num_envs times so env.envs[0] returns this wrapper
        return [self] * self.num_envs

    def task_description(self):
        """Returns task description for LeRobot's add_envs_task() function."""
        return self.task_description_str

    def call(self, method_name, *args, **kwargs):
        """Support env.call() interface for vectorized environments."""
        # LeRobot uses env.call("task_description") to get tasks from all envs
        if method_name == "task_description" or method_name == "task":
            return [self.task_description_str] * self.num_envs
        # Delegate other calls to the underlying environment
        elif hasattr(self.unwrapped, 'call'):
            return self.unwrapped.call(method_name, *args, **kwargs)
        else:
            # Fallback: call the method directly
            method = getattr(self, method_name, None)
            if method is None:
                method = getattr(self.unwrapped, method_name)
            return [method(*args, **kwargs)] * self.num_envs

    def task(self):
        """Alternative method name for task description."""
        return self.task_description_str

    def reset(self, **kwargs):
        """Reset environment and convert observation to LeRobot format."""
        obs, info = self.env.reset(**kwargs)
        obs_lerobot = self._convert_observation(obs)
        return obs_lerobot, info

    def step(self, action):
        """Step environment and convert observation to LeRobot format."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_lerobot = self._convert_observation(obs)
        return obs_lerobot, reward, terminated, truncated, info

    def _convert_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert vectorized ManiSkill observation to LeRobot format.

        Args:
            obs: ManiSkill observation dictionary with batch dimension

        Returns:
            LeRobot-formatted observation dictionary
        """
        try:
            # Extract RGB image from sensor_data
            # ManiSkill vectorized: (num_envs, 128, 128, 3) torch.uint8
            rgb_tensor = obs['sensor_data']['base_camera']['rgb']

            # Safe conversion to numpy with device checks
            if hasattr(rgb_tensor, 'cpu'):
                pixels = rgb_tensor.cpu().numpy()
            elif hasattr(rgb_tensor, 'numpy'):
                pixels = rgb_tensor.numpy()
            else:
                # Already numpy array
                pixels = rgb_tensor

            # Ensure correct data type
            if pixels.dtype != np.uint8:
                pixels = pixels.astype(np.uint8)

            # Extract robot state from agent.qpos
            # ManiSkill vectorized: (num_envs, 9) - we take first 8 dimensions
            qpos = obs['agent']['qpos']

            # Safe conversion with device checks
            if hasattr(qpos, 'cpu'):
                agent_pos = qpos[:, :8].cpu().numpy()
            elif hasattr(qpos, 'numpy'):
                agent_pos = qpos[:, :8].numpy()
            else:
                # Already numpy array
                agent_pos = qpos[:, :8]

            # Ensure correct shape and data type
            agent_pos = agent_pos.astype(np.float32)

            # Return pixels as dict so preprocess_observation maps to observation.images.image
            result = {
                'pixels': {
                    'image': pixels  # This will become observation.images.image
                },
                'agent_pos': agent_pos
            }

            return result

        except Exception as e:
            # Enhanced error reporting
            print(f"\n🔴 ERROR in observation conversion: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
            print(f"   Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")

            # Include what data is available for debugging
            if 'sensor_data' in obs:
                print(f"   sensor_data keys: {list(obs['sensor_data'].keys())}")
                if 'base_camera' in obs['sensor_data']:
                    print(f"   base_camera keys: {list(obs['sensor_data']['base_camera'].keys())}")
            if 'agent' in obs:
                print(f"   agent keys: {list(obs['agent'].keys())}")

            raise RuntimeError(f"Failed to convert observation: {str(e)}") from e
