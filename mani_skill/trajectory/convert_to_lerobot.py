#!/usr/bin/env python3
"""
Converts ManiSkill HDF5 trajectory files to LeRobot v3.0 format.
Memory-efficient streaming version - processes one episode at a time.

Usage:
    python convert_to_lerobot.py --traj-path input.h5 --output-dir output_dir
    python convert_to_lerobot.py --traj-path input.h5 --output-dir output_dir --json-path metadata.json

For more information: https://github.com/huggingface/lerobot
"""

import json
import logging
import numpy as np
import pandas as pd
import cv2
import h5py
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
import tyro
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_FPS = 30
DEFAULT_IMAGE_SIZE = "640x480"
DEFAULT_CHUNKS_SIZE = 1000

# Task mapping: env_id -> (task_index, language_description)
# Single Arm Tasks (16 tasks + 4 ColosseumV2 variants)
SINGLE_ARM_TASK_MAPPING = {
    "CookItemInPan-v1":         (0,  "Cook the item in the pan"),
    "HammerNail-v1":            (1,  "Hammer the nail into the surface"),
    "LiftPegUpright-v1":        (2,  "Lift the peg upright"),
    "OpenCabinet-v1":           (3,  "Open the cabinet door"),
    "OpenDrawer-v1":            (4,  "Open the drawer"),
    "PegInsertionSide-v2":      (5,  "Insert the peg from the side"),
    "PickDishFromRack-v1":      (6,  "Pick up the dish from the rack"),
    "PickSodaFromCabinet-v1":   (7,  "Pick up the soda can from the cabinet"),
    "PlaceBookInShelf-v1":      (8,  "Place the book on the shelf"),
    "PlaceCubeInDrawer-v1":     (9,  "Place the cube in the drawer"),
    "PlaceDishInRack-v1":       (10, "Place the dish in the rack"),
    "PlugCharger-v1":           (11, "Plug in the charger"),
    "RaiseCube-v1":             (12, "Raise the cube up from the table"),
    "RotateArrow-v1":           (13, "Rotate the arrow"),
    "ScoopBanana-v1":           (14, "Scoop the banana"),
    "StackCube-v1":             (15, "Stack one cube on top of another"),
    # ColosseumV2 variants
    "StackCubeColosseumV2-v1":        (16, "Stack one cube on top of another"),
    "LiftPegUprightColosseumV2-v1":   (17, "Lift the peg upright"),
    "PlugChargerColosseumV2-v1":      (18, "Plug in the charger"),
    "PegInsertionSideColosseumV2-v1": (19, "Insert the peg from the side"),
}

# Bimanual Tasks (12 tasks)
BIMANUAL_TASK_MAPPING = {
    "DualArmDrawerOpen-v1":     (0,  "Open the drawer with dual arms"),
    "DualArmDrawerPlace-v1":    (1,  "Place object in drawer with dual arms"),
    "DualArmLiftPot-v1":        (2,  "Lift the pot with dual arms"),
    "DualArmLiftTray-v1":       (3,  "Lift the tray with dual arms"),
    "DualArmPenCap-v1":         (4,  "Cap the pen with dual arms"),
    "DualArmPickBottle-v1":     (5,  "Pick up the bottle with dual arms"),
    "DualArmPickCube-v1":       (6,  "Pick up the cube with dual arms"),
    "DualArmPourPot-v1":        (7,  "Pour from the pot with dual arms"),
    "DualArmPushBox-v1":        (8,  "Push the box with dual arms"),
    "DualArmStack3Cube-v1":     (9,  "Stack three cubes with dual arms"),
    "DualArmStackCube-v1":      (10, "Stack cubes with dual arms"),
    "DualArmThreading-v1":      (11, "Thread the needle with dual arms"),
}

# Combined mapping (auto-detect based on env_id prefix)
TASK_MAPPING = {**SINGLE_ARM_TASK_MAPPING, **BIMANUAL_TASK_MAPPING}


def get_task_info(env_id: str, task_mapping: Dict[str, Tuple[int, str]] = None) -> Tuple[int, str]:
    """Get task_index and language description for an env_id."""
    mapping = task_mapping if task_mapping is not None else TASK_MAPPING
    if env_id in mapping:
        return mapping[env_id]
    else:
        # For unknown tasks, generate a new index and use env_id as description
        logger.warning(f"Unknown env_id: {env_id}, using default mapping")
        return (len(mapping), env_id)


def detect_task_type(episode_to_env: Dict[int, str]) -> Dict[str, Tuple[int, str]]:
    """
    Detect whether the dataset is single-arm or bimanual based on env_ids.
    Returns the appropriate task mapping.
    """
    if not episode_to_env:
        return TASK_MAPPING

    # Check first env_id to determine type
    first_env_id = next(iter(episode_to_env.values()))
    if first_env_id.startswith("DualArm"):
        logger.info("Detected bimanual (dual-arm) dataset")
        return BIMANUAL_TASK_MAPPING
    else:
        logger.info("Detected single-arm dataset")
        return SINGLE_ARM_TASK_MAPPING


@dataclass
class Args:
    traj_path: str
    """Path to ManiSkill .h5 trajectory file"""

    output_dir: str
    """Output directory for LeRobot dataset"""

    json_path: Optional[str] = None
    """Path to metadata JSON file (default: same as h5 file with .json extension)"""

    fps: int = DEFAULT_FPS
    """Video FPS (default: 30)"""

    task_name: Optional[str] = None
    """Task description for single-task mode (ignored if JSON has per-episode env_id)"""

    chunks_size: int = DEFAULT_CHUNKS_SIZE
    """Episodes per chunk (default: 1000)"""

    image_size: str = DEFAULT_IMAGE_SIZE
    """Output image size as WIDTHxHEIGHT or single value for square (default: 640x480)"""

    robot_type: Optional[str] = None
    """Robot type (default: auto-detected, e.g., "panda", "ur5")"""


def load_metadata(h5_file: Path, json_path: Optional[str] = None) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    if json_path:
        json_file = Path(json_path)
    else:
        json_file = h5_file.with_suffix('.json')

    if json_file.exists():
        try:
            with open(json_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")
    return {}


def build_episode_task_mapping(metadata: Dict[str, Any]) -> Dict[int, str]:
    """
    Build a mapping from episode_id to env_id from metadata.

    Returns:
        Dict mapping episode_id (int) -> env_id (str)
    """
    episode_to_env = {}

    if 'episodes' in metadata:
        for ep in metadata['episodes']:
            ep_id = ep.get('episode_id')
            env_id = ep.get('env_id')
            if ep_id is not None and env_id is not None:
                episode_to_env[ep_id] = env_id

    return episode_to_env


def detect_rgb_cameras(obs_group: h5py.Group) -> List[str]:
    cameras = []
    if 'sensor_data' in obs_group:
        sensor_data = obs_group['sensor_data']
        for camera_name in sensor_data.keys():
            if 'rgb' in sensor_data[camera_name]:
                cameras.append(camera_name)
    return cameras


def get_trajectory_info(
    h5_file: Path,
    json_path: Optional[str] = None
) -> Tuple[List[str], Dict[str, Any], Dict[int, str]]:
    """
    Get trajectory keys and metadata without loading episode data.
    Memory-efficient: only reads metadata and structure info.

    Returns:
        traj_keys: List of trajectory keys (sorted numerically)
        info: Dictionary with action_dim, state_dim, rgb_cameras, metadata
        episode_to_env: Mapping from episode_id to env_id
    """
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    metadata = load_metadata(h5_file, json_path)
    episode_to_env = build_episode_task_mapping(metadata)

    with h5py.File(h5_file, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]

        if not traj_keys:
            raise ValueError(f"No trajectories found in {h5_file}. Expected keys starting with 'traj_'")

        # Sort trajectory keys numerically (traj_0, traj_1, ..., traj_10, ...)
        traj_keys = sorted(traj_keys, key=lambda x: int(x.split('_')[1]))

        first_traj = f[traj_keys[0]]
        actions = first_traj['actions'][:]
        action_dim = actions.shape[1]

        rgb_cameras = detect_rgb_cameras(first_traj['obs']) if 'obs' in first_traj else []

        state_dim = None
        if 'obs' in first_traj and 'agent' in first_traj['obs'] and 'qpos' in first_traj['obs']['agent']:
            qpos = first_traj['obs']['agent']['qpos'][:]
            state_dim = qpos.shape[1]

        logger.info(f"Detected: action_dim={action_dim}, state_dim={state_dim}, cameras={rgb_cameras}")

    info = {
        'action_dim': action_dim,
        'state_dim': state_dim,
        'rgb_cameras': rgb_cameras,
        'metadata': metadata
    }

    # Log task distribution
    if episode_to_env:
        log_task_mapping = detect_task_type(episode_to_env)
        task_counts = {}
        for env_id in episode_to_env.values():
            task_counts[env_id] = task_counts.get(env_id, 0) + 1
        logger.info(f"Found {len(task_counts)} different tasks:")
        for env_id, count in sorted(task_counts.items()):
            task_idx, task_desc = get_task_info(env_id, log_task_mapping)
            logger.info(f"  [{task_idx}] {env_id}: {count} episodes -> \"{task_desc}\"")

    return traj_keys, info, episode_to_env


def load_single_episode(
    h5_file: Path,
    traj_key: str,
    rgb_cameras: List[str],
    state_dim: Optional[int]
) -> Dict[str, Any]:
    """
    Load a single episode from HDF5 file.
    Memory-efficient: only loads one episode at a time.

    Returns:
        episode_data: Dictionary with actions, rgb data, robot_state, and episode_id
    """
    with h5py.File(h5_file, 'r') as f:
        traj = f[traj_key]
        actions = traj['actions'][:]

        # Extract episode_id from traj_key (e.g., "traj_0" -> 0)
        episode_id = int(traj_key.split('_')[1])

        episode_data = {
            'actions': actions,
            'episode_id': episode_id,
        }

        if rgb_cameras and 'obs' in traj:
            for camera_name in rgb_cameras:
                rgb = traj['obs']['sensor_data'][camera_name]['rgb'][:]
                episode_data[f'rgb_{camera_name}'] = rgb[:len(actions)]

        if state_dim and 'obs' in traj:
            qpos = traj['obs']['agent']['qpos'][:]
            episode_data['robot_state'] = qpos[:len(actions)]

    return episode_data


def parse_image_size(size_str: str) -> Tuple[int, int]:
    if 'x' in size_str:
        parts = size_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid image size format: {size_str}. Expected 'WIDTHxHEIGHT' or 'SIZE'")
        width, height = int(parts[0]), int(parts[1])
    else:
        width = height = int(size_str)
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got: {width}x{height}")
    
    return width, height


def create_directory_structure(
    output_dir: str, 
    rgb_cameras: List[str], 
    num_episodes: int, 
    chunks_size: int = DEFAULT_CHUNKS_SIZE
) -> Path:
    base_path = Path(output_dir)
    num_chunks = (num_episodes + chunks_size - 1) // chunks_size
    
    for chunk_idx in range(num_chunks):
        (base_path / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        
        for camera_name in rgb_cameras:
            camera_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}"
            camera_path.mkdir(parents=True, exist_ok=True)

    (base_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    return base_path


def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result


def create_video_from_frames(
    frames: np.ndarray,
    output_path: Path,
    fps: int,
    image_width: int,
    image_height: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_size = (image_width, image_height)
    resized_frames = [resize_image_with_padding(frame, target_size) for frame in frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (image_width, image_height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def update_stats_from_video(video_path: Path, stats: 'StreamingImageStats') -> int:
    """
    Decode video frame-by-frame and update statistics incrementally.
    This avoids loading all frames into memory at once.
    Returns the number of frames processed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Update stats with single frame (shape: H, W, 3) -> expand to (1, H, W, 3)
        stats.update_single_frame(frame_rgb)
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frame_count


def process_episode(
    episode_data: Dict[str, np.ndarray], 
    episode_idx: int, 
    has_state: bool, 
    fps: int, 
    task_index: int = 0, 
    task_name: str = "Unknown task"
) -> pd.DataFrame:
    actions = episode_data['actions']
    episode_length = actions.shape[0]
    timestamps = np.arange(episode_length, dtype=np.float32) / fps
    
    df_data = {
        'action': [row.tolist() for row in actions],
        'timestamp': timestamps,
        'frame_index': np.arange(episode_length, dtype=np.int64),
        'episode_index': np.full(episode_length, episode_idx, dtype=np.int64),
        'index': np.arange(episode_length, dtype=np.int64),
        'task_index': np.full(episode_length, task_index, dtype=np.int64),
        'task': [task_name] * episode_length
    }
    
    if has_state and 'robot_state' in episode_data:
        df_data['observation.state'] = [row.tolist() for row in episode_data['robot_state']]
    
    column_order = ['action', 'observation.state', 'timestamp', 'frame_index', 
                    'episode_index', 'index', 'task_index', 'task']
    
    df = pd.DataFrame(df_data)
    
    # Ensure task is stored as string
    if 'task' in df.columns:
        df['task'] = df['task'].astype(str)
    
    return df[[col for col in column_order if col in df.columns]]


class StreamingImageStats:
    """
    Streaming statistics calculator for image data.
    Uses exact calculation (no sampling) - mathematically equivalent to batch calculation.
    Processes frame-by-frame to minimize memory usage.
    """
    def __init__(self):
        self.channel_sum = np.zeros(3, dtype=np.float64)
        self.channel_sq_sum = np.zeros(3, dtype=np.float64)
        self.channel_min = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.channel_max = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.total_frames = 0
        self.total_pixels = 0

    def update_single_frame(self, rgb_frame: np.ndarray):
        """
        Update stats with a single RGB frame. Shape: (H, W, 3)
        Memory efficient - processes one frame at a time.
        """
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0

        # Track pixel count
        num_pixels = normalized[..., 0].size  # H * W

        # Calculate per-channel statistics
        for c in range(3):
            channel_data = normalized[..., c]  # Shape: (H, W)
            self.channel_sum[c] += channel_data.sum()
            self.channel_sq_sum[c] += (channel_data ** 2).sum()
            self.channel_min[c] = min(self.channel_min[c], channel_data.min())
            self.channel_max[c] = max(self.channel_max[c], channel_data.max())

        self.total_frames += 1
        self.total_pixels += num_pixels

    def get_stats(self) -> Dict[str, Any]:
        """Get final statistics."""
        if self.total_pixels == 0:
            return {
                'mean': [[[0.5]], [[0.5]], [[0.5]]],
                'std': [[[0.25]], [[0.25]], [[0.25]]],
                'max': [[[1.0]], [[1.0]], [[1.0]]],
                'min': [[[0.0]], [[0.0]], [[0.0]]],
                'count': [[[0]]] * 3
            }

        mean = self.channel_sum / self.total_pixels
        var = (self.channel_sq_sum / self.total_pixels) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))

        return {
            'mean': [[[float(mean[i])]] for i in range(3)],
            'std': [[[float(std[i])]] for i in range(3)],
            'max': [[[float(self.channel_max[i])]] for i in range(3)],
            'min': [[[float(self.channel_min[i])]] for i in range(3)],
            'count': [[[self.total_frames]]] * 3
        }


def calculate_statistics_streaming(
    all_dataframes: List[pd.DataFrame],
    rgb_cameras: List[str],
    has_state: bool,
    image_stats_by_camera: Optional[Dict[str, 'StreamingImageStats']] = None
) -> Dict[str, Any]:
    """
    Calculate statistics using streaming/incremental approach to save memory.
    """
    stats = {}

    # Calculate action statistics incrementally
    action_sum = None
    action_sq_sum = None
    action_min = None
    action_max = None
    action_count = 0

    state_sum = None
    state_sq_sum = None
    state_min = None
    state_max = None
    state_count = 0

    # Index field accumulators
    field_stats = {field: {'sum': 0, 'sq_sum': 0, 'min': float('inf'), 'max': float('-inf'), 'count': 0}
                   for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']}

    for df in all_dataframes:
        # Action stats
        actions = np.stack(df['action'].values)
        if action_sum is None:
            action_sum = actions.sum(axis=0)
            action_sq_sum = (actions ** 2).sum(axis=0)
            action_min = actions.min(axis=0)
            action_max = actions.max(axis=0)
        else:
            action_sum += actions.sum(axis=0)
            action_sq_sum += (actions ** 2).sum(axis=0)
            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))
        action_count += len(actions)

        # State stats
        if has_state and 'observation.state' in df:
            states = np.stack(df['observation.state'].values)
            if state_sum is None:
                state_sum = states.sum(axis=0)
                state_sq_sum = (states ** 2).sum(axis=0)
                state_min = states.min(axis=0)
                state_max = states.max(axis=0)
            else:
                state_sum += states.sum(axis=0)
                state_sq_sum += (states ** 2).sum(axis=0)
                state_min = np.minimum(state_min, states.min(axis=0))
                state_max = np.maximum(state_max, states.max(axis=0))
            state_count += len(states)

        # Index field stats
        for field in field_stats.keys():
            values = df[field].values.astype(float)
            field_stats[field]['sum'] += values.sum()
            field_stats[field]['sq_sum'] += (values ** 2).sum()
            field_stats[field]['min'] = min(field_stats[field]['min'], values.min())
            field_stats[field]['max'] = max(field_stats[field]['max'], values.max())
            field_stats[field]['count'] += len(values)

    # Compute final action stats
    action_mean = action_sum / action_count
    action_var = (action_sq_sum / action_count) - (action_mean ** 2)
    action_std = np.sqrt(np.maximum(action_var, 0))

    stats['action'] = {
        'mean': action_mean.tolist(),
        'std': action_std.tolist(),
        'max': action_max.tolist(),
        'min': action_min.tolist(),
        'count': [action_count]
    }

    # Compute final state stats
    if has_state and state_sum is not None:
        state_mean = state_sum / state_count
        state_var = (state_sq_sum / state_count) - (state_mean ** 2)
        state_std = np.sqrt(np.maximum(state_var, 0))

        stats['observation.state'] = {
            'mean': state_mean.tolist(),
            'std': state_std.tolist(),
            'max': state_max.tolist(),
            'min': state_min.tolist(),
            'count': [state_count]
        }

    # Get image stats from streaming calculator
    for camera_name in rgb_cameras:
        if image_stats_by_camera and camera_name in image_stats_by_camera:
            stats[f'observation.images.{camera_name}'] = image_stats_by_camera[camera_name].get_stats()
        else:
            # Fallback to default values
            stats[f'observation.images.{camera_name}'] = {
                'mean': [[[0.5]], [[0.5]], [[0.5]]],
                'std': [[[0.25]], [[0.25]], [[0.25]]],
                'max': [[[1.0]], [[1.0]], [[1.0]]],
                'min': [[[0.0]], [[0.0]], [[0.0]]],
                'count': [[[action_count]]] * 3
            }

    # Compute final index field stats
    for field, fs in field_stats.items():
        mean = fs['sum'] / fs['count']
        var = (fs['sq_sum'] / fs['count']) - (mean ** 2)
        std = np.sqrt(max(var, 0))

        if field == 'timestamp':
            stats[field] = {
                'mean': [float(mean)],
                'std': [float(std)],
                'max': [float(fs['max'])],
                'min': [float(fs['min'])],
                'count': [fs['count']]
            }
        else:
            stats[field] = {
                'mean': [float(mean)],
                'std': [float(std)],
                'max': [int(fs['max'])],
                'min': [int(fs['min'])],
                'count': [fs['count']]
            }

    return stats


def create_meta_files(
    base_path: Path,
    episode_lengths: List[int],
    total_frames: int,
    action_dim: int,
    state_dim: Optional[int],
    rgb_cameras: List[str],
    metadata: Dict[str, Any],
    task_descriptions: Dict[int, str],  # task_index -> description
    chunks_size: int,
    fps: int,
    image_width: int,
    image_height: int,
    all_dataframes: List[pd.DataFrame],
    robot_type_override: Optional[str] = None
) -> None:
    num_chunks = (len(episode_lengths) + chunks_size - 1) // chunks_size

    episodes_data = []
    dataset_from_index = 0

    for ep_idx, (length, df) in enumerate(zip(episode_lengths, all_dataframes)):
        chunk_idx = ep_idx // chunks_size

        # Get the task description for this episode
        task_idx = int(df['task_index'].iloc[0])
        task_desc = task_descriptions.get(task_idx, "Unknown task")

        episode_meta = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_from_index + length,
            "tasks": [task_desc],
            "length": length,
        }

        for camera_name in rgb_cameras:
            prefix = f"videos/observation.images.{camera_name}"
            episode_meta[f"{prefix}/chunk_index"] = chunk_idx
            episode_meta[f"{prefix}/file_index"] = ep_idx
            episode_meta[f"{prefix}/from_timestamp"] = float(df['timestamp'].iloc[0])
            episode_meta[f"{prefix}/to_timestamp"] = float(df['timestamp'].iloc[-1])

        actions = np.stack(df['action'].values)
        episode_meta["stats/action/min"] = actions.min(axis=0).tolist()
        episode_meta["stats/action/max"] = actions.max(axis=0).tolist()
        episode_meta["stats/action/mean"] = actions.mean(axis=0).tolist()
        episode_meta["stats/action/std"] = actions.std(axis=0).tolist()
        episode_meta["stats/action/count"] = [length]

        if state_dim and 'observation.state' in df:
            states = np.stack(df['observation.state'].values)
            episode_meta["stats/observation.state/min"] = states.min(axis=0).tolist()
            episode_meta["stats/observation.state/max"] = states.max(axis=0).tolist()
            episode_meta["stats/observation.state/mean"] = states.mean(axis=0).tolist()
            episode_meta["stats/observation.state/std"] = states.std(axis=0).tolist()
            episode_meta["stats/observation.state/count"] = [length]

        # Use default image stats to save memory (don't store all RGB data)
        for camera_name in rgb_cameras:
            prefix = f"stats/observation.images.{camera_name}"
            episode_meta[f"{prefix}/min"] = [[[0.0]], [[0.0]], [[0.0]]]
            episode_meta[f"{prefix}/max"] = [[[1.0]], [[1.0]], [[1.0]]]
            episode_meta[f"{prefix}/mean"] = [[[0.5]], [[0.5]], [[0.5]]]
            episode_meta[f"{prefix}/std"] = [[[0.25]], [[0.25]], [[0.25]]]
            episode_meta[f"{prefix}/count"] = [[[length]]]

        for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            values = df[field].values
            episode_meta[f"stats/{field}/min"] = [int(values.min())] if field != 'timestamp' else [float(values.min())]
            episode_meta[f"stats/{field}/max"] = [int(values.max())] if field != 'timestamp' else [float(values.max())]
            episode_meta[f"stats/{field}/mean"] = [float(values.mean())]
            episode_meta[f"stats/{field}/std"] = [float(values.std())]
            episode_meta[f"stats/{field}/count"] = [length]

        episode_meta["meta/episodes/chunk_index"] = 0
        episode_meta["meta/episodes/file_index"] = 0

        episodes_data.append(episode_meta)
        dataset_from_index += length

    episodes_df = pd.DataFrame(episodes_data)
    episodes_df.to_parquet(base_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    # Create tasks.parquet with all tasks
    # Format: index = task description (str), column = task_index (int)
    tasks_data = []
    for task_idx, task_desc in sorted(task_descriptions.items()):
        tasks_data.append({"task_description": task_desc, "task_index": task_idx})

    tasks_df = pd.DataFrame(tasks_data)
    tasks_df = tasks_df.set_index("task_description")
    tasks_df.index.name = None
    tasks_df.to_parquet(base_path / "meta" / "tasks.parquet", index=True)

    logger.info(f"Created tasks.parquet with {len(task_descriptions)} tasks")
    
    # Determine robot type: use override if provided, otherwise auto-detect
    robot_type = "unknown"
    if robot_type_override:
        robot_type = robot_type_override
    elif metadata and 'env_info' in metadata:
        env_id = metadata['env_info'].get('env_id', 'unknown')
        robot_type = env_id.split('-')[0].lower() if '-' in env_id else 'unknown'

    features = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
            "fps": float(fps)
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)],
            "fps": float(fps)
        } if state_dim else {},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
    }

    # Remove empty observation.state if not present
    if not state_dim:
        del features["observation.state"]

    for camera_name in rgb_cameras:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": image_height,
                "video.width": image_width,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }

    data_files_size = sum(f.stat().st_size for f in (base_path / "data").rglob("*.parquet"))
    data_files_size_mb = int(data_files_size / (1024 * 1024))

    info_data = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": len(task_descriptions),  # Now supports multiple tasks
        "total_videos": len(episode_lengths) * len(rgb_cameras),
        "total_chunks": num_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "data_files_size_in_mb": data_files_size_mb,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features
    }

    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)


def main(args: Args):
    if args.chunks_size <= 0:
        raise ValueError("--chunks-size must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    input_path = Path(args.traj_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        logger.info(f"Loading trajectory info from {input_path} (streaming mode)")
        # Only load metadata, not episode data
        traj_keys, info, episode_to_env = get_trajectory_info(input_path, args.json_path)
        num_episodes = len(traj_keys)
        logger.info(f"Found {num_episodes} episodes")

        # Build task_descriptions dictionary: task_index -> description
        task_descriptions: Dict[int, str] = {}

        # Determine if we have per-episode task info
        has_multi_task = len(episode_to_env) > 0

        # Detect task type (single-arm vs bimanual) and get appropriate mapping
        active_task_mapping = detect_task_type(episode_to_env)

        if has_multi_task:
            logger.info("Multi-task mode: using per-episode env_id from JSON metadata")
            # Collect all unique tasks
            for env_id in set(episode_to_env.values()):
                task_idx, task_desc = get_task_info(env_id, active_task_mapping)
                task_descriptions[task_idx] = task_desc
        else:
            # Single task mode: use task_name argument or auto-detect
            default_task_name = args.task_name
            if not default_task_name and info['metadata'] and 'env_info' in info['metadata']:
                default_task_name = info['metadata']['env_info'].get('env_id', 'Unknown task')
            if not default_task_name:
                default_task_name = "Unknown task"
                logger.warning("No task name provided and couldn't auto-detect. Using 'Unknown task'")

            logger.info(f"Single-task mode: using task '{default_task_name}'")
            task_descriptions[0] = default_task_name

        base_path = create_directory_structure(args.output_dir, info['rgb_cameras'], num_episodes, args.chunks_size)
        image_width, image_height = parse_image_size(args.image_size)

        # Streaming data collectors
        all_dataframes = []
        episode_lengths = []
        global_index = 0

        # Initialize streaming image stats calculators
        image_stats_by_camera = {camera: StreamingImageStats() for camera in info['rgb_cameras']}

        # Store video paths for later stats calculation
        video_paths_by_camera: Dict[str, List[Path]] = {camera: [] for camera in info['rgb_cameras']}

        logger.info("Processing episodes (streaming mode - one episode at a time)...")
        for episode_idx, traj_key in enumerate(tqdm(traj_keys, desc="Processing episodes")):
            chunk_idx = episode_idx // args.chunks_size

            # Load single episode - memory efficient
            episode_data = load_single_episode(
                input_path, traj_key, info['rgb_cameras'], info['state_dim']
            )

            # Get the original episode_id (from h5 traj_X key)
            original_episode_id = episode_data.get('episode_id', episode_idx)

            # Determine task_index and task_name for this episode
            if has_multi_task and original_episode_id in episode_to_env:
                env_id = episode_to_env[original_episode_id]
                task_index, task_name = get_task_info(env_id, active_task_mapping)
            else:
                task_index = 0
                task_name = task_descriptions.get(0, "Unknown task")

            df = process_episode(
                episode_data, episode_idx, info['state_dim'] is not None, args.fps,
                task_index=task_index, task_name=task_name
            )
            episode_length = len(df)
            df['index'] = range(global_index, global_index + episode_length)
            global_index += episode_length

            for camera_name in info['rgb_cameras']:
                rgb_key = f'rgb_{camera_name}'
                if rgb_key in episode_data:
                    rgb_data = episode_data[rgb_key]
                    # Create video
                    video_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}" / f"file-{episode_idx:03d}.mp4"
                    create_video_from_frames(rgb_data, video_path, args.fps, image_width, image_height)
                    video_paths_by_camera[camera_name].append(video_path)

            all_dataframes.append(df)
            episode_lengths.append(episode_length)

            # Free memory after each episode
            del episode_data
            if (episode_idx + 1) % 50 == 0:
                gc.collect()

        # Calculate image stats from decoded video frames (to match LeRobot's stats calculation)
        # Process frame-by-frame to minimize memory usage
        logger.info("Computing image statistics from decoded video frames (frame-by-frame)...")
        for camera_name in info['rgb_cameras']:
            video_paths = video_paths_by_camera[camera_name]
            for i, video_path in enumerate(tqdm(video_paths, desc=f"Reading {camera_name} videos")):
                update_stats_from_video(video_path, image_stats_by_camera[camera_name])
                # Periodic garbage collection to prevent memory buildup
                if (i + 1) % 100 == 0:
                    gc.collect()

        num_chunks = (num_episodes + args.chunks_size - 1) // args.chunks_size
        logger.info(f"Saving data to {num_chunks} chunk(s)")

        import pyarrow as pa
        import pyarrow.parquet as pq

        for chunk_idx in range(num_chunks):
            start_ep = chunk_idx * args.chunks_size
            end_ep = min((chunk_idx + 1) * args.chunks_size, len(all_dataframes))

            chunk_dfs = all_dataframes[start_ep:end_ep]
            combined_df = pd.concat(chunk_dfs, ignore_index=True)

            # Remove task column from parquet (it's dynamically generated from task_index)
            if 'task' in combined_df.columns:
                combined_df = combined_df.drop(columns=['task'])

            parquet_path = base_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"

            schema_fields = []
            for col in combined_df.columns:
                if col in ['action', 'observation.state']:
                    schema_fields.append(pa.field(col, pa.list_(pa.float32())))
                elif col == 'timestamp':
                    schema_fields.append(pa.field(col, pa.float32()))
                elif col in ['frame_index', 'episode_index', 'index', 'task_index']:
                    schema_fields.append(pa.field(col, pa.int64()))

            schema = pa.schema(schema_fields)
            table = pa.Table.from_pandas(combined_df, schema=schema)
            pq.write_table(table, parquet_path)

        logger.info("Calculating statistics (streaming mode to save memory)")
        stats = calculate_statistics_streaming(
            all_dataframes, info['rgb_cameras'], info['state_dim'] is not None,
            image_stats_by_camera=image_stats_by_camera
        )
        with open(base_path / "meta" / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("Creating metadata files")
        total_frames = sum(episode_lengths)
        create_meta_files(
            base_path, episode_lengths, total_frames,
            info['action_dim'], info['state_dim'], info['rgb_cameras'],
            info['metadata'], task_descriptions, args.chunks_size, args.fps,
            image_width, image_height, all_dataframes,
            robot_type_override=args.robot_type
        )

        logger.info(f"\n{'='*80}")
        logger.info("Conversion completed successfully!")
        logger.info(f"{'='*80}")
        logger.info(f"Episodes: {len(episode_lengths)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Tasks: {len(task_descriptions)}")
        logger.info(f"Chunks: {num_chunks}")
        for task_idx, task_desc in sorted(task_descriptions.items()):
            logger.info(f"  [{task_idx}] {task_desc}")
        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    parsed_args = tyro.cli(Args)
    sys.exit(main(parsed_args))