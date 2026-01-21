# Pi0.5 Evaluation for ManiSkill

This directory contains scripts for evaluating Pi0.5 (Vision-Language-Action) models on ManiSkill environments.

## Overview

The evaluation pipeline:
1. Loads a trained Pi0.5 model from HuggingFace
2. Creates ManiSkill environments with LeRobot-compatible wrappers
3. Runs rollouts and computes success metrics

## Files

- `maniskill_env_wrapper.py`: Wrapper to adapt ManiSkill observations to LeRobot format
- `evaluate_pi05.py`: Main evaluation script for Pi0.5 models

## Installation

Make sure you have both ManiSkill and LeRobot installed in the same environment:

```bash
# Already done in the lerobot conda environment
conda activate lerobot

# ManiSkill should be installed in editable mode
# LeRobot should be installed with Pi0.5 support
```

## Usage

### Basic Evaluation

Evaluate a trained Pi0.5 model on PickCube task:

```bash
python evaluate_pi05.py \
    --policy-path pythonsong/pi05_franka_pickcube \
    --dataset-repo pythonsong/franka_maniskill_pickcube_200 \
    --env-id PickCube-v1 \
    --num-episodes 100 \
    --num-envs 10 \
    --device cuda
```

### Arguments

**Model Arguments:**
- `--policy-path`: Path to pretrained Pi0.5 model (HuggingFace repo or local path)
  - Default: `pythonsong/pi05_franka_pickcube`
- `--dataset-repo`: Dataset repo for loading normalization stats
  - Default: `pythonsong/franka_maniskill_pickcube_200`
- `--device`: Device to run on (`cuda` or `cpu`)
  - Default: `cuda`

**Environment Arguments:**
- `--env-id`: ManiSkill environment ID
  - Default: `PickCube-v1`
- `--obs-mode`: Observation mode (`rgb`, `rgbd`, or `state`)
  - Default: `rgb`
- `--sim-backend`: Simulation backend (`auto`, `cpu`, or `gpu`)
  - Default: `auto`
- `--task-description`: Task description for the VLA model
  - Default: `"Pick up the cube and place it at the goal position"`

**Evaluation Arguments:**
- `--num-episodes`: Number of episodes to evaluate
  - Default: `100`
- `--num-envs`: Number of parallel environments
  - Default: `10`
- `--max-episode-steps`: Maximum steps per episode (None = use env default)
  - Default: `None`

**Output Arguments:**
- `--output-dir`: Directory to save evaluation results
  - Default: `./eval_results`
- `--save-video`: Whether to save evaluation videos
  - Default: `False`
- `--video-dir`: Directory to save videos
  - Default: `./videos`

### Examples

**Quick test with 10 episodes:**
```bash
python evaluate_pi05.py \
    --num-episodes 10 \
    --num-envs 5 \
    --device cuda
```

**Full evaluation with 100 episodes:**
```bash
python evaluate_pi05.py \
    --policy-path pythonsong/pi05_franka_pickcube \
    --num-episodes 100 \
    --num-envs 10 \
    --device cuda \
    --output-dir ./eval_results_pickcube
```

**Evaluate on CPU:**
```bash
python evaluate_pi05.py \
    --device cpu \
    --num-episodes 10 \
    --num-envs 2
```

## Output

The script will print:
- Loading progress
- Evaluation progress
- Final metrics (success rate, average return, episode length)

Results are saved to `{output_dir}/evaluation_results.txt`

Example output:
```
================================================================================
Evaluation Results
================================================================================

✓ Success Rate: 85.00% (85/100)
✓ Average Return: 12.45 ± 3.21
✓ Average Episode Length: 156.3 steps

✓ Results saved to: ./eval_results/evaluation_results.txt
```

## How It Works

### 1. ManiSkill Observation Format

ManiSkill outputs observations in this format:
```python
{
    'sensor_data': {
        'base_camera': {
            'rgb': torch.Tensor (num_envs, 128, 128, 3)  # uint8
        }
    },
    'agent': {
        'qpos': torch.Tensor (num_envs, 9)  # float32
    }
}
```

### 2. LeRobot Expected Format

LeRobot's `preprocess_observation` expects:
```python
{
    'pixels': np.ndarray (num_envs, H, W, 3),  # uint8
    'agent_pos': np.ndarray (num_envs, 8)      # float32
}
```

### 3. Wrapper Conversion

`ManiSkillVectorEnvWrapper` converts:
- `sensor_data.base_camera.rgb` → `pixels`
- `agent.qpos[:, :8]` → `agent_pos` (takes first 8 dims from 9-dim qpos)
- Adds `task_description()` method for LeRobot's `add_envs_task()`

### 4. Processing Pipeline

```
ManiSkill obs
    ↓ (wrapper)
LeRobot format obs
    ↓ (preprocess_observation)
Normalized tensors
    ↓ (add_envs_task)
+ task description
    ↓ (preprocessor)
Model input batch
    ↓ (policy.select_action)
Actions
    ↓ (postprocessor)
Denormalized actions
    ↓ (to numpy)
ManiSkill action
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'lerobot'`
- Solution: Make sure LeRobot is installed: `pip install lerobot`

**Issue**: `ModuleNotFoundError: No module named 'mani_skill'`
- Solution: Install ManiSkill: `pip install -e .` from ManiSkill root

**Issue**: Memory error with many parallel envs
- Solution: Reduce `--num-envs` (e.g., try `--num-envs 5` or `--num-envs 2`)

**Issue**: CUDA out of memory
- Solution: Use `--device cpu` or reduce `--num-envs`

**Issue**: Dimension mismatch errors
- Solution: Make sure the model was trained on the same observation/action space

## Notes

- The evaluation uses `policy.select_action()` which automatically handles action queuing for multi-step actions (action horizon = 50 for Pi0.5)
- State normalization is handled automatically by LeRobot's preprocessor using the dataset stats
- The task description must match what the model was trained with
