---
name: mpc-experiments
description: "Understand and work with the MPC experiment framework. Use when: adding/modifying experiments, working with different model types (GNN vs Eulerian), configuring reward functions, debugging experiment runs, or adding new features to the MPC pipeline."
argument-hint: "Specify what you're working on: e.g., 'add new experiment', 'fix gradient issue', 'understand adapter pattern'"
user-invocable: true
---

# MPC Experiments Framework

A model-agnostic gradient-descent MPC (Model Predictive Control) system for object manipulation that supports multiple model types and state representations.

## Architecture Overview

The system uses an **adapter pattern** to make MPC work with different models while keeping the optimization logic unified.

```
run_experiments.py (Batch Runner)
  ├─ Loads experiment config (YAML)
  ├─ Initializes environment
  ├─ For each experiment:
  │   ├─ Loads model (GNN or Eulerian)
  │   ├─ Creates adapter via make_adapter()
  │   ├─ For each episode:
  │   │   ├─ Calls run_simple_mpc() ← Main MPC loop
  │   │   ├─ Saves prediction videos
  │   │   ├─ Computes metrics
  │   └─ Aggregates results
  └─ Generates summary JSONs and plots

simple_mpc/ (Core MPC Package)
  ├─ mpc.py          → Main run_simple_mpc() orchestrator
  ├─ adapters.py     → Model-specific logic (GNN, Eulerian)
  ├─ occupancy_reward.py → Shared reward utility for reporting
  ├─ benchmark.py    → GPU throughput testing
  └─ debug_vis.py    → Debug visualizations
```

## Key Concepts

### 1. **Adapter Pattern**
Each model type implements a consistent interface via adapters in `simple_mpc/adapters.py`:

| Operation | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `obs_to_state()` | Raw obs (H,W,5) | Tensor on device | Convert camera observation to model state (coarse/fast) |
| `obs_to_report_state()` | Raw obs (H,W,5) | Tensor on device | Convert observation for *reporting* reward (dense/correct) |
| `expand_state()` | (1, *S) | (n_sample, *S) | Clone state n times for batch optimization |
| `predict_step()` | State + action | New state | Forward dynamics (differentiable) |
| `compute_reward()` | State batch | Rewards | Score achieved state (higher=better) |
| `get_reward_fn_opt()` | — | callable | Return optimization reward function |
| `get_reward_fn_report()` | — | callable | Return reporting reward function |
| `format_states_pred()` | List of states | Array | Format for output dict |

### 2. **State Representations**
- **GNN Adapter**: States are particle clouds (N, particle_num, 3)
  - `obs_to_state()` returns 50 FPS-sampled particles — intentionally coarse for fast GD optimization
  - `obs_to_report_state()` returns the **full** foreground point cloud from the depth image when `report_type` is `'eulerian'` or `'iou'` (see Critical Warning below); delegates to `obs_to_state` for `'default'`
- **Eulerian Adapter**: States are occupancy grids (64, 64)
  - `obs_to_state()` and `obs_to_report_state()` both use the full depth image — density is already maximal

### 3. **Reward Type System**
Each adapter has a `_REWARD_METHODS` class-level registry:
```python
# GNNAdapter
_REWARD_METHODS = {
    'default':   ('_reward_default', opt=True,  report=True),   # particle scoring
    'iou':       ('_reward_iou',     opt=False, report=True),   # occupancy IoU
    'eulerian':  ('_reward_eulerian', opt=False, report=True),  # occ × score map
}
# EulerianAdapter
_REWARD_METHODS = {
    'default': ('_reward_default', opt=True, report=True),      # occ × score map
    'iou':     ('_reward_iou',     opt=True, report=True),      # occupancy IoU
}
```
YAML override keys (under `overrides:` in experiment spec):
- `mpc.reward.opt_type` — reward used inside the GD optimization loop (default: `'default'`)
- `mpc.reward.report_type` — reward stored in `rewards.npy` and plotted (default: same as `opt_type`)
- `mpc.reward.empty_penalty` — penalise material outside goal region (default: `0.0`)

**During MPC optimization**: each model uses its native reward via `reward_fn_opt = adapter.get_reward_fn_opt()`. Gradients flow through the optimization state.

**For reporting**: `rewards[i]` is computed as `reward_fn_report(adapter.obs_to_report_state(obs_np))`. The `obs_to_report_state` call ensures the reward is computed from a **dense** point cloud regardless of model type.

### ⚠️ Critical Warning: obs_to_report_state vs obs_to_state for GNN
Never call `reward_fn_report(state_cur)` directly for GNN when `report_type in ('eulerian', 'iou')`. The MPC state `state_cur` only contains ~50 FPS-sampled particles, which fills <1.2% of the 64×64 = 4096-voxel occupancy grid. This produces rewards ~100× smaller than Eulerian and high per-step variance that masks real progress. Always use `adapter.obs_to_report_state(obs_np)` to get the dense measurement.

## File Structure & Responsibilities

### `run_experiments.py`
Main entry point for batch experiments.

**Key functions:**
- `main()`: CLI parser, orchestrates experiment loop
- `run_experiment(name, model_spec, cfg)`: Single experiment coordinator
- `run_episode()`: Resets env, calls MPC, saves videos/metrics
- `load_model()`: Loads and caches models (GNN or Eulerian)
- `_save_prediction_video()`: GNN particle comparison videos
- `_save_prediction_video_eulerian()`: Occupancy overlay videos
- `_plot_comparison()`: Cross-experiment reward curves

**Output structure:**
```
outputs/experiments/
├─ <experiment_name>/<timestamp>/
│  ├─ episode_000/
│  │  ├─ rewards.npy
│  │  ├─ prediction_video.avi  (GNN) or prediction_video_eulerian.avi
│  │  └─ metric_*.npy
│  └─ summary.json
├─ compare_<suite>_<timestamp>/
│  ├─ overall_summary.json
│  └─ *.png (comparison plots)
```

### `simple_mpc/mpc.py`
Core MPC optimizer loop.

**Key function:**
- `run_simple_mpc(env, model_dy, subgoal, cfg)`: Main entry point
  - Creates adapter via `make_adapter()` (auto-detects model type)
  - Initializes state from observation
  - Runs N MPC steps with gradient descent optimization
  - Returns dict with rewards, states, actions, etc.

**Returns dict containing:**
- `rewards`: (n_mpc + 1,) array of realized rewards
- `states`: List of observed states (particles or occupancy)
- `states_pred`: List of model-predicted states
- `actions`: (n_mpc, 4) executed actions in world coords
- `raw_obs`: List of raw observations

### `simple_mpc/adapters.py`
Model-specific bridge between MPC loop and model implementations.

**Key classes:**
- `EulerianAdapter`: Wraps `EulerianModelWrapper` (occupancy-grid models)
- `GNNAdapter`: Wraps `PropNetDiffDenModel` (particle-based models)
- `make_adapter()`: Factory function that auto-detects model type, wires reward types from `cfg['mpc']['reward']`, and calls `GNNAdapter.set_occupancy_params()` to enable cross-model metrics

**Setup pattern:**
```python
from simple_mpc.adapters import make_adapter

adapter = make_adapter(
    model_dy=loaded_model,
    env=env,
    subgoal=goal_mask,          # (H, W) binary mask
    cfg=config,                 # Full config dict (reads mpc.reward.opt_type / report_type)
    cam_params=env.get_cam_params(),
    device='cuda'
)

# Get separate reward callables
reward_fn_opt    = adapter.get_reward_fn_opt()    # for GD loop
reward_fn_report = adapter.get_reward_fn_report() # for logging

# In MPC loop — always use obs_to_report_state for reporting:
state = adapter.obs_to_state(obs)                         # coarse, for optimization
report_state = adapter.obs_to_report_state(obs)           # dense, for reporting
reward = reward_fn_report(report_state)
```

### `simple_mpc/occupancy_reward.py`
Shared utility for occupancy-based reward computation.

**Key class:**
- `OccupancyReward`: Converts pixel-space goal masks to occupancy-grid reward maps
  - Used by both adapters to ensure model-invariant metrics
  - Handles distance transforms and empty_penalty tuning

### Configuration System

**Base config**: `config/mpc/config_simple.yaml`
- Dataset parameters (workspace size, camera, particle radius, etc.)
- MPC hyperparameters (n_mpc, n_sample, n_update_iter, learning rate)
- Reward shaping (empty_penalty)
- Debug visualization settings

**Experiment suite**: `config/experiments/compare_models.yaml`
- Defines multiple experiments with per-experiment overrides
- Each experiment specifies:
  - Model type (GNN or Eulerian)
  - Model folder and iteration
  - Optional flags (need_weights, heuristic_type for Eulerian)
  - Optional config overrides

**YAML structure:**
```yaml
base_config: config/mpc/config_simple.yaml
output:
  experiment_name: compare_models
  save_prediction_videos: true
  benchmark_throughput: false

episodes:
  n_episodes: 5

experiments:
  - name: gnn_baseline
    model:
      type: gnn
      folder: 2023-01-28-10-42-05-114323
      iter_num: 1000

  - name: heuristic_splat
    model:
      type: eulerian
      heuristic_type: splat        # 'splat'|'fluid'|'spread'
      need_weights: false          # Skip checkpoint loading

  - name: gnn_report_eulerian
    model:
      type: gnn
      folder: 2023-01-28-10-42-05-114323
      iter_num: 1000
    overrides:
      mpc.reward.opt_type: default    # GNN native reward for GD loop
      mpc.reward.report_type: eulerian  # occ×score map for reporting (must use full dense pcd via obs_to_report_state)
      # GNN-only report types: 'iou', 'eulerian'
      # Eulerian report types: 'default', 'iou'
      # opt_type for GNN must be 'default' (only particle reward has gradients)
```

## Common Tasks

### Adding a New Experiment
1. Edit `config/experiments/compare_models.yaml`
2. Add entry to `experiments:` list with `name`, `model` type/folder, and optional `overrides`
3. Run: `python run_experiments.py config/experiments/compare_models.yaml --only experiment_name`

### Working with Model Types

**GNN Models (Particle-based):**
- State: Particle cloud (N, particle_num, 3)
- Reward: Direct particle-to-pixel scoring (fast, preserves gradients)
- Line: `type: gnn`, `folder: <checkpoint_folder>`, `iter_num: <ckpt_iter>`

**Eulerian Models (Occupancy-based):**
- State: Occupancy grid (64, 64)
- Reward: Occupancy × goal map (coupled to grid representation)
- Learned: `type: eulerian`, `folder: <checkpoint>`, `iter_num: <ckpt_iter>`
- Heuristic: `type: eulerian`, `heuristic_type: splat`, `need_weights: false`

### Debugging Reward Issues

If rewards seem wrong or MPC doesn't improve:
1. **Check adapter choice**: Verify `make_adapter()` returns correct type
2. **Check reward types**: Verify `opt_type` and `report_type` in `cfg['mpc']['reward']`
   - GNN `report_type in ('eulerian', 'iou')`: `rewards.npy` must come from dense point cloud — confirm `obs_to_report_state()` is being called, not `obs_to_state()`
   - Cross-check: `occ_rewards` (from `_compute_occ_reward`) should closely track `rewards` for GNN+eulerian; large divergence means the dense path is not being used
3. **Inspect reward scales**:
   - GNN default: ~[-1, 0], normalized by particle_num (negative: penalty-based)
   - Eulerian default / GNN eulerian-report: ~[0, 400+] unnormalized weighted sum over 4096 voxels
   - GNN iou / Eulerian iou: ~[0, 1] (intersection over union)
4. **Print step diagnostics**: Enable `debug.enabled: true` in config
5. **Visualize occupancy**: Check `_save_prediction_video_eulerian()` overlays

### Adding New Reward Types
1. Add a private method `_reward_<name>(self, state_batch)` to the target adapter
2. Register in `_REWARD_METHODS`: `'name': ('_reward_<name>', valid_for_opt, valid_for_report)`
3. If the reward needs the full dense point cloud (not fixed particle count), set `valid_for_opt=False` and ensure users set `report_type` to it, not `opt_type`
4. In GNNAdapter: if the reward operates on an occupancy grid derived from particles, declare it report-only and let `obs_to_report_state` supply the dense input — **do not** expect the 50-particle MPC state

### Adding New Model Types
1. Create `MyModelAdapter` class in `simple_mpc/adapters.py` with required methods:
   - `obs_to_state(obs_np)`: Render → internal representation (coarse/fast, for optimization)
   - `obs_to_report_state(obs_np)`: Render → dense state for reporting reward (if report reward needs more density than `obs_to_state` provides, extract full point cloud here; otherwise delegate to `obs_to_state`)
   - `expand_state(s, n)`: Clone for batch
   - `predict_step(s_batch, a_batch)`: Forward model
   - `compute_reward(s_batch)`: Default score (used as fallback)
   - `get_reward_fn_opt()` / `get_reward_fn_report()`: Return callables from `_REWARD_METHODS`
2. Define `_REWARD_METHODS` class dict with `(method_name, valid_for_opt, valid_for_report)` entries
3. Register in `make_adapter()`: Add `isinstance(model_dy, MyModel)` branch
4. Update config YAML to specify model type

## Workflow: Running Experiments

### 0. Environment Setup (Required)
Before running any experiments or tests, activate the conda environment and source the setup script:
```bash
conda activate dyn-res-pile-manip
source setup_env.sh
```
This ensures all dependencies (cv2, PyTorch, PyFleX bindings, etc.) are available in the Python path.

**Why this matters:** Running `python run_experiments.py` without this setup will fail with `ModuleNotFoundError` for packages like cv2, even if installed in the environment. The `setup_env.sh` script configures crucial paths for PyFleX integration.

### 1. Check Configuration
```bash
python run_experiments.py config/experiments/compare_models.yaml --dry-run
```
Shows plan (model names, episode counts, output structure) without running.

### 2. Run Experiments
```bash
python run_experiments.py config/experiments/compare_models.yaml
```
Batches all experiments, saves videos, generates comparison plots.

### 3. View Results
- **Per-experiment**: `outputs/experiments/<name>/<timestamp>/summary.json`
- **Overall**: `outputs/experiments/compare_models_<timestamp>/overall_summary.json`
- **Comparison plots**: `outputs/experiments/compare_models_<timestamp>/*.png`

## Key Model Parameters

| Parameter | Default | Use |
|-----------|---------|-----|
| `n_mpc` | 10 | MPC steps per episode |
| `n_sample` | 512 | Candidate actions per optimization step |
| `n_update_iter` | 100 | Adam iterations per MPC step |
| `lr` | 0.05 | Gradient descent learning rate |
| `empty_penalty` | 1.0 | Penalize material in non-goal regions (0=disabled) |
| `particle_num` | 30 | GNN fixed particle count (if using GNN) |

## References
- See `simple_mpc/mpc.py` for main MPC loop structure
- See `simple_mpc/adapters.py` for adapter interfaces and implementations
- See `config/mpc/config_simple.yaml` for all available parameters
- See `config/experiments/compare_models.yaml` for experiment configuration examples
