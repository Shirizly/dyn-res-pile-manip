#!/usr/bin/env python3
"""
experiment_analysis.py — Offline analysis tools for completed MPC experiments.

Primary entry point: recompute rewards for all episodes in a suite using a
new reward function, without re-running the simulator.

Usage
-----
Recompute rewards with a different reward type for every experiment in a suite:

    python experiment_analysis.py outputs/experiments/compare_obj_types_fast_videos_2026-04-30-05-16-26-127965/experiment_suite.yaml \\
        --reward-type eulerian

Save results to a custom output file:

    python experiment_analysis.py <suite.yaml> --reward-type iou --out summary_iou.json

Run only specific experiments within the suite:

    python experiment_analysis.py <suite.yaml> --reward-type eulerian --only gnn_coffee splat_coffee

Design
------
ExperimentSuiteLoader
    Parses an experiment_suite.yaml and builds a complete path tree for all
    experiments and episodes.  No files are loaded at construction time — it
    is a lightweight index.  Future analysis steps (dataset preparation,
    trajectory visualisation, etc.) can use it as a foundation.

recompute_rewards()
    Loads raw_obs.npy for each episode, reconstructs the adapter (and thus the
    reward function) from the saved config, computes per-step rewards, and
    returns a summary dict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# ── project imports ────────────────────────────────────────────────────────────
from utils import (
    load_yaml,
    gen_goal_shape,
    gen_subgoal,
)
from run_experiments import apply_overrides, _deep_set


# ──────────────────────────────────────────────────────── path tree ───────────

@dataclass
class EpisodePaths:
    """Filesystem paths for all files saved by a single episode."""
    episode_dir:    str

    @property
    def metrics_json(self)    -> str: return os.path.join(self.episode_dir, 'metrics.json')
    @property
    def rewards_npy(self)     -> str: return os.path.join(self.episode_dir, 'rewards.npy')
    @property
    def occ_rewards_npy(self) -> str: return os.path.join(self.episode_dir, 'occ_rewards.npy')
    @property
    def actions_npy(self)     -> str: return os.path.join(self.episode_dir, 'actions.npy')
    @property
    def raw_obs_npy(self)     -> str: return os.path.join(self.episode_dir, 'raw_obs.npy')
    @property
    def states_npy(self)      -> str: return os.path.join(self.episode_dir, 'states.npy')
    @property
    def episode_data_npz(self)-> str: return os.path.join(self.episode_dir, 'episode_data.npz')

    def exists(self, name: str) -> bool:
        return os.path.exists(getattr(self, name))


@dataclass
class ExperimentPaths:
    """Filesystem paths for one experiment (N episodes) within a suite run."""
    experiment_dir:  str            # e.g. outputs/experiments/gnn_coffee/<timestamp>
    name:            str
    model_spec:      dict
    cfg:             dict           # fully-merged config (base + overrides)
    n_episodes:      int
    episodes:        list[EpisodePaths] = field(default_factory=list)

    @property
    def summary_json(self)       -> str: return os.path.join(self.experiment_dir, 'summary.json')
    @property
    def rewards_all_npy(self)    -> str: return os.path.join(self.experiment_dir, 'rewards_all.npy')
    @property
    def occ_rewards_all_npy(self)-> str: return os.path.join(self.experiment_dir, 'occ_rewards_all.npy')

    def load_summary(self) -> dict:
        with open(self.summary_json) as f:
            return json.load(f)


class ExperimentSuiteLoader:
    """
    Parse a saved experiment_suite.yaml and index all output paths.

    The suite YAML is the copy saved inside the timestamped compare_dir
    (``outputs/experiments/<suite_name>_<timestamp>/experiment_suite.yaml``).
    It contains the original source YAML content, which includes the
    ``base_config`` path, ``output.root_dir``, and all ``experiments``.

    The timestamp is extracted from the compare_dir name; individual
    experiment results are stored under:
        <root_dir>/<experiment_name>/<timestamp>/

    Parameters
    ----------
    suite_yaml_path : str
        Absolute or relative path to the saved ``experiment_suite.yaml``.
    only : list[str] | None
        If given, restrict to these experiment names.
    """

    def __init__(self, suite_yaml_path: str, only: list[str] | None = None):
        self.suite_yaml_path = os.path.abspath(suite_yaml_path)
        self.suite_dir       = os.path.dirname(self.suite_yaml_path)  # compare_dir

        # Extract timestamp from the compare_dir name
        # e.g.  compare_obj_types_fast_videos_2026-04-30-05-16-26-127965
        #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        m = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+)$',
                      os.path.basename(self.suite_dir))
        self.timestamp = m.group(1) if m else None

        suite       = load_yaml(self.suite_yaml_path)
        self.suite  = suite

        base_cfg_path   = suite.get('base_config', 'config/mpc/config_simple.yaml')
        self.base_cfg   = load_yaml(base_cfg_path)
        self.output_cfg = suite.get('output', {})
        self.root_dir   = self.output_cfg.get('root_dir', 'outputs/experiments')
        self.episodes_cfg = suite.get('episodes', {})

        all_specs = suite.get('experiments', [])
        if only:
            known = {e['name'] for e in all_specs}
            bad   = set(only) - known
            if bad:
                raise ValueError(f"Unknown experiment(s): {sorted(bad)}. "
                                 f"Available: {sorted(known)}")
            all_specs = [e for e in all_specs if e['name'] in only]

        self.experiments: list[ExperimentPaths] = []
        for spec in all_specs:
            self.experiments.append(self._build_experiment_paths(spec))

    # ── index building ─────────────────────────────────────────────────────

    def _build_experiment_paths(self, spec: dict) -> ExperimentPaths:
        name       = spec['name']
        overrides  = spec.get('overrides', {})
        n_episodes = spec.get('n_episodes',
                               self.episodes_cfg.get('n_episodes', 5))

        cfg = apply_overrides(self.base_cfg, overrides)

        model_spec = spec.get('model', {})
        if 'folder' in model_spec:
            _deep_set(cfg, 'mpc.model_folder', model_spec['folder'])
        if 'iter_num' in model_spec:
            _deep_set(cfg, 'mpc.iter_num', model_spec['iter_num'])

        ts = self.timestamp or ''
        exp_dir = os.path.join(self.root_dir, name, ts)

        episodes = [
            EpisodePaths(os.path.join(exp_dir, f'episode_{ep:03d}'))
            for ep in range(n_episodes)
        ]

        return ExperimentPaths(
            experiment_dir=exp_dir,
            name=name,
            model_spec=model_spec,
            cfg=cfg,
            n_episodes=n_episodes,
            episodes=episodes,
        )

    # ── convenience accessors ─────────────────────────────────────────────

    @property
    def experiment_names(self) -> list[str]:
        return [e.name for e in self.experiments]

    def get_experiment(self, name: str) -> ExperimentPaths:
        for exp in self.experiments:
            if exp.name == name:
                return exp
        raise KeyError(f"No experiment named '{name}'")

    def check_files(self, file_attr: str = 'raw_obs_npy') -> dict[str, list[bool]]:
        """Return a dict {experiment_name: [episode_exists, ...]}."""
        result = {}
        for exp in self.experiments:
            result[exp.name] = [
                ep.exists(file_attr) for ep in exp.episodes
            ]
        return result

    def __repr__(self) -> str:
        lines = [f"ExperimentSuiteLoader({os.path.basename(self.suite_dir)})"]
        for exp in self.experiments:
            lines.append(f"  {exp.name:40s}  {exp.n_episodes} eps  "
                         f"model={exp.model_spec.get('type','?')}")
        return '\n'.join(lines)


# ──────────────────────────────────────────────── reward recomputation ────────

def _make_reward_fn(exp: ExperimentPaths,
                    reward_type: str,
                    device: str = 'cuda'):
    """
    Construct a reward function for *exp* using *reward_type*.

    We cannot instantiate the full adapter (it needs a live env + model), so we
    build a lightweight stand-alone reward callable that works directly on the
    raw observation arrays saved to disk.

    The callable has signature:
        reward_fn(obs_np: np.ndarray) -> float
    where obs_np is (H, W, 5).

    Currently supported reward types
    ---------------------------------
    eulerian
        Particles extracted from depth channel → hard-voxel occupancy grid →
        dot-product with pre-computed goal score map. Same path as
        EulerianAdapter._reward_default / GNNAdapter._reward_eulerian.
    iou
        Same occupancy grid → intersection-over-union with the goal region.
    """
    from utils import depth2fgpcd
    from model.eulerian_wrapper import _particles_to_occupancy, EulerianModelWrapper

    cfg      = exp.cfg
    ds       = cfg['dataset']
    mpc_task = cfg['mpc']['task']

    global_scale = ds['global_scale']
    wkspc_w      = ds['wkspc_w']

    # Reconstruct goal image at the render resolution recorded in config.
    # Use 720 × 720 as fallback (matches config_simple.yaml default).
    H = int(ds.get('render_height', 720))
    W = int(ds.get('render_width',  720))

    task_type = mpc_task.get('type', 'target_shape')
    if task_type == 'target_shape':
        subgoal, _ = gen_goal_shape(mpc_task['target_char'], h=H, w=W)
    else:
        subgoal, _ = gen_subgoal(mpc_task['goal_row'], mpc_task['goal_col'],
                                 mpc_task['goal_r'], h=H, w=W)

    _FG_DEPTH_THRESHOLD = 0.599 / 0.8

    # ── build occupancy-based score tensor for eulerian / iou ────────────────
    bounds   = EulerianModelWrapper.default_bounds(cfg)
    grid_res = (64, 64)

    # Camera parameters must be reconstructed from config.
    import math
    cam_idx = int(ds.get('cam_idx', 0))
    rad     = math.radians(cam_idx * 20.)
    cam_dis = 0.0 * global_scale / 8.0
    cam_h   = 6.0 * global_scale / 8.0
    cam_pos = np.array([math.sin(rad) * cam_dis, cam_h, math.cos(rad) * cam_dis])
    cam_angle = np.array([rad, -math.radians(90.), 0.])

    # Replicate FlexEnv.get_cam_params() — returns (fx, fy, cx, cy)
    # Using PyFleX's projection conventions from the training code.
    # The exact intrinsics depend on PyFleX internals; load from a saved
    # summary if available, otherwise fall back to a reasonable default.
    # NOTE: For accurate offline reward computation, cam_params should ideally
    # be serialised alongside the episode. Here we recompute them from cfg.
    focal = W / (2.0 * math.tan(math.radians(70.) / 2.0))   # fov=70° (pyflex default)
    cam_params = (focal, focal, W / 2.0, H / 2.0)

    # Pre-compute goal score tensor via a temporary EulerianModelWrapper.
    # We only need the cfg-derived bounds/grid — no weights are loaded.
    # cam_extrinsic does not affect prepare_goal_reward (subgoal is projected
    # via cam_params directly), so a placeholder identity is fine here.
    from model.eulerian_wrapper import EulerianModelWrapper, SplatPushModel
    _push_model  = SplatPushModel(width=wkspc_w, sigma=0.5)
    _cam_ext     = np.eye(4, dtype=np.float64)
    _wrapper     = EulerianModelWrapper(_push_model, bounds, grid_res,
                                        _cam_ext, global_scale)
    occ_score = _wrapper.prepare_goal_reward(
        subgoal, cam_params, device=device, empty_penalty=0.0)  # (Nx, Ny)

    # ── actual reward closures ────────────────────────────────────────────────

    def _obs_to_full_occ(obs_np: np.ndarray) -> torch.Tensor:
        depth  = obs_np[..., -1] / global_scale
        pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, cam_params)
        if pts_np.shape[0] == 0:
            return torch.zeros((1, grid_res[0], grid_res[1]),
                                device=device, dtype=torch.float32)
        pts_t = torch.from_numpy(pts_np.astype(np.float32)).to(device).unsqueeze(0)
        return _particles_to_occupancy(pts_t, bounds, grid_res, sigma=0.0)  # (1, Nx, Ny)

    if reward_type == 'eulerian':
        def reward_fn(obs_np: np.ndarray) -> float:
            occ = _obs_to_full_occ(obs_np)
            return float((occ.clamp(0., 1.) * occ_score).view(1, -1).sum().item())

    elif reward_type == 'iou':
        goal_mask = (occ_score > 0).float().view(-1)  # (Nx*Ny,)
        def reward_fn(obs_np: np.ndarray) -> float:
            occ  = _obs_to_full_occ(obs_np)                     # (1, Nx, Ny)
            pred = (occ > 0.5).float().view(1, -1)               # (1, Nx*Ny)
            inter = (pred * goal_mask).sum()
            union = torch.maximum(pred, goal_mask.unsqueeze(0)).sum()
            return float((inter / (union + 1e-6)).item())

    else:
        raise ValueError(
            f"Unsupported reward_type for offline recomputation: '{reward_type}'. "
            f"Supported: 'eulerian', 'iou'.")

    return reward_fn


def recompute_rewards(
    suite_yaml_path: str,
    reward_type: str,
    only: list[str] | None = None,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Recompute per-step rewards for every episode in a suite using *reward_type*.

    Loads ``raw_obs.npy`` for each episode and re-evaluates the reward function
    from scratch.  Does not require the simulator or model to be loaded.

    Parameters
    ----------
    suite_yaml_path : str
        Path to the ``experiment_suite.yaml`` saved inside the compare_dir.
    reward_type : str
        One of ``'eulerian'``, ``'iou'``.
    only : list[str] | None
        If given, only recompute for these experiment names.
    device : str
        ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    dict with keys per experiment name, each mapping to::
        {
          'rewards_all':   np.ndarray (n_episodes, n_steps+1),
          'reward_gain':   {'mean', 'std', 'min', 'max', 'median'},
          'reward_final':  {'mean', ...},
          'episodes': [
              {'episode': int, 'rewards': list[float],
               'reward_gain': float, 'reward_final': float}, ...
          ],
        }
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        if verbose:
            print("CUDA not available — using CPU")

    loader = ExperimentSuiteLoader(suite_yaml_path, only=only)

    if verbose:
        print(loader)
        print(f"\nRecomputing rewards with type='{reward_type}' on {device}\n")

    results = {}

    for exp in loader.experiments:
        if verbose:
            print(f"{'='*60}")
            print(f"Experiment: {exp.name}  ({exp.n_episodes} episodes)")

        # Check raw_obs availability upfront
        missing = [ep.episode_dir for ep in exp.episodes
                   if not os.path.exists(ep.raw_obs_npy)]
        if missing:
            print(f"  WARNING: raw_obs.npy missing for {len(missing)} episode(s). "
                  f"Skipping.\n  First missing: {missing[0]}")
            results[exp.name] = {'error': f'{len(missing)} episodes missing raw_obs.npy'}
            continue

        # Build reward function once per experiment (not per episode).
        try:
            reward_fn = _make_reward_fn(exp, reward_type, device=device)
        except Exception as exc:
            print(f"  ERROR building reward function: {exc}")
            results[exp.name] = {'error': str(exc)}
            continue

        all_rewards: list[np.ndarray] = []
        ep_records:  list[dict]       = []

        for ep_paths in exp.episodes:
            raw_obs = np.load(ep_paths.raw_obs_npy, allow_pickle=True)
            # raw_obs is (n_steps+1, H, W, C) or a Python list saved as object array
            if raw_obs.dtype == object:
                raw_obs = np.stack([np.array(x) for x in raw_obs])

            n_obs = raw_obs.shape[0]
            rewards = np.empty(n_obs, dtype=np.float32)

            # Per-step reward from raw obs
            for t, obs_np in enumerate(raw_obs):
                rewards[t] = reward_fn(np.array(obs_np))

            ep_idx = exp.episodes.index(ep_paths)
            gain   = float(rewards[-1] - rewards[0])
            all_rewards.append(rewards)
            ep_records.append({
                'episode':      ep_idx,
                'rewards':      rewards.tolist(),
                'reward_initial': float(rewards[0]),
                'reward_final':   float(rewards[-1]),
                'reward_gain':    gain,
            })

            if verbose:
                print(f"  ep {ep_idx:03d}: "
                      f"init={rewards[0]:+.4f}  "
                      f"final={rewards[-1]:+.4f}  "
                      f"gain={gain:+.4f}")

        # Aggregate
        mat   = np.stack(all_rewards)   # (n_episodes, n_steps+1)
        gains = mat[:, -1] - mat[:, 0]

        def _stats(a: np.ndarray) -> dict:
            return {'mean': float(a.mean()), 'std': float(a.std()),
                    'min':  float(a.min()),  'max': float(a.max()),
                    'median': float(np.median(a))}

        exp_result = {
            'name':          exp.name,
            'reward_type':   reward_type,
            'n_episodes':    exp.n_episodes,
            'rewards_all':   mat,
            'reward_gain':   _stats(gains),
            'reward_final':  _stats(mat[:, -1]),
            'reward_initial': _stats(mat[:, 0]),
            'episodes':      ep_records,
        }
        results[exp.name] = exp_result

        if verbose:
            s = exp_result['reward_gain']
            print(f"  → gain  {s['mean']:+.4f} ± {s['std']:.4f}  "
                  f"[{s['min']:+.4f}, {s['max']:+.4f}]")

    return results


def save_recomputed_summary(results: dict, out_path: str) -> None:
    """Serialise *results* to JSON (numpy arrays → lists)."""
    def _jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    out = {}
    for name, rec in results.items():
        out[name] = _jsonify({k: v for k, v in rec.items() if k != 'rewards_all'})
        if 'rewards_all' in rec:
            out[name]['rewards_all'] = _jsonify(rec['rewards_all'])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved recomputed summary → {out_path}")


# ──────────────────────────────────────────────────────────── CLI ─────────────

def main():
    parser = argparse.ArgumentParser(
        description='Offline reward recomputation for completed MPC experiments.')
    parser.add_argument(
        'suite_yaml',
        help='Path to experiment_suite.yaml saved in the compare_dir.')
    parser.add_argument(
        '--reward-type', default='eulerian',
        choices=['eulerian', 'iou'],
        help='Reward function to apply (default: eulerian).')
    parser.add_argument(
        '--only', nargs='+', metavar='NAME',
        help='Only recompute for these experiment names.')
    parser.add_argument(
        '--out', metavar='FILE',
        help='JSON file to write the recomputed summary to. '
             'Defaults to <compare_dir>/recomputed_rewards_<reward_type>.json.')
    parser.add_argument(
        '--device', default='cuda', choices=['cuda', 'cpu'],
        help='Torch device (default: cuda).')
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-episode output.')
    args = parser.parse_args()

    results = recompute_rewards(
        suite_yaml_path=args.suite_yaml,
        reward_type=args.reward_type,
        only=args.only,
        device=args.device,
        verbose=not args.quiet,
    )

    suite_dir = os.path.dirname(os.path.abspath(args.suite_yaml))
    out_path  = args.out or os.path.join(
        suite_dir, f'recomputed_rewards_{args.reward_type}.json')
    save_recomputed_summary(results, out_path)


if __name__ == '__main__':
    main()
