"""
Visualize GNN dynamics model prediction quality during MPC.

For each MPC step this script produces a side-by-side comparison panel:

  Left  – CURRENT state:
           The actual rendered image with the extracted particle representation
           overlaid in green.

  Right – PREDICTION vs REALITY (after taking the chosen action):
           The actual rendered image of the resulting state, with:
             • cyan  dots = model-predicted particle positions
             • red   dots = actual extracted particle positions

Configuration is read from config/mpc/config_simple.yaml, with an optional
sub-key  mpc.prediction_viz for display overrides:

  prediction_viz:
    output_dir:     outputs/prediction_viz
    video_fps:      2

Usage:
    python visualize_prediction.py
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.flex_env import FlexEnv
from model.gnn_dyn import PropNetDiffDenModel
from utils import (load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms,
                   pcd2pix, gen_goal_shape, gen_subgoal,
                   scale_subgoal_to_material_pixels)
from simple_mpc import run_simple_mpc

SIMPLE_MPC_CONFIG = 'config/mpc/config_simple.yaml'


# ─────────────────────────────────────────────────────────────── helpers ─────

def _draw_particles(img_bgr: np.ndarray,
                    pts_3d: np.ndarray,
                    cam_params,
                    color_bgr: tuple,
                    radius: int = 5,
                    alpha: float = 0.75) -> np.ndarray:
    """Project 3-D particles to pixel space and draw circles on img_bgr."""
    if pts_3d is None or len(pts_3d) == 0:
        return img_bgr.copy()
    pix = pcd2pix(pts_3d, cam_params)   # (N, 2) — (row, col)
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for r, c in pix:
        if 0 <= r < H and 0 <= c < W:
            cv2.circle(overlay, (int(c), int(r)), radius, color_bgr, -1, cv2.LINE_AA)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)


def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,   0,   0), 1, cv2.LINE_AA)
    return out


def make_panel(img_cur_rgb: np.ndarray,
               pts_cur: np.ndarray,
               img_next_rgb: np.ndarray,
               pts_pred: np.ndarray,
               pts_actual_next: np.ndarray,
               cam_params,
               step: int,
               reward_cur: float,
               reward_next: float,
               goal_img_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return a BGR comparison panel.

    Layout  (when goal_img_rgb is supplied, a third tile is appended):
      [ current-state tile | next-state tile | (goal tile) ]
      [ legend bar                                         ]
    """
    H, W = img_cur_rgb.shape[:2]

    # BGR colour palette
    GREEN = (50,  200,  0)
    CYAN  = (200, 180,  0)
    RED   = (0,    50, 220)

    img_cur_bgr  = cv2.cvtColor(img_cur_rgb.astype(np.uint8),  cv2.COLOR_RGB2BGR)
    img_next_bgr = cv2.cvtColor(img_next_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    tile_cur  = _draw_particles(img_cur_bgr,  pts_cur,        cam_params, GREEN)
    tile_next = _draw_particles(img_next_bgr, pts_pred,       cam_params, CYAN)
    tile_next = _draw_particles(tile_next,    pts_actual_next, cam_params, RED, radius=3)

    tile_cur  = _label(tile_cur,  f"Step {step}  r={reward_cur:.3f}")
    tile_next = _label(tile_next, f"After action  r={reward_next:.3f}")

    tiles = [tile_cur, tile_next]

    if goal_img_rgb is not None:
        goal_bgr = cv2.cvtColor(goal_img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        goal_bgr = cv2.resize(goal_bgr, (W, H))
        goal_bgr = _label(goal_bgr, "Goal")
        tiles.append(goal_bgr)

    main_row = np.hstack(tiles)

    # legend
    total_w = main_row.shape[1]
    legend_h = 28
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8)
    items = [(8, GREEN, "current particles"),
             (220, CYAN,  "predicted (model)"),
             (430, RED,   "actual (sim)")]
    for x, col, txt in items:
        cv2.circle(legend, (x + 6, legend_h // 2), 6, col, -1)
        cv2.putText(legend, txt, (x + 16, legend_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

    return np.vstack([main_row, legend])


# ─────────────────────────────────────────────────────────────── main ────────

def main():
    config = load_yaml(SIMPLE_MPC_CONFIG)

    # ── prediction_viz display overrides ──────────────────────────────────
    pv            = config['mpc'].get('prediction_viz', {})
    output_base   = pv.get('output_dir',    'outputs/prediction_viz')
    video_fps     = pv.get('video_fps',     2)

    # ── load GNN model ────────────────────────────────────────────────────────
    model_root   = 'data/gnn_dyn_model/'
    model_folder = os.path.join(model_root, config['mpc']['model_folder'])
    model_iter   = config['mpc']['iter_num']

    GNN = PropNetDiffDenModel(config, True)
    ckpt = (f'{model_folder}/net_best.pth' if model_iter == -1
            else f'{model_folder}/net_epoch_0_iter_{model_iter}.pth')
    GNN.load_state_dict(torch.load(ckpt), strict=False)
    GNN = GNN.cuda()

    # ── build environment + goal ──────────────────────────────────────────────
    env = FlexEnv(config)
    screenH = screenW = 720

    task_type = config['mpc']['task']['type']
    if task_type == 'target_control':
        subgoal, goal_mask = gen_subgoal(config['mpc']['task']['goal_row'],
                                         config['mpc']['task']['goal_col'],
                                         config['mpc']['task']['goal_r'],
                                         h=screenH, w=screenW)
        goal_img_rgb = np.stack([goal_mask * 255] * 3, axis=-1).astype(np.uint8)
    elif task_type == 'target_shape':
        subgoal, goal_img_rgb = gen_goal_shape(config['mpc']['task']['target_char'],
                                               h=screenH, w=screenW)
    else:
        raise NotImplementedError(f"Unknown task type: {task_type}")

    env.reset()

    # Scale goal so its pixel footprint matches the material area.
    obs_init = env.render()
    subgoal  = scale_subgoal_to_material_pixels(
        subgoal, obs_init[..., -1], config['dataset']['global_scale'])

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = get_current_YYYY_MM_DD_hh_mm_ss_ms()
    run_dir = os.path.join(output_base, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    # Render at the save resolution to avoid downscaling artefacts.
    env.set_save_render_mode()

    # ── run MPC (model-agnostic via simple_mpc adapter) ───────────────────────
    print(f"Running MPC with GNN model via run_simple_mpc\n")
    subg_output = run_simple_mpc(env, GNN, subgoal, config)

    env.restore_native_render_mode()

    # ── unpack results ────────────────────────────────────────────────────────
    rewards      = subg_output['rewards']         # (n_mpc+1,)
    raw_obs      = subg_output['raw_obs']         # (n_mpc+1, H, W, 5)
    states       = subg_output['states']          # list[n_mpc+1] of (particle_num, 3)
    states_pred  = subg_output['states_pred']     # list[n_mpc]   of (particle_num, 3)
    rew_means    = subg_output['rew_means']       # (n_mpc, 1, n_update_iter*gd_loop)

    cam_params = env.get_cam_params()

    # ── build per-step panels ─────────────────────────────────────────────────
    panels = []
    for i in range(len(states_pred)):           # = n_mpc (actual steps completed)
        img_cur_rgb  = raw_obs[i][..., :3]      # RGB before step i
        img_next_rgb = raw_obs[i + 1][..., :3]  # RGB after  step i

        pts_cur         = states[i]             # actual particles before step i
        pts_pred        = states_pred[i]        # model prediction of state after step i
        pts_actual_next = states[i + 1]         # actual particles after step i

        # ── diagnostic prints ─────────────────────────────────────────────────
        rew_i  = rew_means[i, 0, :]
        valid  = rew_i[rew_i != 0]
        rew_rng = f"{valid.min():.3f}→{valid.max():.3f}" if len(valid) > 1 else "n/a"
        print(f"  step {i + 1}  reward: {rew_rng}  "
              f"(std={float(rew_means[i, 0, :].std()):.4f})")

        goal_for_panel = goal_img_rgb if i == 0 else None

        panel = make_panel(
            img_cur_rgb,
            pts_cur,
            img_next_rgb,
            pts_pred,
            pts_actual_next,
            cam_params,
            step=i + 1,
            reward_cur=float(rewards[i]),
            reward_next=float(rewards[i + 1]),
            goal_img_rgb=goal_for_panel,
        )

        out_path = os.path.join(run_dir, f'step_{i + 1:03d}.png')
        cv2.imwrite(out_path, panel)
        print(f"  step {i + 1:3d}  r_cur={rewards[i]:.4f}  r_next={rewards[i+1]:.4f}"
              f"  → {out_path}")
        panels.append(panel)

    # ── save video ────────────────────────────────────────────────────────────
    if panels:
        # Pad all frames to the same size (first panel may be wider due to goal tile)
        max_w = max(p.shape[1] for p in panels)
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            ph, pw = p.shape[:2]
            if ph < max_h or pw < max_w:
                canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                canvas[:ph, :pw] = p
                padded.append(canvas)
            else:
                padded.append(p)

        vid_path = os.path.join(run_dir, 'prediction_comparison.avi')
        fps = max(1, video_fps)
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, (max_w, max_h))
        hold_frames = fps * 2      # 2 seconds per step
        final_frames = fps * 5     # 5 seconds on last frame
        for idx, f in enumerate(padded):
            n_repeat = final_frames if idx == len(padded) - 1 else hold_frames
            for _ in range(n_repeat):
                writer.write(f)
        writer.release()
        print(f"\nSaved comparison video → {vid_path}")

    # ── reward plot ───────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o')
    plt.xlabel('MPC step')
    plt.ylabel('Reward')
    plt.title('Reward over MPC steps (GNN model)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'rewards.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved reward plot → {plot_path}")

    # ── per-step optimizer convergence plot ───────────────────────────────────
    n_steps_ran = rew_means.shape[0]
    n_cols = min(n_steps_ran, 5)
    n_rows = int(np.ceil(n_steps_ran / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for i in range(n_steps_ran):
        ax = axes[i // n_cols][i % n_cols]
        curve = rew_means[i, 0, :]
        ax.plot(curve)
        ax.set_title(f"Step {i+1}  Δr={curve[-1]-curve[0]:.3f}", fontsize=9)
        ax.set_xlabel('iter', fontsize=8)
        ax.set_ylabel('mean reward', fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(n_steps_ran, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle('Optimizer convergence per MPC step (GNN model)', fontsize=11)
    fig.tight_layout()
    conv_path = os.path.join(run_dir, 'optimizer_convergence.png')
    fig.savefig(conv_path, dpi=150)
    plt.close(fig)
    print(f"Saved optimizer convergence plot → {conv_path}")

    print(f"\nDone. All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
