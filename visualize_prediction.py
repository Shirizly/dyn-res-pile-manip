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

The script delegates all MPC logic to env.step_subgoal_ptcl() and then
generates the per-step visualisations from the returned data.

Configuration is read from config/mpc/config.yaml, with an optional
sub-key  mpc.prediction_viz  for overrides:

  prediction_viz:
    n_mpc:          5       # number of MPC steps (overrides mpc.n_mpc)
    n_sample:       20      # rollout samples per optimisation iteration
    n_update_iter:  50      # gradient-descent iterations per MPC step
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
                   pcd2pix, gen_goal_shape, gen_subgoal)


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
    config = load_yaml("config/mpc/config.yaml")

    # ── read prediction_viz overrides (fall back to mpc-level values) ────────
    pv            = config['mpc'].get('prediction_viz', {})
    n_mpc         = pv.get('n_mpc',         config['mpc']['n_mpc'])
    n_look_ahead  = pv.get('n_look_ahead',  config['mpc']['n_look_ahead'])
    n_sample      = pv.get('n_sample',      config['mpc']['n_sample'])
    n_update_iter = pv.get('n_update_iter', config['mpc']['n_update_iter'])
    gd_loop       = pv.get('gd_loop',       config['mpc']['gd_loop'])
    mpc_type      = pv.get('mpc_type',      config['mpc']['mpc_type'])
    time_lim      = pv.get('time_lim',      config['mpc'].get('time_lim', float('inf')))
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

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = get_current_YYYY_MM_DD_hh_mm_ss_ms()
    run_dir = os.path.join(output_base, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    funnel_dist = np.zeros_like(subgoal)
    action_seq_init = np.load(f'init_action/init_action_{n_sample}.npy')[np.newaxis, ...]
    action_label_init = np.zeros(1)

    # Render at the save resolution to avoid downscaling artefacts.
    env.set_save_render_mode()

    # ── run MPC ───────────────────────────────────────────────────────────────
    print(f"Running MPC: {n_mpc} steps, {n_sample} samples, {n_update_iter} update iters\n")
    subg_output = env.step_subgoal_ptcl(
        subgoal,
        GNN,
        None,                           # init_pos
        n_mpc=n_mpc,
        n_look_ahead=n_look_ahead,
        n_sample=n_sample,
        n_update_iter=n_update_iter,
        mpc_type=mpc_type,
        gd_loop=gd_loop,
        particle_num=-1,
        funnel_dist=funnel_dist,
        action_seq_mpc_init=action_seq_init,
        action_label_seq_mpc_init=action_label_init,
        time_lim=time_lim,
        auto_particle_r=True,
    )

    env.restore_native_render_mode()

    # ── unpack results ────────────────────────────────────────────────────────
    rewards      = subg_output['rewards']         # (n_mpc+1,)
    raw_obs      = subg_output['raw_obs']         # (n_mpc+1, H, W, 5)
    states       = subg_output['states']          # list[n_mpc+1] of (particle_num, 3)
    states_pred  = subg_output['states_pred']     # list[n_mpc]   of (particle_num, 3)

    cam_params = env.get_cam_params()

    # ── build per-step panels ─────────────────────────────────────────────────
    panels = []
    for i in range(len(states_pred)):           # = n_mpc (actual steps completed)
        img_cur_rgb  = raw_obs[i][..., :3]      # RGB before step i
        img_next_rgb = raw_obs[i + 1][..., :3]  # RGB after  step i

        pts_cur         = states[i]             # actual particles before step i
        pts_pred        = states_pred[i]        # model prediction of state after step i
        pts_actual_next = states[i + 1]         # actual particles after step i

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
        H_v, W_v = panels[0].shape[:2]
        vid_path = os.path.join(run_dir, 'prediction_comparison.avi')
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                 max(1, video_fps), (W_v, H_v))
        for f in panels:
            writer.write(f)
        writer.release()
        print(f"\nSaved comparison video → {vid_path}")

    # ── reward plot ───────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o')
    plt.xlabel('MPC step')
    plt.ylabel('Reward')
    plt.title('Reward over MPC steps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'rewards.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved reward plot → {plot_path}")

    print(f"\nDone. All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
