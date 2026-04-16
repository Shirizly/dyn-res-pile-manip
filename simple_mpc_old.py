"""Simple bare-bones gradient-descent MPC for EulerianModelWrapper.

Critical differences from the full env.step_subgoal_ptcl pipeline:

  1. **Random independent action initialisation** for all n_sample candidates.
     The original code stacks the same sequence n_sample times → std=0 → Adam
     sees identical gradients for every sample → no diversity, oscillation, and
     Δr < 0 every step.  Here each candidate is drawn independently and
     uniformly from the workspace, giving the optimizer a diverse landscape to
     climb.

  2. **Full depth-image point cloud** for Eulerian occupancy seeding.
     The original code uses obs2ptcl_fixed_num_batch (FPS sub-sampling to a
     fixed particle count), discarding density information and introducing
     stochastic noise.  Here the complete foreground depth point cloud is used
     directly.

  3. **No dynamic particle-count module (res_rgr / auto_particle_r).**
     Irrelevant for the Eulerian representation whose state is a fixed-size 2-D
     occupancy grid regardless of how many particles are used for seeding.

  4. **Consistent reward inside and outside the optimizer.**
     Both use  r = (occ * score_map).sum()  so the logged per-step rewards are
     directly comparable to the optimizer objective.

Return value is a dict with the same keys as step_subgoal_ptcl so all
visualize_*.py scripts work without modification — just swap the call:

    # in visualize_mpc.py / visualize_prediction_eulerian.py
    USE_SIMPLE_MPC = True          # ← one-line switch
    SIMPLE_MPC_CONFIG = 'config/mpc/config_simple.yaml'

    if USE_SIMPLE_MPC:
        from simple_mpc import run_simple_mpc
        config = load_yaml(SIMPLE_MPC_CONFIG)   # replaces config.yaml
        ...
        subg_output = run_simple_mpc(env, model_dy, subgoal, config,
                                     video_recorder=video_recorder)
    else:
        subg_output = env.step_subgoal_ptcl(...)
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.optim as optim

from utils import load_yaml, depth2fgpcd
from model.eulerian_wrapper import EulerianModelWrapper, _action_to_cam_3d


# ── public helpers ────────────────────────────────────────────────────────────

def load_simple_config(path: str = 'config/mpc/config_simple.yaml') -> dict:
    """Load the simple-MPC config file."""
    return load_yaml(path)


def benchmark_push_throughput(
    model_dy,
    wkspc_w: float = 5.0,
    batch_sizes: list = None,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = 'cuda',
) -> None:
    """
    Benchmark the push model (predict_one_step_occ) throughput for a range of
    batch sizes.  Run this after constructing the EulerianModelWrapper to check
    how many candidates/second the GPU can process.

    Output columns
    --------------
    batch  – number of parallel candidates (≈ n_sample in the MPC config)
    ms/step – wall-clock milliseconds for one full forward pass of the push model
    cand/s  – throughput in candidates per second
    ms/iter – expected time per Adam iteration at this batch size (forward only)

    Usage
    -----
    >>> from simple_mpc import benchmark_push_throughput
    >>> benchmark_push_throughput(model_dy, wkspc_w=5.0)
    """
    import time

    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 50, 64, 100, 128, 200, 256, 512, 1024, 2048, 4096]

    Nx, Ny = model_dy.grid_res
    act_lo = np.array([-wkspc_w, -wkspc_w, -wkspc_w * 0.85, -wkspc_w * 0.85],
                      dtype=np.float32)
    act_hi = -act_lo

    print(f"\n── push throughput benchmark ────────────────────────────────────")
    print(f"   model : {type(model_dy.user_model).__name__}")
    print(f"   grid  : {Nx} × {Ny}   device: {device}")
    print(f"   warmup: {n_warmup}   timed runs: {n_runs}")
    print(f"{'batch':>7}  {'ms/step':>9}  {'cand/s':>10}  {'ms/iter':>9}  {'status':>6}")
    print(f"{'─'*7}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*6}")

    for B in batch_sizes:
        try:
            occ = torch.rand(B, Nx, Ny, device=device)
            act_np = np.random.uniform(act_lo, act_hi, (B, 4)).astype(np.float32)
            act = torch.tensor(act_np, device=device)

            # warmup (including JIT / cuBLAS plan caches)
            for _ in range(n_warmup):
                with torch.no_grad():
                    model_dy.predict_one_step_occ(occ, act)
            if device == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    model_dy.predict_one_step_occ(occ, act)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1e3 / n_runs

            cand_per_s = B / (elapsed_ms * 1e-3)
            print(f"{B:>7}  {elapsed_ms:>9.2f}  {cand_per_s:>10.0f}  "
                  f"{elapsed_ms:>9.2f}     ok")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = 'OOM' if 'out of memory' in str(exc).lower() else 'ERR'
            print(f"{B:>7}  {'':>9}  {'':>10}  {'':>9}  {msg:>6}")
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f"─────────────────────────────────────────────────────────────────\n")


# ── debug visualization helpers ───────────────────────────────────────────────
# These are pure numpy/cv2 — no dependency on visualize_*.py.

def _heatmap_bgr(
    arr: np.ndarray,                      # (Nx, Ny) float
    size: tuple,                          # (H, W)
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Render a 2-D grid as a false-colour BGR image (x=right, y=down)."""
    t  = arr.T   # (Ny, Nx)
    lo, hi = t.min(), t.max()
    u8 = ((t - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(u8, colormap),
                      (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def _diff_heatmap_bgr(diff: np.ndarray, size: tuple) -> np.ndarray:
    """Occupancy change: green = material arrived, red = material left."""
    H, W = size
    gain = cv2.resize((np.clip( diff, 0, 1).T * 255).astype(np.uint8),
                      (W, H), interpolation=cv2.INTER_NEAREST)
    loss = cv2.resize((np.clip(-diff, 0, 1).T * 255).astype(np.uint8),
                      (W, H), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 1] = gain   # green channel → gain  (BGR)
    img[:, :, 2] = loss   # red   channel → loss  (BGR)
    return img


def _stamp(img: np.ndarray, lines) -> np.ndarray:
    """Stamp text lines at the top-left corner of a copy of img."""
    out = img.copy()
    for k, text in enumerate(lines):
        y = 15 + k * 15
        cv2.putText(out, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0,   0,   0), 1, cv2.LINE_AA)
    return out


def _save_debug_candidates(
    path:              str,
    occ_init_np:       np.ndarray,        # (Nx, Ny)
    topk_occ_nps:      list,              # K × (Nx, Ny)
    topk_acts:         np.ndarray,        # (K, 4)  world-2D
    topk_rews:         np.ndarray,        # (K,)
    score_np:          np.ndarray,        # (Nx, Ny)  goal score map
    step: int,
    it:   int,
    tile: int = 160,
    topk_start_grids:  np.ndarray = None, # (K, 2)  grid-index coords
    topk_end_grids:    np.ndarray = None, # (K, 2)
) -> None:
    """
    Save a vertical stack of top-K candidate tiles.
    Each row: [input occ | predicted occ + action arrow | pred*score | score map]

    Columns:
      col 0 – input occupancy (same for all rows)
      col 1 – predicted occupancy after the push, with a cyan arrow showing
               where the push action falls IN GRID SPACE.  If the arrow
               misses the occupied region the optimizer has nothing to push.
      col 2 – pred * goal_score (reward density): WHERE reward comes from.
               Compare with col 3 to verify alignment with the goal.
      col 3 – goal score map (same for all rows, for reference).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    Nx, Ny    = occ_init_np.shape
    inp_base  = _heatmap_bgr(occ_init_np, (tile, tile))
    score_lbl = _stamp(_heatmap_bgr(score_np, (tile, tile), cv2.COLORMAP_HOT),
                       ['score map (goal)'])
    rows = []
    for rank, (occ_p, act, rew) in enumerate(
            zip(topk_occ_nps, topk_acts, topk_rews)):
        inp_l    = _stamp(inp_base.copy(),
                          ['input occ'] if rank == 0 else ['  (same)'])
        pred_img = _heatmap_bgr(occ_p, (tile, tile))
        # -- draw push arrow in grid space (cyan = start, tip = end) ----------
        # Grid dim 0 (cam-x) → image column; dim 1 (cam-y) → image row.
        if topk_start_grids is not None and topk_end_grids is not None:
            sx_g = float(topk_start_grids[rank, 0])
            sy_g = float(topk_start_grids[rank, 1])
            ex_g = float(topk_end_grids[rank, 0])
            ey_g = float(topk_end_grids[rank, 1])
            sc   = (tile - 1) / max(Nx - 1, 1)
            sp   = (int(np.clip(sx_g * sc, 0, tile - 1)),
                    int(np.clip(sy_g * sc, 0, tile - 1)))  # (col, row) for cv2
            ep   = (int(np.clip(ex_g * sc, 0, tile - 1)),
                    int(np.clip(ey_g * sc, 0, tile - 1)))
            cv2.arrowedLine(pred_img, sp, ep, (0, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.3)
            cv2.circle(pred_img, sp, 4, (0, 180, 255), -1, cv2.LINE_AA)
        pred_l = _stamp(pred_img,
                        [f'rank{rank+1}  r={rew:.3f}',
                         f's({act[0]:.2f},{act[1]:.2f})',
                         f'e({act[2]:.2f},{act[3]:.2f})'])
        # Use clamped occ for reward density — matches optimizer objective.
        rew_l  = _stamp(_heatmap_bgr(np.clip(occ_p, 0.0, 1.0) * score_np,
                                     (tile, tile), cv2.COLORMAP_HOT),
                        ['pred*score (clamped)'])
        rows.append(np.hstack([inp_l, pred_l, rew_l, score_lbl]))
    hdr = np.zeros((20, rows[0].shape[1], 3), dtype=np.uint8)
    cv2.putText(hdr, f'Step {step}  iter {it}  top-{len(rows)} candidates',
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.imwrite(path, np.vstack([hdr] + rows))


def _save_predicted_trajectory_video(
    path:        str,
    occ_init_np: np.ndarray,       # (Nx, Ny)
    occ_seq:     list,             # list of (Nx, Ny) per look-ahead step
    score_np:    np.ndarray,       # (Nx, Ny)
    step:        int,
    tile:        int = 320,
    fps:         int = 4,
    hold_seconds:  float = 2.0,
    final_seconds: float = 5.0,
) -> None:
    """Save a video showing the predicted occupancy trajectory for one MPC step."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    all_occ = [occ_init_np] + list(occ_seq)
    frames = []
    for t, occ in enumerate(all_occ):
        occ_img = _heatmap_bgr(occ, (tile, tile))
        rew_img = _heatmap_bgr(np.clip(occ, 0, 1) * score_np,
                               (tile, tile), cv2.COLORMAP_HOT)
        score_img = _heatmap_bgr(score_np, (tile, tile), cv2.COLORMAP_HOT)
        r = float((np.clip(occ, 0, 1) * score_np).sum())
        lbl = f'Step {step}  t={t} (init)' if t == 0 else f'Step {step}  t={t}  r={r:.3f}'
        occ_img   = _stamp(occ_img,   [lbl])
        rew_img   = _stamp(rew_img,   ['pred*score'])
        score_img = _stamp(score_img, ['goal score'])
        frames.append(np.hstack([occ_img, rew_img, score_img]))
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    hold  = max(1, int(fps * hold_seconds))
    final = max(1, int(fps * final_seconds))
    for idx, f in enumerate(frames):
        n = final if idx == len(frames) - 1 else hold
        for _ in range(n):
            writer.write(f)
    writer.release()


def _save_debug_winner(
    path:        str,
    occ_init_np: np.ndarray,   # (Nx, Ny)
    occ_pred_np: np.ndarray,   # (Nx, Ny)
    score_np:    np.ndarray,   # (Nx, Ny)
    best_act:    np.ndarray,   # (4,)
    predicted_r: float,        # best reward from optimizer
    actual_r:    float,        # real Eulerian reward after env.step
    step:        int,
    tile:        int = 240,
) -> None:
    """
    6-tile winner panel (2 rows x 3 cols):
      Row 0: [input occ | predicted occ | goal score map]
      Row 1: [input*score  | pred*score  | occ change (pred - input)]

    Reading guide:
      - Compare 'input*score' vs 'pred*score': is the reward density
        expected to increase?  If yes but act_r drops, the PUSH MODEL is
        the problem (it predicts material movement that doesn't happen).
      - Compare 'pred*score' shape with 'goal score map': if material is
        predicted to land in dark (low-score) regions, the SCORING is wrong.
      - 'act_gain' printed in the diff tile: negative means the environment
        moved material AWAY from the goal even though the MPC predicted gain.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    T       = tile
    cur_r   = float((occ_init_np * score_np).sum())
    pred_gain = predicted_r - cur_r
    act_gain  = actual_r    - cur_r

    r0 = np.hstack([
        _stamp(_heatmap_bgr(occ_init_np, (T, T)),
               ['input occ', f'r={cur_r:.3f}']),
        _stamp(_heatmap_bgr(occ_pred_np, (T, T)),
               ['predicted occ',
                f's({best_act[0]:.2f},{best_act[1]:.2f})',
                f'e({best_act[2]:.2f},{best_act[3]:.2f})']),
        _stamp(_heatmap_bgr(score_np, (T, T), cv2.COLORMAP_HOT),
               ['goal score map']),
    ])
    r1 = np.hstack([
        _stamp(_heatmap_bgr(occ_init_np * score_np, (T, T), cv2.COLORMAP_HOT),
               ['cur rew density', f'sum={cur_r:.3f}']),
        # Use clamped occ for reward density — matches optimizer objective.
        _stamp(_heatmap_bgr(np.clip(occ_pred_np, 0.0, 1.0) * score_np,
                            (T, T), cv2.COLORMAP_HOT),
               ['pred*score (clamped)',
                f'pred_r={predicted_r:.3f}',
                f'pred_gain={pred_gain:+.3f}']),
        _stamp(_diff_heatmap_bgr(occ_pred_np - occ_init_np, (T, T)),
               ['occ change (pred-input)',
                'green=gain  red=loss',
                f'act_r={actual_r:.3f}',
                f'act_gain={act_gain:+.3f}']),
    ])
    title_h = 24
    title = np.zeros((title_h, r0.shape[1], 3), dtype=np.uint8)
    cv2.putText(title,
                (f'Step {step} | pred_r={predicted_r:.4f}  act_r={actual_r:.4f} | '
                 f'pred_gain={pred_gain:+.3f}  act_gain={act_gain:+.3f}'),
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
    cv2.imwrite(path, np.vstack([title, r0, r1]))


# ── main entry point ──────────────────────────────────────────────────────────

def run_simple_mpc(
    env,                        # FlexEnv (already reset; set_save_render_mode
                                #   before calling if you want saved frames)
    model_dy,                   # EulerianModelWrapper
    subgoal: np.ndarray,        # (H, W)  float;  0 inside goal region
    cfg:     dict,              # full config dict (e.g. from load_simple_config)
    video_recorder=None,        # same optional arg as step_subgoal_ptcl
) -> dict:
    """
    Run simple gradient-descent MPC and return a result dict identical to
    env.step_subgoal_ptcl so that visualize_*.py scripts need no changes.

    Parameters
    ----------
    cfg : dict
        Must contain (at minimum):
          cfg['dataset']['wkspc_w']
          cfg['mpc']['n_mpc']
          cfg['mpc']['n_look_ahead']
          cfg['mpc']['n_sample']
          cfg['mpc']['n_update_iter']
          cfg['mpc']['gd']['lr']
    """
    assert isinstance(model_dy, EulerianModelWrapper), (
        "run_simple_mpc only supports EulerianModelWrapper. "
        "For the GNN model use env.step_subgoal_ptcl()."
    )

    # ── read config ────────────────────────────────────────────────────────────
    mpc_cfg       = cfg['mpc']
    n_mpc         = mpc_cfg['n_mpc']
    n_look_ahead  = mpc_cfg['n_look_ahead']
    n_sample      = mpc_cfg['n_sample']
    n_update_iter = mpc_cfg['n_update_iter']
    gd_lr         = mpc_cfg['gd']['lr']
    device        = 'cuda'

    debug_cfg          = mpc_cfg.get('debug', {})
    debug_enabled      = bool(debug_cfg.get('enabled', False))
    debug_dir          = debug_cfg.get('output_dir', 'outputs/simple_mpc_debug')
    debug_top_k        = int(debug_cfg.get('top_k', 5))
    debug_every        = int(debug_cfg.get('save_every_n_iters', 25))
    debug_bench        = bool(debug_cfg.get('benchmark_throughput', False))

    reward_cfg    = mpc_cfg.get('reward', {})
    empty_penalty = float(reward_cfg.get('empty_penalty', 0.0))

    wkspc_w = cfg['dataset']['wkspc_w']
    cam_params = env.get_cam_params()

    # ── workspace bounds for action clipping ──────────────────────────────────
    # Action format: [sx, sy, ex, ey] in world-2D coords.
    # Start point: full workspace.  End point: 15 % inset to avoid clipping.
    x0, x1 = -wkspc_w, wkspc_w
    y0, y1 = -wkspc_w, wkspc_w
    xd, yd = x1 - x0, y1 - y0
    act_lo = np.array([x0,              y0,              x0 + 0.15 * xd, y0 + 0.15 * yd],
                      dtype=np.float32)
    act_hi = np.array([x1,              y1,              x1 - 0.15 * xd, y1 - 0.15 * yd],
                      dtype=np.float32)

    print(f"simple_mpc: n_mpc={n_mpc}, n_look_ahead={n_look_ahead}, "
          f"n_sample={n_sample}, n_update_iter={n_update_iter}, "
          f"lr={gd_lr}, workspace=[-{wkspc_w},{wkspc_w}]^2")

    if debug_bench:
        benchmark_push_throughput(model_dy, wkspc_w=wkspc_w, device=device)

    # ── pre-compute goal score map once ────────────────────────────────────────
    # score_tensor(*grid_res): higher = closer to / inside goal.
    score_tensor = model_dy.prepare_goal_reward(
        subgoal, cam_params, device=device, empty_penalty=empty_penalty)
    score_np_cpu = score_tensor.cpu().numpy()   # (Nx, Ny) — kept on CPU for debug vis

    # -- score map sanity check -----------------------------------------------
    _goal_vox = int((score_np_cpu > 0).sum())
    print(f"  score map: {_goal_vox}/{score_np_cpu.size} voxels in goal region "
          f"({100*_goal_vox/score_np_cpu.size:.1f}%),  "
          f"range=[{score_np_cpu.min():.3f}, {score_np_cpu.max():.3f}]  "
          f"empty_penalty={empty_penalty}  shape={score_np_cpu.shape}")

    # ── preallocate return arrays ──────────────────────────────────────────────
    H, W        = env.screenHeight, env.screenWidth
    rewards     = np.zeros(n_mpc + 1,           dtype=np.float32)
    raw_obs     = np.zeros((n_mpc + 1, H, W, 5), dtype=np.float32)
    states      = []   # list[n_mpc+1] of (N, 3) camera-space point clouds
    actions     = np.zeros((n_mpc, 4),           dtype=np.float32)
    states_pred = []   # list[n_mpc] of (n_look_ahead, Nx, Ny)
    rew_means   = np.zeros((n_mpc, 1, n_update_iter), dtype=np.float32)
    rew_stds    = np.zeros((n_mpc, 1, n_update_iter), dtype=np.float32)
    total_time = rollout_time = optim_time = 0.0
    iter_num   = 0

    # ── helper: rendered obs → foreground pts + Eulerian reward ───────────────
    def _pts_and_reward(obs: np.ndarray):
        """
        Convert a 5-channel rendered observation to:
          pts   : (N, 3) float32  foreground point cloud in camera coords
          r_val : float           Eulerian reward  (occ · score_tensor).sum()
        """
        depth = obs[..., -1] / env.global_scale                      # normalised depth
        pts   = depth2fgpcd(depth, depth < 0.599 / 0.8, cam_params)  # (N, 3)
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().to(device).unsqueeze(0)  # (1, N, 3)
            occ   = model_dy.initial_occ_from_particles(pts_t)              # (1, Nx, Ny)
            r_val = float((occ[0] * score_tensor).sum().item())
        return pts, r_val

    # ── t = 0: render and seed initial state ──────────────────────────────────
    obs_cur    = env.render()
    raw_obs[0] = obs_cur
    pts0, r0   = _pts_and_reward(obs_cur)
    rewards[0] = r0
    states.append(pts0)

    print(f"  initial reward: {r0:.4f}")

    # ── MPC loop ──────────────────────────────────────────────────────────────
    for i in range(n_mpc):
        t_step_start = time.time()
        n_ahead = min(n_look_ahead, n_mpc - i)

        pts_i    = states[i]   # (N, 3) camera-space foreground points
        pts_t    = torch.from_numpy(pts_i).float().to(device).unsqueeze(0)   # (1, N, 3)
        occ_init = model_dy.initial_occ_from_particles(pts_t).detach()        # (1, Nx, Ny)

        # -- material coverage + centroid (are there any voxels to push?) -----
        _occ_01  = occ_init[0].cpu().numpy()
        _mask_01 = _occ_01 > 0.5
        _cov_pct = _mask_01.mean() * 100
        if _mask_01.any():
            _ci, _cj = np.where(_mask_01)
            print(f"  step {i+1}: material {_cov_pct:.1f}% of grid, "
                  f"centroid voxel ({_ci.mean():.1f}, {_cj.mean():.1f}) "
                  f"of {_occ_01.shape}  "
                  f"[action range: x_grid 0-{_occ_01.shape[0]-1}, "
                  f"y_grid 0-{_occ_01.shape[1]-1}]")
        else:
            print(f"  step {i+1}: WARNING — occupancy grid is entirely empty! "
                  "Check depth threshold and grid bounds.")

        debug_step_dir = os.path.join(debug_dir, f'step_{i+1:03d}')

        # -- random independent action initialisation  -------------------------
        # KEY FIX vs. existing code: draw n_sample DIFFERENT starting points.
        # np.random.uniform broadcasts act_lo/act_hi over (n_sample, n_look_ahead).
        act_np   = np.random.uniform(
            act_lo, act_hi, (n_sample, n_ahead, 4)).astype(np.float32)
        act_seqs = torch.tensor(act_np, device=device, requires_grad=True)
        optimizer = optim.Adam([act_seqs], lr=gd_lr, betas=(0.9, 0.999))

        best_reward = -float('inf')
        best_act    = act_seqs[0].detach().clone()   # (n_ahead, 4)

        t_rollout = 0.0
        t_optim   = 0.0

        for it in range(n_update_iter):
            # -- rollout n_sample candidates in grid space --------------------
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

            occ = occ_init.expand(n_sample, *model_dy.grid_res).clone()
            for s in range(n_ahead):
                occ = model_dy.predict_one_step_occ(occ, act_seqs[:, s, :])

            ev1.record()
            torch.cuda.synchronize()
            t_rollout += ev0.elapsed_time(ev1)

            # -- reward: element-wise product with goal score map -------------
            #
            # EXPERIMENTAL FIX — per-voxel clamp to [0, 1] before scoring.
            #
            # The push model (differentiable_push_splat) does not clamp its
            # output: all swept mass piles onto the few endpoint voxels,
            # producing values >> 1.  Without clamping the dot product
            #   (occ * score_tensor).sum()
            # is dominated by those voxels, so the optimizer can obtain
            # arbitrarily high predicted reward by concentrating mass at a
            # single high-score voxel — regardless of whether the action
            # contacts real material.
            #
            # Clamping per-voxel to [0, 1] makes this reward comparable to
            # the environment reward, which comes from
            # initial_occ_from_particles() and is already clamped via
            # scatter_add + clamp.  This removes the degenerate incentive and
            # ensures the optimizer is maximizing the same quantity that is
            # measured in the real environment.
            #
            # Limitation: clamp() has zero gradient where occ > 1, so the
            # optimizer cannot improve through the bilinear-deposit gradient
            # once a destination voxel saturates.  It can still improve
            # through the swept-mask gradient (i.e., which pixels are swept
            # and where the action starts/ends).
            #
            # A previous attempt used global mass normalisation
            # (occ * init_mass / pred_mass), but that created a perverse
            # gradient: because pred_mass is differentiable, the optimizer
            # was incentivised to push mass OFF the grid boundary (reducing
            # pred_mass inflates the normalisation factor).
            #
            # TODO: enforce mass conservation inside the push model itself
            # (e.g. a physics-based advection kernel that is exactly
            # conservative) so clamping is never needed.
            rew_seqs = (occ.clamp(0.0, 1.0) * score_tensor).view(n_sample, -1).sum(dim=-1)   # (n_sample,)
            best_i   = int(rew_seqs.argmax())
            if rew_seqs[best_i].item() > best_reward:
                best_reward = rew_seqs[best_i].item()
                best_act    = act_seqs[best_i].detach().clone()

            rew_means[i, 0, it] = float(rew_seqs.mean().item())
            rew_stds[i,  0, it] = float(rew_seqs.std().item())

            # -- debug: snapshot top-K candidates (before gradient step) ------
            if debug_enabled and (
                    it == 0
                    or (it + 1) % debug_every == 0
                    or it == n_update_iter - 1):
                _k = min(debug_top_k, n_sample)
                topk_idx = rew_seqs.detach().cpu().numpy().argsort()[::-1][:_k].copy()
                # Compute grid-space coords for the topk actions (for arrow drawing)
                _topk_act_t = act_seqs.detach()[topk_idx, 0, :]   # (K, 4) world 2D
                _s_cam, _e_cam = _action_to_cam_3d(
                    _topk_act_t, model_dy.cam_extrinsic, model_dy.global_scale)
                _sg = model_dy._cam3d_to_grid(_s_cam).cpu().numpy()   # (K, 3)
                _eg = model_dy._cam3d_to_grid(_e_cam).cpu().numpy()   # (K, 3)
                _save_debug_candidates(
                    path=os.path.join(debug_step_dir,
                                      f'iter_{it:04d}_top{_k}.png'),
                    occ_init_np=occ_init[0].cpu().numpy(),
                    topk_occ_nps=[occ.detach().cpu().numpy()[j] for j in topk_idx],
                    topk_acts=act_seqs.detach().cpu().numpy()[topk_idx, 0, :],
                    topk_rews=rew_seqs.detach().cpu().numpy()[topk_idx],
                    score_np=score_np_cpu,
                    step=i + 1, it=it,
                    topk_start_grids=_sg[:, :2],
                    topk_end_grids=_eg[:, :2],
                )

            # -- gradient step ------------------------------------------------
            ev2 = torch.cuda.Event(enable_timing=True)
            ev3 = torch.cuda.Event(enable_timing=True)
            ev2.record()

            loss = -rew_seqs.sum()
            optimizer.zero_grad()
            loss.backward()
            # -- gradient norm: near-zero means the push misses the material --
            if debug_enabled and (
                    it == 0
                    or (it + 1) % debug_every == 0
                    or it == n_update_iter - 1):
                _gn = (act_seqs.grad.abs().mean().item()
                       if act_seqs.grad is not None else float('nan'))
                print(f"    iter {it+1:4d}: best={best_reward:.4f}  "
                      f"mean={rew_seqs.mean().item():.4f}  "
                      f"std={rew_seqs.std().item():.4f}  "
                      f"grad={_gn:.5f}")
            optimizer.step()

            # clip all candidates into workspace bounds (dimension-by-dimension
            # to avoid any tensor-bound broadcasting issues across pyTorch versions)
            with torch.no_grad():
                for d in range(4):
                    act_seqs.data[:, :, d].clamp_(float(act_lo[d]), float(act_hi[d]))

            ev3.record()
            torch.cuda.synchronize()
            t_optim += ev2.elapsed_time(ev3)

        rollout_time += t_rollout / 1000.0   # ms → s
        optim_time   += t_optim  / 1000.0
        iter_num     += n_update_iter

        # -- rollout best action sequence for states_pred ---------------------
        with torch.no_grad():
            occ_seq = []
            occ_run = occ_init.clone()   # (1, Nx, Ny)
            best_t  = best_act.unsqueeze(0)   # (1, n_ahead, 4)
            for s in range(n_ahead):
                occ_run = model_dy.predict_one_step_occ(occ_run, best_t[:, s, :])
                occ_seq.append(occ_run[0].cpu().numpy())   # (Nx, Ny)
        states_pred.append(np.stack(occ_seq))   # (n_look_ahead, Nx, Ny)

        # -- save predicted trajectory video ---------------------------------
        if debug_enabled:
            _save_predicted_trajectory_video(
                path=os.path.join(debug_step_dir, 'predicted_trajectory.avi'),
                occ_init_np=occ_init[0].cpu().numpy(),
                occ_seq=occ_seq,
                score_np=score_np_cpu,
                step=i + 1,
            )

        best_action_np = best_act[0].cpu().numpy()   # (4,) — first look-ahead

        delta_r = rew_means[i, 0, -1] - rew_means[i, 0, 0]
        print(f"  step {i+1:3d}/{n_mpc}  best_rew={best_reward:.4f}  "
              f"Δr_optim={delta_r:.4f}  std_last={rew_stds[i,0,-1]:.4f}  "
              f"act={np.round(best_action_np, 2)}")

        # -- execute best action in simulator ---------------------------------
        obs_next = env.step(best_action_np, video_recorder=video_recorder)
        if obs_next is None:
            print("WARNING: simulation exploded — terminating MPC early")
            # Zero-fill remaining entries so output dict shape is consistent
            for j in range(i + 1, n_mpc):
                states.append(states[-1])
                states_pred.append(states_pred[-1])
            break

        raw_obs[i + 1] = obs_next
        pts_next, r_next = _pts_and_reward(obs_next)
        states.append(pts_next)
        actions[i]     = best_action_np
        rewards[i + 1] = r_next

        # -- debug: winner panel (predicted vs actual, using Eulerian reward) --
        if debug_enabled:
            _save_debug_winner(
                path=os.path.join(debug_step_dir, 'winner.png'),
                occ_init_np=occ_init[0].cpu().numpy(),
                occ_pred_np=states_pred[-1][0],
                score_np=score_np_cpu,
                best_act=best_action_np,
                predicted_r=best_reward,
                actual_r=float(r_next),
                step=i + 1,
            )

        total_time += time.time() - t_step_start
        print(f"          env reward after step: {r_next:.4f}  "
              f"(predicted: {best_reward:.4f}  gap: {float(r_next)-best_reward:+.4f})")

    print(f"\nsimple_mpc done: r_init={rewards[0]:.4f}  r_final={rewards[n_mpc]:.4f}  "
          f"total_time={total_time:.1f}s")

    return {
        'rewards':          rewards,
        'raw_obs':          raw_obs,
        'states':           states,
        'actions':          actions,
        'states_pred':      states_pred,
        'rew_means':        rew_means,
        'rew_stds':         rew_stds,
        'total_time':       total_time,
        'rollout_time':     rollout_time,
        'optim_time':       optim_time,
        'iter_num':         iter_num,
        'particle_den_seq': [],   # unused; present for API compatibility
    }
