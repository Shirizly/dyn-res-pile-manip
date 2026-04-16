"""Core gradient-descent MPC loop for EulerianModelWrapper.

This module contains the main MPC entry point ``run_simple_mpc`` and the
config loader ``load_simple_config``.  Debug visualisation and benchmarking
live in sibling modules (``debug_vis`` and ``benchmark``).
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim

from utils import load_yaml, depth2fgpcd
from model.eulerian_wrapper import EulerianModelWrapper, _action_to_cam_3d

from simple_mpc.benchmark import benchmark_push_throughput
from simple_mpc.debug_vis import (
    save_debug_candidates,
    save_predicted_trajectory_video,
    save_debug_winner,
)


# ── public helpers ────────────────────────────────────────────────────────────

def load_simple_config(path: str = 'config/mpc/config_simple.yaml') -> dict:
    """Load the simple-MPC config file."""
    return load_yaml(path)


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
        # np.random.uniform broadcasts act_lo/act_hi over (n_sample, n_ahead).
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
                save_debug_candidates(
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
        states_pred.append(np.stack(occ_seq))   # (n_ahead, Nx, Ny)

        # -- save predicted trajectory video ---------------------------------
        if debug_enabled:
            save_predicted_trajectory_video(
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
            save_debug_winner(
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
