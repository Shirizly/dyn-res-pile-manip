"""
EulerianModelWrapper
====================
Adapts a user-supplied Eulerian (occupancy-field) dynamics model so that it can
be dropped into the existing particle-based MPC pipeline in place of
PropNetDiffDenModel.

Pipeline position
-----------------
The MPC planner calls the model inside ``PlannerGD.ptcl_model_rollout``, which
will dispatch to this wrapper whenever ``isinstance(model_dy, EulerianModelWrapper)``
is true.  The planner passes the raw 4-D action ``[sx, sy, ex, ey]`` directly to
``predict_one_step``, so the wrapper never has to reconstruct it from s_delta.

User model contract
-------------------
The model passed as ``user_model`` must implement::

    forward(occ_grid, action_start, action_end) -> occ_grid_pred

where:
    occ_grid      : torch.Tensor  (B, *grid_shape)  – occupancy field [0..1]
    action_start  : torch.Tensor  (B, 3)            – tool start in *grid coordinates*
                                                       (voxel indices, float)
    action_end    : torch.Tensor  (B, 3)            – tool end   in *grid coordinates*
    returns:
    occ_grid_pred : torch.Tensor  (B, *grid_shape)  – predicted occupancy field [0..1]

Grid coordinates are defined as (ix, iy, iz) in [0, grid_res-1] along each axis,
mapping linearly from the supplied ``grid_bounds``.  For a 2-D grid supply a
``grid_res`` of length 2; the wrapper will operate in the x-y plane of normalized
camera space (which corresponds to the horizontal table plane for a top-down camera).

Coordinate conventions
-----------------------
*Normalized camera space* – the coordinate system used for particle positions
throughout the existing code – is obtained by:

1. Unprojecting each depth pixel with the pinhole camera model (``depth2fgpcd``).
2. Dividing by ``global_scale``.

The x-z axes span the horizontal plane (the table surface); y points upward.
For the default top-down camera the table is at approximately
``z_cam ≈ 0.745`` in these units.

The 4-D MPC action ``[sx, sy, ex, ey]`` lives in *world 2-D* coordinates
(the horizontal plane, range ≈ [-wkspc_w, wkspc_w]).  The wrapper converts it
to normalized camera 3-D using the same ``world2cam`` logic as the planner's
``gen_s_delta``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _particles_to_occupancy(
    s_cur: torch.Tensor,                  # (B, N, 3) normalized cam coords
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, ...],
    sigma: float = 0.0,                   # >0 → soft Gaussian splat; 0 → hard voxel
) -> torch.Tensor:
    """
    Convert a batch of particle point-clouds to an occupancy grid.

    Returns
    -------
    occ : (B, *grid_res)  float32, values in [0, 1].
    """
    B, N, _ = s_cur.shape
    device = s_cur.device
    ndim = len(grid_res)

    # Axis names and order: x (grid dim 0), then y / z depending on ndim
    axes = _get_axes(ndim)
    lo  = torch.tensor([grid_bounds[f'{a}_min'] for a in axes], device=device, dtype=s_cur.dtype)
    hi  = torch.tensor([grid_bounds[f'{a}_max'] for a in axes], device=device, dtype=s_cur.dtype)
    res = torch.tensor(grid_res, device=device, dtype=s_cur.dtype)

    # Compute the axis index (dim) in the 3-D particle coordinate for each grid axis
    axis_idx = torch.tensor(_axis_indices(axes), device=device, dtype=torch.long)

    # Particle coords projected onto grid axes  (B, N, ndim)
    pts = s_cur[..., axis_idx]  # (B, N, ndim)

    # Normalise to [0, grid_res-1]
    pts_norm = (pts - lo) / (hi - lo) * (res - 1)  # (B, N, ndim)

    occ = torch.zeros([B] + list(grid_res), device=device, dtype=torch.float32)

    if sigma <= 0.0:
        # Hard voxel: scatter-add a '1' to each occupied cell
        idx = pts_norm.round().long().clamp(
            torch.zeros(ndim, device=device, dtype=torch.long),
            (res.long() - 1))
        for b in range(B):
            flat_idx = _ravel_idx(idx[b], grid_res)  # (N,)
            occ[b].view(-1).scatter_add_(0, flat_idx, torch.ones(N, device=device))
        # Clamp to [0, 1]
        occ = occ.clamp(0.0, 1.0)
    else:
        # Soft Gaussian splat – expensive but differentiable
        # Build grid of voxel centres  (*grid_res, ndim)
        grid_pts = _make_grid_coords(grid_res, device)  # (*grid_res, ndim)  in voxel idx
        for b in range(B):
            # pts_norm[b]: (N, ndim), grid_pts: (*grid_res, ndim)
            # dist2: (N, *grid_res)
            diff = pts_norm[b].unsqueeze(1).unsqueeze(1) - grid_pts.unsqueeze(0)  # (N, *grid_res, ndim)
            dist2 = (diff ** 2).sum(-1)             # (N, *grid_res)
            contrib = torch.exp(-dist2 / (2 * sigma ** 2)).sum(0)  # (*grid_res)
            occ[b] = contrib.clamp(0.0, 1.0)

    return occ  # (B, *grid_res)


def _occupancy_to_particles(
    occ: torch.Tensor,         # (B, *grid_res)  float, in [0, 1]
    n_particles: int,
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, ...],
    thresh: float = 0.5,
) -> torch.Tensor:
    """
    Convert occupancy grids back to a fixed-size particle set via FPS.

    Steps
    -----
    1. Threshold at ``thresh`` to get occupied voxel centres.
    2. Farthest-point-sample ``n_particles`` from those centres.
       If fewer occupied voxels than n_particles, repeat voxel centres.
    3. Return particles in normalised camera coordinates.

    Returns
    -------
    s_pred : (B, n_particles, 3) float32 – in normalized camera coords.
             The 'missing' axis (y for 2-D grids) is set to the mid-point of
             its bound.
    """
    B = occ.shape[0]
    ndim = len(grid_res)
    device = occ.device
    dtype = occ.dtype

    axes = _get_axes(ndim)
    lo  = np.array([grid_bounds[f'{a}_min'] for a in axes])
    hi  = np.array([grid_bounds[f'{a}_max'] for a in axes])
    res = np.array(grid_res)
    voxel_size = (hi - lo) / (res - 1)  # (ndim,)

    # Depth (z) fill value for 2-D grids: all particles sit at ~constant table depth.
    if ndim == 2:
        z_mid = 0.5 * (grid_bounds.get('z_min', 0.745) + grid_bounds.get('z_max', 0.745))

    s_pred = torch.zeros(B, n_particles, 3, device=device, dtype=dtype)

    for b in range(B):
        occ_b = occ[b]  # (*grid_res)

        # Voxel indices of occupied cells
        occ_np = occ_b.detach().cpu().numpy()
        mask = occ_np >= thresh
        occupied_idx = np.argwhere(mask)  # (M, ndim)

        if occupied_idx.shape[0] == 0:
            # Fallback: use all voxel centres
            occupied_idx = np.argwhere(np.ones_like(occ_np, dtype=bool))

        # Convert voxel indices → world coordinates
        voxel_coords = lo + occupied_idx * voxel_size  # (M, ndim)

        # Reconstruct 3-D from 2-D / 3-D grid
        pts_3d = _to_3d(voxel_coords, axes, z_mid if ndim == 2 else None)  # (M, 3)

        # FPS to n_particles
        pts_3d = _fps_np(pts_3d, n_particles)  # (n_particles, 3)

        s_pred[b] = torch.from_numpy(pts_3d).to(device=device, dtype=dtype)

    return s_pred


def _action_to_cam_3d(
    action: torch.Tensor,      # (B, 4) [sx, sy, ex, ey] in world 2-D
    cam_extrinsic: np.ndarray, # (4, 4) OpenGL view matrix
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 4-D planar action to 3-D normalized camera-space start/end.

    Mirrors the logic of ``PlannerGD.world2cam`` and ``gen_s_delta``.

    Returns
    -------
    s_3d_cam, e_3d_cam : each (B, 3)
    """
    device = action.device
    dtype  = action.dtype
    B = action.shape[0]

    h = torch.zeros(B, 1, device=device, dtype=dtype)  # height = 0 (table plane)

    s_world = torch.cat([action[:, 0:1], h, -action[:, 1:2]], dim=1)  # (B, 3)
    e_world = torch.cat([action[:, 2:3], h, -action[:, 3:4]], dim=1)  # (B, 3)

    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0,-1, 0, 0],
                                [0, 0,-1, 0],
                                [0, 0, 0, 1]], dtype=np.float64)
    opencv_T_world     = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    opencv_T_world_inv = np.linalg.inv(opencv_T_world)
    M = torch.tensor(opencv_T_world_inv, device=device, dtype=dtype)  # (4, 4)

    def _transform(pts):
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        homog = torch.cat([pts, ones], dim=1)  # (B, 4)
        out = (M @ homog.T).T                  # (B, 4)
        return out[:, :3] / global_scale

    return _transform(s_world), _transform(e_world)


# ----- small numpy FPS (avoids importing the utils module) -----------------

def _fps_np(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sample *n* points from pts (M, d).  Repeats if M < n."""
    M = pts.shape[0]
    if M == 0:
        return np.zeros((n, pts.shape[1]), dtype=pts.dtype)
    if M <= n:
        reps = int(np.ceil(n / M))
        pts = np.tile(pts, (reps, 1))
    rand_idx = np.random.randint(pts.shape[0])
    selected = [pts[rand_idx]]
    dist = np.linalg.norm(pts - selected[0], axis=1)
    while len(selected) < n:
        farthest = pts[dist.argmax()]
        selected.append(farthest)
        dist = np.minimum(dist, np.linalg.norm(pts - farthest, axis=1))
    return np.stack(selected[:n])


# ----- coordinate bookkeeping helpers --------------------------------------

def _get_axes(ndim: int):
    """Return the particle-coordinate axes used by the grid.

    For a top-down camera ``depth2fgpcd`` places:
      dim 0 = camera x  (horizontal, proportional to pixel column)
      dim 1 = camera y  (horizontal, proportional to pixel row)
      dim 2 = camera z  (depth, ≈ constant 0.75 for all table particles)
    So a 2-D top-down grid should span dims 0 and 1, not 0 and 2.
    """
    if ndim == 2:
        return ('x', 'y')   # camera x and y span the horizontal table plane
    elif ndim == 3:
        return ('x', 'y', 'z')
    else:
        raise ValueError(f"grid_res must have 2 or 3 elements, got {ndim}")


def _axis_indices(axes):
    """Map axis names to indices in the 3-D particle coordinate vector."""
    mapping = {'x': 0, 'y': 1, 'z': 2}
    return [mapping[a] for a in axes]


def _to_3d(voxel_coords: np.ndarray, axes, depth_fill: float | None) -> np.ndarray:
    """Promote a (M, ndim) grid-coordinate array to (M, 3).

    For 2-D grids ``depth_fill`` is written into the depth axis (dim 2, camera z)
    so that returned points can be used as normalized camera-space coordinates.
    """
    M = voxel_coords.shape[0]
    pts = np.zeros((M, 3), dtype=voxel_coords.dtype)
    for k, a in enumerate(axes):
        idx = {'x': 0, 'y': 1, 'z': 2}[a]
        pts[:, idx] = voxel_coords[:, k]
    if len(axes) == 2 and depth_fill is not None:
        pts[:, 2] = depth_fill   # fill the depth (camera z) axis
    return pts


def _ravel_idx(idx: torch.Tensor, grid_res: Tuple[int, ...]) -> torch.Tensor:
    """Ravel multi-dim indices (N, ndim) to flat index (N,) for a C-order grid."""
    strides = torch.tensor(
        [int(np.prod(grid_res[k+1:])) for k in range(len(grid_res))],
        device=idx.device, dtype=torch.long)
    return (idx * strides).sum(-1)


def _make_grid_coords(grid_res: Tuple[int, ...], device) -> torch.Tensor:
    """Return a grid of voxel-index coordinates (*grid_res, ndim)."""
    ranges = [torch.arange(r, device=device, dtype=torch.float32) for r in grid_res]
    mesh = torch.meshgrid(*ranges, indexing='ij')  # each: (*grid_res)
    return torch.stack(mesh, dim=-1)               # (*grid_res, ndim)


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class EulerianModelWrapper(nn.Module):
    """
    Wraps a user-supplied Eulerian dynamics model for the particle-based MPC
    pipeline.

    Parameters
    ----------
    user_model : nn.Module
        Must implement ``forward(occ_grid, action_start, action_end) -> occ_grid_pred``
        as documented at the top of this file.
    grid_bounds : dict
        World-extent of the occupancy grid in *normalized camera coordinates*.
        Required keys for a 2-D grid: ``x_min, x_max, z_min, z_max``.
        Also requires ``y_min, y_max`` (used only to reconstruct the y coordinate
        when converting back to 3-D particles).
        For a 3-D grid all six keys must be present.
        Use ``EulerianModelWrapper.default_bounds(config)`` to get a sensible
        default derived from the environment config.
    grid_res : tuple[int, ...]
        Number of voxels along each grid axis, e.g. ``(64, 64)`` (2-D) or
        ``(32, 32, 16)`` (3-D).
    cam_extrinsic : np.ndarray  (4, 4)
        OpenGL view matrix from ``env.get_cam_extrinsics()``.
    global_scale : float
        From ``config['dataset']['global_scale']``.
    splat_sigma : float, optional
        If > 0, use a Gaussian splat when converting particles → occupancy
        (differentiable but slower).  Default 0 → hard voxel assignment.
    occ_threshold : float, optional
        Occupancy threshold used when converting the predicted grid back to
        particles.  Default 0.5.
    """

    def __init__(
        self,
        user_model: nn.Module,
        grid_bounds: Dict[str, float],
        grid_res: Sequence[int],
        cam_extrinsic: np.ndarray,
        global_scale: float,
        splat_sigma: float = 0.0,
        occ_threshold: float = 0.5,
    ):
        super().__init__()
        self.user_model    = user_model
        self.grid_bounds   = dict(grid_bounds)
        self.grid_res      = tuple(grid_res)
        self.cam_extrinsic = cam_extrinsic.copy()
        self.global_scale  = float(global_scale)
        self.splat_sigma   = splat_sigma
        self.occ_threshold = occ_threshold

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def default_bounds(config: dict) -> Dict[str, float]:
        """
        Return grid bounds that cover the workspace in normalized camera coords.

        The workspace half-width ``wkspc_w`` is used for x and z.  The y
        (vertical) extent is set to a thin slab around the table surface.

        These are estimates; adjust for tighter or looser coverage.
        """
        gs  = config['dataset']['global_scale']
        w   = config['dataset']['wkspc_w']   # world units
        w_n = w / gs                          # normalized

        # Camera is straight above; table depth ≈ cam_height /gs ≈ 0.75
        # Objects sit at depth slightly below that.
        z_table = 6.0 / 8.0          # cam_height / gs  (for cam_idx=0)
        z_margin = 0.05

        # x, y cover the horizontal extent of the workspace in normalized camera space.
        # z covers the narrow depth band where table-surface particles live.
        # These are estimates; calibrate with EulerianModelWrapper.calibrate_from_particles()
        # if the default bounds do not fully contain your observed workspace.
        return {
            'x_min': -w_n * 1.2, 'x_max':  w_n * 1.2,
            'y_min': -w_n * 1.2, 'y_max':  w_n * 1.2,
            'z_min': z_table - z_margin, 'z_max': z_table + z_margin,  # depth slab
        }

    # ------------------------------------------------------------------
    # Core interface – called by ptcl_model_rollout
    # ------------------------------------------------------------------

    def predict_one_step(
        self,
        s_cur: torch.Tensor,    # (B, N, 3)  particles in normalized cam coords
        action: torch.Tensor,   # (B, 4)     [sx, sy, ex, ey] in world 2-D
    ) -> torch.Tensor:
        """
        Predict the next particle state using the Eulerian model.

        Steps
        -----
        1. Convert particles to an occupancy grid  (B, *grid_res).
        2. Convert raw action [sx, sy, ex, ey] to 3-D camera-space start/end,
           then to grid-coordinate start/end (B, 3).
        3. Run ``user_model(occ, start_grid, end_grid)`` → predicted occupancy.
        4. Convert the predicted occupancy back to (B, N, 3) particles.

        Returns
        -------
        s_pred : (B, N, 3)  predicted particle positions in normalized cam coords.
        """
        B, N, _ = s_cur.shape
        occ = _particles_to_occupancy(
            s_cur, self.grid_bounds, self.grid_res, sigma=self.splat_sigma)
        occ_pred = self.predict_one_step_occ(occ, action)
        return _occupancy_to_particles(
            occ_pred, n_particles=N,
            grid_bounds=self.grid_bounds, grid_res=self.grid_res,
            thresh=self.occ_threshold)

    def predict_one_step_occ(
        self,
        occ_cur: torch.Tensor,   # (B, *grid_res)  current occupancy field
        action:  torch.Tensor,   # (B, 4)           [sx, sy, ex, ey] world 2-D
    ) -> torch.Tensor:
        """
        Single prediction step that stays entirely in occupancy space.

        This is the method used by the Eulerian MPC optimizer for multi-step
        rollouts.  It avoids the lossy ``occ → FPS-particles → occ``
        round-trip that ``predict_one_step`` performs for interface
        compatibility.

        The gradient path is:
            act_seqs_tensor → action_to_cam_3d → cam3d_to_grid
                            → user_model → occ_pred → reward

        Returns
        -------
        occ_pred : (B, *grid_res)  next predicted occupancy field.
        """
        s_3d_cam, e_3d_cam = _action_to_cam_3d(
            action, self.cam_extrinsic, self.global_scale)
        start_grid = self._cam3d_to_grid(s_3d_cam)  # (B, 3)
        end_grid   = self._cam3d_to_grid(e_3d_cam)  # (B, 3)
        return self.user_model(occ_cur, start_grid, end_grid)

    def initial_occ_from_particles(
        self,
        s_cur: torch.Tensor,   # (B, N, 3)  particles in normalized cam coords
    ) -> torch.Tensor:
        """
        Convert a batch of particle observations to occupancy grids.

        Call this once at the start of each MPC step to seed the Eulerian
        optimizer from the current environment observation.

        Returns
        -------
        occ : (B, *grid_res)  float32, detached (not part of the grad graph).
        """
        with torch.no_grad():
            return _particles_to_occupancy(
                s_cur, self.grid_bounds, self.grid_res, sigma=self.splat_sigma)

    def prepare_goal_reward(
        self,
        subgoal: np.ndarray,   # (H, W)  0 = object should be here
        cam_params,            # [fx, fy, cx, cy]  from env.get_cam_params()
        device: str = 'cuda',
        empty_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        Precompute a (*grid_res) score tensor from the pixel-space subgoal.

        Call this **once** before the MPC optimizer loop; the tensor is then
        used inside the loop as a fixed reward landscape:

            reward_per_sample = (occ_pred * score_tensor).sum()

        Parameters
        ----------
        empty_penalty : float, optional
            Controls the reward assigned to voxels that should be empty.

            0.0 (default, backward-compatible)
                ``score -= score.min()`` is applied so all values are ≥ 0.
                Empty voxels carry 0 reward — the optimizer has no incentive
                to move material *away* from non-goal regions.

            > 0.0
                ``dist_from_goal`` is normalized to [0, 1] and subtracted
                with weight ``empty_penalty``.  Score ∈ [-empty_penalty, +1]:
                goal voxels ≈ +1, the farthest empty voxel ≈ -empty_penalty.
                The optimizer is now penalized for placing material in regions
                that should be empty, which discourages minimalist no-op
                actions (pushing material nowhere useful still hurts).

        Returns
        -------
        score : torch.Tensor (*grid_res) on ``device``, higher = better.
        """
        from scipy.ndimage import distance_transform_edt

        occ_goal = self.subgoal_mask_to_occupancy(subgoal, cam_params)  # (1, *grid_res)
        occ_goal_np = occ_goal[0].numpy()                               # (*grid_res)

        occupied = (occ_goal_np > 0.5)
        # Zero on occupied voxels, positive elsewhere.
        dist_from_goal = distance_transform_edt(~occupied).astype(np.float32)

        if empty_penalty > 0.0:
            # Normalize distance to [0, 1] so the penalty is bounded regardless
            # of grid size.  Score: goal ≈ +1, farthest empty ≈ -empty_penalty.
            max_dist = dist_from_goal.max()
            dist_norm = dist_from_goal / max_dist if max_dist > 0 else dist_from_goal
            score = occ_goal_np.astype(np.float32) - empty_penalty * dist_norm
        else:
            score = occ_goal_np.astype(np.float32) - dist_from_goal
            score -= score.min()   # shift: 0 = farthest from goal, max = at goal

        return torch.from_numpy(score).to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def obs_to_occupancy(
        self,
        obs: np.ndarray,
        cam_params,
        depth_thresh: float = 0.599 / 0.8,
    ) -> torch.Tensor:
        """
        Convert a raw simulation observation (H, W, 5) directly to an
        occupancy grid, bypassing the particle intermediate representation.

        This is useful for computing the *goal* occupancy if the subgoal is
        given as a binary pixel mask (the existing ``subgoal`` array has value
        0 where the object should be and 1 elsewhere).

        Parameters
        ----------
        obs : np.ndarray (H, W, 5)
            Raw observation from ``env.render()`` (or a goal image similarly
            structured).
        cam_params : [fx, fy, cx, cy]
            From ``env.get_cam_params()``.
        depth_thresh : float
            Normalized depth threshold below which pixels are foreground.

        Returns
        -------
        occ : torch.Tensor (1, *grid_res)
        """
        from utils import depth2fgpcd
        depth = obs[..., -1] / self.global_scale
        mask  = depth < depth_thresh
        fgpcd = depth2fgpcd(depth, mask, cam_params)   # (M, 3)
        s_cur = torch.from_numpy(fgpcd).float().unsqueeze(0)  # (1, M, 3)
        return _particles_to_occupancy(s_cur, self.grid_bounds, self.grid_res,
                                       sigma=self.splat_sigma)

    def subgoal_mask_to_occupancy(
        self,
        subgoal: np.ndarray,   # (H, W)  0 = object should be here, 1 = background
        cam_params,
    ) -> torch.Tensor:
        """
        Convert the binary pixel-space subgoal mask to a goal occupancy grid.

        The convention in the existing code is that ``subgoal < 0.5`` marks
        the target region.  This method back-projects those pixels into 3-D
        at a fixed depth (the mean of ``z_min`` and ``z_max`` in grid_bounds)
        and then voxelizes them.

        Returns
        -------
        occ_goal : torch.Tensor (1, *grid_res)
        """
        H, W = subgoal.shape
        fx, fy, cx, cy = cam_params

        # Pixels where the goal region is active
        ys, xs = np.where(subgoal < 0.5)   # pixel rows and columns

        # Use a fixed depth equal to the z-extent midpoint
        z_mid = 0.5 * (self.grid_bounds.get('z_min', 0.7)
                       + self.grid_bounds.get('z_max', 0.8))
        depth_val = z_mid * self.global_scale   # un-normalize for projection

        # Mirrors depth2fgpcd: particle.x = (col - cx) * depth_norm / fx
        X = (xs - cx) * z_mid / fx
        Y = (ys - cy) * z_mid / fy
        Z = np.full_like(X, z_mid, dtype=np.float32)
        fgpcd = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (M, 3)

        if fgpcd.shape[0] == 0:
            return torch.zeros([1] + list(self.grid_res))

        s_goal = torch.from_numpy(fgpcd).float().unsqueeze(0)
        return _particles_to_occupancy(s_goal, self.grid_bounds, self.grid_res,
                                       sigma=self.splat_sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cam3d_to_grid(self, pts_cam: torch.Tensor) -> torch.Tensor:
        """
        Map (B, 3) normalized camera coords to (B, 3) grid-index coords.

        The mapping is linear from [bound_min, bound_max] to [0, grid_res-1].
        The output uses the same axis ordering as the grid (x, [y,] z).
        """
        axes = _get_axes(len(self.grid_res))
        axis_idx = _axis_indices(axes)
        lo  = torch.tensor([self.grid_bounds[f'{a}_min'] for a in axes],
                           device=pts_cam.device, dtype=pts_cam.dtype)
        hi  = torch.tensor([self.grid_bounds[f'{a}_max'] for a in axes],
                           device=pts_cam.device, dtype=pts_cam.dtype)
        res = torch.tensor(self.grid_res,
                           device=pts_cam.device, dtype=pts_cam.dtype)
        pts_sel = pts_cam[:, axis_idx]                         # (B, ndim)
        grid_coords = (pts_sel - lo) / (hi - lo) * (res - 1)  # (B, ndim)

        # Return as 3-D; fill the missing axis with 0 for 2-D grids
        out = torch.zeros(pts_cam.shape[0], 3, device=pts_cam.device, dtype=pts_cam.dtype)
        for k, a in enumerate(axes):
            dim3d = {'x': 0, 'y': 1, 'z': 2}[a]
            out[:, dim3d] = grid_coords[:, k]
        return out


# ---------------------------------------------------------------------------
# Built-in push models for use with EulerianModelWrapper
# ---------------------------------------------------------------------------

class SplatPushModel(nn.Module):
    """
    Differentiable push model based on bilinear splatting.

    Wraps ``differentiable_push_splat`` (and optionally
    ``differentiable_redistribute``) from ``model/diff_mass_push.py`` so that
    it satisfies the ``EulerianModelWrapper`` user-model contract::

        forward(occ, action_start, action_end) -> occ_pred

    Usage
    -----
    >>> model = SplatPushModel(width=5, sigma=1.5, redistribute=True)
    >>> wrapper = EulerianModelWrapper(model, bounds, grid_res, cam_ext, gs)

    Coordinate note
    ---------------
    ``EulerianModelWrapper`` stores the grid as ``(B, Nx, Ny)`` where dim 1
    indexes camera-x (column direction) and dim 2 indexes camera-y (row
    direction).  The ``diff_mass_push`` functions use the standard image
    convention ``(H=rows, W=cols)`` = ``(cam_y, cam_x)``.  This class
    handles the transpose internally; callers do not need to worry about it.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).  Smaller → sharper edges
        but weaker gradients near the boundary.
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push to diffuse excess
        occupancy forward along the push direction.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_splat_batch

        # Transpose entire batch at once: (B, Nx, Ny) → (B, Ny, Nx) so that
        # dim 1 = rows = cam_y and dim 2 = cols = cam_x, matching the (H, W)
        # image convention expected by the push function.  This replaces the
        # previous Python ``for b in range(B)`` loop (one serial GPU call per
        # sample) with a single fully-vectorised kernel call.
        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]              # (B, 2) [cam_x, cam_y]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_splat_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0                   # (B, 2)
            d_norm = d.norm(dim=-1)            # (B,)
            # redistribute operates on single frames — loop is unavoidable here;
            # this path is only taken when redistribute=True (off by default)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class FluidPushModel(nn.Module):
    """
    Differentiable push model using a velocity-field (fluid) approach.

    Wraps ``fluid_push`` (and optionally ``differentiable_redistribute``) from
    ``model/diff_mass_push.py`` so that it satisfies the
    ``EulerianModelWrapper`` user-model contract::

        forward(occ, action_start, action_end) -> occ_pred

    Usage
    -----
    >>> model = FluidPushModel(width=5, n_steps=20)
    >>> wrapper = EulerianModelWrapper(model, bounds, grid_res, cam_ext, gs)

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    n_steps : int
        Number of velocity propagation iterations (controls influence radius).
    decay : float
        Per-step velocity attenuation factor (0–1).
    media_sharpness : float
        Steepness of the soft media-presence gate; higher → velocity stays
        more confined to already-occupied regions.
    blur_sigma : float
        Gaussian σ for the propagation blur kernel (voxels).
    correct_divergence : bool
        Apply the Jacobian divergence correction for approximate mass
        conservation during the advection step.
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 5.0,
        sigma: float = 1.0,
        n_steps: int = 20,
        decay: float = 0.95,
        media_sharpness: float = 5.0,
        blur_sigma: float = 1.0,
        correct_divergence: bool = False,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.n_steps = n_steps
        self.decay = decay
        self.media_sharpness = media_sharpness
        self.blur_sigma = blur_sigma
        self.correct_divergence = correct_divergence
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import fluid_push
        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute

        B = occ.shape[0]
        results = []
        for b in range(B):
            rho = occ[b].T              # (Ny, Nx) – image (rows, cols) convention
            p0  = action_start[b, :2]
            p1  = action_end[b, :2]

            rho_new, _, _ = fluid_push(
                rho, p0, p1,
                width=self.width,
                sigma=self.sigma,
                n_steps=self.n_steps,
                decay=self.decay,
                media_sharpness=self.media_sharpness,
                blur_sigma=self.blur_sigma,
                correct_divergence=self.correct_divergence,
            )

            if self.redistribute:
                d = p1 - p0
                d_norm = d.norm()
                if d_norm.item() > 1e-6:
                    rho_new = differentiable_redistribute(
                        rho_new, d / d_norm,
                        max_iters=self.redistribute_iters)

            results.append(rho_new.T)   # back to (Nx, Ny)
        return torch.stack(results, dim=0)


class SpreadPushModel(nn.Module):
    """
    Differentiable push with linear proportional spread (Approach B).

    Instead of depositing all swept mass at a single line (like
    ``SplatPushModel``), spreads the pile forward past the tool end: material
    originally near the start of the sweep is deposited farthest from p1,
    material near p1 stays close.  Pile length equals total_mass / (2*width),
    which is exact for uniform-density input.

    Wraps ``differentiable_push_spread_batch`` from ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_spread_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_spread_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class SplatPushModel2(nn.Module):
    """
    Destination-aware spread push with optional isotropic blur.

    Extends ``SpreadPushModel`` to handle the case where the deposit region
    already contains material (which would otherwise cause stacking).

    Two anti-stacking mechanisms:

    1.  **Destination extension** – the landing zone (rectangle past p1,
        width = tool width, length = pile_depth) is probed for existing
        occupancy.  The deposit band is extended forward by
        ``extra_depth = existing_mass / (2 * width)``, so swept material
        leapfrogs pre-existing material.

    2.  **Isotropic blur** (optional, ``blur_sigma > 0``) – a mild Gaussian
        blur is applied *only* to the deposited mass before it is added
        back to the cleared field.  Simulates granular scatter and smooths
        residual peaks the first-order extension misses.

    Wraps ``differentiable_push_spread2_batch`` from
    ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    blur_sigma : float
        If > 0, Gaussian σ (voxels) for post-deposit isotropic blur on the
        deposited mass only.  Default 0 (off).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        blur_sigma: float = 0.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.blur_sigma = blur_sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_spread2_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_spread2_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma,
            blur_sigma=self.blur_sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class CumulativePushModel(nn.Module):
    """
    Differentiable push with cumulative-mass forward spread (Approach A).

    Instead of depositing all swept mass at a single line (like
    ``SplatPushModel``), computes the cumulative mass ahead of each swept
    pixel along the push direction and deposits it at
    ``p1 + cum_ahead * d_unit``.  This implements the snow-plow formula and
    is exact for arbitrary (non-uniform) density distributions.

    The cumulative scan uses K sequential ``grid_sample`` calls where
    K ≈ push length in pixels, so it is more expensive than
    ``SpreadPushModel`` but more physically accurate for non-uniform input.

    Wraps ``differentiable_push_cumulative_batch`` from
    ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_cumulative_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_cumulative_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)
