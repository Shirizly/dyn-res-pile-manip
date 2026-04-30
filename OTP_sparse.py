import numpy as np
import ot
import matplotlib.pyplot as plt
import time

def _tic():
    return time.perf_counter()

def _toc(t0, label):
    dt = time.perf_counter() - t0
    print(f"  {label:<40s} {dt*1000:8.2f} ms")
    return dt

# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------
n = 100
x, y = np.arange(n), np.arange(n)
X, Y = np.meshgrid(x, y)
coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (n*n, 2)  col-first (x=col, y=row)

# ---------------------------------------------------------------------------
# 1. Source distribution: uniform random pile inside a square
# ---------------------------------------------------------------------------
N = 300
rng = np.random.default_rng(42)
sq_lo, sq_hi = int(0.1 * n), int(0.7 * n)
pile_2d = np.zeros((n, n))
pts = rng.integers(sq_lo, sq_hi, size=(N, 2))
for py, px in pts:
    pile_2d[py, px] += 1

# ---------------------------------------------------------------------------
# 2. Goal distribution: grid of points forming the letter "T"
# ---------------------------------------------------------------------------
goal_2d = np.zeros((n, n))
bar_row    = int(0.15 * n)
bar_col_l  = int(0.15 * n)
bar_col_r  = int(0.85 * n)
stem_col   = int(0.50 * n)
stem_row_b = int(0.75 * n)
step = 1

for col in range(bar_col_l, bar_col_r + 1, step):
    goal_2d[bar_row, col] += 1
for row in range(bar_row + step, stem_row_b + 1, step):
    goal_2d[row, stem_col] += 1

# ---------------------------------------------------------------------------
# 3. Sparse-support OT
#
# Only cells with actual mass are included.  This means:
#   - No floor mass (eps_floor) needed → no fabricated marginals
#   - The cost matrix is (n_src × n_tgt) instead of (n² × n²)
#   - Every output vector is defined exactly where particles exist, nowhere else
# ---------------------------------------------------------------------------
print("Computation times:")
t_total = _tic()

t0 = _tic()
src_idx = np.where(pile_2d.flatten() > 0)[0]   # indices of occupied source cells
tgt_idx = np.where(goal_2d.flatten() > 0)[0]   # indices of occupied goal  cells

src_coords = coords[src_idx]                    # (n_src, 2)
tgt_coords = coords[tgt_idx]                    # (n_tgt, 2)

a_sparse = pile_2d.flatten()[src_idx].astype(float)
b_sparse = goal_2d.flatten()[tgt_idx].astype(float)
a_sparse /= a_sparse.sum()
b_sparse /= b_sparse.sum()
_toc(t0, f"support extraction  ({len(src_idx)} src, {len(tgt_idx)} tgt cells)")

t0 = _tic()
C_sparse = ot.dist(src_coords, tgt_coords, metric='sqeuclidean')
C_sparse = C_sparse / C_sparse.max()
_toc(t0, f"cost matrix  ({len(src_idx)}×{len(tgt_idx)})")

t0 = _tic()
reg = 0.001
P_sparse, log = ot.sinkhorn(a_sparse, b_sparse, C_sparse, reg=reg, log=True)
_toc(t0, f"Sinkhorn  (reg={reg}, {log.get('niter', '?')} iters)")

# ---------------------------------------------------------------------------
# 4. Barycentric projection → transport vector field (sparse)
# ---------------------------------------------------------------------------
# For each occupied source cell i:
#   transported position  = sum_j P_ij * tgt_coords_j  /  a_i
#   displacement vector   = transported position - src_coords_i
t0 = _tic()
T_x_sparse = (P_sparse @ tgt_coords) / a_sparse[:, None]
v_sparse = T_x_sparse - src_coords        # (n_src, 2) — all meaningful, none fabricated

# Scatter back into a dense grid for visualisation and divergence
vectors_2d = np.zeros((n, n, 2))
vectors_2d.reshape(-1, 2)[src_idx] = v_sparse
_toc(t0, "barycentric projection + scatter")

# ---------------------------------------------------------------------------
# 5. Divergence of the vector field (central finite differences)
# ---------------------------------------------------------------------------
t0 = _tic()
vx = vectors_2d[..., 0]            # ∂-direction: col  (x)
vy = vectors_2d[..., 1]            # ∂-direction: row  (y)

dvx_dx = np.gradient(vx, axis=1)   # ∂vx/∂col
dvy_dy = np.gradient(vy, axis=0)   # ∂vy/∂row
div     = dvx_dx + dvy_dy          # signed divergence
div_mag = np.abs(div)              # magnitude  (small = coherent / laminar)

# Zero divergence outside the source support to avoid boundary artefacts
source_mask = pile_2d > 0
div     = div     * source_mask
div_mag = div_mag * source_mask
_toc(t0, "divergence")

_toc(t_total, "TOTAL")
print()

# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def plot_distributions(pile, goal):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(pile, origin='lower', cmap='Blues')
    axes[0].set_title('Source distribution (pile)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(goal, origin='lower', cmap='Oranges')
    axes[1].set_title('Goal distribution (T shape)')
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    return fig


def plot_vector_field(pile, goal, vectors_2d, src_coords, v_sparse, stride=2):
    """Two panels: (left) quiver over source; (right) projected positions over goal."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- left: transport plan arrows ---
    # Draw one arrow per occupied source cell (no stride subsampling on the sparse set).
    # src_coords columns: [col (x), row (y)]
    axes[0].imshow(pile, origin='lower', cmap='Blues', alpha=0.6)
    axes[0].quiver(src_coords[:, 0], src_coords[:, 1],
                   v_sparse[:, 0],   v_sparse[:, 1],
                   color='crimson', scale=1.0,
                   scale_units='xy', angles='xy', width=0.003)
    axes[0].set_title('OT transport plan  (one arrow per occupied cell)')

    # --- right: projected positions overlaid on goal ---
    projected = src_coords + v_sparse          # shape (n_src, 2): col, row
    proj_grid = np.zeros((n, n))
    for col, row in np.round(projected).astype(int):
        r = np.clip(row, 0, n - 1)
        c = np.clip(col, 0, n - 1)
        proj_grid[r, c] += 1
    axes[1].imshow(goal, origin='lower', cmap='Oranges', alpha=0.6)
    # Show projected cells as blue squares at the same pixel scale as the grid
    proj_rgba = np.zeros((n, n, 4))
    proj_rgba[..., 2] = 0.27       # B  (steelblue-ish)
    proj_rgba[..., 0] = 0.13       # R
    proj_rgba[..., 1] = 0.47       # G
    proj_rgba[..., 3] = np.where(proj_grid > 0, 0.5, 0.0)   # alpha: transparent where empty
    axes[1].imshow(proj_rgba, origin='lower', interpolation='nearest')
    axes[1].set_title('Projected positions over goal distribution')

    fig.tight_layout()
    return fig


def plot_divergence(div, div_mag):
    """Side-by-side: signed divergence and its magnitude (source support only)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmax = np.percentile(np.abs(div[div != 0]), 99) if (div != 0).any() else 1.0
    im0 = axes[0].imshow(div, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Signed divergence  (red=source, blue=sink)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(div_mag, origin='lower', cmap='hot_r')
    axes[1].set_title('Divergence magnitude  (dark = coherent / laminar)')
    plt.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    plot_distributions(pile_2d, goal_2d)
    plot_vector_field(pile_2d, goal_2d, vectors_2d, src_coords, v_sparse, stride=2)
    # plot_divergence(div, div_mag)
    plt.show()
