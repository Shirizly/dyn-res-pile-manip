import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------
n = 10
x, y = np.arange(n), np.arange(n)
X, Y = np.meshgrid(x, y)
coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (n*n, 2)

# ---------------------------------------------------------------------------
# 1. Source distribution: uniform random pile inside a square
# ---------------------------------------------------------------------------
N = 10
rng = np.random.default_rng(42)
sq_lo, sq_hi = int(0.0 * n), int(0.5 * n)
pile_2d = np.zeros((n, n))
pts = rng.integers(sq_lo, sq_hi, size=(N, 2))
for py, px in pts:
    pile_2d[py, px] += 1

print("Source distribution (pile):")
print(pile_2d)
# ---------------------------------------------------------------------------
# 2. Goal distribution: grid of points forming the letter "T"
# ---------------------------------------------------------------------------
goal_2d = np.zeros((n, n))
# # T parameters (in grid units)
# bar_row   = int(0.25 * n)          # row of the horizontal bar (top of T)
# bar_col_l = int(0.25 * n)          # left  end of bar
# bar_col_r = int(0.75 * n)          # right end of bar
# stem_col  = int(0.50 * n)          # column of the vertical stem
# stem_row_t = bar_row               # stem top (same as bar)
# stem_row_b = int(0.75 * n)         # stem bottom
# step = 1                           # spacing between grid points

# # Horizontal bar
# for col in range(bar_col_l, bar_col_r + 1, step):
#     goal_2d[bar_row, col] += 1

# # Vertical stem (skip top row, already drawn)
# for row in range(stem_row_t + step, stem_row_b + 1, step):
#     goal_2d[row, stem_col] += 1
N = 10
sq_lo, sq_hi = int(0.5 * n), int(1.0 * n)
pts = rng.integers(sq_lo, sq_hi, size=(N, 2))
for py, px in pts:
    goal_2d[py, px] += 1
print("Goal distribution:")
print(goal_2d)

# ---------------------------------------------------------------------------
# 3. Ground cost matrix (squared Euclidean, normalised)
# ---------------------------------------------------------------------------
C = ot.dist(coords, coords, metric='sqeuclidean')
C = C / C.max()

# ---------------------------------------------------------------------------
# 4. Sinkhorn OT
# ---------------------------------------------------------------------------
reg = 0.02
# Add a small uniform floor so Sinkhorn never sees exact zeros in the marginals
eps_floor = 1e-3
a_raw = pile_2d.flatten() + eps_floor
b_raw = goal_2d.flatten() + eps_floor
a = a_raw / a_raw.sum()
b = b_raw / b_raw.sum()
P, log = ot.sinkhorn(a, b, C, reg, log=True)

# ---------------------------------------------------------------------------
# 5. Transport vector field
# ---------------------------------------------------------------------------
# For every source cell i: v_i = (sum_j P_ij * x_j) / a_i  -  x_i
safe_a = np.where(a > 0, a, 1.0)          # avoid divide-by-zero
T_x = np.dot(P, coords) / safe_a[:, None]
vectors = np.where(a[:, None] > 0, T_x - coords, 0.0)
vectors_2d = vectors.reshape(n, n, 2)      # shape (n, n, 2): axis 2 = (dx, dy)

# ---------------------------------------------------------------------------
# 5b. Optional: mask the vector field in empty source regions
# ---------------------------------------------------------------------------
# Why are vectors in empty regions NOT already near zero?
# The Sinkhorn solver works on a slightly smoothed `a` (eps_floor > 0 everywhere).
# Entropy regularisation then spreads transport proportional to exp(-C/reg),
# which is purely cost-driven and ignores whether a cell actually has particles.
# Empty cells get a displacement pointing toward the cost-weighted barycenter of
# the goal — a large, physically meaningless vector.
# Masking removes these artifacts before computing divergence.
mask_empty   = True   # set False to skip masking
blur_sigma   = 1.5    # Gaussian blur radius (grid cells)
mask_thresh  = 0.05   # fraction of peak blurred value below which to zero out

if mask_empty:
    blurred_pile = gaussian_filter(pile_2d.astype(float), sigma=blur_sigma)
    source_mask  = blurred_pile >= mask_thresh * blurred_pile.max()  # (n, n) bool
    vectors_2d   = vectors_2d * source_mask[..., np.newaxis]

# ---------------------------------------------------------------------------
# 6. Divergence of the vector field  (central finite differences)
# ---------------------------------------------------------------------------
vx = vectors_2d[..., 0]   # x-component  (indexed as [row, col])
vy = vectors_2d[..., 1]   # y-component

dvx_dx = np.gradient(vx, axis=1)   # ∂vx/∂col
dvy_dy = np.gradient(vy, axis=0)   # ∂vy/∂row
div = dvx_dx + dvy_dy              # signed divergence
div_mag = np.abs(div)              # magnitude  (small = laminar / coherent)

# ---------------------------------------------------------------------------
# 7. Visualisation
# ---------------------------------------------------------------------------

def plot_distributions(pile, goal, n):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(pile, origin='lower', cmap='Blues')
    axes[0].set_title('Source distribution (pile)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(goal, origin='lower', cmap='Oranges')
    axes[1].set_title('Goal distribution (T shape)')
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    return fig


def plot_vector_field(pile, vectors_2d, n, stride=3, source_mask=None):
    """Quiver plot of the OT displacement field overlaid on the source density."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pile, origin='lower', cmap='Blues', alpha=0.6)
    if source_mask is not None:
        # Overlay mask boundary so it's easy to see what was zeroed out
        ax.contour(source_mask.astype(float), levels=[0.5], colors='gold',
                   linewidths=1, linestyles='--')
    rows = np.arange(0, n, stride)
    cols = np.arange(0, n, stride)
    C_grid, R_grid = np.meshgrid(cols, rows)
    U = vectors_2d[::stride, ::stride, 0]   # col / x component
    V = vectors_2d[::stride, ::stride, 1]   # row / y component
    ax.quiver(C_grid, R_grid, U, V, color='crimson', scale=None,
              scale_units='xy', angles='xy', width=0.003)
    ax.set_title('OT transport vector field')
    fig.tight_layout()
    return fig


def plot_divergence(div, div_mag):
    """Side-by-side: signed divergence and its magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmax = np.percentile(np.abs(div), 99)
    im0 = axes[0].imshow(div, origin='lower', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    axes[0].set_title('Signed divergence  (red=source, blue=sink)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(div_mag, origin='lower', cmap='hot_r')
    axes[1].set_title('Divergence magnitude  (dark = coherent/laminar)')
    plt.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    _mask = source_mask if mask_empty else None
    plot_distributions(pile_2d, goal_2d, n)
    plot_vector_field(pile_2d, vectors_2d, n, stride=1, source_mask=_mask)
    plot_divergence(div, div_mag)
    plt.show()