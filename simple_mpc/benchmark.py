"""GPU throughput benchmark for the Eulerian push model."""

import time
import numpy as np
import torch


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
