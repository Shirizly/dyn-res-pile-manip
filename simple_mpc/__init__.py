"""Simple gradient-descent MPC for EulerianModelWrapper.

Public API
----------
run_simple_mpc          – Run MPC and return a result dict compatible with
                          env.step_subgoal_ptcl().
load_simple_config      – Load the simple-MPC YAML config file.
benchmark_push_throughput – Measure push-model GPU throughput.
"""

from simple_mpc.mpc import run_simple_mpc, load_simple_config
from simple_mpc.benchmark import benchmark_push_throughput

__all__ = [
    'run_simple_mpc',
    'load_simple_config',
    'benchmark_push_throughput',
]
