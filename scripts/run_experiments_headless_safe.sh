#!/usr/bin/env bash

is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  is_sourced=1
fi

fail() {
  echo "$1" >&2
  if [[ "$is_sourced" -eq 1 ]]; then
    return 1
  fi
  exit 1
}

main() {
  if [[ $# -lt 1 ]]; then
    fail "Usage: ${BASH_SOURCE[0]} <suite_config.yaml> [extra run_experiments args...]"
    return 1
  fi

  local root_dir suite_config
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  suite_config="$1"
  shift || true

  (
    set -euo pipefail
    cd "$root_dir"
    source setup_env.sh

    # Force non-native-headless mode in configs when using Xvfb path.
    export PYFLEX_HEADLESS_OVERRIDE=0

    xvfb-run -a -s "-screen 0 1280x1024x24" \
      /home/shirizly/anaconda3/envs/dyn-res-pile-manip/bin/python \
      run_experiments.py "$suite_config" "$@"
  )
}

# Run PyFleX experiments without a physical display by using Xvfb.
# This avoids the unstable native PyFleX headless path (EGL shader issues on
# some systems) while still creating a valid OpenGL context.
#
# Usage:
#   bash scripts/run_experiments_headless_safe.sh \
#     config/experiments/compare_models_fast.yaml

main "$@"
status=$?

if [[ "$is_sourced" -eq 1 ]]; then
  return "$status"
fi

exit "$status"
