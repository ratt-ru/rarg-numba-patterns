"""Shared fixtures live here. Prefer fixtures over setup/teardown; keep tests
deterministic (see the shared Python conventions)."""

import os
import sys


def _init_numba_cache_debugging_with_capture(cache_dir, stdout_path, stderr_path):
  assert "numba" not in sys.modules
  os.environ["NUMBA_CACHE_DIR"] = cache_dir
  os.environ["NUMBA_DEBUG_CACHE"] = "1"

  # These streams must outlive this call so numba's cache-debug output lands in
  # them, so they deliberately are not context-managed.
  sys.stdout = open(stdout_path, "a", buffering=1)  # noqa: SIM115
  sys.stderr = open(stderr_path, "a", buffering=1)  # noqa: SIM115
