"""A kernel built the three-layer way, and the helpers it uses."""

import multiprocessing as mp
from dataclasses import dataclass

import numba
import numpy as np
import pytest
from conftest import _init_numba_cache_debugging_with_capture
from numba.core.errors import TypingError
from numba.extending import overload

from rarg_numba_patterns import (
  Datum,
  HostDescription,
  KernelConfiguration,
  configuration,
  detect_host,
  threads,
)

JIT_OPTIONS = {"nogil": True, "error_model": "numpy"}


@dataclass(frozen=True)
class ScaleConfiguration(KernelConfiguration):
  factor: float = 2.0


@dataclass(frozen=True)
class OtherConfiguration(KernelConfiguration):
  pass


def scale(x, *, factor=2.0, nthreads=None, host=None):
  """Layer 1: scale x into a new array. Returns (result, numba thread id per row)."""
  config = ScaleConfiguration(
    host=host or detect_host(), parallel=nthreads not in (None, 1), factor=factor
  )
  out = np.empty_like(x)
  tids = np.empty(x.shape[0], np.int64)
  with threads(nthreads):
    _scale(x, out, tids, Datum(config))
  return out, tids


@numba.njit(cache=True, **JIT_OPTIONS)
def _scale(x, out, tids, config):
  """Layer 2"""
  scale_impl(x, out, tids, config)


def scale_impl(x, out, tids, config):
  pass


def scale_rows_serial(x, out, tids, config):
  pass


def scale_rows_parallel(x, out, tids, config):
  pass


@overload(scale_impl, prefer_literal=True, jit_options=JIT_OPTIONS)
def _scale_impl(x, out, tids, config):
  """Layer 3"""
  CONFIG = configuration(config, ScaleConfiguration)
  if CONFIG.parallel:
    return lambda x, out, tids, config: scale_rows_parallel(x, out, tids, config)
  return lambda x, out, tids, config: scale_rows_serial(x, out, tids, config)


def _rows(parallel):
  def factory(x, out, tids, config):
    FACTOR = configuration(config, ScaleConfiguration).factor
    if parallel:

      def impl(x, out, tids, config):
        for i in numba.prange(x.shape[0]):
          tids[i] = numba.get_thread_id()
          for j in range(x.shape[1]):
            out[i, j] = FACTOR * x[i, j]

      return impl

    # No get_thread_id here: outside a parallel region, a cached kernel that calls
    # it segfaults when loaded from cache, because nothing has launched numba's
    # threading layer (numba 0.68.1)
    def impl(x, out, tids, config):
      for i in range(x.shape[0]):
        tids[i] = 0
        for j in range(x.shape[1]):
          out[i, j] = FACTOR * x[i, j]

    return impl

  return factory


overload(scale_rows_serial, jit_options=JIT_OPTIONS)(_rows(False))
# Not inlined, so it compiles with its own parallel flag
overload(scale_rows_parallel, jit_options={**JIT_OPTIONS, "parallel": True})(
  _rows(True)
)


@pytest.fixture
def x():
  return np.arange(256 * 8, dtype=np.float64).reshape(256, 8)


def test_serial_kernel(x):
  out, _ = scale(x, factor=3.0)
  np.testing.assert_array_equal(out, 3.0 * x)


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2, reason="needs 2 threads")
def test_parallel_overload_runs_threaded_from_a_serial_entry(x):
  out, tids = scale(x, factor=3.0, nthreads=2)
  np.testing.assert_array_equal(out, 3.0 * x)
  assert len(set(tids)) == 2


def test_configuration_rejects_another_class():
  host = HostDescription(simd_bytes=32, l1d=32768, l2=1 << 20)
  config = numba.typeof(Datum(OtherConfiguration(host=host)))
  with pytest.raises(TypingError, match="ScaleConfiguration"):
    configuration(config, ScaleConfiguration)


def test_configuration_rejects_a_non_literal():
  with pytest.raises(TypingError):
    configuration(numba.types.float64, ScaleConfiguration)


def test_kernel_rejects_a_wrong_configuration(x):
  host = detect_host()
  with pytest.raises(TypingError, match="ScaleConfiguration"):
    _scale(x, np.empty_like(x), np.empty(256, np.int64), Datum(OtherConfiguration(host=host)))


def test_configuration_type_name_is_module_qualified():
  host = HostDescription(simd_bytes=32, l1d=32768, l2=1 << 20)
  name = numba.typeof(Datum(ScaleConfiguration(host=host))).name
  assert f"{__name__}.ScaleConfiguration(" in name


def test_threads_restores_the_thread_count():
  before = numba.get_num_threads()
  with threads(1):
    assert numba.get_num_threads() == before
  if numba.config.NUMBA_NUM_THREADS >= 2:
    with threads(2):
      assert numba.get_num_threads() == 2
  assert numba.get_num_threads() == before


@pytest.mark.parametrize("nthreads", [0, numba.config.NUMBA_NUM_THREADS + 1])
def test_threads_rejects_out_of_range(nthreads):
  with pytest.raises(ValueError), threads(nthreads):
    pass


def _caching_worker(host):
  x = np.ones((4, 2))
  out, _ = scale(x, host=host)
  return float(out.sum())


HOST_A = HostDescription(simd_bytes=32, l1d=32768, l2=1 << 20, l3=1 << 20)
HOST_B = HostDescription(simd_bytes=32, l1d=32768, l2=1 << 20, l3=1 << 21)


@pytest.mark.parametrize("hosts,saved,loaded", [((HOST_A, HOST_A), 1, 1), ((HOST_A, HOST_B), 2, 0)])
def test_host_description_is_part_of_the_cache_key(tmp_path, hosts, saved, loaded):
  stdout_f = tmp_path / "stdout.txt"
  stderr_f = tmp_path / "stderr.txt"
  for host in hosts:
    # A fresh process per call, so the second can only reuse the on-disk cache
    with mp.get_context("spawn").Pool(
      1,
      initializer=_init_numba_cache_debugging_with_capture,
      initargs=(str(tmp_path), str(stdout_f), str(stderr_f)),
    ) as p:
      assert p.apply(_caching_worker, args=(host,)) == 16.0

  combined = stdout_f.read_text() + stderr_f.read_text()
  assert combined.count(f"data saved to '{tmp_path}") == saved, combined
  assert combined.count(f"data loaded from '{tmp_path}") == loaded, combined
