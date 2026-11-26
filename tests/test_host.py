import os
import platform
import subprocess
import sys

import pytest

from rarg_numba_patterns.host import (
  HostDescription,
  _floor_pow2,
  _linux_caches,
  _parse_cpu_list,
  _parse_size,
  _simd_bytes,
  detect_host,
)

KB = 1024
MB = 1024 * KB


@pytest.mark.parametrize(
  "n,expected",
  [(1, 1), (32 * KB, 32 * KB), (48 * KB, 32 * KB), (1280 * KB, MB), (12 * MB, 8 * MB)],
)
def test_floor_pow2_rounds_down(n, expected):
  assert _floor_pow2(n) == expected


def test_floor_pow2_rejects_zero():
  with pytest.raises(ValueError):
    _floor_pow2(0)


@pytest.mark.parametrize(
  "text,expected", [("48K\n", 48 * KB), ("1280K", 1280 * KB), ("12M", 12 * MB)]
)
def test_parse_size(text, expected):
  assert _parse_size(text) == expected


def test_parse_cpu_list():
  assert _parse_cpu_list("0-3,8\n") == [0, 1, 2, 3, 8]


@pytest.mark.parametrize(
  "features,expected",
  [
    ({"sse2", "avx", "avx2", "avx512f"}, 64),
    ({"sse2", "avx", "avx2"}, 32),
    ({"sse2"}, 16),
    ({"neon"}, 16),
    (set(), 16),
  ],
)
def test_simd_bytes_takes_the_widest_feature(features, expected):
  assert _simd_bytes(frozenset(features)) == expected


def _write_cpu(root, cpu, core, caches):
  """caches: (level, type, size, shared_cpu_list) per index"""
  topology = root / f"cpu{cpu}" / "topology"
  topology.mkdir(parents=True)
  (topology / "physical_package_id").write_text("0\n")
  (topology / "core_id").write_text(f"{core}\n")
  for i, (level, kind, size, shared) in enumerate(caches):
    index = root / f"cpu{cpu}" / "cache" / f"index{i}"
    index.mkdir(parents=True)
    (index / "level").write_text(f"{level}\n")
    (index / "type").write_text(f"{kind}\n")
    (index / "size").write_text(f"{size}\n")
    (index / "shared_cpu_list").write_text(f"{shared}\n")


@pytest.fixture
def hybrid_sysfs(tmp_path):
  """One performance core with SMT siblings 0-1, and efficiency cores 2-5 sharing
  an L2. All six CPUs share a 12M L3."""
  for cpu in (0, 1):
    _write_cpu(
      tmp_path,
      cpu,
      core=0,
      caches=[
        (1, "Data", "48K", "0-1"),
        (1, "Instruction", "64K", "0-1"),
        (2, "Unified", "1280K", "0-1"),
        (3, "Unified", "12M", "0-5"),
      ],
    )
  for cpu in range(2, 6):
    _write_cpu(
      tmp_path,
      cpu,
      core=cpu,
      caches=[
        (1, "Data", "32K", str(cpu)),
        (2, "Unified", "2M", "2-5"),
        (3, "Unified", "12M", "0-5"),
      ],
    )
  return tmp_path


def test_linux_caches_takes_the_smallest_per_core_share(hybrid_sysfs):
  # L2: 1280K on the performance core, 2M / 4 on the efficiency cores. L3: 12M
  # shared by 5 cores, since SMT siblings count once.
  l1d, l2, l3, l3_total = _linux_caches(hybrid_sysfs, tuple(range(6)))
  assert (l1d, l2, l3, l3_total) == (32 * KB, 512 * KB, 12 * MB // 5, 12 * MB)


def test_linux_caches_follows_affinity(hybrid_sysfs):
  l1d, l2, l3, l3_total = _linux_caches(hybrid_sysfs, (0, 1))
  assert (l1d, l2, l3, l3_total) == (48 * KB, 1280 * KB, 12 * MB // 5, 12 * MB)


def test_linux_caches_without_l3(tmp_path):
  _write_cpu(tmp_path, 0, core=0, caches=[(1, "Data", "32K", "0"), (2, "Unified", "1M", "0")])
  assert _linux_caches(tmp_path, (0,)) == (32 * KB, MB, None, None)


def test_linux_caches_needs_l1d_and_l2(tmp_path):
  _write_cpu(tmp_path, 0, core=0, caches=[(1, "Data", "32K", "0")])
  with pytest.raises(ValueError):
    _linux_caches(tmp_path, (0,))


def _is_pow2(n):
  return n > 0 and n & (n - 1) == 0


def test_detect_host_rounds_to_powers_of_two():
  host = detect_host()
  assert host.simd_bytes in (16, 32, 64)
  sizes = [host.l1d, host.l2, host.l3, host.l3_total]
  assert all(_is_pow2(s) for s in sizes if s is not None)


def test_detect_host_is_memoised():
  assert detect_host() is detect_host()


def test_host_description_is_hashable_and_compares_by_value():
  a = HostDescription(simd_bytes=32, l1d=32 * KB, l2=MB)
  b = HostDescription(simd_bytes=32, l1d=32 * KB, l2=MB)
  assert a == b and hash(a) == hash(b)


@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="x86 only")
def test_generic_cpu_name_gives_baseline_vectors():
  code = "from rarg_numba_patterns import detect_host; print(detect_host().simd_bytes)"
  env = {**os.environ, "NUMBA_CPU_NAME": "generic"}
  out = subprocess.run(
    [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
  )
  assert out.stdout.strip() == "16"
