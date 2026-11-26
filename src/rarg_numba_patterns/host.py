"""Describe the host CPU that numba kernels specialise on.

A :class:`HostDescription` holds the vector width and data cache sizes of the CPU a
kernel runs on. Kernels carry it in their configuration as a compile-time literal,
so they can derive block sizes from it as constants. Because it's part of the
kernel's type, it's also part of numba's cache key.

- The vector width comes from the CPU features numba compiles for, so it honours
  ``NUMBA_CPU_FEATURES``, and ``NUMBA_CPU_NAME=generic``, which clears them. Another
  ``NUMBA_CPU_NAME`` on its own keeps the host's features.
- Cache sizes come from the operating system: sysfs on Linux and ``sysctl`` on
  macOS, with a conservative fallback elsewhere. Neither numba nor the standard
  library exposes them portably.
- On hybrid CPUs each level reports the smallest **per-core share** across the CPUs
  this process may run on: a cache shared by k cores counts as size / k, and SMT
  siblings count as one core.
- Every cache size is rounded down to a power of two. Similar hosts then share one
  description, and so one compiled kernel.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["HostDescription", "detect_host"]

# Bytes per vector register by instruction set, widest first
_SIMD_BYTES_BY_FEATURE = (("avx512f", 64), ("avx", 32), ("sse2", 16), ("neon", 16))
# Every 64-bit target numba supports has 128-bit SIMD
_SIMD_BYTES_DEFAULT = 16

_KB = 1024
_MB = 1024 * _KB


@dataclass(frozen=True)
class HostDescription:
  """Vector width and data cache sizes of a CPU, in bytes.

  ``l1d``, ``l2`` and ``l3`` are per-core shares. ``l3_total`` is the whole L3 that a
  single thread can use. ``l3`` and ``l3_total`` are None when the host has no L3,
  or when its size couldn't be determined.
  """

  simd_bytes: int
  l1d: int
  l2: int
  l3: int | None = None
  l3_total: int | None = None


# Used when the operating system can't be asked: small enough for any recent CPU
_FALLBACK_CACHES = (32 * _KB, 256 * _KB, None, None)


def detect_host() -> HostDescription:
  """Describe the CPU this process is running on.

  Call this where the kernel runs, on the worker, not on a client that ships the
  result to workers. The result is memoised per CPU affinity set, so pinning the
  process to other CPUs gives a fresh description.
  """
  try:
    affinity: tuple[int, ...] | None = tuple(sorted(os.sched_getaffinity(0)))
  except AttributeError:  # not Linux
    affinity = None
  return _detect_host(affinity)


@functools.cache
def _detect_host(affinity: tuple[int, ...] | None) -> HostDescription:
  l1d, l2, l3, l3_total = _cache_sizes(affinity)
  return HostDescription(
    simd_bytes=_simd_bytes(_cpu_features()),
    l1d=_floor_pow2(l1d),
    l2=_floor_pow2(l2),
    l3=None if l3 is None else _floor_pow2(l3),
    l3_total=None if l3_total is None else _floor_pow2(l3_total),
  )


def _floor_pow2(n: int) -> int:
  """The largest power of two no greater than n, for n >= 1"""
  if n < 1:
    raise ValueError(f"cache size must be positive, got {n}")
  return 1 << (n.bit_length() - 1)


def _cpu_features() -> frozenset[str]:
  """Enabled features of the CPU numba compiles for, such as {"avx2", "fma"}"""
  from numba.core.registry import cpu_target

  # A private numba API: llvmlite's public host query ignores NUMBA_CPU_FEATURES
  features = cpu_target.target_context.codegen()._get_host_cpu_features()
  return frozenset(f[1:] for f in features.split(",") if f.startswith("+"))


def _simd_bytes(features: frozenset[str]) -> int:
  """Width of the widest vector register the features allow, in bytes.

  On some AVX-512 CPUs LLVM's loop vectoriser prefers 256-bit vectors anyway. This
  reports what the instruction set allows, which is what explicit vector code in an
  intrinsic can use.
  """
  for feature, nbytes in _SIMD_BYTES_BY_FEATURE:
    if feature in features:
      return nbytes
  return _SIMD_BYTES_DEFAULT


def _cache_sizes(
  affinity: tuple[int, ...] | None,
) -> tuple[int, int, int | None, int | None]:
  """(l1d, l2, l3, l3_total) before rounding, or the fallback"""
  system = platform.system()
  try:
    if system == "Linux" and affinity:
      return _linux_caches(Path("/sys/devices/system/cpu"), affinity)
    if system == "Darwin":
      return _darwin_caches()
  except (OSError, ValueError, subprocess.SubprocessError):
    pass
  return _FALLBACK_CACHES


def _parse_size(text: str) -> int:
  """A sysfs cache size such as "48K", "1280K" or "12M", in bytes"""
  text = text.strip()
  scale = {"K": _KB, "M": _MB, "G": 1024 * _MB}.get(text[-1].upper(), 1)
  return int(text.rstrip("KMGkmg")) * scale


def _parse_cpu_list(text: str) -> list[int]:
  """A sysfs CPU list such as "0-3,8", as [0, 1, 2, 3, 8]"""
  cpus: list[int] = []
  for part in text.strip().split(","):
    if "-" in part:
      lo, hi = part.split("-")
      cpus.extend(range(int(lo), int(hi) + 1))
    elif part:
      cpus.append(int(part))
  return cpus


def _linux_caches(
  root: Path, cpus: tuple[int, ...]
) -> tuple[int, int, int | None, int | None]:
  def core_of(cpu: int) -> tuple[str, str]:
    topology = root / f"cpu{cpu}" / "topology"
    package = (topology / "physical_package_id").read_text().strip()
    return package, (topology / "core_id").read_text().strip()

  shares: dict[int, list[int]] = {1: [], 2: [], 3: []}
  l3_totals = []
  for cpu in cpus:
    for index in sorted((root / f"cpu{cpu}" / "cache").glob("index*")):
      if (index / "type").read_text().strip() not in ("Data", "Unified"):
        continue
      level = int((index / "level").read_text())
      if level not in shares:
        continue
      size = _parse_size((index / "size").read_text())
      sharing = _parse_cpu_list((index / "shared_cpu_list").read_text())
      cores = {core_of(c) for c in sharing}
      shares[level].append(size // max(1, len(cores)))
      if level == 3:
        l3_totals.append(size)
  if not shares[1] or not shares[2]:
    raise ValueError("no L1d or L2 cache information in sysfs")
  return (
    min(shares[1]),
    min(shares[2]),
    min(shares[3]) if shares[3] else None,
    min(l3_totals) if l3_totals else None,
  )


def _sysctl(name: str) -> int | None:
  out = subprocess.run(
    ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False
  )
  value = out.stdout.strip()
  return int(value) if out.returncode == 0 and value.isdigit() else None


def _darwin_caches() -> tuple[int, int, int | None, int | None]:
  """macOS: per performance level on Apple silicon (hw.perflevelN.*), else hw.*.

  Untested: written from the sysctl documentation, not run on a Mac.
  """
  levels = []
  for n in range(_sysctl("hw.nperflevels") or 0):
    p = f"hw.perflevel{n}."
    l1, l2 = _sysctl(p + "l1dcachesize"), _sysctl(p + "l2cachesize")
    if l1 and l2:
      per_l2 = max(1, _sysctl(p + "cpusperl2") or 1)
      levels.append((l1, l2 // per_l2, _sysctl(p + "l3cachesize")))
  if not levels:
    l1, l2 = _sysctl("hw.l1dcachesize"), _sysctl("hw.l2cachesize")
    if not (l1 and l2):
      raise ValueError("no cache information from sysctl")
    levels = [(l1, l2, _sysctl("hw.l3cachesize"))]
  l3s = [l3 for _, _, l3 in levels if l3]
  l3 = min(l3s) if l3s else None
  # macOS reports no per-core sharing for L3
  return min(lv[0] for lv in levels), min(lv[1] for lv in levels), l3, l3
