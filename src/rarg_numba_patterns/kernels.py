"""Parameterise numba kernels with one compile-time kernel configuration.

A kernel has three layers:

1. A public Python function builds the configuration, wraps it in a
   :class:`~rarg_numba_patterns.literals.Datum` and sets the thread count with
   :func:`threads`.
2. A thin, serial ``@njit(cache=True)`` entry passes everything to the overload.
3. An ``@overload`` reads the configuration at typing time with
   :func:`configuration` and builds the impl, folding its values in as constants.

When the configuration's ``parallel`` is true, the overload calls a separate
``prange`` overload compiled with ``jit_options={"parallel": True}``. That overload
must not be inlined: an inlined overload takes its caller's flags, so it would run
serially from the serial entry.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import TypeVar

import numba
from numba.core.errors import TypingError

from rarg_numba_patterns.host import HostDescription
from rarg_numba_patterns.literals import DatumLiteral

__all__ = ["KernelConfiguration", "configuration", "threads"]


@dataclass(frozen=True, kw_only=True)
class KernelConfiguration:
  """Base class for a kernel's configuration.

  Subclass it once per kernel, at module level so that it pickles, and decorate the
  subclass with ``@dataclass(frozen=True)``. Every field must be hashable.

  Attributes:
    parallel: whether the kernel runs its parallel implementation.
    host: the CPU the kernel runs on, normally from
      :func:`~rarg_numba_patterns.host.detect_host`.
  """

  host: HostDescription
  parallel: bool = False

  def __str__(self) -> str:
    # A DatumLiteral's numba type name is built from str(value). Qualify the class
    # so that two configuration classes with the same name, in different modules,
    # are different numba types.
    cls = type(self)
    values = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
    return f"{cls.__module__}.{cls.__qualname__}({values})"


C = TypeVar("C", bound=KernelConfiguration)


def configuration(config: object, cls: type[C]) -> C:
  """Return the configuration wrapped by a ``DatumLiteral`` type, at typing time.

  Call it in an ``@overload`` on the type of the configuration argument.

  Raises:
    TypingError: unless ``config`` is a ``DatumLiteral`` wrapping an instance of
      ``cls``.
  """
  if not isinstance(config, DatumLiteral) or not isinstance(config.literal_value, cls):
    raise TypingError(
      f"the kernel configuration must be a Datum wrapping a {cls.__qualname__}, "
      f"got {config}"
    )
  return config.literal_value


@contextlib.contextmanager
def threads(nthreads: int | None) -> Iterator[None]:
  """Run numba's parallel regions in this block on ``nthreads`` threads.

  ``None`` and 1 change nothing: the kernel runs its serial implementation, so
  ``parallel`` should be ``nthreads not in (None, 1)``. Larger values set numba's
  thread count for the calling thread and restore it on exit. Kernels can't set it
  themselves, because a kernel that calls ``numba.set_num_threads`` can't be cached.

  Raises:
    ValueError: if ``nthreads`` is less than 1, or more than ``NUMBA_NUM_THREADS``.
  """
  if nthreads is None or nthreads == 1:
    yield
    return
  if nthreads < 1:
    raise ValueError(f"nthreads must be at least 1, got {nthreads}")
  if nthreads > numba.config.NUMBA_NUM_THREADS:
    raise ValueError(
      f"nthreads ({nthreads}) is more than NUMBA_NUM_THREADS "
      f"({numba.config.NUMBA_NUM_THREADS})"
    )
  previous = numba.get_num_threads()
  numba.set_num_threads(nthreads)
  try:
    yield
  finally:
    numba.set_num_threads(previous)
