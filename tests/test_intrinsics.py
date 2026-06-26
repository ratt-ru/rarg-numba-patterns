import numba
import numpy as np

from rarg_numba_patterns.intrinsics import accumulate_data, load_data


def test_load_data():
  shape = (5, 4)

  @numba.njit
  def load(a, i):
    return load_data(a, (i,), shape[1], -1)

  data = np.arange(np.prod(shape)).reshape(shape)

  for i in range(shape[0]):
    assert load(data, i) == tuple(i * shape[1] + j for j in range(shape[1]))


def test_accumulate_data():
  shape = (5, 4)

  @numba.njit
  def accumulate(d, a, i):
    return accumulate_data(d, a, (i,), -1)

  data = np.zeros(shape)

  for i in range(shape[0]):
    accumulate((i,) * shape[1], data, i)
    accumulate((i,) * shape[1], data, i)

  assert np.all(np.broadcast_to(np.arange(shape[0])[:, None], shape) * 2 == data)