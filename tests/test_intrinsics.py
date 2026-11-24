import numba
import numpy as np
import pytest
from numba.core.errors import TypingError

from rarg_numba_patterns.intrinsics import accumulate_data, load_data, store_data


def test_load_data():
  shape = (5, 4)

  @numba.njit
  def load(a, i):
    return load_data(a, (i,), shape[1], -1)

  data = np.arange(np.prod(shape)).reshape(shape)

  for i in range(shape[0]):
    assert load(data, i) == tuple(i * shape[1] + j for j in range(shape[1]))


def test_store_data():
  shape = (5, 4)

  @numba.njit
  def store(d, a, i):
    return store_data(d, a, (i,), -1)

  data = np.zeros(shape)

  for i in range(shape[0]):
    # Storing twice overwrites (unlike accumulate, which sums).
    store((i,) * shape[1], data, i)
    store((i,) * shape[1], data, i)

  assert np.all(np.broadcast_to(np.arange(shape[0])[:, None], shape) == data)


def test_store_load_round_trip():
  shape = (5, 4)

  @numba.njit
  def round_trip(d, a, i):
    store_data(d, a, (i,), -1)
    return load_data(a, (i,), shape[1], -1)

  data = np.zeros(shape)

  for i in range(shape[0]):
    row = tuple(float(i * shape[1] + j) for j in range(shape[1]))
    assert round_trip(row, data, i) == row


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


# Data shorter than the index: catches axes inserted at the wrong position
NDIM_SHAPE = (2, 3, 4, 2)


def _data_index(axis, index):
  """The NumPy index selecting the data along `axis` at `index`"""
  ax = axis + len(NDIM_SHAPE) if axis < 0 else axis
  return index[:ax] + (slice(None),) + index[ax:]


@pytest.mark.parametrize("axis", [-1, -2, -4, 0, 1, 3])
def test_load_data_axis(axis):
  shape = NDIM_SHAPE
  ndata = shape[axis]

  @numba.njit
  def load(a, index):
    return load_data(a, index, ndata, axis)

  data = np.arange(np.prod(shape)).reshape(shape)
  index = (1, 2, 1)
  assert load(data, index) == tuple(data[_data_index(axis, index)])


@pytest.mark.parametrize("axis", [-1, -2, 0, 3])
def test_store_data_axis(axis):
  shape = NDIM_SHAPE

  @numba.njit
  def store(d, a, index):
    store_data(d, a, index, axis)

  data = np.zeros(shape)
  index = (1, 2, 1)
  values = tuple(float(v + 1) for v in range(shape[axis]))
  store(values, data, index)
  expected = np.zeros(shape)
  expected[_data_index(axis, index)] = values
  np.testing.assert_array_equal(data, expected)


@pytest.mark.parametrize("axis", [-1, -2, 0, 3])
def test_accumulate_data_axis(axis):
  shape = NDIM_SHAPE

  @numba.njit
  def accumulate(d, a, index):
    accumulate_data(d, a, index, axis)

  data = np.ones(shape)
  index = (1, 2, 1)
  values = tuple(float(v + 1) for v in range(shape[axis]))
  accumulate(values, data, index)
  expected = np.ones(shape)
  expected[_data_index(axis, index)] += values
  np.testing.assert_array_equal(data, expected)


@pytest.mark.parametrize("axis", [4, -5])
def test_load_data_axis_out_of_range(axis):
  @numba.njit
  def load(a, index):
    return load_data(a, index, 2, axis)

  with pytest.raises(TypingError, match="out of range"):
    load(np.zeros(NDIM_SHAPE), (1, 2, 1))
