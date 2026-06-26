import rarg_numba_patterns


def test_package_exposes_version() -> None:
  assert rarg_numba_patterns.__version__
