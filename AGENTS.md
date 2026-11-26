# rarg-numba-patterns: general-purpose numba intrinsics and literals patterns

[ratt-ru/rarg-numba-patterns](https://github.com/ratt-ru/rarg-numba-patterns) provides
reusable low-level building blocks for numba `@njit` code: intrinsics, literal
types, and atomic-lock primitives shared across RARG codebases.

## Layout

- [`intrinsics.py`](src/rarg_numba_patterns/intrinsics.py): `@intrinsic`
  helpers and their `@overload_method` wrappers.
  - `load_data` / `store_data` / `accumulate_data` read, write, and `+=` a
    tuple of values along one axis of an array, at a compile-time-literal `ndata`
    and `axis`.
  - `field_ptr` / `item_ptr` return `CPointer`s into a record field or an array
    element (exposed as `.field_ptr(...)` / `.item_ptr(...)` methods).
  - `atomic_rmw_intrinsic` lowers to llvmlite's `atomic_rmw`, exposed as the
    `.atomic_rmw(op, value, ordering)` method on `CPointer`.
- [`literals.py`](src/rarg_numba_patterns/literals.py): passing arbitrary Python
  objects into jitted code as compile-time literals. `Schema`/`SchemaLiteral`,
  `Datum`/`DatumLiteral` (and the typed subclasses `BooleanDatumLiteral`,
  `IntegerDatumLiteral`, `FloatDatumLiteral` and `StringDatumLiteral`, with
  `lower_cast` to native constants), and
  `LiteralStructRef`, a `StructRef` that specialises on its literal-typed fields.
- [`host.py`](src/rarg_numba_patterns/host.py): `HostDescription` and
  `detect_host()`, the vector width and power-of-two cache sizes of the CPU a
  kernel runs on.
- [`kernels.py`](src/rarg_numba_patterns/kernels.py): `KernelConfiguration`, the
  base for a kernel's single compile-time configuration (passed as a `Datum`),
  `configuration()` to read it in an `@overload`, and `threads()` to set numba's
  thread count around a parallel call.
- [`concurrency/mutex.py`](src/rarg_numba_patterns/concurrency/mutex.py):
  `atomic_lock`, `lock_int_op` and `lock_array_op`, cmpxchg-based spinlocks that
  call the OS yield function (`sched_yield` / `SwitchToThread`). This module is
  **not** re-exported from the top-level package.

## Conventions

- ruff for formatting and linting: **2-space indent**, line length 88, imports
  sorted by the ruff `I` rules.
- mypy runs in non-strict mode, because numba's decorators are untyped. Still
  type public signatures.
- pytest: one behaviour per test, named for the behaviour; prefer fixtures over
  setup/teardown; keep tests deterministic.

## Testing notes

The tests need no external data. They already work around these numba
gotchas; follow them:

- Test modules that put `assert` inside `@njit` functions carry a
  `PYTEST_DONT_REWRITE` docstring so pytest doesn't rewrite asserts inside numba
  code (see [`test_pointers.py`](tests/test_pointers.py)).
- A cached `@njit` function that calls `numba.get_thread_id()` outside a
  `prange` segfaults when loaded from cache, since nothing has launched the
  threading layer (numba 0.68.1). Only call it inside parallel regions.
- Numba cache-debugging env vars must be set **before** numba is imported (see
  `_init_numba_cache_debugging_with_capture` in [`conftest.py`](tests/conftest.py)).

## Tooling

Use `uv` to interact with this package.

- Install: `uv sync`
- Test: `uv run --group test pytest tests/`
- Pre-commit hooks (ruff, mypy and file hygiene) gate commits; run them before
  proposing a change as done: `uv run --group dev pre-commit run -a`
- Versioning: tbump (never hand-edit version strings; they live in
  `pyproject.toml` and `__init__.py`): `uv run --group dev tbump <new-version>`.

CI is delegated to the reusable `ratt-ru/rarg-gh-workflows` python-ci workflow;
tagged pushes publish to PyPI via trusted publishing (no API tokens). Update
[`CHANGELOG.rst`](CHANGELOG.rst) for user-visible changes.
