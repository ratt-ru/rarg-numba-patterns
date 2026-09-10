=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========

Added
-----

- The pre-commit hooks now run in CI, through the reusable ``precommit.yml``
  workflow. ``deploy`` gates on them alongside the test suite, so a tagged
  release cannot publish with the lint and type gate red (:pr:`4`)

Changed
-------

- mypy no longer runs in ``strict`` mode. numba's ``@intrinsic``,
  ``@overload_method`` and ``@njit`` decorators are untyped, so strict mode
  flagged every decorated function and every lowering closure (:pr:`4`)
- Pre-commit hook revisions updated, and the deprecated ``ruff`` hook id
  renamed to ``ruff-check``. ruff 0.16 widens its default rule set, which
  brings assorted lint and formatting changes with it (:pr:`4`)

Fixed
-----

- The ``mypy`` pre-commit hook failed on every run, so the commit gate never
  passed. ``python_version`` was pinned below the 3.12 that numpy's stubs
  require, and the hook resolved none of the project's dependencies, leaving
  every numpy symbol as ``Any`` (:pr:`4`)

0.0.2
=====

Added
-----

- ``store_data`` intrinsic, the write counterpart of ``load_data``, which stores
  a tuple of values into an array at a given axis and index using plain
  assignment (:pr:`2`)

0.0.1
=====

- Initial release.
