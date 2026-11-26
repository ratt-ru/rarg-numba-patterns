"""General-purpose numba intrinsics and literals patterns"""

from rarg_numba_patterns.host import HostDescription, detect_host
from rarg_numba_patterns.intrinsics import (
  accumulate_data,
  atomic_rmw_intrinsic,
  field_ptr,
  item_ptr,
  load_data,
  overload_atomic_rmw,
  overload_field_ptr,
  overload_item_ptr,
  store_data,
)
from rarg_numba_patterns.kernels import KernelConfiguration, configuration, threads
from rarg_numba_patterns.literals import (
  BooleanDatumLiteral,
  Datum,
  DatumLiteral,
  FloatDatumLiteral,
  IntegerDatumLiteral,
  LiteralStructRef,
  Schema,
  SchemaLiteral,
  StringDatumLiteral,
  is_datum_literal,
)

__all__ = [
  "BooleanDatumLiteral",
  "Datum",
  "DatumLiteral",
  "FloatDatumLiteral",
  "HostDescription",
  "IntegerDatumLiteral",
  "KernelConfiguration",
  "LiteralStructRef",
  "Schema",
  "SchemaLiteral",
  "StringDatumLiteral",
  "accumulate_data",
  "atomic_rmw_intrinsic",
  "configuration",
  "detect_host",
  "field_ptr",
  "is_datum_literal",
  "item_ptr",
  "load_data",
  "overload_atomic_rmw",
  "overload_field_ptr",
  "overload_item_ptr",
  "store_data",
  "threads",
]

__version__ = "0.0.2"
