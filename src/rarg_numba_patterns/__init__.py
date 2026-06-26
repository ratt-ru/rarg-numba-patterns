"""General-purpose numba intrinsics and literals patterns"""

from rarg_numba_patterns.intrinsics import (
  accumulate_data,
  atomic_rmw_intrinsic,
  field_ptr,
  item_ptr,
  load_data,
  overload_atomic_rmw,
  overload_field_ptr,
  overload_item_ptr,
)
from rarg_numba_patterns.literals import (
  BooleanDatumLiteral,
  Datum,
  DatumLiteral,
  FloatDatumLiteral,
  IntegerDatumLiteral,
  is_datum_literal,
  LiteralStructRef,
  Schema,
  SchemaLiteral,
  StringDatumLiteral,
)

__all__ = [
  "accumulate_data",
  "atomic_rmw_intrinsic",
  "BooleanDatumLiteral",
  "Datum",
  "DatumLiteral",
  "field_ptr",
  "FloatDatumLiteral",
  "IntegerDatumLiteral",
  "is_datum_literal",
  "item_ptr",
  "LiteralStructRef",
  "load_data",
  "overload_atomic_rmw",
  "overload_field_ptr",
  "overload_item_ptr",
  "Schema",
  "SchemaLiteral",
  "StringDatumLiteral",
]

__version__ = "0.0.1"