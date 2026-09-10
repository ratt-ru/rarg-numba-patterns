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
  store_data,
)
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
  "IntegerDatumLiteral",
  "LiteralStructRef",
  "Schema",
  "SchemaLiteral",
  "StringDatumLiteral",
  "accumulate_data",
  "atomic_rmw_intrinsic",
  "field_ptr",
  "is_datum_literal",
  "item_ptr",
  "load_data",
  "overload_atomic_rmw",
  "overload_field_ptr",
  "overload_item_ptr",
  "store_data",
]

__version__ = "0.0.2"
