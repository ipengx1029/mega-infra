"""Typed, deterministic kernel parameter schemas.

Kernel-specific metadata is a byte blob to the generic runtime, but every
backend variant must declare the schema used to pack that blob. This keeps the
device command compact without falling back to unversioned Python tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct
from typing import Any, Mapping

from .errors import AbiError


class ParamType(Enum):
    U32 = "I"
    I32 = "i"
    U64 = "Q"
    I64 = "q"
    F32 = "f"
    F64 = "d"


@dataclass(frozen=True)
class ParamField:
    name: str
    type: ParamType

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise AbiError(f"invalid parameter field name {self.name!r}")


class ParameterSchema:
    """A packed little-endian parameter struct with explicit field types."""

    def __init__(self, *fields: ParamField) -> None:
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise AbiError("parameter schema contains duplicate field names")
        self.fields = tuple(fields)
        self._struct = struct.Struct(
            "<" + "".join(field.type.value for field in fields)
        )

    @property
    def size(self) -> int:
        return self._struct.size

    def pack(self, values: Mapping[str, Any]) -> bytes:
        expected = {field.name for field in self.fields}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise AbiError(f"parameter keys differ; missing={missing}, extra={extra}")
        try:
            return self._struct.pack(*(values[field.name] for field in self.fields))
        except (struct.error, TypeError, ValueError) as exc:
            raise AbiError(f"could not pack kernel parameters: {exc}") from exc

    def unpack(self, payload: bytes | memoryview) -> dict[str, Any]:
        if len(payload) != self.size:
            raise AbiError(
                f"parameter payload has {len(payload)} bytes; expected {self.size}"
            )
        values = self._struct.unpack(payload)
        return {field.name: value for field, value in zip(self.fields, values)}


EMPTY_PARAMETERS = ParameterSchema()
