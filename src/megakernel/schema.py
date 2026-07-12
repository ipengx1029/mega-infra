"""Semantic op schemas and shape inference.

Op schemas are frontend contracts. Kernel variants live in a separate registry,
so adding an SM90/SM100 implementation never changes graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

from .errors import RegistryError, ShapeError
from .ir import OutputSpec, TensorSpec


InferFn = Callable[[tuple[TensorSpec, ...], Mapping[str, Any]], tuple[OutputSpec, ...]]


@dataclass(frozen=True)
class OpSchema:
    name: str
    infer: InferFn
    defaults: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    metadata_only: bool = False

    def infer_outputs(
        self, inputs: Sequence[TensorSpec], attrs: Mapping[str, Any] | None = None
    ) -> tuple[tuple[OutputSpec, ...], dict[str, Any]]:
        merged = dict(self.defaults)
        merged.update(attrs or {})
        outputs = tuple(self.infer(tuple(inputs), merged))
        return outputs, merged


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: dict[str, OpSchema] = {}

    def register(self, schema: OpSchema, *, replace: bool = False) -> None:
        if schema.name in self._schemas and not replace:
            raise RegistryError(f"op schema {schema.name!r} is already registered")
        self._schemas[schema.name] = schema

    def get(self, name: str) -> OpSchema:
        try:
            return self._schemas[name]
        except KeyError as exc:
            raise RegistryError(f"unknown op schema {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._schemas))


def _expect_arity(name: str, inputs: tuple[TensorSpec, ...], count: int) -> None:
    if len(inputs) != count:
        raise ShapeError(f"{name} expects {count} inputs, got {len(inputs)}")


def _same_shape_binary(
    name: str, inputs: tuple[TensorSpec, ...], _: Mapping[str, Any]
) -> tuple[OutputSpec, ...]:
    _expect_arity(name, inputs, 2)
    if inputs[0] != inputs[1]:
        raise ShapeError(f"{name} inputs must have identical tensor specs")
    return (OutputSpec(inputs[0]),)


def _unary(name: str) -> InferFn:
    def infer(
        inputs: tuple[TensorSpec, ...], _: Mapping[str, Any]
    ) -> tuple[OutputSpec, ...]:
        _expect_arity(name, inputs, 1)
        return (OutputSpec(inputs[0]),)

    return infer


def _matmul(
    inputs: tuple[TensorSpec, ...], _: Mapping[str, Any]
) -> tuple[OutputSpec, ...]:
    _expect_arity("matmul", inputs, 2)
    lhs, rhs = inputs
    if lhs.rank != 2 or rhs.rank != 2:
        raise ShapeError("phase-1 matmul expects rank-2 tensors")
    if lhs.shape[1] != rhs.shape[0]:
        raise ShapeError(
            f"matmul K dimensions differ: {lhs.shape[1]} vs {rhs.shape[0]}"
        )
    if lhs.dtype != rhs.dtype:
        raise ShapeError("matmul inputs must have the same dtype")
    return (
        OutputSpec(TensorSpec((lhs.shape[0], rhs.shape[1]), lhs.dtype, lhs.layout)),
    )


def _rms_norm(
    inputs: tuple[TensorSpec, ...], attrs: Mapping[str, Any]
) -> tuple[OutputSpec, ...]:
    _expect_arity("rms_norm", inputs, 2)
    x, weight = inputs
    if x.rank != 2 or weight.rank != 1:
        raise ShapeError("rms_norm expects x[M, N] and weight[N]")
    if x.shape[1] != weight.shape[0]:
        raise ShapeError("rms_norm hidden size does not match weight")
    if isinstance(attrs["eps"], bool):
        raise ShapeError("rms_norm eps must be a finite number")
    try:
        eps = float(attrs["eps"])
    except (TypeError, ValueError) as exc:
        raise ShapeError("rms_norm eps must be a finite number") from exc
    if not math.isfinite(eps) or eps <= 0:
        raise ShapeError("rms_norm eps must be positive")
    return (OutputSpec(x),)


def _copy_inplace(
    inputs: tuple[TensorSpec, ...], _: Mapping[str, Any]
) -> tuple[OutputSpec, ...]:
    _expect_arity("copy_inplace", inputs, 2)
    destination, source = inputs
    if destination != source:
        raise ShapeError("copy_inplace source and destination specs must match")
    return (OutputSpec(destination, alias_input=0, inherit_view=True),)


def _view(
    inputs: tuple[TensorSpec, ...], attrs: Mapping[str, Any]
) -> tuple[OutputSpec, ...]:
    _expect_arity("view", inputs, 1)
    source = inputs[0]
    if "shape" not in attrs:
        raise ShapeError("view requires a shape attribute")
    shape = tuple(attrs["shape"])
    strides = tuple(attrs["strides"]) if attrs.get("strides") is not None else None
    result = TensorSpec(
        shape,
        source.dtype,
        source.layout,
        storage_dtype=source.storage_dtype,
        quant=source.quant,
    )
    return (
        OutputSpec(
            result,
            alias_input=0,
            byte_offset=attrs.get("byte_offset", 0),
            strides=strides,
        ),
    )


def create_default_schema_registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register(OpSchema("add", lambda i, a: _same_shape_binary("add", i, a)))
    registry.register(OpSchema("mul", lambda i, a: _same_shape_binary("mul", i, a)))
    registry.register(OpSchema("silu", _unary("silu")))
    registry.register(OpSchema("matmul", _matmul))
    registry.register(OpSchema("rms_norm", _rms_norm, defaults={"eps": 1e-6}))
    registry.register(
        OpSchema(
            "copy_inplace",
            _copy_inplace,
            description="Reference state-buffer update used to validate alias hazards.",
        )
    )
    registry.register(
        OpSchema(
            "view",
            _view,
            defaults={"byte_offset": 0, "strides": None},
            description="Metadata-only storage alias; emits no runtime task.",
            metadata_only=True,
        )
    )
    return registry
