"""Ergonomic Python graph builder."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .ir import (
    EffectMode,
    EffectResource,
    EffectUse,
    Graph,
    Node,
    StorageKind,
    TensorSpec,
    Value,
)
from .schema import SchemaRegistry, create_default_schema_registry


class GraphBuilder:
    """Build a semantic graph through registered, backend-neutral op schemas."""

    def __init__(
        self,
        name: str = "graph",
        *,
        schemas: SchemaRegistry | None = None,
    ) -> None:
        self.graph = Graph(name)
        self.schemas = schemas or create_default_schema_registry()

    def input(self, name: str, spec: TensorSpec) -> Value:
        return self.graph.input(name, spec, kind=StorageKind.INPUT)

    def parameter(self, name: str, spec: TensorSpec) -> Value:
        return self.graph.input(name, spec, kind=StorageKind.CONSTANT)

    def state(self, name: str, spec: TensorSpec) -> Value:
        return self.graph.input(name, spec, kind=StorageKind.STATE)

    def effect(self, name: str, *, lifetime: str = "invocation") -> EffectResource:
        return self.graph.effect(name, lifetime=lifetime)

    def call(
        self,
        op: str,
        *inputs: Value,
        attrs: Mapping[str, Any] | None = None,
        control_deps: Iterable[Node | int] = (),
        effects: Iterable[EffectUse | tuple[EffectResource, EffectMode]] = (),
        name: str | None = None,
    ) -> Value | tuple[Value, ...]:
        schema = self.schemas.get(op)
        output_specs, merged_attrs = schema.infer_outputs(
            [value.spec for value in inputs], attrs
        )
        return self.graph.add_node(
            op,
            inputs,
            output_specs,
            attrs=merged_attrs,
            control_deps=control_deps,
            effects=effects,
            metadata_only=schema.metadata_only,
            name=name,
        )

    def output(self, *values: Value) -> Graph:
        self.graph.mark_output(*values)
        self.graph.validate()
        return self.graph

    def add(self, lhs: Value, rhs: Value, *, name: str | None = None) -> Value:
        return self.call("add", lhs, rhs, name=name)  # type: ignore[return-value]

    def mul(self, lhs: Value, rhs: Value, *, name: str | None = None) -> Value:
        return self.call("mul", lhs, rhs, name=name)  # type: ignore[return-value]

    def silu(self, value: Value, *, name: str | None = None) -> Value:
        return self.call("silu", value, name=name)  # type: ignore[return-value]

    def matmul(self, lhs: Value, rhs: Value, *, name: str | None = None) -> Value:
        return self.call("matmul", lhs, rhs, name=name)  # type: ignore[return-value]

    def rms_norm(
        self, value: Value, weight: Value, *, eps: float = 1e-6, name: str | None = None
    ) -> Value:
        return self.call("rms_norm", value, weight, attrs={"eps": eps}, name=name)  # type: ignore[return-value]

    def copy_inplace(
        self, destination: Value, source: Value, *, name: str | None = None
    ) -> Value:
        return self.call("copy_inplace", destination, source, name=name)  # type: ignore[return-value]

    def view(
        self,
        value: Value,
        shape: Sequence[int | str],
        *,
        byte_offset: int = 0,
        strides: Sequence[int] | None = None,
        name: str | None = None,
    ) -> Value:
        return self.call(
            "view",
            value,
            attrs={
                "shape": tuple(shape),
                "byte_offset": byte_offset,
                "strides": tuple(strides) if strides is not None else None,
            },
            name=name,
        )  # type: ignore[return-value]


def tensor(
    shape: Sequence[int | str],
    *,
    dtype=None,
):
    """Short TensorSpec helper kept intentionally tiny for frontend examples."""

    if dtype is None:
        return TensorSpec(shape)
    return TensorSpec(shape, dtype=dtype)
