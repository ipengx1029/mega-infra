"""Backend-neutral semantic graph IR.

The graph is SSA at the value level, while ``storage_id`` captures aliasing.
This distinction is important for KV-cache updates: a new logical value may be
produced in-place in an existing state buffer without losing WAR/WAW hazards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from operator import index as integer_index, mul
from typing import Any, Iterable, Mapping, Sequence
from types import MappingProxyType

from .errors import GraphValidationError, ShapeError


MAX_TENSOR_RANK = 8


def _freeze_attr(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_attr(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_attr(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_attr(item) for item in value)
    return deepcopy(value)


class DType(IntEnum):
    """Stable dtype codes shared with the command ABI."""

    BOOL = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    FP16 = 10
    BF16 = 11
    FP32 = 12
    FP8_E4M3 = 20
    FP8_E5M2 = 21
    FP8_E8M0 = 22
    FP4_E2M1 = 30

    @property
    def bits(self) -> int:
        return {
            DType.BOOL: 8,
            DType.INT32: 32,
            DType.INT64: 64,
            DType.UINT8: 8,
            DType.FP16: 16,
            DType.BF16: 16,
            DType.FP32: 32,
            DType.FP8_E4M3: 8,
            DType.FP8_E5M2: 8,
            DType.FP8_E8M0: 8,
            DType.FP4_E2M1: 4,
        }[self]


class Layout(IntEnum):
    """Logical tensor layout. OPAQUE lets a backend own the physical format."""

    ROW_MAJOR = 1
    COLUMN_MAJOR = 2
    OPAQUE = 255


class StorageKind(IntEnum):
    INPUT = 1
    CONSTANT = 2
    TEMPORARY = 3
    STATE = 4


class EffectMode(IntEnum):
    READ = 1
    WRITE = 2
    READ_WRITE = 3


@dataclass(frozen=True)
class EffectResource:
    id: int
    name: str
    lifetime: str = "invocation"


@dataclass(frozen=True)
class EffectUse:
    resource_id: int
    mode: EffectMode


@dataclass(frozen=True, order=True)
class Symbol:
    """A specialize-at-compile-time dimension.

    Runtime-ragged work (for example MoE expert token counts) is deliberately
    not represented as a Symbol. It is handled by dynamic task expansion in a
    later lowering stage, so normal dense shapes remain easy to validate.
    """

    name: str
    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ShapeError(f"invalid symbol name: {self.name!r}")
        if self.minimum < 0:
            raise ShapeError(f"symbol {self.name!r} minimum must be non-negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ShapeError(f"symbol {self.name!r} maximum is below its minimum")

    def resolve(self, bindings: Mapping[str, int]) -> int:
        if self.name not in bindings:
            raise ShapeError(f"missing value for symbolic dimension {self.name!r}")
        raw_value = bindings[self.name]
        if isinstance(raw_value, bool):
            raise ShapeError(f"symbol {self.name!r} must bind to an integer, not bool")
        try:
            value = integer_index(raw_value)
        except TypeError as exc:
            raise ShapeError(f"symbol {self.name!r} must bind to an integer") from exc
        if value < self.minimum or (self.maximum is not None and value > self.maximum):
            raise ShapeError(
                f"symbol {self.name!r}={value} is outside "
                f"[{self.minimum}, {self.maximum if self.maximum is not None else 'inf'}]"
            )
        return value


ShapeDim = int | Symbol


def _normalize_dim(dim: int | str | Symbol) -> ShapeDim:
    if isinstance(dim, str):
        return Symbol(dim)
    if isinstance(dim, Symbol):
        return dim
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ShapeError(f"invalid tensor dimension: {dim!r}")
    return dim


def _stable_u32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


@dataclass(frozen=True)
class QuantSpec:
    """Logical-to-packed storage contract for FP8/FP4 style tensors.

    The scale tensor itself remains a normal graph Value and is bound by the
    op/kernel ABI. This descriptor captures the reusable format semantics.
    """

    format: str
    storage_dtype: DType
    block_shape: tuple[int, ...]
    scale_dtype: DType
    accumulator_dtype: DType = DType.FP32

    def __post_init__(self) -> None:
        if not isinstance(self.format, str) or not self.format:
            raise ShapeError("quantization format must not be empty")
        raw_block_shape = tuple(self.block_shape)
        normalized_block_shape: list[int] = []
        for raw_value in raw_block_shape:
            if isinstance(raw_value, bool):
                raise ShapeError("quantization block dimensions must be integers")
            try:
                value = integer_index(raw_value)
            except TypeError as exc:
                raise ShapeError(
                    "quantization block dimensions must be integers"
                ) from exc
            normalized_block_shape.append(value)
        if not normalized_block_shape or any(
            value <= 0 for value in normalized_block_shape
        ):
            raise ShapeError("quantization block shape must be positive")
        try:
            object.__setattr__(self, "storage_dtype", DType(self.storage_dtype))
            object.__setattr__(self, "scale_dtype", DType(self.scale_dtype))
            object.__setattr__(self, "accumulator_dtype", DType(self.accumulator_dtype))
        except ValueError as exc:
            raise ShapeError(
                "quantization descriptor contains an invalid dtype"
            ) from exc
        object.__setattr__(self, "block_shape", tuple(normalized_block_shape))

    @property
    def stable_id(self) -> int:
        fields = ",".join(str(value) for value in self.block_shape)
        return _stable_u32(
            f"{self.format}:{int(self.storage_dtype)}:{fields}:"
            f"{int(self.scale_dtype)}:{int(self.accumulator_dtype)}"
        )


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[ShapeDim, ...]
    dtype: DType = DType.FP32
    layout: Layout = Layout.ROW_MAJOR
    storage_dtype: DType = DType.FP32
    physical_shape: tuple[ShapeDim, ...] = ()
    quant: QuantSpec | None = None

    def __init__(
        self,
        shape: Sequence[int | str | Symbol],
        dtype: DType = DType.FP32,
        layout: Layout = Layout.ROW_MAJOR,
        *,
        storage_dtype: DType | None = None,
        physical_shape: Sequence[int | str | Symbol] | None = None,
        quant: QuantSpec | None = None,
    ) -> None:
        normalized = tuple(_normalize_dim(dim) for dim in shape)
        if len(normalized) > MAX_TENSOR_RANK:
            raise ShapeError(
                f"rank {len(normalized)} exceeds ABI limit {MAX_TENSOR_RANK}"
            )
        object.__setattr__(self, "shape", normalized)
        object.__setattr__(self, "dtype", DType(dtype))
        object.__setattr__(self, "layout", Layout(layout))
        resolved_storage_dtype = (
            DType(storage_dtype) if storage_dtype is not None else DType(dtype)
        )
        normalized_physical = (
            tuple(_normalize_dim(dim) for dim in physical_shape)
            if physical_shape is not None
            else normalized
        )
        if len(normalized_physical) > MAX_TENSOR_RANK:
            raise ShapeError("physical tensor rank exceeds ABI limit")
        if quant is not None and quant.storage_dtype != resolved_storage_dtype:
            raise ShapeError("quant spec and tensor storage dtype disagree")
        object.__setattr__(self, "storage_dtype", resolved_storage_dtype)
        object.__setattr__(self, "physical_shape", normalized_physical)
        object.__setattr__(self, "quant", quant)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def concrete_shape(
        self, bindings: Mapping[str, int] | None = None
    ) -> tuple[int, ...]:
        bindings = bindings or {}
        return tuple(
            dim.resolve(bindings) if isinstance(dim, Symbol) else dim
            for dim in self.shape
        )

    def numel(self, bindings: Mapping[str, int] | None = None) -> int:
        return reduce(mul, self.concrete_shape(bindings), 1)

    def nbytes(self, bindings: Mapping[str, int] | None = None) -> int:
        bindings = bindings or {}
        physical = tuple(
            dim.resolve(bindings) if isinstance(dim, Symbol) else dim
            for dim in self.physical_shape
        )
        bits = reduce(mul, physical, 1) * self.storage_dtype.bits
        return (bits + 7) // 8

    def contiguous_strides(
        self, bindings: Mapping[str, int] | None = None
    ) -> tuple[int, ...]:
        shape = self.concrete_shape(bindings)
        if not shape:
            return ()
        if self.layout == Layout.OPAQUE:
            return tuple(0 for _ in shape)
        strides = [1] * len(shape)
        if self.layout == Layout.ROW_MAJOR:
            for index in range(len(shape) - 2, -1, -1):
                strides[index] = strides[index + 1] * shape[index + 1]
        else:
            for index in range(1, len(shape)):
                strides[index] = strides[index - 1] * shape[index - 1]
        return tuple(strides)

    def bind(self, bindings: Mapping[str, int]) -> "TensorSpec":
        physical = tuple(
            dim.resolve(bindings) if isinstance(dim, Symbol) else dim
            for dim in self.physical_shape
        )
        return TensorSpec(
            self.concrete_shape(bindings),
            self.dtype,
            self.layout,
            storage_dtype=self.storage_dtype,
            physical_shape=physical,
            quant=self.quant,
        )

    def physical_strides(
        self, bindings: Mapping[str, int] | None = None
    ) -> tuple[int, ...]:
        physical = TensorSpec(
            tuple(
                dim.resolve(bindings or {}) if isinstance(dim, Symbol) else dim
                for dim in self.physical_shape
            ),
            self.storage_dtype,
            self.layout,
        )
        return physical.contiguous_strides()


@dataclass(frozen=True)
class OutputSpec:
    """Schema-inferred node output, optionally aliasing an input storage."""

    tensor: TensorSpec
    alias_input: int | None = None
    byte_offset: int = 0
    strides: tuple[int, ...] | None = None
    inherit_view: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.byte_offset, bool):
            raise ShapeError("view byte offset must be an integer")
        try:
            byte_offset = integer_index(self.byte_offset)
        except TypeError as exc:
            raise ShapeError("view byte offset must be an integer") from exc
        if byte_offset < 0:
            raise ShapeError("view byte offset must be non-negative")
        object.__setattr__(self, "byte_offset", byte_offset)
        if self.strides is not None:
            normalized_strides: list[int] = []
            for raw_value in self.strides:
                if isinstance(raw_value, bool):
                    raise ShapeError("view strides must be integers")
                try:
                    value = integer_index(raw_value)
                except TypeError as exc:
                    raise ShapeError("view strides must be integers") from exc
                normalized_strides.append(value)
            if len(normalized_strides) != self.tensor.rank or any(
                value < 0 for value in normalized_strides
            ):
                raise ShapeError(
                    "view strides must be non-negative and match tensor rank"
                )
            object.__setattr__(self, "strides", tuple(normalized_strides))
        if self.inherit_view and self.alias_input is None:
            raise ShapeError("inherit_view requires an aliased output")


@dataclass(frozen=True)
class Storage:
    id: int
    name: str
    spec: TensorSpec
    kind: StorageKind


@dataclass(frozen=True)
class Value:
    id: int
    name: str
    spec: TensorSpec
    storage_id: int
    producer: int | None
    output_index: int = 0
    byte_offset: int = 0
    strides: tuple[int, ...] | None = None
    alias_input: int | None = None

    def resolved_strides(
        self, bindings: Mapping[str, int] | None = None
    ) -> tuple[int, ...]:
        return self.strides or self.spec.contiguous_strides(bindings)


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    op: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    control_deps: tuple[int, ...] = ()
    effects: tuple[EffectUse, ...] = ()
    metadata_only: bool = False


class Graph:
    """An append-only semantic graph with explicit values and storages."""

    def __init__(self, name: str = "graph") -> None:
        if not name:
            raise GraphValidationError("graph name must not be empty")
        self.name = name
        self._values: list[Value] = []
        self._storages: list[Storage] = []
        self._nodes: list[Node] = []
        self._effects: list[EffectResource] = []
        self._input_ids: list[int] = []
        self._output_ids: list[int] = []
        self._names: set[str] = set()
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _require_mutable(self) -> None:
        if self._frozen:
            raise GraphValidationError(f"graph {self.name!r} is frozen")

    def freeze(self) -> "Graph":
        self.validate()
        self._frozen = True
        return self

    @property
    def values(self) -> tuple[Value, ...]:
        return tuple(self._values)

    @property
    def storages(self) -> tuple[Storage, ...]:
        return tuple(self._storages)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    @property
    def effects(self) -> tuple[EffectResource, ...]:
        return tuple(self._effects)

    @property
    def inputs(self) -> tuple[Value, ...]:
        return tuple(self._values[index] for index in self._input_ids)

    @property
    def outputs(self) -> tuple[Value, ...]:
        return tuple(self._values[index] for index in self._output_ids)

    def value(self, value_id: int) -> Value:
        if value_id < 0:
            raise GraphValidationError(f"unknown value id {value_id}")
        try:
            return self._values[value_id]
        except IndexError as exc:
            raise GraphValidationError(f"unknown value id {value_id}") from exc

    def storage(self, storage_id: int) -> Storage:
        if storage_id < 0:
            raise GraphValidationError(f"unknown storage id {storage_id}")
        try:
            return self._storages[storage_id]
        except IndexError as exc:
            raise GraphValidationError(f"unknown storage id {storage_id}") from exc

    def _claim_name(self, requested: str) -> str:
        if not requested:
            raise GraphValidationError("IR names must not be empty")
        if requested in self._names:
            suffix = 1
            while f"{requested}_{suffix}" in self._names:
                suffix += 1
            requested = f"{requested}_{suffix}"
        self._names.add(requested)
        return requested

    def effect(self, name: str, *, lifetime: str = "invocation") -> EffectResource:
        self._require_mutable()
        if not name or any(item.name == name for item in self._effects):
            raise GraphValidationError(
                f"effect name must be unique and non-empty: {name!r}"
            )
        if lifetime not in {"invocation", "sequence", "request", "model"}:
            raise GraphValidationError(f"unsupported effect lifetime {lifetime!r}")
        resource = EffectResource(len(self._effects), name, lifetime)
        self._effects.append(resource)
        return resource

    def input(
        self,
        name: str,
        spec: TensorSpec,
        *,
        kind: StorageKind = StorageKind.INPUT,
    ) -> Value:
        self._require_mutable()
        if kind not in (StorageKind.INPUT, StorageKind.CONSTANT, StorageKind.STATE):
            raise GraphValidationError(
                f"graph input cannot use storage kind {kind.name}"
            )
        name = self._claim_name(name)
        storage_id = len(self._storages)
        self._storages.append(Storage(storage_id, name, spec, kind))
        value = Value(len(self._values), name, spec, storage_id, None)
        self._values.append(value)
        self._input_ids.append(value.id)
        return value

    def add_node(
        self,
        op: str,
        inputs: Sequence[Value],
        output_specs: Sequence[OutputSpec],
        *,
        attrs: Mapping[str, Any] | None = None,
        control_deps: Iterable[Node | int] = (),
        effects: Iterable[EffectUse | tuple[EffectResource, EffectMode]] = (),
        metadata_only: bool = False,
        name: str | None = None,
    ) -> Value | tuple[Value, ...]:
        """Atomically append a node; failed validation leaves the graph unchanged."""

        self._require_mutable()

        snapshot = (
            len(self._values),
            len(self._storages),
            len(self._nodes),
            set(self._names),
        )
        try:
            return self._add_node_impl(
                op,
                inputs,
                output_specs,
                attrs=attrs,
                control_deps=control_deps,
                effects=effects,
                metadata_only=metadata_only,
                name=name,
            )
        except Exception:
            value_count, storage_count, node_count, names = snapshot
            del self._values[value_count:]
            del self._storages[storage_count:]
            del self._nodes[node_count:]
            self._names.clear()
            self._names.update(names)
            raise

    def _add_node_impl(
        self,
        op: str,
        inputs: Sequence[Value],
        output_specs: Sequence[OutputSpec],
        *,
        attrs: Mapping[str, Any] | None = None,
        control_deps: Iterable[Node | int] = (),
        effects: Iterable[EffectUse | tuple[EffectResource, EffectMode]] = (),
        metadata_only: bool = False,
        name: str | None = None,
    ) -> Value | tuple[Value, ...]:
        if not op:
            raise GraphValidationError("node op must not be empty")
        for value in inputs:
            if (
                value.id < 0
                or value.id >= len(self._values)
                or self._values[value.id] is not value
            ):
                raise GraphValidationError(
                    f"input {value!r} does not belong to graph {self.name!r}"
                )
        node_id = len(self._nodes)
        node_name = self._claim_name(name or f"{op}_{node_id}")
        raw_control_deps = tuple(control_deps)
        dep_ids = tuple(
            dep.id if isinstance(dep, Node) else int(dep) for dep in raw_control_deps
        )
        for dep in raw_control_deps:
            if isinstance(dep, Node) and (
                dep.id < 0 or dep.id >= node_id or self._nodes[dep.id] is not dep
            ):
                raise GraphValidationError(
                    f"node {node_name!r} has a control dependency from another graph"
                )
        if any(dep < 0 or dep >= node_id for dep in dep_ids):
            raise GraphValidationError(
                f"node {node_name!r} has a control dependency that is not an earlier node"
            )
        effect_uses: list[EffectUse] = []
        for effect in effects:
            if isinstance(effect, EffectUse):
                use = effect
            else:
                resource, mode = effect
                if (
                    resource.id < 0
                    or resource.id >= len(self._effects)
                    or self._effects[resource.id] is not resource
                ):
                    raise GraphValidationError(
                        f"node {node_name!r} uses an effect from another graph"
                    )
                use = EffectUse(resource.id, EffectMode(mode))
            if use.resource_id < 0 or use.resource_id >= len(self._effects):
                raise GraphValidationError(f"node {node_name!r} uses an unknown effect")
            effect_uses.append(use)
        outputs: list[Value] = []
        for output_index, inferred in enumerate(output_specs):
            value_name = self._claim_name(
                node_name if len(output_specs) == 1 else f"{node_name}:{output_index}"
            )
            if inferred.alias_input is None:
                storage_id = len(self._storages)
                self._storages.append(
                    Storage(
                        storage_id, value_name, inferred.tensor, StorageKind.TEMPORARY
                    )
                )
            else:
                if inferred.alias_input < 0 or inferred.alias_input >= len(inputs):
                    raise GraphValidationError(
                        f"node {node_name!r} output {output_index} aliases invalid input "
                        f"{inferred.alias_input}"
                    )
                aliased = inputs[inferred.alias_input]
                storage_id = aliased.storage_id
            value = Value(
                len(self._values),
                value_name,
                inferred.tensor,
                storage_id,
                node_id,
                output_index,
                (
                    inputs[inferred.alias_input].byte_offset
                    if inferred.alias_input is not None
                    else 0
                )
                + inferred.byte_offset,
                (
                    inputs[inferred.alias_input].strides
                    if inferred.alias_input is not None and inferred.inherit_view
                    else inferred.strides
                ),
                inferred.alias_input,
            )
            self._values.append(value)
            outputs.append(value)
        node = Node(
            node_id,
            node_name,
            op,
            tuple(value.id for value in inputs),
            tuple(value.id for value in outputs),
            _freeze_attr(dict(attrs or {})),
            tuple(dict.fromkeys(dep_ids)),
            tuple(effect_uses),
            metadata_only,
        )
        self._nodes.append(node)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def mark_output(self, *values: Value) -> None:
        self._require_mutable()
        for value in values:
            if (
                value.id < 0
                or value.id >= len(self._values)
                or self._values[value.id] is not value
            ):
                raise GraphValidationError(
                    f"output {value!r} does not belong to graph {self.name!r}"
                )
            if value.id not in self._output_ids:
                self._output_ids.append(value.id)

    def validate(self) -> None:
        if not self._output_ids:
            raise GraphValidationError("graph has no outputs")
        if any(value.id != index for index, value in enumerate(self._values)):
            raise GraphValidationError("value ids are not contiguous")
        if any(storage.id != index for index, storage in enumerate(self._storages)):
            raise GraphValidationError("storage ids are not contiguous")
        if len(self._input_ids) != len(set(self._input_ids)):
            raise GraphValidationError("graph input ids contain duplicates")
        if any(
            value_id < 0 or value_id >= len(self._values)
            for value_id in self._input_ids
        ):
            raise GraphValidationError("graph input id is out of bounds")
        if any(
            value_id < 0 or value_id >= len(self._values)
            for value_id in self._output_ids
        ):
            raise GraphValidationError("graph output id is out of bounds")
        input_ids = set(self._input_ids)
        for value in self._values:
            if value.storage_id < 0 or value.storage_id >= len(self._storages):
                raise GraphValidationError(
                    f"value {value.name!r} references invalid storage"
                )
            if value.producer is None:
                if value.id not in input_ids:
                    raise GraphValidationError(
                        f"value {value.name!r} has no producer and is not a graph input"
                    )
            else:
                if value.producer < 0 or value.producer >= len(self._nodes):
                    raise GraphValidationError(
                        f"value {value.name!r} has an invalid producer"
                    )
                if value.id not in self._nodes[value.producer].outputs:
                    raise GraphValidationError(
                        f"value {value.name!r} is not owned by its producer node"
                    )
        for node_id, node in enumerate(self._nodes):
            if node.id != node_id:
                raise GraphValidationError("node ids are not contiguous")
            for value_id in node.inputs:
                if value_id < 0 or value_id >= len(self._values):
                    raise GraphValidationError(
                        f"node {node.name!r} has invalid input id"
                    )
                value = self.value(value_id)
                if value.producer is not None and value.producer >= node.id:
                    raise GraphValidationError(
                        f"node {node.name!r} consumes a value from a non-earlier node"
                    )
            for value_id in node.outputs:
                if value_id < 0 or value_id >= len(self._values):
                    raise GraphValidationError(
                        f"node {node.name!r} has invalid output id"
                    )
                output = self.value(value_id)
                if output.producer != node.id:
                    raise GraphValidationError(
                        f"node {node.name!r} has an invalid output owner"
                    )
                if output.alias_input is not None:
                    if output.alias_input < 0 or output.alias_input >= len(node.inputs):
                        raise GraphValidationError(
                            f"node {node.name!r} has invalid alias metadata"
                        )
                    if (
                        output.storage_id
                        != self.value(node.inputs[output.alias_input]).storage_id
                    ):
                        raise GraphValidationError(
                            f"node {node.name!r} alias storage does not match"
                        )
            if any(dep >= node.id for dep in node.control_deps):
                raise GraphValidationError(
                    f"node {node.name!r} has a forward control edge"
                )
            if any(use.resource_id >= len(self._effects) for use in node.effects):
                raise GraphValidationError(
                    f"node {node.name!r} has an invalid effect use"
                )
            if node.metadata_only:
                if node.control_deps or node.effects:
                    raise GraphValidationError(
                        f"metadata-only node {node.name!r} cannot carry control/effect edges"
                    )
                if any(
                    self.value(value_id).alias_input is None
                    for value_id in node.outputs
                ):
                    raise GraphValidationError(
                        f"metadata-only node {node.name!r} must alias all outputs"
                    )

    def bind(self, bindings: Mapping[str, int]) -> "Graph":
        """Clone the graph with all symbolic dimensions specialized."""

        clone = Graph(self.name)
        value_map: dict[int, Value] = {}
        node_map: dict[int, Node] = {}
        effect_map = {
            effect.id: clone.effect(effect.name, lifetime=effect.lifetime)
            for effect in self.effects
        }
        for value in self.inputs:
            storage = self.storage(value.storage_id)
            value_map[value.id] = clone.input(
                value.name, value.spec.bind(bindings), kind=storage.kind
            )
        for node in self.nodes:
            outputs = clone.add_node(
                node.op,
                [value_map[value_id] for value_id in node.inputs],
                [
                    OutputSpec(
                        self.value(value_id).spec.bind(bindings),
                        self.value(value_id).alias_input,
                        self.value(value_id).byte_offset
                        - (
                            self.value(
                                node.inputs[self.value(value_id).alias_input]
                            ).byte_offset
                            if self.value(value_id).alias_input is not None
                            else 0
                        ),
                        self.value(value_id).strides,
                    )
                    for value_id in node.outputs
                ],
                attrs=node.attrs,
                control_deps=[node_map[dep] for dep in node.control_deps],
                effects=[
                    (effect_map[use.resource_id], use.mode) for use in node.effects
                ],
                metadata_only=node.metadata_only,
                name=node.name,
            )
            output_values = (outputs,) if isinstance(outputs, Value) else outputs
            for old_id, new_value in zip(node.outputs, output_values):
                value_map[old_id] = new_value
            node_map[node.id] = clone.nodes[-1]
        clone.mark_output(*(value_map[value.id] for value in self.outputs))
        clone.validate()
        return clone
