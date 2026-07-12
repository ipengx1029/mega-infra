"""Task IR, access regions, runtime guards, and dependency derivation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
import math
from operator import index as integer_index
from typing import Any, Mapping, Sequence

from .errors import CompilationError
from .ir import DType


class AccessMode(IntEnum):
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    REDUCE = 4


class ResourceKind(IntEnum):
    BUFFER = 1
    EFFECT = 2
    WORKLIST = 3


@dataclass(frozen=True, order=True)
class ResourceRef:
    kind: ResourceKind
    id: int

    def __post_init__(self) -> None:
        if self.id < 0:
            raise CompilationError("resource id must be non-negative")

    @staticmethod
    def buffer(buffer_id: int) -> "ResourceRef":
        return ResourceRef(ResourceKind.BUFFER, buffer_id)


@dataclass(frozen=True)
class Interval:
    begin: int
    end: int

    def __post_init__(self) -> None:
        if self.begin < 0 or self.end < self.begin:
            raise CompilationError(
                f"invalid half-open interval [{self.begin}, {self.end})"
            )

    def overlaps(self, other: "Interval") -> bool:
        return self.begin < other.end and other.begin < self.end


@dataclass(frozen=True)
class Region:
    """A rectangular logical region; ``None`` in Access means whole/unknown."""

    axes: tuple[Interval, ...]

    def __init__(self, axes: Sequence[Interval | tuple[int, int]]) -> None:
        object.__setattr__(
            self,
            "axes",
            tuple(
                axis if isinstance(axis, Interval) else Interval(*axis) for axis in axes
            ),
        )

    @staticmethod
    def full(shape: Sequence[int]) -> "Region":
        return Region(tuple((0, extent) for extent in shape))

    def overlaps(self, other: "Region") -> bool:
        if len(self.axes) != len(other.axes):
            # Different-rank aliases/views need a byte-level proof. Until that
            # pass exists, conservatively preserve the dependency.
            return True
        return all(lhs.overlaps(rhs) for lhs, rhs in zip(self.axes, other.axes))


@dataclass(frozen=True)
class Access:
    resource: ResourceRef
    mode: AccessMode
    region: Region | None = None
    value_id: int | None = None
    reduction_op: str | None = None
    atomic: bool = False
    region_is_storage: bool = False

    def __post_init__(self) -> None:
        if self.mode == AccessMode.REDUCE and not self.reduction_op:
            raise CompilationError("REDUCE access requires a non-empty reduction_op")
        if self.mode != AccessMode.REDUCE and (
            self.reduction_op is not None or self.atomic
        ):
            raise CompilationError(
                "reduction metadata is only valid for REDUCE accesses"
            )

    def conflicts(self, previous: "Access") -> bool:
        if self.resource != previous.resource:
            return False
        if self.region is not None and previous.region is not None:
            if not self.region.overlaps(previous.region):
                return False
        if self.mode == AccessMode.READ and previous.mode == AccessMode.READ:
            return False
        if (
            self.mode == AccessMode.REDUCE
            and previous.mode == AccessMode.REDUCE
            and self.atomic
            and previous.atomic
            and self.reduction_op == previous.reduction_op
        ):
            return False
        return True


@dataclass(frozen=True)
class OperandBinding:
    value_id: int
    buffer_id: int
    mode: AccessMode


class GuardOp(IntEnum):
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6


def stable_u32(text: str) -> int:
    """FNV-1a, used for stable plugin and runtime-scalar identifiers."""

    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


@dataclass(frozen=True)
class Guard:
    """A simple runtime predicate for bounded dynamic work."""

    scalar: str
    op: GuardOp
    value: int

    def __post_init__(self) -> None:
        if not self.scalar or not self.scalar.isidentifier():
            raise CompilationError(f"invalid runtime scalar name {self.scalar!r}")
        object.__setattr__(self, "op", GuardOp(self.op))
        if isinstance(self.value, bool):
            raise CompilationError(
                "guard comparison value must be an integer, not bool"
            )
        try:
            value = integer_index(self.value)
        except TypeError as exc:
            raise CompilationError("guard comparison value must be an integer") from exc
        if value < -(1 << 63) or value >= (1 << 63):
            raise CompilationError("guard comparison value is outside int64")
        object.__setattr__(self, "value", value)

    @property
    def scalar_id(self) -> int:
        return stable_u32(self.scalar)

    def evaluate(self, runtime_scalars: Mapping[str, int]) -> bool:
        if self.scalar not in runtime_scalars:
            raise CompilationError(
                f"runtime scalar {self.scalar!r} required by task guard"
            )
        raw_lhs = runtime_scalars[self.scalar]
        if isinstance(raw_lhs, bool):
            raise CompilationError(
                f"runtime scalar {self.scalar!r} must be an integer, not bool"
            )
        try:
            lhs = integer_index(raw_lhs)
        except TypeError as exc:
            raise CompilationError(
                f"runtime scalar {self.scalar!r} must be an integer"
            ) from exc
        rhs = self.value
        return {
            GuardOp.EQ: lhs == rhs,
            GuardOp.NE: lhs != rhs,
            GuardOp.LT: lhs < rhs,
            GuardOp.LE: lhs <= rhs,
            GuardOp.GT: lhs > rhs,
            GuardOp.GE: lhs >= rhs,
        }[self.op]


@dataclass(frozen=True)
class WorklistSpec:
    """A bounded ragged/indirect launch domain for sparse attention and MoE."""

    name: str
    indices_buffer: int
    count_scalar: str
    capacity: int
    index_dtype: DType = DType.INT32
    indptr_buffer: int | None = None
    weights_buffer: int | None = None
    sentinel: int = -1
    logical_index_space: str = "flat"

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise CompilationError("worklist capacity must be non-negative")
        if not self.name or not self.count_scalar:
            raise CompilationError("worklist name and count scalar must not be empty")
        if self.indices_buffer < 0 or any(
            value is not None and value < 0
            for value in (self.indptr_buffer, self.weights_buffer)
        ):
            raise CompilationError("worklist buffer ids must be non-negative")
        try:
            object.__setattr__(self, "index_dtype", DType(self.index_dtype))
        except ValueError as exc:
            raise CompilationError("worklist has an invalid index dtype") from exc
        if self.index_dtype not in (DType.INT32, DType.INT64):
            raise CompilationError("worklist index dtype must be INT32 or INT64")


@dataclass(frozen=True)
class TaskDraft:
    node_id: int
    coordinates: tuple[int, int, int]
    operands: tuple[OperandBinding, ...]
    accesses: tuple[Access, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    estimated_cost: float = 1.0
    guard: Guard | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _validate_coordinates(self.coordinates))
        if isinstance(self.estimated_cost, bool):
            raise CompilationError(
                "task estimated cost must be a finite positive number"
            )
        try:
            cost = float(self.estimated_cost)
        except (TypeError, ValueError) as exc:
            raise CompilationError("task estimated cost must be numeric") from exc
        if not math.isfinite(cost) or cost <= 0:
            raise CompilationError(
                "task estimated cost must be a finite positive number"
            )
        object.__setattr__(self, "estimated_cost", cost)
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "accesses", tuple(self.accesses))


@dataclass(frozen=True)
class Task:
    id: int
    node_id: int
    kernel_id: int
    kernel_name: str
    coordinates: tuple[int, int, int]
    operands: tuple[OperandBinding, ...]
    accesses: tuple[Access, ...]
    params: bytes
    estimated_cost: float
    guard: Guard | None = None

    def __post_init__(self) -> None:
        if self.id < 0 or self.node_id < 0:
            raise CompilationError("task and node ids must be non-negative")
        object.__setattr__(self, "coordinates", _validate_coordinates(self.coordinates))
        if not math.isfinite(self.estimated_cost) or self.estimated_cost <= 0:
            raise CompilationError("task estimated cost must be finite and positive")
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "accesses", tuple(self.accesses))


def _validate_coordinates(coordinates: Sequence[int]) -> tuple[int, int, int]:
    if len(coordinates) != 3:
        raise CompilationError("task coordinates must be an xyz tuple")
    normalized: list[int] = []
    for coordinate in coordinates:
        if isinstance(coordinate, bool):
            raise CompilationError("task coordinates must be integers, not bool")
        try:
            value = integer_index(coordinate)
        except TypeError as exc:
            raise CompilationError("task coordinates must be integers") from exc
        if value < 0 or value > (1 << 32) - 1:
            raise CompilationError("task coordinate is outside uint32")
        normalized.append(value)
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True)
class TaskDependencies:
    predecessors: tuple[tuple[int, ...], ...]
    successors: tuple[tuple[int, ...], ...]

    @property
    def edge_count(self) -> int:
        return sum(len(items) for items in self.predecessors)


def _reachable(
    start: int,
    target: int,
    successors: Mapping[int, set[int]],
    ignored: tuple[int, int],
) -> bool:
    stack = [start]
    seen = {start}
    while stack:
        current = stack.pop()
        for child in successors.get(current, set()):
            if (current, child) == ignored:
                continue
            if child == target:
                return True
            if child not in seen and child < target:
                seen.add(child)
                stack.append(child)
    return False


def derive_dependencies(
    tasks: Sequence[Task],
    *,
    node_control_deps: Mapping[int, Sequence[int]] | None = None,
    transitive_reduction: bool = True,
) -> TaskDependencies:
    """Derive RAW/WAR/WAW edges from one ordered access stream.

    Task IDs must be topological candidates (all graph producers are lowered
    first). Region overlap allows a consumer tile to start before unrelated
    producer tiles finish.
    """

    if any(task.id != index for index, task in enumerate(tasks)):
        raise CompilationError(
            "task ids must be contiguous before dependency derivation"
        )
    successors: dict[int, set[int]] = defaultdict(set)
    history: dict[ResourceRef, list[tuple[int, Access]]] = defaultdict(list)
    regional_resources: set[ResourceRef] = set()
    whole_writers: dict[ResourceRef, dict[int, Access]] = defaultdict(dict)
    whole_readers: dict[ResourceRef, dict[int, Access]] = defaultdict(dict)
    for task in tasks:
        for access in task.accesses:
            resource = access.resource
            if resource not in regional_resources and access.region is None:
                writers = whole_writers[resource]
                readers = whole_readers[resource]
                if access.mode == AccessMode.READ:
                    for prior_id, prior_access in writers.items():
                        if prior_id != task.id and access.conflicts(prior_access):
                            successors[prior_id].add(task.id)
                    readers[task.id] = access
                    continue

                for prior_id, prior_access in writers.items():
                    if prior_id != task.id and access.conflicts(prior_access):
                        successors[prior_id].add(task.id)
                for prior_id in readers:
                    if prior_id != task.id:
                        successors[prior_id].add(task.id)
                if access.mode == AccessMode.REDUCE and access.atomic:
                    # Unordered compatible reductions must each retain the
                    # same pre-group writer/reader prerequisites; depending on
                    # a sibling reduction would incorrectly serialize them.
                    surviving_writers = {**writers, **readers}
                else:
                    surviving_writers = {
                        prior_id: prior_access
                        for prior_id, prior_access in writers.items()
                        if not access.conflicts(prior_access)
                    }
                surviving_writers[task.id] = access
                whole_writers[resource] = surviving_writers
                whole_readers[resource] = {}
                continue

            if resource not in regional_resources:
                history[resource] = [
                    *whole_writers[resource].items(),
                    *whole_readers[resource].items(),
                ]
                regional_resources.add(resource)
            for prior_id, prior_access in history[resource]:
                if prior_id != task.id and access.conflicts(prior_access):
                    successors[prior_id].add(task.id)
            if access.region is None and access.mode in (
                AccessMode.WRITE,
                AccessMode.READ_WRITE,
            ):
                history[resource] = [(task.id, access)]
            else:
                history[resource].append((task.id, access))

    if node_control_deps:
        by_node: dict[int, list[int]] = defaultdict(list)
        for task in tasks:
            by_node[task.node_id].append(task.id)
        for node_id, dependencies in node_control_deps.items():
            for before_node in dependencies:
                for before in by_node.get(before_node, ()):
                    for after in by_node.get(node_id, ()):
                        successors[before].add(after)

    for parent, children in successors.items():
        if any(child <= parent for child in children):
            raise CompilationError("dependency builder produced a backward edge")

    if transitive_reduction:
        for parent in range(len(tasks)):
            for child in sorted(tuple(successors.get(parent, set()))):
                if _reachable(parent, child, successors, (parent, child)):
                    successors[parent].remove(child)

    predecessors: list[list[int]] = [[] for _ in tasks]
    normalized_successors: list[tuple[int, ...]] = []
    for parent in range(len(tasks)):
        children = tuple(sorted(successors.get(parent, set())))
        normalized_successors.append(children)
        for child in children:
            predecessors[child].append(parent)
    return TaskDependencies(
        tuple(tuple(sorted(items)) for items in predecessors),
        tuple(normalized_successors),
    )
