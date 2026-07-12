"""Multi-variant backend kernel registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Protocol

from .errors import RegistryError
from .parameters import EMPTY_PARAMETERS, ParameterSchema
from .task import TaskDraft, stable_u32

if TYPE_CHECKING:
    from .ir import Graph, Node
    from .runtime.interpreter import ExecutionContext


class Architecture(IntEnum):
    CPU = 0
    SM90 = 90
    SM100 = 100


@dataclass(frozen=True)
class WarpRole:
    name: str
    first_warp: int
    warp_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise RegistryError("warp role name must not be empty")
        if self.first_warp < 0 or self.warp_count <= 0:
            raise RegistryError("warp role range must be non-negative and non-empty")


@dataclass(frozen=True)
class KernelResources:
    threads_per_cta: int = 1
    dynamic_smem_bytes: int = 0
    registers_per_thread_hint: int = 0
    cluster_shape: tuple[int, int, int] = (1, 1, 1)
    pipeline_stages: int = 1
    tmem_columns: int = 0
    warp_roles: tuple[WarpRole, ...] = ()
    requires_cooperative_launch: bool = False

    def __post_init__(self) -> None:
        if self.threads_per_cta <= 0 or self.threads_per_cta > 1024:
            raise RegistryError("threads_per_cta must be in [1, 1024]")
        if self.dynamic_smem_bytes < 0 or self.pipeline_stages <= 0:
            raise RegistryError("kernel resource values must be non-negative")
        if self.registers_per_thread_hint < 0 or self.tmem_columns < 0:
            raise RegistryError("register and TMEM resource hints must be non-negative")
        if len(self.cluster_shape) != 3 or any(
            value <= 0 for value in self.cluster_shape
        ):
            raise RegistryError("cluster_shape must be a positive xyz tuple")
        claimed_warps: set[int] = set()
        max_warps = (self.threads_per_cta + 31) // 32
        for role in self.warp_roles:
            role_warps = set(range(role.first_warp, role.first_warp + role.warp_count))
            if role_warps & claimed_warps:
                raise RegistryError("warp role ranges overlap")
            if role_warps and max(role_warps) >= max_warps:
                raise RegistryError("warp role exceeds threads_per_cta")
            claimed_warps.update(role_warps)


@dataclass(frozen=True)
class LoweringContext:
    graph: "Graph"
    node: "Node"
    target: Architecture


class LoweringFn(Protocol):
    def __call__(self, context: LoweringContext) -> Iterable[TaskDraft]: ...


class ExecutorFn(Protocol):
    def __call__(self, context: "ExecutionContext") -> None: ...


ConstraintFn = Callable[[LoweringContext], bool]


def _always(_: LoweringContext) -> bool:
    return True


@dataclass(frozen=True)
class KernelVariant:
    """A backend plugin contract for one semantic op implementation."""

    name: str
    op: str
    backend: str
    architectures: frozenset[Architecture]
    lower: LoweringFn
    parameters: ParameterSchema = EMPTY_PARAMETERS
    executor: ExecutorFn | None = None
    resources: KernelResources = KernelResources()
    constraint: ConstraintFn = _always
    priority: int = 0
    abi_version: int = 1
    tags: frozenset[str] = frozenset()
    empty_is_noop: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.op or not self.backend:
            raise RegistryError("kernel name, op, and backend must not be empty")
        if not self.architectures:
            raise RegistryError("kernel variant must support at least one architecture")
        if self.abi_version <= 0:
            raise RegistryError("kernel ABI version must be positive")

    @property
    def stable_id(self) -> int:
        return stable_u32(f"{self.backend}:{self.name}:v{self.abi_version}")


class KernelRegistry:
    def __init__(self) -> None:
        self._by_op: dict[str, list[KernelVariant]] = {}
        self._by_id: dict[int, KernelVariant] = {}

    def register(self, variant: KernelVariant) -> None:
        existing = self._by_id.get(variant.stable_id)
        if existing is not None:
            if existing == variant:
                raise RegistryError(
                    f"kernel variant {variant.name!r} is already registered"
                )
            raise RegistryError(
                f"stable kernel id collision: {variant.name!r} and {existing.name!r}"
            )
        self._by_id[variant.stable_id] = variant
        self._by_op.setdefault(variant.op, []).append(variant)

    def get(self, stable_id: int) -> KernelVariant:
        try:
            return self._by_id[stable_id]
        except KeyError as exc:
            raise RegistryError(f"unknown kernel stable id 0x{stable_id:08x}") from exc

    def variants(self, op: str) -> tuple[KernelVariant, ...]:
        return tuple(self._by_op.get(op, ()))

    def select(self, context: LoweringContext) -> KernelVariant:
        candidates = [
            variant
            for variant in self._by_op.get(context.node.op, ())
            if context.target in variant.architectures and variant.constraint(context)
        ]
        if not candidates:
            available = [
                variant.name for variant in self._by_op.get(context.node.op, ())
            ]
            raise RegistryError(
                f"no kernel variant for op={context.node.op!r}, target={context.target.name}; "
                f"registered={available}"
            )
        return min(
            candidates,
            key=lambda item: (
                -item.priority,
                item.name,
                item.backend,
                -item.abi_version,
                item.stable_id,
            ),
        )

    def manifest(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "id": stable_id,
                "name": variant.name,
                "op": variant.op,
                "backend": variant.backend,
                "abi_version": variant.abi_version,
                "architectures": [arch.name for arch in sorted(variant.architectures)],
                "empty_is_noop": variant.empty_is_noop,
            }
            for stable_id, variant in sorted(self._by_id.items())
        )
