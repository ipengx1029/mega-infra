"""Graph -> TaskGraph -> ExecutionPlan -> CommandProgram compilation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .errors import CompilationError
from .ir import Graph, Layout, StorageKind
from .registry import Architecture, KernelRegistry, LoweringContext
from .runtime.abi import (
    ALIGNMENT,
    GUARD_STRUCT,
    UINT32_MAX,
    BufferDescriptor,
    BufferFlags,
    CommandProgram,
    OperandDescriptor,
    TaskFlags,
    TaskInstruction,
    ValueDescriptor,
)
from .scheduler import ExecutionPlan, schedule_tasks
from .task import (
    Access,
    AccessMode,
    ResourceKind,
    ResourceRef,
    Task,
    TaskDependencies,
    derive_dependencies,
)


@dataclass(frozen=True)
class CompileOptions:
    target: Architecture = Architecture.CPU
    num_workers: int = 4
    symbolic_bindings: Mapping[str, int] | None = None
    transitive_reduction: bool = True


@dataclass(frozen=True)
class CompiledArtifact:
    graph: Graph
    tasks: tuple[Task, ...]
    dependencies: TaskDependencies
    schedule: ExecutionPlan
    program: CommandProgram
    registry: KernelRegistry
    runtime_scalars: Mapping[int, str]

    def serialize(self) -> bytes:
        return self.program.to_bytes()


def _buffer_flags(graph: Graph, storage_id: int) -> BufferFlags:
    storage = graph.storage(storage_id)
    flags = {
        StorageKind.INPUT: BufferFlags.INPUT,
        StorageKind.CONSTANT: BufferFlags.INPUT | BufferFlags.CONSTANT,
        StorageKind.STATE: BufferFlags.INPUT | BufferFlags.STATE,
        StorageKind.TEMPORARY: BufferFlags.TEMPORARY,
    }[storage.kind]
    if any(value.storage_id == storage_id for value in graph.outputs):
        flags |= BufferFlags.OUTPUT
    if storage.spec.layout == Layout.OPAQUE:
        flags |= BufferFlags.OPAQUE_LAYOUT
    return flags


def _align_blob(blob: bytearray, alignment: int = ALIGNMENT) -> None:
    blob.extend(b"\0" * ((-len(blob)) % alignment))


def _access_covers_operand(access_mode: AccessMode, operand_mode: AccessMode) -> bool:
    return {
        AccessMode.READ: access_mode in (AccessMode.READ, AccessMode.READ_WRITE),
        AccessMode.WRITE: access_mode
        in (AccessMode.WRITE, AccessMode.READ_WRITE, AccessMode.REDUCE),
        AccessMode.READ_WRITE: access_mode == AccessMode.READ_WRITE,
        AccessMode.REDUCE: access_mode == AccessMode.REDUCE,
    }[operand_mode]


def _operand_covers_access(operand_mode: AccessMode, access_mode: AccessMode) -> bool:
    return {
        AccessMode.READ: operand_mode in (AccessMode.READ, AccessMode.READ_WRITE),
        AccessMode.WRITE: operand_mode in (AccessMode.WRITE, AccessMode.READ_WRITE),
        AccessMode.READ_WRITE: operand_mode == AccessMode.READ_WRITE,
        AccessMode.REDUCE: operand_mode == AccessMode.REDUCE,
    }[access_mode]


def _build_program(
    graph: Graph,
    tasks: tuple[Task, ...],
    dependencies: TaskDependencies,
    schedule: ExecutionPlan,
    target: Architecture,
) -> tuple[CommandProgram, dict[int, str]]:
    buffers = tuple(
        BufferDescriptor(
            storage.id,
            _buffer_flags(graph, storage.id),
            storage.spec.storage_dtype,
            tuple(int(dim) for dim in storage.spec.physical_shape),
            storage.spec.physical_strides(),
            storage.spec.nbytes(),
            storage.spec.quant.stable_id if storage.spec.quant is not None else 0,
        )
        for storage in graph.storages
    )
    values = tuple(
        ValueDescriptor(
            value.id,
            value.storage_id,
            value.spec.dtype,
            value.spec.concrete_shape(),
            value.resolved_strides(),
            value.byte_offset,
            value.spec.quant.stable_id if value.spec.quant is not None else 0,
        )
        for value in graph.values
    )
    operands: list[OperandDescriptor] = []
    predecessor_table: list[int] = []
    successor_table: list[int] = []
    params = bytearray()
    instructions: list[TaskInstruction] = []
    runtime_scalars: dict[int, str] = {}
    for task in tasks:
        operand_start = len(operands)
        operands.extend(
            OperandDescriptor(item.buffer_id, item.value_id, item.mode)
            for item in task.operands
        )
        dependency_start = len(predecessor_table)
        predecessor_table.extend(dependencies.predecessors[task.id])
        successor_start = len(successor_table)
        successor_table.extend(dependencies.successors[task.id])
        _align_blob(params)
        param_offset = len(params)
        flags = TaskFlags.NONE
        if task.guard is not None:
            flags |= TaskFlags.HAS_GUARD
            previous = runtime_scalars.get(task.guard.scalar_id)
            if previous is not None and previous != task.guard.scalar:
                raise CompilationError("runtime scalar stable-id collision")
            runtime_scalars[task.guard.scalar_id] = task.guard.scalar
            params.extend(
                GUARD_STRUCT.pack(
                    task.guard.scalar_id, int(task.guard.op), task.guard.value
                )
            )
        params.extend(task.params)
        x, y, z = task.coordinates
        instructions.append(
            TaskInstruction(
                task.id,
                task.kernel_id,
                flags,
                schedule.entry(task.id).worker_hint
                if schedule.num_workers > 0
                else UINT32_MAX,
                dependency_start,
                len(dependencies.predecessors[task.id]),
                successor_start,
                len(dependencies.successors[task.id]),
                operand_start,
                len(task.operands),
                param_offset,
                len(params) - param_offset,
                x,
                y,
                z,
            )
        )
    program = CommandProgram(
        target,
        buffers,
        values,
        tuple(instructions),
        tuple(operands),
        tuple(predecessor_table),
        tuple(successor_table),
        bytes(params),
    )
    program.validate()
    return program, runtime_scalars


def compile_graph(
    graph: Graph,
    registry: KernelRegistry,
    options: CompileOptions | None = None,
) -> CompiledArtifact:
    options = options or CompileOptions()
    graph.validate()
    if any(
        not isinstance(dim, int)
        for dimensions in (
            *(value.spec.shape for value in graph.values),
            *(storage.spec.physical_shape for storage in graph.storages),
        )
        for dim in dimensions
    ):
        if options.symbolic_bindings is None:
            raise CompilationError(
                "symbolic graph requires CompileOptions.symbolic_bindings"
            )
        graph = graph.bind(options.symbolic_bindings)
    else:
        graph = graph.bind(options.symbolic_bindings or {})
    graph.freeze()

    lowered: list[Task] = []
    for node in graph.nodes:
        if node.metadata_only:
            if (
                node.control_deps
                or node.effects
                or any(
                    graph.value(value_id).alias_input is None
                    for value_id in node.outputs
                )
            ):
                raise CompilationError("invalid metadata-only node contract")
            continue
        context = LoweringContext(graph, node, options.target)
        variant = registry.select(context)
        drafts = tuple(variant.lower(context))
        if not drafts:
            empty_dense_output = bool(node.outputs) and all(
                graph.value(value_id).spec.numel() == 0 for value_id in node.outputs
            )
            if not empty_dense_output or node.effects or not variant.empty_is_noop:
                raise CompilationError(
                    f"kernel {variant.name!r} produced no tasks for node {node.name!r}"
                )
            continue
        for draft in drafts:
            if draft.node_id != node.id:
                raise CompilationError(
                    f"kernel {variant.name!r} emitted task for node {draft.node_id}, expected {node.id}"
                )
            legal_values = set(node.inputs) | set(node.outputs)
            for operand in draft.operands:
                if operand.value_id not in legal_values:
                    raise CompilationError(
                        f"kernel {variant.name!r} task operand value {operand.value_id} "
                        f"is not an input/output of node {node.name!r}"
                    )
                value = graph.value(operand.value_id)
                if value.storage_id != operand.buffer_id:
                    raise CompilationError(
                        f"kernel {variant.name!r} operand buffer/value storage disagree"
                    )
                if not any(
                    access.resource == ResourceRef.buffer(operand.buffer_id)
                    and _access_covers_operand(access.mode, operand.mode)
                    for access in draft.accesses
                ):
                    raise CompilationError(
                        f"kernel {variant.name!r} operand {operand.value_id} has no "
                        f"matching {operand.mode.name} access declaration"
                    )
            for access in draft.accesses:
                if access.resource.kind != ResourceKind.BUFFER:
                    continue
                if access.value_id is None or access.value_id not in legal_values:
                    raise CompilationError(
                        f"kernel {variant.name!r} buffer access must name a node input/output"
                    )
                value = graph.value(access.value_id)
                if value.storage_id != access.resource.id:
                    raise CompilationError(
                        f"kernel {variant.name!r} access resource/value storage disagree"
                    )
                if not any(
                    operand.buffer_id == access.resource.id
                    and _operand_covers_access(operand.mode, access.mode)
                    for operand in draft.operands
                ):
                    raise CompilationError(
                        f"kernel {variant.name!r} buffer access has no matching operand"
                    )

            normalized_accesses = []
            for access in draft.accesses:
                if (
                    access.resource.kind == ResourceKind.BUFFER
                    and access.value_id is not None
                    and access.region is not None
                    and not access.region_is_storage
                ):
                    value = graph.value(access.value_id)
                    storage = graph.storage(value.storage_id)
                    is_base_view = (
                        value.byte_offset == 0
                        and value.spec == storage.spec
                        and value.resolved_strides()
                        == storage.spec.contiguous_strides()
                    )
                    if not is_base_view:
                        # Exact strided-view intersection is a later compiler pass.
                        # Whole-resource fallback is conservative and cannot miss a hazard.
                        access = replace(access, region=None)
                normalized_accesses.append(access)
            params = variant.parameters.pack(draft.params)
            effect_accesses = tuple(
                Access(
                    ResourceRef(ResourceKind.EFFECT, use.resource_id),
                    AccessMode(int(use.mode)),
                )
                for use in node.effects
            )
            lowered.append(
                Task(
                    len(lowered),
                    node.id,
                    variant.stable_id,
                    variant.name,
                    draft.coordinates,
                    draft.operands,
                    tuple(normalized_accesses) + effect_accesses,
                    params,
                    draft.estimated_cost,
                    draft.guard,
                )
            )
        for output_id in node.outputs:
            output = graph.value(output_id)
            if not any(
                access.resource.kind == ResourceKind.BUFFER
                and access.resource.id == output.storage_id
                and access.mode
                in (AccessMode.WRITE, AccessMode.READ_WRITE, AccessMode.REDUCE)
                for task in lowered
                if task.node_id == node.id
                for access in task.accesses
            ):
                raise CompilationError(
                    f"kernel {variant.name!r} did not declare a write for output {output.name!r}"
                )

    tasks = tuple(lowered)

    taskful_nodes = {task.node_id for task in tasks}
    nearest_ancestors: dict[int, set[int]] = {}
    for node in graph.nodes:
        if node.id in taskful_nodes:
            nearest_ancestors[node.id] = {node.id}
            continue
        parents = set(node.control_deps)
        parents.update(
            producer
            for value_id in node.inputs
            if (producer := graph.value(value_id).producer) is not None
        )
        nearest_ancestors[node.id] = (
            set().union(*(nearest_ancestors[parent] for parent in parents))
            if parents
            else set()
        )

    effective_control_deps: dict[int, tuple[int, ...]] = {}
    for node in graph.nodes:
        if node.id not in taskful_nodes:
            continue
        expanded: set[int] = set()
        for dependency in node.control_deps:
            expanded.update(nearest_ancestors[dependency])
        effective_control_deps[node.id] = tuple(sorted(expanded))

    dependencies = derive_dependencies(
        tasks,
        node_control_deps=effective_control_deps,
        transitive_reduction=options.transitive_reduction,
    )
    schedule = schedule_tasks(tasks, dependencies, num_workers=options.num_workers)
    program, runtime_scalars = _build_program(
        graph, tasks, dependencies, schedule, options.target
    )
    return CompiledArtifact(
        graph, tasks, dependencies, schedule, program, registry, runtime_scalars
    )
