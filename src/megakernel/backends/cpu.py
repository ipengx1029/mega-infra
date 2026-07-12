"""NumPy backend plugins used as the semantic oracle for phase 1."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..ir import DType, Layout
from ..parameters import ParamField, ParameterSchema, ParamType
from ..registry import (
    Architecture,
    KernelRegistry,
    KernelResources,
    KernelVariant,
    LoweringContext,
)
from ..runtime.interpreter import ExecutionContext
from ..task import (
    Access,
    AccessMode,
    OperandBinding,
    Region,
    ResourceRef,
    TaskDraft,
)


ELEMENTWISE_PARAMS = ParameterSchema(
    ParamField("row_begin", ParamType.U32),
    ParamField("row_end", ParamType.U32),
    ParamField("col_begin", ParamType.U32),
    ParamField("col_end", ParamType.U32),
)
MATMUL_PARAMS = ParameterSchema(
    ParamField("m_begin", ParamType.U32),
    ParamField("m_end", ParamType.U32),
    ParamField("n_begin", ParamType.U32),
    ParamField("n_end", ParamType.U32),
    ParamField("k", ParamType.U32),
)
RMS_NORM_PARAMS = ParameterSchema(
    ParamField("row_begin", ParamType.U32),
    ParamField("row_end", ParamType.U32),
    ParamField("hidden", ParamType.U32),
    ParamField("eps", ParamType.F32),
)


def _cpu_fp32(context: LoweringContext) -> bool:
    graph = context.graph
    node = context.node
    values = [graph.value(value_id) for value_id in (*node.inputs, *node.outputs)]
    return (
        all(
            value.spec.dtype == DType.FP32
            and value.spec.layout == Layout.ROW_MAJOR
            and value.spec.storage_dtype == DType.FP32
            and value.spec.quant is None
            and value.spec.physical_shape == value.spec.shape
            and value.byte_offset == 0
            and value.resolved_strides() == value.spec.contiguous_strides()
            and value.spec.shape == graph.storage(value.storage_id).spec.shape
            for value in values
        )
        and all(
            storage.spec.storage_dtype == DType.FP32
            and storage.spec.quant is None
            and storage.spec.physical_shape == storage.spec.shape
            for storage in {
                graph.storage(value.storage_id).id: graph.storage(value.storage_id)
                for value in values
            }.values()
        )
        and (
            context.node.op == "rms_norm"
            or all(value.spec.rank == 2 for value in values)
        )
    )


def _tiles(extent: int, size: int):
    for begin in range(0, extent, size):
        yield begin, min(begin + size, extent)


def _elementwise_lower(tile_m: int, tile_n: int):
    def lower(context: LoweringContext) -> Iterable[TaskDraft]:
        graph, node = context.graph, context.node
        inputs = [graph.value(value_id) for value_id in node.inputs]
        output = graph.value(node.outputs[0])
        m, n = output.spec.concrete_shape()
        for tile_row, (row_begin, row_end) in enumerate(_tiles(m, tile_m)):
            for tile_col, (col_begin, col_end) in enumerate(_tiles(n, tile_n)):
                region = Region(((row_begin, row_end), (col_begin, col_end)))
                operands = tuple(
                    OperandBinding(value.id, value.storage_id, AccessMode.READ)
                    for value in inputs
                ) + (OperandBinding(output.id, output.storage_id, AccessMode.WRITE),)
                accesses = tuple(
                    Access(
                        ResourceRef.buffer(value.storage_id),
                        AccessMode.READ,
                        region,
                        value.id,
                    )
                    for value in inputs
                ) + (
                    Access(
                        ResourceRef.buffer(output.storage_id),
                        AccessMode.WRITE,
                        region,
                        output.id,
                    ),
                )
                yield TaskDraft(
                    node.id,
                    (tile_row, tile_col, 0),
                    operands,
                    accesses,
                    {
                        "row_begin": row_begin,
                        "row_end": row_end,
                        "col_begin": col_begin,
                        "col_end": col_end,
                    },
                    estimated_cost=max(
                        1.0, (row_end - row_begin) * (col_end - col_begin)
                    ),
                )

    return lower


def _matmul_lower(tile_m: int, tile_n: int):
    def lower(context: LoweringContext) -> Iterable[TaskDraft]:
        graph, node = context.graph, context.node
        lhs, rhs = (graph.value(value_id) for value_id in node.inputs)
        output = graph.value(node.outputs[0])
        m, k = lhs.spec.concrete_shape()
        _, n = rhs.spec.concrete_shape()
        for tile_row, (m_begin, m_end) in enumerate(_tiles(m, tile_m)):
            for tile_col, (n_begin, n_end) in enumerate(_tiles(n, tile_n)):
                lhs_region = Region(((m_begin, m_end), (0, k)))
                rhs_region = Region(((0, k), (n_begin, n_end)))
                output_region = Region(((m_begin, m_end), (n_begin, n_end)))
                yield TaskDraft(
                    node.id,
                    (tile_row, tile_col, 0),
                    (
                        OperandBinding(lhs.id, lhs.storage_id, AccessMode.READ),
                        OperandBinding(rhs.id, rhs.storage_id, AccessMode.READ),
                        OperandBinding(output.id, output.storage_id, AccessMode.WRITE),
                    ),
                    (
                        Access(
                            ResourceRef.buffer(lhs.storage_id),
                            AccessMode.READ,
                            lhs_region,
                            lhs.id,
                        ),
                        Access(
                            ResourceRef.buffer(rhs.storage_id),
                            AccessMode.READ,
                            rhs_region,
                            rhs.id,
                        ),
                        Access(
                            ResourceRef.buffer(output.storage_id),
                            AccessMode.WRITE,
                            output_region,
                            output.id,
                        ),
                    ),
                    {
                        "m_begin": m_begin,
                        "m_end": m_end,
                        "n_begin": n_begin,
                        "n_end": n_end,
                        "k": k,
                    },
                    estimated_cost=max(
                        1.0, 2.0 * (m_end - m_begin) * (n_end - n_begin) * k
                    ),
                )

    return lower


def _rms_norm_lower(tile_m: int):
    def lower(context: LoweringContext) -> Iterable[TaskDraft]:
        graph, node = context.graph, context.node
        value, weight = (graph.value(value_id) for value_id in node.inputs)
        output = graph.value(node.outputs[0])
        m, hidden = value.spec.concrete_shape()
        for tile_row, (row_begin, row_end) in enumerate(_tiles(m, tile_m)):
            value_region = Region(((row_begin, row_end), (0, hidden)))
            weight_region = Region(((0, hidden),))
            yield TaskDraft(
                node.id,
                (tile_row, 0, 0),
                (
                    OperandBinding(value.id, value.storage_id, AccessMode.READ),
                    OperandBinding(weight.id, weight.storage_id, AccessMode.READ),
                    OperandBinding(output.id, output.storage_id, AccessMode.WRITE),
                ),
                (
                    Access(
                        ResourceRef.buffer(value.storage_id),
                        AccessMode.READ,
                        value_region,
                        value.id,
                    ),
                    Access(
                        ResourceRef.buffer(weight.storage_id),
                        AccessMode.READ,
                        weight_region,
                        weight.id,
                    ),
                    Access(
                        ResourceRef.buffer(output.storage_id),
                        AccessMode.WRITE,
                        value_region,
                        output.id,
                    ),
                ),
                {
                    "row_begin": row_begin,
                    "row_end": row_end,
                    "hidden": hidden,
                    "eps": float(node.attrs["eps"]),
                },
                estimated_cost=max(1.0, 5.0 * (row_end - row_begin) * hidden),
            )

    return lower


def _copy_lower(tile_m: int, tile_n: int):
    def lower(context: LoweringContext) -> Iterable[TaskDraft]:
        graph, node = context.graph, context.node
        destination, source = (graph.value(value_id) for value_id in node.inputs)
        output = graph.value(node.outputs[0])
        m, n = output.spec.concrete_shape()
        for tile_row, (row_begin, row_end) in enumerate(_tiles(m, tile_m)):
            for tile_col, (col_begin, col_end) in enumerate(_tiles(n, tile_n)):
                region = Region(((row_begin, row_end), (col_begin, col_end)))
                yield TaskDraft(
                    node.id,
                    (tile_row, tile_col, 0),
                    (
                        OperandBinding(
                            destination.id, destination.storage_id, AccessMode.WRITE
                        ),
                        OperandBinding(source.id, source.storage_id, AccessMode.READ),
                    ),
                    (
                        Access(
                            ResourceRef.buffer(source.storage_id),
                            AccessMode.READ,
                            region,
                            source.id,
                        ),
                        Access(
                            ResourceRef.buffer(destination.storage_id),
                            AccessMode.WRITE,
                            region,
                            output.id,
                        ),
                    ),
                    {
                        "row_begin": row_begin,
                        "row_end": row_end,
                        "col_begin": col_begin,
                        "col_end": col_end,
                    },
                    estimated_cost=max(
                        1.0, (row_end - row_begin) * (col_end - col_begin)
                    ),
                )

    return lower


def _slice(parameters):
    return (
        slice(parameters["row_begin"], parameters["row_end"]),
        slice(parameters["col_begin"], parameters["col_end"]),
    )


def _execute_add(context: ExecutionContext) -> None:
    tile = _slice(context.parameters)
    context.array(2)[tile] = context.array(0)[tile] + context.array(1)[tile]


def _execute_mul(context: ExecutionContext) -> None:
    tile = _slice(context.parameters)
    context.array(2)[tile] = context.array(0)[tile] * context.array(1)[tile]


def _execute_silu(context: ExecutionContext) -> None:
    tile = _slice(context.parameters)
    values = context.array(0)[tile]
    context.array(1)[tile] = values / (1.0 + np.exp(-values))


def _execute_matmul(context: ExecutionContext) -> None:
    p = context.parameters
    rows = slice(p["m_begin"], p["m_end"])
    cols = slice(p["n_begin"], p["n_end"])
    context.array(2)[rows, cols] = (
        context.array(0)[rows, : p["k"]] @ context.array(1)[: p["k"], cols]
    )


def _execute_rms_norm(context: ExecutionContext) -> None:
    p = context.parameters
    rows = slice(p["row_begin"], p["row_end"])
    values = context.array(0)[rows, : p["hidden"]]
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    context.array(2)[rows, : p["hidden"]] = (
        values * (1.0 / np.sqrt(variance + p["eps"])) * context.array(1)[: p["hidden"]]
    )


def _execute_copy(context: ExecutionContext) -> None:
    tile = _slice(context.parameters)
    context.array(0)[tile] = context.array(1)[tile]


def create_cpu_registry(*, tile_m: int = 2, tile_n: int = 4) -> KernelRegistry:
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError("CPU tile dimensions must be positive")
    registry = KernelRegistry()
    common = {
        "backend": "numpy",
        "architectures": frozenset({Architecture.CPU}),
        "resources": KernelResources(threads_per_cta=1),
        "constraint": _cpu_fp32,
        "empty_is_noop": True,
    }
    registry.register(
        KernelVariant(
            "numpy.add.tile2d",
            "add",
            lower=_elementwise_lower(tile_m, tile_n),
            parameters=ELEMENTWISE_PARAMS,
            executor=_execute_add,
            **common,
        )
    )
    registry.register(
        KernelVariant(
            "numpy.mul.tile2d",
            "mul",
            lower=_elementwise_lower(tile_m, tile_n),
            parameters=ELEMENTWISE_PARAMS,
            executor=_execute_mul,
            **common,
        )
    )
    registry.register(
        KernelVariant(
            "numpy.silu.tile2d",
            "silu",
            lower=_elementwise_lower(tile_m, tile_n),
            parameters=ELEMENTWISE_PARAMS,
            executor=_execute_silu,
            **common,
        )
    )
    registry.register(
        KernelVariant(
            "numpy.matmul.tile2d",
            "matmul",
            lower=_matmul_lower(tile_m, tile_n),
            parameters=MATMUL_PARAMS,
            executor=_execute_matmul,
            **common,
        )
    )
    registry.register(
        KernelVariant(
            "numpy.rms_norm.rows",
            "rms_norm",
            lower=_rms_norm_lower(tile_m),
            parameters=RMS_NORM_PARAMS,
            executor=_execute_rms_norm,
            **common,
        )
    )
    registry.register(
        KernelVariant(
            "numpy.copy_inplace.tile2d",
            "copy_inplace",
            lower=_copy_lower(tile_m, tile_n),
            parameters=ELEMENTWISE_PARAMS,
            executor=_execute_copy,
            **common,
        )
    )
    return registry
