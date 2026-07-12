"""CPU reference runtime for compiled task programs."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

import numpy as np

from ..errors import RuntimeExecutionError
from ..ir import DType, Layout, StorageKind
from ..registry import KernelVariant
from ..task import OperandBinding, Task

if TYPE_CHECKING:
    from ..compiler import CompiledArtifact


def _numpy_dtype(dtype: DType):
    mapping = {
        DType.BOOL: np.bool_,
        DType.INT32: np.int32,
        DType.INT64: np.int64,
        DType.UINT8: np.uint8,
        DType.FP16: np.float16,
        DType.FP32: np.float32,
    }
    if dtype not in mapping:
        raise RuntimeExecutionError(
            f"CPU reference runtime does not support {dtype.name}"
        )
    return mapping[dtype]


@dataclass
class ExecutionContext:
    task: Task
    variant: KernelVariant
    arrays: MutableMapping[int, np.ndarray]
    parameters: Mapping[str, Any]
    operands: tuple[OperandBinding, ...]
    runtime_scalars: Mapping[str, int]

    def array(self, operand_index: int) -> np.ndarray:
        try:
            return self.arrays[self.operands[operand_index].buffer_id]
        except (IndexError, KeyError) as exc:
            raise RuntimeExecutionError(
                f"task {self.task.id} has no bound operand {operand_index}"
            ) from exc


class CPUInterpreter:
    def run(
        self,
        artifact: CompiledArtifact,
        bindings: Mapping[str, np.ndarray],
        *,
        runtime_scalars: Mapping[str, int] | None = None,
    ) -> dict[str, np.ndarray]:
        if artifact.program.target.name != "CPU":
            raise RuntimeExecutionError(
                "CPUInterpreter can only execute CPU command programs"
            )
        runtime_scalars = runtime_scalars or {}
        arrays: dict[int, np.ndarray] = {}
        for storage in artifact.graph.storages:
            spec = storage.spec
            if (
                spec.layout != Layout.ROW_MAJOR
                or spec.quant is not None
                or spec.dtype != spec.storage_dtype
                or spec.physical_shape != spec.shape
            ):
                raise RuntimeExecutionError(
                    f"storage {storage.name!r} uses a physical/logical layout unsupported "
                    "by the phase-1 CPU runtime"
                )
            _numpy_dtype(spec.storage_dtype)
        for value in artifact.graph.values:
            storage = artifact.graph.storage(value.storage_id)
            if (
                value.spec.dtype != storage.spec.storage_dtype
                or value.spec.quant is not None
            ):
                raise RuntimeExecutionError(
                    f"value {value.name!r} requires unsupported dtype reinterpretation"
                )
        expected_names = {
            value.name
            for value in artifact.graph.inputs
            if artifact.graph.storage(value.storage_id).kind
            in (StorageKind.INPUT, StorageKind.CONSTANT, StorageKind.STATE)
        }
        missing = expected_names - set(bindings)
        extra = set(bindings) - expected_names
        if missing or extra:
            raise RuntimeExecutionError(
                f"binding names differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for value in artifact.graph.inputs:
            array = np.asarray(bindings[value.name])
            shape = value.spec.concrete_shape()
            if array.shape != shape:
                raise RuntimeExecutionError(
                    f"binding {value.name!r} shape {array.shape} does not match {shape}"
                )
            expected_dtype = np.dtype(_numpy_dtype(value.spec.dtype))
            if array.dtype != expected_dtype:
                raise RuntimeExecutionError(
                    f"binding {value.name!r} dtype {array.dtype} does not match {expected_dtype}"
                )
            if not array.flags.c_contiguous:
                raise RuntimeExecutionError(
                    f"binding {value.name!r} must be C-contiguous in the phase-1 CPU runtime"
                )
            arrays[value.storage_id] = array
        for storage in artifact.graph.storages:
            if storage.id not in arrays:
                arrays[storage.id] = np.empty(
                    storage.spec.concrete_shape(),
                    dtype=_numpy_dtype(storage.spec.dtype),
                )

        indegree = [len(items) for items in artifact.dependencies.predecessors]
        dispatch_rank = {
            task_id: index
            for index, task_id in enumerate(artifact.schedule.dispatch_order)
        }
        ready: list[tuple[int, int]] = []
        for task_id, count in enumerate(indegree):
            if count == 0:
                heapq.heappush(ready, (dispatch_rank[task_id], task_id))
        completed = 0
        while ready:
            _, task_id = heapq.heappop(ready)
            task = artifact.tasks[task_id]
            try:
                if task.guard is None or task.guard.evaluate(runtime_scalars):
                    variant = artifact.registry.get(task.kernel_id)
                    if variant.executor is None:
                        raise RuntimeExecutionError(
                            f"kernel {variant.name!r} has no CPU executor"
                        )
                    context = ExecutionContext(
                        task,
                        variant,
                        arrays,
                        variant.parameters.unpack(task.params),
                        task.operands,
                        runtime_scalars,
                    )
                    variant.executor(context)
            except RuntimeExecutionError:
                raise
            except Exception as exc:
                raise RuntimeExecutionError(
                    f"task {task.id} ({task.kernel_name}) failed: {exc}"
                ) from exc
            completed += 1
            for child in artifact.dependencies.successors[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (dispatch_rank[child], child))
        if completed != len(artifact.tasks):
            raise RuntimeExecutionError("runtime task graph did not make progress")
        outputs: dict[str, np.ndarray] = {}
        for value in artifact.graph.outputs:
            try:
                base = arrays[value.storage_id]
                shape = value.spec.concrete_shape()
                strides = value.resolved_strides()
                if (
                    value.byte_offset == 0
                    and shape == base.shape
                    and strides == value.spec.contiguous_strides()
                ):
                    outputs[value.name] = base
                else:
                    dtype = np.dtype(_numpy_dtype(value.spec.dtype))
                    outputs[value.name] = np.ndarray(
                        shape,
                        dtype=dtype,
                        buffer=base.data,
                        offset=value.byte_offset,
                        strides=tuple(stride * dtype.itemsize for stride in strides),
                    )
            except RuntimeExecutionError:
                raise
            except Exception as exc:
                raise RuntimeExecutionError(
                    f"could not materialize output view {value.name!r}: {exc}"
                ) from exc
        return outputs
