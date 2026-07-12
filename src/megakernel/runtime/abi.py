"""Versioned, relocatable command-buffer ABI.

The GPU-facing representation contains no raw pointers. Tasks reference a
separate buffer binding table by stable integer IDs; host code supplies actual
addresses for each invocation. All offsets are relative to the command buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from operator import index as integer_index
import struct
import zlib

from ..errors import AbiError
from ..ir import DType, MAX_TENSOR_RANK
from ..registry import Architecture
from ..task import AccessMode, GuardOp


MAGIC = b"MKCB"
ABI_MAJOR = 1
ABI_MINOR = 0
ALIGNMENT = 16
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

HEADER_STRUCT = struct.Struct("<4sHH" + "I" * 30)  # 128 bytes
BUFFER_STRUCT = struct.Struct("<IIIIIIQ" + "I" * 8)  # 64 bytes
VALUE_STRUCT = struct.Struct("<" + "I" * 8 + "Q" + "I" * 6)  # 64 bytes
TASK_STRUCT = struct.Struct("<" + "I" * 16)  # 64 bytes
OPERAND_STRUCT = struct.Struct("<IIII")  # 16 bytes
U32_STRUCT = struct.Struct("<I")
I64_STRUCT = struct.Struct("<q")
GUARD_STRUCT = struct.Struct("<IIq")  # scalar id, comparison opcode, rhs
CRC_OFFSET = 20

assert HEADER_STRUCT.size == 128
assert BUFFER_STRUCT.size == 64
assert VALUE_STRUCT.size == 64
assert TASK_STRUCT.size == 64
assert OPERAND_STRUCT.size == 16
assert GUARD_STRUCT.size == 16


class BufferFlags(IntFlag):
    INPUT = 1 << 0
    CONSTANT = 1 << 1
    STATE = 1 << 2
    OUTPUT = 1 << 3
    TEMPORARY = 1 << 4
    OPAQUE_LAYOUT = 1 << 5


class TaskFlags(IntFlag):
    NONE = 0
    HAS_GUARD = 1 << 0


def _abi_integer(value, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise AbiError(f"{label} must be an integer, not bool")
    try:
        normalized = integer_index(value)
    except TypeError as exc:
        raise AbiError(f"{label} must be an integer") from exc
    if normalized < 0 or normalized > maximum:
        raise AbiError(f"{label} is outside its ABI integer field")
    return normalized


def _abi_i64_tuple(values, label: str) -> tuple[int, ...]:
    return tuple(_abi_integer(value, label, (1 << 63) - 1) for value in values)


@dataclass(frozen=True)
class BufferDescriptor:
    id: int
    flags: BufferFlags
    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    nbytes: int
    quant_spec_id: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "flags",
                BufferFlags(_abi_integer(self.flags, "buffer flags", UINT32_MAX)),
            )
            object.__setattr__(
                self,
                "dtype",
                DType(_abi_integer(self.dtype, "buffer dtype", UINT32_MAX)),
            )
        except ValueError as exc:
            raise AbiError("buffer descriptor has an invalid enum value") from exc
        object.__setattr__(self, "id", _abi_integer(self.id, "buffer id", UINT32_MAX))
        object.__setattr__(self, "shape", _abi_i64_tuple(self.shape, "buffer shape"))
        object.__setattr__(
            self, "strides", _abi_i64_tuple(self.strides, "buffer stride")
        )
        object.__setattr__(
            self, "nbytes", _abi_integer(self.nbytes, "buffer size", UINT64_MAX)
        )
        object.__setattr__(
            self,
            "quant_spec_id",
            _abi_integer(self.quant_spec_id, "buffer quantization id", UINT32_MAX),
        )
        if len(self.shape) != len(self.strides):
            raise AbiError("buffer shape and strides have different ranks")
        if len(self.shape) > MAX_TENSOR_RANK:
            raise AbiError("buffer rank exceeds the ABI maximum")
        if _span_bits(self.shape, self.strides, self.dtype.bits) > self.nbytes * 8:
            raise AbiError("buffer descriptor strides exceed its allocation size")


@dataclass(frozen=True)
class ValueDescriptor:
    id: int
    buffer_id: int
    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_offset: int = 0
    quant_spec_id: int = 0
    flags: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "dtype",
                DType(_abi_integer(self.dtype, "value dtype", UINT32_MAX)),
            )
        except ValueError as exc:
            raise AbiError("value descriptor has an invalid dtype") from exc
        object.__setattr__(self, "id", _abi_integer(self.id, "value id", UINT32_MAX))
        object.__setattr__(
            self,
            "buffer_id",
            _abi_integer(self.buffer_id, "value buffer id", UINT32_MAX),
        )
        object.__setattr__(self, "shape", _abi_i64_tuple(self.shape, "value shape"))
        object.__setattr__(
            self, "strides", _abi_i64_tuple(self.strides, "value stride")
        )
        object.__setattr__(
            self,
            "byte_offset",
            _abi_integer(self.byte_offset, "value byte offset", UINT64_MAX),
        )
        object.__setattr__(
            self,
            "quant_spec_id",
            _abi_integer(self.quant_spec_id, "value quantization id", UINT32_MAX),
        )
        object.__setattr__(
            self, "flags", _abi_integer(self.flags, "value flags", UINT32_MAX)
        )
        if len(self.shape) != len(self.strides) or len(self.shape) > MAX_TENSOR_RANK:
            raise AbiError("value shape/stride rank is invalid")


def _span_bits(
    shape: tuple[int, ...], strides: tuple[int, ...], element_bits: int
) -> int:
    if any(extent == 0 for extent in shape):
        return 0
    max_element = sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    return (max_element + 1) * element_bits


@dataclass(frozen=True)
class OperandDescriptor:
    buffer_id: int
    value_id: int
    mode: AccessMode
    flags: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "mode",
                AccessMode(_abi_integer(self.mode, "operand mode", UINT32_MAX)),
            )
        except ValueError as exc:
            raise AbiError("operand has an invalid access mode") from exc
        object.__setattr__(
            self,
            "buffer_id",
            _abi_integer(self.buffer_id, "operand buffer id", UINT32_MAX),
        )
        object.__setattr__(
            self,
            "value_id",
            _abi_integer(self.value_id, "operand value id", UINT32_MAX),
        )
        object.__setattr__(
            self, "flags", _abi_integer(self.flags, "operand flags", UINT32_MAX)
        )


@dataclass(frozen=True)
class TaskInstruction:
    task_id: int
    kernel_id: int
    flags: TaskFlags
    worker_hint: int
    dependency_start: int
    dependency_count: int
    successor_start: int
    successor_count: int
    operand_start: int
    operand_count: int
    param_offset: int
    param_size: int
    logical_x: int
    logical_y: int
    logical_z: int
    reserved: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "flags",
            TaskFlags(_abi_integer(self.flags, "task flags", UINT32_MAX)),
        )

    def words(self) -> tuple[int, ...]:
        raw_words = (
            self.task_id,
            self.kernel_id,
            int(self.flags),
            self.worker_hint,
            self.dependency_start,
            self.dependency_count,
            self.successor_start,
            self.successor_count,
            self.operand_start,
            self.operand_count,
            self.param_offset,
            self.param_size,
            self.logical_x,
            self.logical_y,
            self.logical_z,
            self.reserved,
        )
        words: list[int] = []
        for raw_value in raw_words:
            if isinstance(raw_value, bool):
                raise AbiError("task instruction fields must be integers, not bool")
            try:
                value = integer_index(raw_value)
            except TypeError as exc:
                raise AbiError("task instruction field is not an integer") from exc
            if value < 0 or value > UINT32_MAX:
                raise AbiError("task instruction field is outside uint32")
            words.append(value)
        return tuple(words)


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _enum_value(enum_type, value: int, label: str):
    try:
        return enum_type(value)
    except ValueError as exc:
        raise AbiError(f"invalid {label} value {value}") from exc


def _append_aligned(target: bytearray, payload: bytes) -> int:
    padding = _align(len(target)) - len(target)
    target.extend(b"\0" * padding)
    offset = len(target)
    target.extend(payload)
    return offset


@dataclass(frozen=True)
class CommandProgram:
    target: Architecture
    buffers: tuple[BufferDescriptor, ...]
    values: tuple[ValueDescriptor, ...]
    tasks: tuple[TaskInstruction, ...]
    operands: tuple[OperandDescriptor, ...]
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    params: bytes

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "target",
                Architecture(
                    _abi_integer(self.target, "target architecture", UINT32_MAX)
                ),
            )
        except ValueError as exc:
            raise AbiError(
                "command program has an invalid target architecture"
            ) from exc
        for field_name, expected_type in (
            ("buffers", BufferDescriptor),
            ("values", ValueDescriptor),
            ("tasks", TaskInstruction),
            ("operands", OperandDescriptor),
        ):
            items = tuple(getattr(self, field_name))
            if any(not isinstance(item, expected_type) for item in items):
                raise AbiError(f"command {field_name} contains an invalid descriptor")
            object.__setattr__(self, field_name, items)
        object.__setattr__(
            self,
            "predecessors",
            tuple(
                _abi_integer(item, "predecessor task id", UINT32_MAX)
                for item in self.predecessors
            ),
        )
        object.__setattr__(
            self,
            "successors",
            tuple(
                _abi_integer(item, "successor task id", UINT32_MAX)
                for item in self.successors
            ),
        )
        try:
            object.__setattr__(self, "params", bytes(self.params))
        except (TypeError, ValueError) as exc:
            raise AbiError("command parameter blob must be bytes-like") from exc

    def validate(self) -> None:
        if any(buffer.id != index for index, buffer in enumerate(self.buffers)):
            raise AbiError("buffer ids must be contiguous")
        if any(value.id != index for index, value in enumerate(self.values)):
            raise AbiError("value ids must be contiguous")
        if any(task.task_id != index for index, task in enumerate(self.tasks)):
            raise AbiError("task ids must be contiguous")
        for task in self.tasks:
            task.words()
            if task.reserved != 0:
                raise AbiError(f"task {task.task_id} has a non-zero reserved field")
            if int(task.flags) & ~int(TaskFlags.HAS_GUARD):
                raise AbiError(f"task {task.task_id} uses unknown task flags")
            if task.dependency_start + task.dependency_count > len(self.predecessors):
                raise AbiError(
                    f"task {task.task_id} predecessor slice is out of bounds"
                )
            if task.successor_start + task.successor_count > len(self.successors):
                raise AbiError(f"task {task.task_id} successor slice is out of bounds")
            if task.operand_start + task.operand_count > len(self.operands):
                raise AbiError(f"task {task.task_id} operand slice is out of bounds")
            if task.param_offset + task.param_size > len(self.params):
                raise AbiError(f"task {task.task_id} parameter slice is out of bounds")
            if task.param_offset % ALIGNMENT:
                raise AbiError(f"task {task.task_id} parameter payload is unaligned")
            if task.flags & TaskFlags.HAS_GUARD and task.param_size < GUARD_STRUCT.size:
                raise AbiError(f"task {task.task_id} guard payload is truncated")
            if task.flags & TaskFlags.HAS_GUARD:
                _, guard_opcode, _ = GUARD_STRUCT.unpack_from(
                    self.params, task.param_offset
                )
                try:
                    GuardOp(guard_opcode)
                except ValueError as exc:
                    raise AbiError(
                        f"task {task.task_id} has an invalid guard opcode"
                    ) from exc
        if any(
            item >= len(self.tasks) for item in (*self.predecessors, *self.successors)
        ):
            raise AbiError("dependency table references an unknown task")
        if any(operand.buffer_id >= len(self.buffers) for operand in self.operands):
            raise AbiError("operand references an unknown buffer")
        if any(operand.value_id >= len(self.values) for operand in self.operands):
            raise AbiError("operand references an unknown value")
        if any(value.buffer_id >= len(self.buffers) for value in self.values):
            raise AbiError("value references an unknown buffer")
        known_buffer_flags = int(
            BufferFlags.INPUT
            | BufferFlags.CONSTANT
            | BufferFlags.STATE
            | BufferFlags.OUTPUT
            | BufferFlags.TEMPORARY
            | BufferFlags.OPAQUE_LAYOUT
        )
        if any(int(buffer.flags) & ~known_buffer_flags for buffer in self.buffers):
            raise AbiError("buffer uses unknown flags")
        for value in self.values:
            if value.flags != 0:
                raise AbiError("value uses unknown flags")
            buffer = self.buffers[value.buffer_id]
            required_bits = value.byte_offset * 8 + _span_bits(
                value.shape, value.strides, value.dtype.bits
            )
            if required_bits > buffer.nbytes * 8:
                raise AbiError(f"value {value.id} view exceeds buffer {buffer.id}")
        for operand in self.operands:
            if self.values[operand.value_id].buffer_id != operand.buffer_id:
                raise AbiError("operand buffer id disagrees with its value descriptor")

        predecessor_edges: set[tuple[int, int]] = set()
        successor_edges: set[tuple[int, int]] = set()
        for task in self.tasks:
            predecessors = self.predecessors[
                task.dependency_start : task.dependency_start + task.dependency_count
            ]
            successors = self.successors[
                task.successor_start : task.successor_start + task.successor_count
            ]
            if len(predecessors) != len(set(predecessors)) or len(successors) != len(
                set(successors)
            ):
                raise AbiError(f"task {task.task_id} has duplicate dependency entries")
            if any(parent >= task.task_id for parent in predecessors):
                raise AbiError(f"task {task.task_id} has a non-topological predecessor")
            if any(child <= task.task_id for child in successors):
                raise AbiError(f"task {task.task_id} has a non-topological successor")
            predecessor_edges.update((parent, task.task_id) for parent in predecessors)
            successor_edges.update((task.task_id, child) for child in successors)
        if predecessor_edges != successor_edges:
            raise AbiError("predecessor and successor tables describe different edges")

    def to_bytes(self) -> bytes:
        self.validate()
        dimensions: list[int] = []
        buffer_bytes = bytearray()
        for buffer in self.buffers:
            shape_offset = len(dimensions)
            dimensions.extend(buffer.shape)
            stride_offset = len(dimensions)
            dimensions.extend(buffer.strides)
            buffer_bytes.extend(
                BUFFER_STRUCT.pack(
                    buffer.id,
                    int(buffer.flags),
                    int(buffer.dtype),
                    len(buffer.shape),
                    shape_offset,
                    stride_offset,
                    buffer.nbytes,
                    buffer.quant_spec_id,
                    *([0] * 7),
                )
            )
        value_bytes = bytearray()
        for value in self.values:
            shape_offset = len(dimensions)
            dimensions.extend(value.shape)
            stride_offset = len(dimensions)
            dimensions.extend(value.strides)
            value_bytes.extend(
                VALUE_STRUCT.pack(
                    value.id,
                    value.buffer_id,
                    value.flags,
                    int(value.dtype),
                    len(value.shape),
                    shape_offset,
                    stride_offset,
                    value.quant_spec_id,
                    value.byte_offset,
                    *([0] * 6),
                )
            )
        task_bytes = b"".join(TASK_STRUCT.pack(*task.words()) for task in self.tasks)
        operand_bytes = b"".join(
            OPERAND_STRUCT.pack(
                operand.buffer_id, operand.value_id, int(operand.mode), operand.flags
            )
            for operand in self.operands
        )
        predecessor_bytes = b"".join(
            U32_STRUCT.pack(item) for item in self.predecessors
        )
        successor_bytes = b"".join(U32_STRUCT.pack(item) for item in self.successors)
        dimension_bytes = b"".join(I64_STRUCT.pack(item) for item in dimensions)

        blob = bytearray(b"\0" * HEADER_STRUCT.size)
        offsets = [
            _append_aligned(blob, bytes(buffer_bytes)),
            _append_aligned(blob, bytes(value_bytes)),
            _append_aligned(blob, task_bytes),
            _append_aligned(blob, operand_bytes),
            _append_aligned(blob, predecessor_bytes),
            _append_aligned(blob, successor_bytes),
            _append_aligned(blob, dimension_bytes),
            _append_aligned(blob, self.params),
        ]
        total_size = len(blob)
        values = [
            int(self.target),
            HEADER_STRUCT.size,
            total_size,
            0,
            len(self.buffers),
            len(self.values),
            len(self.tasks),
            len(self.operands),
            len(self.predecessors),
            len(self.successors),
            len(dimensions),
            len(self.params),
            *offsets,
            *([0] * 10),
        ]
        if len(values) != 30:
            raise AssertionError("internal ABI header field count changed")
        blob[: HEADER_STRUCT.size] = HEADER_STRUCT.pack(
            MAGIC, ABI_MAJOR, ABI_MINOR, *values
        )
        checksum = zlib.crc32(blob) & UINT32_MAX
        values[3] = checksum
        blob[: HEADER_STRUCT.size] = HEADER_STRUCT.pack(
            MAGIC, ABI_MAJOR, ABI_MINOR, *values
        )
        return bytes(blob)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CommandProgram":
        if len(payload) < HEADER_STRUCT.size:
            raise AbiError("command buffer is shorter than its header")
        unpacked = HEADER_STRUCT.unpack_from(payload)
        magic, major, minor = unpacked[:3]
        values = unpacked[3:]
        if magic != MAGIC:
            raise AbiError(f"invalid command magic {magic!r}")
        if major != ABI_MAJOR:
            raise AbiError(f"unsupported command ABI major {major}")
        if minor > ABI_MINOR:
            raise AbiError(
                f"command ABI minor {minor} is newer than runtime {ABI_MINOR}"
            )
        (
            target,
            header_size,
            total_size,
            expected_crc,
            buffer_count,
            value_count,
            task_count,
            operand_count,
            predecessor_count,
            successor_count,
            dimension_count,
            param_size,
            buffer_offset,
            value_offset,
            task_offset,
            operand_offset,
            predecessor_offset,
            successor_offset,
            dimension_offset,
            param_offset,
            *_reserved,
        ) = values
        if header_size != HEADER_STRUCT.size or total_size != len(payload):
            raise AbiError("command header/total size is inconsistent")
        checksum_payload = bytearray(payload)
        struct.pack_into("<I", checksum_payload, CRC_OFFSET, 0)
        actual_crc = zlib.crc32(checksum_payload) & UINT32_MAX
        if actual_crc != expected_crc:
            raise AbiError("command buffer checksum mismatch")
        offsets = (
            buffer_offset,
            value_offset,
            task_offset,
            operand_offset,
            predecessor_offset,
            successor_offset,
            dimension_offset,
            param_offset,
        )
        if any(offset < HEADER_STRUCT.size or offset % ALIGNMENT for offset in offsets):
            raise AbiError("command section offset is invalid or unaligned")
        section_sizes = (
            buffer_count * BUFFER_STRUCT.size,
            value_count * VALUE_STRUCT.size,
            task_count * TASK_STRUCT.size,
            operand_count * OPERAND_STRUCT.size,
            predecessor_count * U32_STRUCT.size,
            successor_count * U32_STRUCT.size,
            dimension_count * I64_STRUCT.size,
            param_size,
        )
        previous_end = HEADER_STRUCT.size
        for offset, size in zip(offsets, section_sizes):
            if offset < _align(previous_end):
                raise AbiError("command sections overlap or are out of order")
            previous_end = offset + size
        if previous_end != total_size:
            raise AbiError("command sections do not account for the total buffer size")

        def checked_slice(offset: int, size: int) -> memoryview:
            if size < 0 or offset + size > len(payload):
                raise AbiError("command section exceeds buffer bounds")
            return memoryview(payload)[offset : offset + size]

        dimension_view = checked_slice(
            dimension_offset, dimension_count * I64_STRUCT.size
        )
        dimensions = tuple(
            I64_STRUCT.unpack_from(dimension_view, index * I64_STRUCT.size)[0]
            for index in range(dimension_count)
        )
        buffers: list[BufferDescriptor] = []
        buffer_view = checked_slice(buffer_offset, buffer_count * BUFFER_STRUCT.size)
        for index in range(buffer_count):
            record = BUFFER_STRUCT.unpack_from(buffer_view, index * BUFFER_STRUCT.size)
            buffer_id, flags, dtype, rank, shape_start, stride_start, nbytes = record[
                :7
            ]
            if rank > MAX_TENSOR_RANK:
                raise AbiError("encoded buffer rank exceeds ABI limit")
            if shape_start + rank > len(dimensions) or stride_start + rank > len(
                dimensions
            ):
                raise AbiError("encoded buffer dimension slice is out of bounds")
            buffers.append(
                BufferDescriptor(
                    buffer_id,
                    BufferFlags(flags),
                    _enum_value(DType, dtype, "buffer dtype"),
                    dimensions[shape_start : shape_start + rank],
                    dimensions[stride_start : stride_start + rank],
                    nbytes,
                    record[7],
                )
            )

        values_out: list[ValueDescriptor] = []
        value_view = checked_slice(value_offset, value_count * VALUE_STRUCT.size)
        for index in range(value_count):
            record = VALUE_STRUCT.unpack_from(value_view, index * VALUE_STRUCT.size)
            (
                value_id,
                buffer_id,
                flags,
                dtype,
                rank,
                shape_start,
                stride_start,
                quant_spec_id,
                byte_offset,
            ) = record[:9]
            if rank > MAX_TENSOR_RANK:
                raise AbiError("encoded value rank exceeds ABI limit")
            if shape_start + rank > len(dimensions) or stride_start + rank > len(
                dimensions
            ):
                raise AbiError("encoded value dimension slice is out of bounds")
            values_out.append(
                ValueDescriptor(
                    value_id,
                    buffer_id,
                    _enum_value(DType, dtype, "value dtype"),
                    dimensions[shape_start : shape_start + rank],
                    dimensions[stride_start : stride_start + rank],
                    byte_offset,
                    quant_spec_id,
                    flags,
                )
            )

        task_view = checked_slice(task_offset, task_count * TASK_STRUCT.size)
        decoded_tasks: list[TaskInstruction] = []
        for index in range(task_count):
            words = list(TASK_STRUCT.unpack_from(task_view, index * TASK_STRUCT.size))
            words[2] = _enum_value(TaskFlags, words[2], "task flags")
            decoded_tasks.append(TaskInstruction(*words))
        tasks = tuple(decoded_tasks)
        operand_view = checked_slice(
            operand_offset, operand_count * OPERAND_STRUCT.size
        )
        operands = tuple(
            OperandDescriptor(
                buffer_id,
                value_id,
                _enum_value(AccessMode, mode, "operand access mode"),
                flags,
            )
            for index in range(operand_count)
            for buffer_id, value_id, mode, flags in (
                OPERAND_STRUCT.unpack_from(operand_view, index * OPERAND_STRUCT.size),
            )
        )

        def read_u32_table(offset: int, count: int) -> tuple[int, ...]:
            view = checked_slice(offset, count * U32_STRUCT.size)
            return tuple(
                U32_STRUCT.unpack_from(view, index * U32_STRUCT.size)[0]
                for index in range(count)
            )

        program = cls(
            _enum_value(Architecture, target, "target architecture"),
            tuple(buffers),
            tuple(values_out),
            tasks,
            operands,
            read_u32_table(predecessor_offset, predecessor_count),
            read_u32_table(successor_offset, successor_count),
            bytes(checked_slice(param_offset, param_size)),
        )
        program.validate()
        return program
