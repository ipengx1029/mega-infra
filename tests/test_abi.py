import unittest
from dataclasses import replace

from megakernel import GraphBuilder, TensorSpec, compile_graph, create_cpu_registry
from megakernel.errors import AbiError
from megakernel.runtime.abi import (
    BUFFER_STRUCT,
    HEADER_STRUCT,
    OPERAND_STRUCT,
    TASK_STRUCT,
    VALUE_STRUCT,
    BufferDescriptor,
    CommandProgram,
)


def compiled_program():
    builder = GraphBuilder("abi")
    lhs = builder.input("lhs", TensorSpec((3, 5)))
    rhs = builder.parameter("rhs", TensorSpec((5, 7)))
    output = builder.matmul(lhs, rhs)
    graph = builder.output(output)
    return compile_graph(graph, create_cpu_registry(tile_m=2, tile_n=4))


class CommandAbiTests(unittest.TestCase):
    def test_records_have_fixed_sizes(self):
        self.assertEqual(HEADER_STRUCT.size, 128)
        self.assertEqual(BUFFER_STRUCT.size, 64)
        self.assertEqual(VALUE_STRUCT.size, 64)
        self.assertEqual(TASK_STRUCT.size, 64)
        self.assertEqual(OPERAND_STRUCT.size, 16)

    def test_round_trip_is_lossless_and_deterministic(self):
        artifact = compiled_program()
        first = artifact.serialize()
        decoded = CommandProgram.from_bytes(first)
        self.assertEqual(decoded, artifact.program)
        self.assertEqual(decoded.to_bytes(), first)
        self.assertEqual(len(decoded.values), len(artifact.graph.values))

    def test_corruption_is_rejected_by_checksum(self):
        payload = bytearray(compiled_program().serialize())
        payload[-1] ^= 0x7F
        with self.assertRaisesRegex(AbiError, "checksum"):
            CommandProgram.from_bytes(bytes(payload))

    def test_header_corruption_is_also_checksummed(self):
        payload = bytearray(compiled_program().serialize())
        payload[8] = 90  # target architecture word: CPU -> SM90
        with self.assertRaisesRegex(AbiError, "checksum"):
            CommandProgram.from_bytes(bytes(payload))

    def test_value_view_cannot_exceed_its_buffer(self):
        program = compiled_program().program
        invalid_value = replace(program.values[0], shape=(10_000,), strides=(1,))
        invalid_program = replace(
            program,
            values=(invalid_value,) + program.values[1:],
        )
        with self.assertRaisesRegex(AbiError, "exceeds buffer"):
            invalid_program.validate()

    def test_non_integer_instruction_field_is_rejected_before_struct_pack(self):
        program = compiled_program().program
        invalid_task = replace(program.tasks[0], logical_x=1.5)
        invalid_program = replace(program, tasks=(invalid_task,) + program.tasks[1:])
        with self.assertRaisesRegex(AbiError, "not an integer"):
            invalid_program.validate()

    def test_descriptor_rejects_non_integer_shape_and_stride(self):
        with self.assertRaisesRegex(AbiError, "integer"):
            BufferDescriptor(0, 1, 12, (1.5,), (1,), 4)
        with self.assertRaisesRegex(AbiError, "integer"):
            BufferDescriptor(0, 1, 12, (1,), (True,), 4)

    def test_view_descriptor_is_relocatable(self):
        builder = GraphBuilder("view_abi")
        base = builder.input("base", TensorSpec((4, 4)))
        view = builder.view(base, (2, 2), byte_offset=16, strides=(4, 1))
        artifact = compile_graph(builder.output(view), create_cpu_registry())
        descriptor = artifact.program.values[view.id]
        self.assertEqual(descriptor.buffer_id, base.storage_id)
        self.assertEqual(descriptor.byte_offset, 16)
        self.assertEqual(descriptor.strides, (4, 1))
        self.assertEqual(len(artifact.program.tasks), 0)


if __name__ == "__main__":
    unittest.main()
