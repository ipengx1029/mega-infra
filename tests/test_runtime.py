import unittest

import numpy as np

from megakernel import (
    CPUInterpreter,
    CompileOptions,
    GraphBuilder,
    Symbol,
    TensorSpec,
    compile_graph,
    create_cpu_registry,
)
from megakernel.runtime.abi import CommandProgram
from megakernel.errors import CompilationError, RegistryError, RuntimeExecutionError
from megakernel import DType


class RuntimeTests(unittest.TestCase):
    def test_end_to_end_gated_mlp_residual(self):
        builder = GraphBuilder("gated_mlp")
        x = builder.input("x", TensorSpec((3, 4)))
        norm_weight = builder.parameter("norm_weight", TensorSpec((4,)))
        up_weight = builder.parameter("up_weight", TensorSpec((4, 6)))
        gate_weight = builder.parameter("gate_weight", TensorSpec((4, 6)))
        down_weight = builder.parameter("down_weight", TensorSpec((6, 4)))

        normalized = builder.rms_norm(x, norm_weight, eps=1e-5)
        up = builder.matmul(normalized, up_weight)
        gate = builder.silu(builder.matmul(normalized, gate_weight))
        hidden = builder.mul(up, gate)
        projected = builder.matmul(hidden, down_weight)
        output = builder.add(x, projected)
        graph = builder.output(output)

        rng = np.random.default_rng(7)
        bindings = {
            "x": rng.normal(size=(3, 4)).astype(np.float32),
            "norm_weight": rng.normal(size=(4,)).astype(np.float32),
            "up_weight": rng.normal(size=(4, 6)).astype(np.float32),
            "gate_weight": rng.normal(size=(4, 6)).astype(np.float32),
            "down_weight": rng.normal(size=(6, 4)).astype(np.float32),
        }
        artifact = compile_graph(
            graph,
            create_cpu_registry(tile_m=2, tile_n=3),
            CompileOptions(num_workers=3),
        )
        actual = next(iter(CPUInterpreter().run(artifact, bindings).values()))

        values = bindings["x"]
        variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        normalized_ref = (
            values / np.sqrt(variance + np.float32(1e-5)) * bindings["norm_weight"]
        )
        up_ref = normalized_ref @ bindings["up_weight"]
        gate_ref = normalized_ref @ bindings["gate_weight"]
        gate_ref = gate_ref / (1.0 + np.exp(-gate_ref))
        expected = values + (up_ref * gate_ref) @ bindings["down_weight"]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertGreater(artifact.dependencies.edge_count, 0)
        self.assertEqual(
            CommandProgram.from_bytes(artifact.serialize()), artifact.program
        )

    def test_state_update_aliases_input_storage(self):
        builder = GraphBuilder("state")
        cache = builder.state("cache", TensorSpec((2, 4)))
        update = builder.input("update", TensorSpec((2, 4)))
        result = builder.copy_inplace(cache, update)
        artifact = compile_graph(
            builder.output(result), create_cpu_registry(tile_m=1, tile_n=2)
        )

        cache_array = np.zeros((2, 4), dtype=np.float32)
        update_array = np.arange(8, dtype=np.float32).reshape(2, 4)
        outputs = CPUInterpreter().run(
            artifact, {"cache": cache_array, "update": update_array}
        )
        self.assertIs(next(iter(outputs.values())), cache_array)
        np.testing.assert_array_equal(cache_array, update_array)

    def test_symbolic_specialization(self):
        tokens = Symbol("tokens", maximum=8)
        builder = GraphBuilder("dynamic_tokens")
        lhs = builder.input("lhs", TensorSpec((tokens, 4)))
        rhs = builder.parameter("rhs", TensorSpec((4, 3)))
        graph = builder.output(builder.matmul(lhs, rhs))
        artifact = compile_graph(
            graph,
            create_cpu_registry(),
            CompileOptions(symbolic_bindings={"tokens": 5}),
        )
        self.assertEqual(artifact.graph.inputs[0].spec.shape, (5, 4))

    def test_metadata_only_view_output(self):
        builder = GraphBuilder("view_runtime")
        base = builder.input("base", TensorSpec((4, 4)))
        view = builder.view(base, (2, 2), byte_offset=16, strides=(4, 1))
        artifact = compile_graph(builder.output(view), create_cpu_registry())
        base_array = np.arange(16, dtype=np.float32).reshape(4, 4)
        output = next(
            iter(CPUInterpreter().run(artifact, {"base": base_array}).values())
        )
        np.testing.assert_array_equal(output, base_array[1:3, :2])

    def test_symbolic_view_also_requires_specialization(self):
        rows = Symbol("rows", minimum=0, maximum=4)
        builder = GraphBuilder("symbolic_view")
        base = builder.input("base", TensorSpec((4, 4)))
        view = builder.view(base, (rows, 4), strides=(4, 1))
        graph = builder.output(view)
        with self.assertRaises(CompilationError):
            compile_graph(graph, create_cpu_registry())
        artifact = compile_graph(
            graph,
            create_cpu_registry(),
            CompileOptions(symbolic_bindings={"rows": 2}),
        )
        self.assertEqual(artifact.program.values[view.id].shape, (2, 4))

    def test_zero_extent_dense_output_is_a_valid_no_work_program(self):
        builder = GraphBuilder("zero_work")
        x = builder.input("x", TensorSpec((0, 4)))
        output = builder.silu(x)
        artifact = compile_graph(builder.output(output), create_cpu_registry())
        self.assertEqual(artifact.tasks, ())
        result = next(
            iter(
                CPUInterpreter()
                .run(artifact, {"x": np.empty((0, 4), dtype=np.float32)})
                .values()
            )
        )
        self.assertEqual(result.shape, (0, 4))

    def test_cpu_backend_rejects_mismatched_physical_storage(self):
        builder = GraphBuilder("physical_mismatch")
        x = builder.input(
            "x",
            TensorSpec(
                (2, 2),
                DType.FP32,
                storage_dtype=DType.INT32,
                physical_shape=(1,),
            ),
        )
        graph = builder.output(builder.silu(x))
        with self.assertRaises(RegistryError):
            compile_graph(graph, create_cpu_registry())

    def test_cpu_runtime_checks_storage_even_without_compute_nodes(self):
        builder = GraphBuilder("direct_physical_mismatch")
        x = builder.input(
            "x",
            TensorSpec((2, 2), DType.FP32, storage_dtype=DType.INT32),
        )
        artifact = compile_graph(builder.output(x), create_cpu_registry())
        with self.assertRaisesRegex(RuntimeExecutionError, "unsupported"):
            CPUInterpreter().run(artifact, {"x": np.zeros((2, 2), dtype=np.float32)})

    def test_noncontiguous_binding_fails_with_runtime_error(self):
        builder = GraphBuilder("noncontiguous")
        base = builder.input("base", TensorSpec((4, 4)))
        view = builder.view(base, (2, 2), strides=(4, 1))
        artifact = compile_graph(builder.output(view), create_cpu_registry())
        noncontiguous = np.arange(32, dtype=np.float32).reshape(4, 8)[:, ::2]
        self.assertEqual(noncontiguous.shape, (4, 4))
        with self.assertRaisesRegex(RuntimeExecutionError, "C-contiguous"):
            CPUInterpreter().run(artifact, {"base": noncontiguous})


if __name__ == "__main__":
    unittest.main()
