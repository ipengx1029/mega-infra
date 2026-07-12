import unittest
from dataclasses import replace
import struct

import numpy as np

from megakernel import (
    CPUInterpreter,
    EffectMode,
    GraphBuilder,
    Guard,
    GuardOp,
    KernelVariant,
    OpSchema,
    TensorSpec,
    compile_graph,
    create_cpu_registry,
    create_default_schema_registry,
)
from megakernel.ir import OutputSpec
from megakernel.parameters import EMPTY_PARAMETERS
from megakernel.registry import Architecture, LoweringContext
from megakernel.runtime.abi import TaskFlags
from megakernel.runtime.interpreter import ExecutionContext
from megakernel.errors import AbiError, RuntimeExecutionError
from megakernel.task import (
    Access,
    AccessMode,
    OperandBinding,
    Region,
    ResourceRef,
    TaskDraft,
)


def conditional_copy_schema(inputs, attrs):
    if len(inputs) != 2 or inputs[0] != inputs[1]:
        raise ValueError("conditional_copy expects identical state/source specs")
    return (OutputSpec(inputs[0], alias_input=0),)


def conditional_copy_lower(context: LoweringContext):
    graph, node = context.graph, context.node
    state, source = (graph.value(value_id) for value_id in node.inputs)
    output = graph.value(node.outputs[0])
    region = Region.full(output.spec.concrete_shape())
    yield TaskDraft(
        node.id,
        (0, 0, 0),
        (
            OperandBinding(state.id, state.storage_id, AccessMode.WRITE),
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
                ResourceRef.buffer(state.storage_id),
                AccessMode.WRITE,
                region,
                output.id,
            ),
        ),
        {},
        guard=Guard("compress_ready", GuardOp.NE, 0),
    )


def conditional_copy_execute(context: ExecutionContext):
    np.copyto(context.array(0), context.array(1))


def side_effect_lower(context: LoweringContext):
    state = context.graph.value(context.node.inputs[0])
    region = Region.full(state.spec.concrete_shape())
    yield TaskDraft(
        context.node.id,
        (0, 0, 0),
        (OperandBinding(state.id, state.storage_id, AccessMode.READ_WRITE),),
        (
            Access(
                ResourceRef.buffer(state.storage_id),
                AccessMode.READ_WRITE,
                region,
                state.id,
            ),
        ),
        {},
    )


def side_effect_execute(context: ExecutionContext):
    context.array(0)[:] += 1


class PluginAndDynamicTests(unittest.TestCase):
    def test_custom_schema_and_kernel_plugin_execute_without_core_changes(self):
        schemas = create_default_schema_registry()
        schemas.register(OpSchema("conditional_copy", conditional_copy_schema))
        builder = GraphBuilder("plugin", schemas=schemas)
        state = builder.state("state", TensorSpec((2, 3)))
        source = builder.input("source", TensorSpec((2, 3)))
        output = builder.call("conditional_copy", state, source)
        graph = builder.output(output)

        registry = create_cpu_registry()
        registry.register(
            KernelVariant(
                "example.conditional_copy",
                "conditional_copy",
                "numpy-plugin",
                frozenset({Architecture.CPU}),
                conditional_copy_lower,
                parameters=EMPTY_PARAMETERS,
                executor=conditional_copy_execute,
            )
        )
        artifact = compile_graph(graph, registry)
        self.assertEqual(artifact.tasks[0].kernel_name, "example.conditional_copy")
        self.assertEqual(artifact.program.tasks[0].flags, TaskFlags.HAS_GUARD)
        self.assertEqual(tuple(artifact.runtime_scalars.values()), ("compress_ready",))

        invalid_params = bytearray(artifact.program.params)
        struct.pack_into(
            "<I", invalid_params, artifact.program.tasks[0].param_offset + 4, 999
        )
        with self.assertRaisesRegex(AbiError, "guard opcode"):
            replace(artifact.program, params=bytes(invalid_params)).validate()

        with self.assertRaisesRegex(RuntimeExecutionError, "runtime scalar"):
            CPUInterpreter().run(
                artifact,
                {
                    "state": np.zeros((2, 3), dtype=np.float32),
                    "source": np.zeros((2, 3), dtype=np.float32),
                },
            )

        initial = np.zeros((2, 3), dtype=np.float32)
        source_array = np.arange(6, dtype=np.float32).reshape(2, 3)
        CPUInterpreter().run(
            artifact,
            {"state": initial, "source": source_array},
            runtime_scalars={"compress_ready": 0},
        )
        np.testing.assert_array_equal(initial, 0)

        CPUInterpreter().run(
            artifact,
            {"state": initial, "source": source_array},
            runtime_scalars={"compress_ready": 1},
        )
        np.testing.assert_array_equal(initial, source_array)

    def test_write_effect_is_a_conservative_op_barrier(self):
        builder = GraphBuilder("effect_barrier")
        value = builder.input("value", TensorSpec((4, 4)))
        effect = builder.effect("opaque_state", lifetime="sequence")
        output = builder.call("silu", value, effects=((effect, EffectMode.WRITE),))
        artifact = compile_graph(
            builder.output(output), create_cpu_registry(tile_m=2, tile_n=2)
        )
        self.assertEqual(len(artifact.tasks), 4)
        self.assertEqual(artifact.dependencies.edge_count, 3)
        self.assertEqual(artifact.dependencies.successors[0], (1,))

    def test_side_effect_only_op_compiles_and_runs(self):
        schemas = create_default_schema_registry()
        schemas.register(OpSchema("increment_state", lambda inputs, attrs: ()))
        builder = GraphBuilder("side_effect_runtime", schemas=schemas)
        state = builder.state("state", TensorSpec((2, 2)))
        effect = builder.effect("cache_version", lifetime="sequence")
        builder.call(
            "increment_state", state, effects=((effect, EffectMode.READ_WRITE),)
        )
        graph = builder.output(state)

        registry = create_cpu_registry()
        registry.register(
            KernelVariant(
                "example.increment_state",
                "increment_state",
                "numpy-plugin",
                frozenset({Architecture.CPU}),
                side_effect_lower,
                parameters=EMPTY_PARAMETERS,
                executor=side_effect_execute,
            )
        )
        artifact = compile_graph(graph, registry)
        value = np.zeros((2, 2), dtype=np.float32)
        CPUInterpreter().run(artifact, {"state": value})
        np.testing.assert_array_equal(value, 1)


if __name__ == "__main__":
    unittest.main()
