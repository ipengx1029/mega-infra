import unittest

from megakernel import (
    DType,
    GraphBuilder,
    OpSchema,
    TensorSpec,
    compile_graph,
    create_cpu_registry,
    create_default_schema_registry,
)
from megakernel.errors import CompilationError, GraphValidationError, RegistryError
from megakernel.ir import OutputSpec
from megakernel.parameters import EMPTY_PARAMETERS
from megakernel.registry import (
    Architecture,
    KernelRegistry,
    KernelResources,
    KernelVariant,
    LoweringContext,
    WarpRole,
)
from megakernel.scheduler import schedule_tasks
from megakernel.task import (
    Access,
    AccessMode,
    OperandBinding,
    Region,
    ResourceRef,
    Task,
    TaskDependencies,
    TaskDraft,
)


class RegistryAndSchedulerTests(unittest.TestCase):
    def test_registry_selects_highest_priority_then_name(self):
        builder = GraphBuilder("select")
        x = builder.input("x", TensorSpec((1, 1)))
        graph = builder.output(builder.silu(x))
        node = graph.nodes[0]

        def no_tasks(context):
            return ()

        registry = KernelRegistry()
        registry.register(
            KernelVariant(
                "z_variant",
                "silu",
                "test",
                frozenset({Architecture.CPU}),
                no_tasks,
                priority=3,
            )
        )
        registry.register(
            KernelVariant(
                "a_variant",
                "silu",
                "test",
                frozenset({Architecture.CPU}),
                no_tasks,
                priority=3,
            )
        )
        selected = registry.select(LoweringContext(graph, node, Architecture.CPU))
        self.assertEqual(selected.name, "a_variant")

    def test_duplicate_variant_is_rejected(self):
        registry = create_cpu_registry()
        duplicate = registry.variants("add")[0]
        with self.assertRaises(RegistryError):
            registry.register(duplicate)

    def test_kernel_resource_contract_rejects_invalid_roles_and_hints(self):
        with self.assertRaises(RegistryError):
            WarpRole("", 0, 1)
        with self.assertRaises(RegistryError):
            KernelResources(registers_per_thread_hint=-1)
        with self.assertRaises(RegistryError):
            KernelResources(tmem_columns=-1)
        with self.assertRaises(RegistryError):
            KernelResources(
                threads_per_cta=64,
                warp_roles=(WarpRole("load", 0, 2), WarpRole("math", 1, 1)),
            )

    def test_schedule_respects_dependency_finish_and_worker_bounds(self):
        tasks = tuple(
            Task(index, index, 1, "k", (index, 0, 0), (), (), b"", cost)
            for index, cost in enumerate((4.0, 2.0, 1.0, 3.0))
        )
        dependencies = TaskDependencies(
            predecessors=((), (), (0,), (1, 2)),
            successors=((2,), (3,), (3,), ()),
        )
        plan = schedule_tasks(tasks, dependencies, num_workers=2)
        self.assertGreaterEqual(plan.entry(2).start, plan.entry(0).finish)
        self.assertGreaterEqual(plan.entry(3).start, plan.entry(2).finish)
        self.assertTrue(all(0 <= entry.worker_hint < 2 for entry in plan.entries))

    def test_compilation_is_deterministic(self):
        builder = GraphBuilder("deterministic")
        x = builder.input("x", TensorSpec((4, 4)))
        y = builder.input("y", TensorSpec((4, 4)))
        graph = builder.output(builder.add(x, y))
        registry = create_cpu_registry()
        first = compile_graph(graph, registry).serialize()
        second = compile_graph(graph, registry).serialize()
        self.assertEqual(first, second)

    def test_kernel_abi_versions_can_coexist_and_newest_is_selected(self):
        builder = GraphBuilder("versions")
        x = builder.input("x", TensorSpec((1, 1)))
        graph = builder.output(builder.silu(x))

        def no_tasks(context):
            return ()

        registry = KernelRegistry()
        old = KernelVariant(
            "same_kernel",
            "silu",
            "plugin",
            frozenset({Architecture.CPU}),
            no_tasks,
            abi_version=1,
        )
        new = KernelVariant(
            "same_kernel",
            "silu",
            "plugin",
            frozenset({Architecture.CPU}),
            no_tasks,
            abi_version=2,
        )
        registry.register(old)
        registry.register(new)
        self.assertIs(registry.get(old.stable_id), old)
        self.assertIs(registry.get(new.stable_id), new)
        selected = registry.select(
            LoweringContext(graph, graph.nodes[0], Architecture.CPU)
        )
        self.assertIs(selected, new)

    def test_operand_without_access_is_rejected(self):
        builder = GraphBuilder("bad_contract")
        x = builder.input("x", TensorSpec((1, 1)))
        graph = builder.output(builder.silu(x))

        def bad_lower(context):
            source = context.graph.value(context.node.inputs[0])
            output = context.graph.value(context.node.outputs[0])
            region = Region.full((1, 1))
            yield TaskDraft(
                context.node.id,
                (0, 0, 0),
                (
                    OperandBinding(source.id, source.storage_id, AccessMode.READ),
                    OperandBinding(output.id, output.storage_id, AccessMode.WRITE),
                ),
                (
                    Access(
                        ResourceRef.buffer(output.storage_id),
                        AccessMode.WRITE,
                        region,
                        output.id,
                    ),
                ),
                {},
            )

        registry = KernelRegistry()
        registry.register(
            KernelVariant(
                "bad.silu",
                "silu",
                "test",
                frozenset({Architecture.CPU}),
                bad_lower,
                parameters=EMPTY_PARAMETERS,
            )
        )
        with self.assertRaisesRegex(CompilationError, "no matching READ access"):
            compile_graph(graph, registry)

    def test_replaced_non_metadata_view_is_not_silently_skipped(self):
        schemas = create_default_schema_registry()
        schemas.register(
            OpSchema("view", lambda inputs, attrs: (OutputSpec(inputs[0]),)),
            replace=True,
        )
        builder = GraphBuilder("replaced_view", schemas=schemas)
        x = builder.input("x", TensorSpec((1, 1)))
        graph = builder.output(builder.call("view", x))
        self.assertFalse(graph.nodes[0].metadata_only)
        with self.assertRaises(RegistryError):
            compile_graph(graph, create_cpu_registry())

    def test_control_dependency_through_metadata_node_is_preserved(self):
        builder = GraphBuilder("metadata_control")
        x = builder.input("x", TensorSpec((1, 1)))
        first = builder.silu(x)
        builder.view(first, (1, 1))
        view_node = builder.graph.nodes[-1]
        y = builder.input("y", TensorSpec((1, 1)))
        final = builder.call("silu", y, control_deps=(view_node,))
        artifact = compile_graph(builder.output(final), create_cpu_registry())
        self.assertEqual([task.node_id for task in artifact.tasks], [0, 2])
        self.assertEqual(artifact.dependencies.successors[0], (1,))

    def test_control_dependency_through_empty_node_is_preserved(self):
        builder = GraphBuilder("empty_control")
        x = builder.input("x", TensorSpec((1, 1)))
        builder.silu(x)
        first_node = builder.graph.nodes[-1]
        empty = builder.input("empty", TensorSpec((0, 1)))
        builder.call("silu", empty, control_deps=(first_node,))
        empty_node = builder.graph.nodes[-1]
        y = builder.input("y", TensorSpec((1, 1)))
        final = builder.call("silu", y, control_deps=(empty_node,))
        artifact = compile_graph(builder.output(final), create_cpu_registry())
        self.assertEqual([task.node_id for task in artifact.tasks], [0, 2])
        self.assertEqual(artifact.dependencies.successors[0], (1,))

    def test_compilation_snapshots_and_freezes_graph(self):
        builder = GraphBuilder("snapshot")
        x = builder.input("x", TensorSpec((1, 1)))
        graph = builder.output(builder.silu(x))
        artifact = compile_graph(graph, create_cpu_registry())
        builder.input("late", TensorSpec((1, 1)))
        self.assertEqual(len(artifact.graph.inputs), 1)
        self.assertEqual(len(artifact.program.buffers), 2)
        with self.assertRaises(GraphValidationError):
            artifact.graph.input("forbidden", TensorSpec((1, 1)))
        with self.assertRaises(TypeError):
            artifact.graph.nodes[0].attrs["mutate"] = True

    def test_non_base_view_region_falls_back_to_whole_storage(self):
        schemas = create_default_schema_registry()
        schemas.register(
            OpSchema(
                "consume_reinterpret", lambda inputs, attrs: (OutputSpec(inputs[0]),)
            )
        )
        builder = GraphBuilder("reinterpret_region", schemas=schemas)
        base = builder.input("base", TensorSpec((4, 4), DType.FP32))
        reinterpret = builder.graph.add_node(
            "reinterpret",
            (base,),
            (
                OutputSpec(
                    TensorSpec((4, 4), DType.FP16),
                    alias_input=0,
                    strides=(4, 1),
                ),
            ),
            metadata_only=True,
        )
        output = builder.call("consume_reinterpret", reinterpret)
        graph = builder.output(output)

        def lower(context):
            source = context.graph.value(context.node.inputs[0])
            result = context.graph.value(context.node.outputs[0])
            region = Region(((2, 4), (0, 4)))
            yield TaskDraft(
                context.node.id,
                (0, 0, 0),
                (
                    OperandBinding(source.id, source.storage_id, AccessMode.READ),
                    OperandBinding(result.id, result.storage_id, AccessMode.WRITE),
                ),
                (
                    Access(
                        ResourceRef.buffer(source.storage_id),
                        AccessMode.READ,
                        region,
                        source.id,
                    ),
                    Access(
                        ResourceRef.buffer(result.storage_id),
                        AccessMode.WRITE,
                        region,
                        result.id,
                    ),
                ),
                {},
            )

        registry = KernelRegistry()
        registry.register(
            KernelVariant(
                "test.consume_reinterpret",
                "consume_reinterpret",
                "test",
                frozenset({Architecture.CPU}),
                lower,
                parameters=EMPTY_PARAMETERS,
            )
        )
        artifact = compile_graph(graph, registry)
        self.assertIsNone(artifact.tasks[0].accesses[0].region)

    def test_empty_output_requires_explicit_kernel_noop_contract(self):
        schemas = create_default_schema_registry()
        schemas.register(
            OpSchema(
                "unsafe_empty",
                lambda inputs, attrs: (OutputSpec(TensorSpec((0, 1))),),
            )
        )
        builder = GraphBuilder("unsafe_empty", schemas=schemas)
        x = builder.input("x", TensorSpec((1024, 1)))
        graph = builder.output(builder.call("unsafe_empty", x))
        registry = KernelRegistry()
        registry.register(
            KernelVariant(
                "test.unsafe_empty",
                "unsafe_empty",
                "test",
                frozenset({Architecture.CPU}),
                lambda context: (),
            )
        )
        with self.assertRaisesRegex(CompilationError, "produced no tasks"):
            compile_graph(graph, registry)

    def test_long_metadata_chain_does_not_use_python_recursion(self):
        builder = GraphBuilder("long_views")
        x = builder.input("x", TensorSpec((1, 1)))
        first = builder.silu(x)
        value = first
        for _ in range(1100):
            value = builder.view(value, (1, 1))
        last_view_node = builder.graph.nodes[-1]
        y = builder.input("y", TensorSpec((1, 1)))
        final = builder.call("silu", y, control_deps=(last_view_node,))
        artifact = compile_graph(builder.output(final), create_cpu_registry())
        self.assertEqual(len(artifact.tasks), 2)
        self.assertEqual(artifact.dependencies.successors[0], (1,))


if __name__ == "__main__":
    unittest.main()
