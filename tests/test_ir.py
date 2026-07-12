import unittest

from megakernel import (
    DType,
    EffectMode,
    GraphBuilder,
    OpSchema,
    QuantSpec,
    Symbol,
    TensorSpec,
    create_default_schema_registry,
)
from megakernel.errors import GraphValidationError, ShapeError
from megakernel.ir import OutputSpec


class GraphIRTests(unittest.TestCase):
    def test_symbolic_graph_binding_preserves_alias(self):
        batch = Symbol("batch", minimum=1, maximum=8)
        builder = GraphBuilder("symbolic")
        state = builder.state("state", TensorSpec((batch, 4)))
        update = builder.input("update", TensorSpec((batch, 4)))
        result = builder.copy_inplace(state, update, name="update_state")
        graph = builder.output(result)

        bound = graph.bind({"batch": 3})
        self.assertEqual(bound.inputs[0].spec.shape, (3, 4))
        self.assertEqual(bound.outputs[0].storage_id, bound.inputs[0].storage_id)
        self.assertEqual(bound.outputs[0].spec.nbytes(), 48)

    def test_symbol_bounds_are_checked(self):
        symbol = Symbol("tokens", minimum=1, maximum=4)
        with self.assertRaises(ShapeError):
            TensorSpec((symbol, 8)).concrete_shape({"tokens": 5})
        for invalid in (True, 2.5, "2"):
            with self.assertRaises(ShapeError):
                TensorSpec((symbol, 8)).concrete_shape({"tokens": invalid})

    def test_multi_output_custom_schema(self):
        schemas = create_default_schema_registry()
        schemas.register(
            OpSchema(
                "duplicate",
                lambda inputs, attrs: (OutputSpec(inputs[0]), OutputSpec(inputs[0])),
            )
        )
        builder = GraphBuilder("multi", schemas=schemas)
        value = builder.input("value", TensorSpec((2, 3)))
        first, second = builder.call("duplicate", value, name="pair")
        graph = builder.output(first, second)
        self.assertEqual(len(graph.outputs), 2)
        self.assertNotEqual(first.storage_id, second.storage_id)

    def test_quantized_logical_and_physical_shapes_are_separate(self):
        quant = QuantSpec(
            "nvfp4-e2m1x2-block32",
            DType.UINT8,
            (1, 32),
            DType.FP8_E8M0,
        )
        spec = TensorSpec(
            (8, 64),
            DType.FP4_E2M1,
            storage_dtype=DType.UINT8,
            physical_shape=(8, 32),
            quant=quant,
        )
        self.assertEqual(spec.numel(), 512)
        self.assertEqual(spec.nbytes(), 256)
        self.assertNotEqual(spec.dtype, spec.storage_dtype)
        self.assertNotEqual(quant.stable_id, 0)
        generator_quant = QuantSpec(
            "generated-block",
            DType.UINT8,
            (value for value in (1, 32)),
            DType.FP8_E8M0,
        )
        self.assertEqual(generator_quant.block_shape, (1, 32))
        for invalid in ((True, 32), (1.5, 32), ()):
            with self.assertRaises(ShapeError):
                QuantSpec("invalid", DType.UINT8, invalid, DType.FP8_E8M0)

    def test_view_retains_storage_identity_and_strides(self):
        builder = GraphBuilder("views")
        base = builder.input("base", TensorSpec((4, 4)))
        view = builder.view(base, (2, 2), byte_offset=16, strides=(4, 1), name="slice")
        graph = builder.output(view)
        self.assertEqual(view.storage_id, base.storage_id)
        self.assertEqual(view.byte_offset, 16)
        self.assertEqual(view.resolved_strides(), (4, 1))
        graph.validate()
        for offset, strides in ((1.5, (4, 1)), (0, (1.5, 1)), (True, (4, 1))):
            invalid_builder = GraphBuilder("invalid_view")
            invalid_base = invalid_builder.input("base", TensorSpec((4, 4)))
            with self.assertRaises(ShapeError):
                invalid_builder.view(
                    invalid_base,
                    (2, 2),
                    byte_offset=offset,
                    strides=strides,
                )

    def test_binding_preserves_exact_alias_input_when_inputs_share_storage(self):
        builder = GraphBuilder("alias_choice")
        base = builder.input("base", TensorSpec((4, 4)))
        first = builder.view(base, (2, 2), byte_offset=0, strides=(4, 1), name="first")
        second = builder.view(
            base, (2, 2), byte_offset=16, strides=(4, 1), name="second"
        )
        output = builder.graph.add_node(
            "alias_second",
            (first, second),
            (OutputSpec(second.spec, alias_input=1, strides=(4, 1)),),
        )
        graph = builder.output(output)
        rebound = graph.bind({})
        self.assertEqual(rebound.outputs[0].alias_input, 1)
        self.assertEqual(rebound.outputs[0].byte_offset, 16)

    def test_side_effect_only_schema_is_representable(self):
        schemas = create_default_schema_registry()
        schemas.register(OpSchema("touch", lambda inputs, attrs: ()))
        builder = GraphBuilder("side_effect", schemas=schemas)
        state = builder.state("state", TensorSpec((1, 1)))
        effect = builder.effect("state_version", lifetime="sequence")
        result = builder.call("touch", state, effects=((effect, EffectMode.WRITE),))
        self.assertEqual(result, ())
        graph = builder.output(state)
        self.assertEqual(graph.nodes[0].outputs, ())

    def test_explicit_effect_has_lifetime_and_mode(self):
        builder = GraphBuilder("effect")
        value = builder.input("value", TensorSpec((2, 2)))
        cache_epoch = builder.effect("kv_cache_epoch", lifetime="sequence")
        output = builder.call(
            "silu",
            value,
            effects=((cache_epoch, EffectMode.WRITE),),
            name="stateful_silu",
        )
        graph = builder.output(output)
        self.assertEqual(graph.effects[0].lifetime, "sequence")
        self.assertEqual(graph.nodes[0].effects[0].mode, EffectMode.WRITE)

    def test_graph_requires_an_output(self):
        builder = GraphBuilder("bad")
        builder.input("x", TensorSpec((1, 1)))
        with self.assertRaises(GraphValidationError):
            builder.graph.validate()

    def test_rms_norm_rejects_non_finite_epsilon(self):
        for epsilon in (float("nan"), float("inf"), True, "bad"):
            builder = GraphBuilder("bad_epsilon")
            x = builder.input("x", TensorSpec((1, 4)))
            weight = builder.parameter("weight", TensorSpec((4,)))
            with self.assertRaises(ShapeError):
                builder.rms_norm(x, weight, eps=epsilon)

    def test_structurally_equal_value_from_another_graph_is_rejected(self):
        first = GraphBuilder("first")
        first_value = first.input("x", TensorSpec((1, 1)))
        second = GraphBuilder("second")
        second_value = second.input("x", TensorSpec((1, 1)))
        self.assertEqual(first_value, second_value)
        with self.assertRaises(GraphValidationError):
            first.graph.add_node(
                "identity",
                (second_value,),
                (OutputSpec(second_value.spec),),
            )

    def test_failed_node_append_is_transactional(self):
        builder = GraphBuilder("transaction")
        value = builder.input("x", TensorSpec((1, 1)))
        before = (
            len(builder.graph.values),
            len(builder.graph.storages),
            len(builder.graph.nodes),
        )
        with self.assertRaises(GraphValidationError):
            builder.graph.add_node(
                "broken_multi_output",
                (value,),
                (OutputSpec(value.spec), OutputSpec(value.spec, alias_input=7)),
            )
        after = (
            len(builder.graph.values),
            len(builder.graph.storages),
            len(builder.graph.nodes),
        )
        self.assertEqual(before, after)
        builder.output(value).validate()

    def test_negative_value_and_storage_ids_are_rejected(self):
        builder = GraphBuilder("negative_ids")
        value = builder.input("x", TensorSpec((1, 1)))
        builder.output(value)
        with self.assertRaises(GraphValidationError):
            builder.graph.value(-1)
        with self.assertRaises(GraphValidationError):
            builder.graph.storage(-1)

    def test_inplace_alias_inherits_view_stride(self):
        builder = GraphBuilder("view_update")
        base = builder.state("base", TensorSpec((4, 4)))
        destination = builder.view(
            base, (2, 2), byte_offset=16, strides=(4, 1), name="destination"
        )
        source = builder.input("source", TensorSpec((2, 2)))
        result = builder.copy_inplace(destination, source)
        graph = builder.output(result)
        self.assertEqual(graph.outputs[0].byte_offset, 16)
        self.assertEqual(graph.outputs[0].resolved_strides(), (4, 1))


if __name__ == "__main__":
    unittest.main()
