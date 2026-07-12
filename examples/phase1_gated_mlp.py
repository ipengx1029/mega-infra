"""Phase 1 end-to-end example: RMSNorm + gated MLP + residual."""

import numpy as np

from megakernel import (
    CPUInterpreter,
    CompileOptions,
    GraphBuilder,
    TensorSpec,
    compile_graph,
    create_cpu_registry,
)


def build_graph():
    builder = GraphBuilder("phase1_gated_mlp")
    x = builder.input("x", TensorSpec((3, 8)))
    norm_weight = builder.parameter("norm_weight", TensorSpec((8,)))
    up_weight = builder.parameter("up_weight", TensorSpec((8, 12)))
    gate_weight = builder.parameter("gate_weight", TensorSpec((8, 12)))
    down_weight = builder.parameter("down_weight", TensorSpec((12, 8)))

    normalized = builder.rms_norm(x, norm_weight, eps=1e-5)
    up = builder.matmul(normalized, up_weight)
    gate = builder.silu(builder.matmul(normalized, gate_weight))
    gated = builder.mul(up, gate)
    projected = builder.matmul(gated, down_weight)
    output = builder.add(x, projected)
    return builder.output(output)


def reference(bindings):
    x = bindings["x"]
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    normalized = x / np.sqrt(variance + np.float32(1e-5))
    normalized *= bindings["norm_weight"]
    up = normalized @ bindings["up_weight"]
    gate = normalized @ bindings["gate_weight"]
    gate = gate / (1.0 + np.exp(-gate))
    return x + (up * gate) @ bindings["down_weight"]


def main():
    graph = build_graph()
    rng = np.random.default_rng(7)
    bindings = {
        "x": rng.normal(size=(3, 8)).astype(np.float32),
        "norm_weight": rng.normal(size=(8,)).astype(np.float32),
        "up_weight": rng.normal(size=(8, 12)).astype(np.float32),
        "gate_weight": rng.normal(size=(8, 12)).astype(np.float32),
        "down_weight": rng.normal(size=(12, 8)).astype(np.float32),
    }

    artifact = compile_graph(
        graph,
        create_cpu_registry(tile_m=2, tile_n=4),
        CompileOptions(num_workers=4),
    )
    actual = next(iter(CPUInterpreter().run(artifact, bindings).values()))
    expected = reference(bindings)
    max_abs_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    print(f"graph: {graph.name} ({len(graph.nodes)} nodes, {len(graph.values)} values)")
    print(f"tasks: {len(artifact.tasks)}")
    print(f"edges: {artifact.dependencies.edge_count}")
    print(f"command bytes: {len(artifact.serialize())}")
    print(f"max abs error: {max_abs_error:.3e}")


if __name__ == "__main__":
    main()
