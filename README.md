# mega-infra

`mega-infra` 面向单卡 megakernel，目标是把“模型语义与任务划分”和“硬件 kernel 实现”分开：前端描述模型数据流，编译层生成带依赖的细粒度任务与稳定命令 ABI，后端只需按统一契约注册 lowering、资源约束和执行入口。目标硬件为 Hopper（H20/SM90）与 Blackwell（B300/SM100）。

## Phase 1

当前已完成一条可执行的基础链路：

- Python `GraphBuilder`、算子 schema 和带 storage/value 语义的 Graph IR；
- kernel 注册与选择、op 到 tile task 的 lowering；
- 基于读写 region、控制边和 effect 的依赖推导及确定性调度；
- 可版本化、可序列化的 command ABI；
- NumPy CPU 语义后端与解释器，用作后续 GPU 后端的正确性 oracle；
- 覆盖 IR、依赖、调度、ABI、插件和端到端执行的测试。

## 目录

- `src/megakernel/`：前端、IR、编译、调度、kernel registry、命令 ABI 与参考 runtime。
- `tests/`：Phase 1 单元测试和端到端测试。
- `examples/`：可直接运行的小模型图示例。
- `docs/`：架构、数据结构、后端契约和演进路线。
- `Megakernels/`、`mirage/`、`Triton-distributed/`：用于设计研究的上游参考项目。
- `deepseekv4/`：后续复杂模型接入的参考模型定义。

## 快速运行

需要 Python 3.10+、NumPy。测试可直接用标准库 `unittest` 运行；也可安装可选的 pytest 依赖。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
PYTHONPATH=src python examples/phase1_gated_mlp.py
```

示例构建并执行 `RMSNorm -> gated MLP -> residual`，输出 Graph、task、依赖边、序列化命令大小和 NumPy 参考误差。

## 当前非目标与限制

Phase 1 是后端中立的基础设施验证，不是 GPU 性能实现。目前没有生产级 SM90/SM100 persistent kernel、warp specialization/codegen、多卡通信或自动性能搜索；CPU 参考后端仅覆盖少量连续 FP32 算子，动态 shape 主要通过编译期绑定专化。调度结果表达依赖与 worker hint，尚未替代真实 GPU runtime 的并发、同步和资源管理。

## 文档

- [总体架构](docs/architecture.md)
- [核心数据结构](docs/data_structures.md)
- [后端开发契约](docs/backend_contract.md)
- [分阶段路线与 DeepSeekV4 接入考虑](docs/roadmap.md)
