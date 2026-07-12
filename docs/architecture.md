# Phase 1 架构：从模型语义到可执行命令

## 1. 目标与当前边界

本仓库的长期目标是把“模型如何组网、如何切 task、task 如何同步”和“某个硬件上的 kernel 如何流水化”分开。前端只描述语义、数据流与副作用；后端插件负责把一个语义算子切成适合目标架构的 task，并声明每个 task 访问的数据区域；通用编译器据此构造依赖、调度并生成 runtime 可解析的命令。

Phase 1 已经打通以下参考链路：

```text
Python GraphBuilder
        │  schema inference
        ▼
Graph IR ──variant selection/lowering──▶ Task IR + Task DAG
        │                                      │
        │                              critical-path scheduling
        │                                      ▼
        └──────────────────────────────▶ ExecutionPlan
                                               │
                                               ▼
                                      CommandProgram (MKCB v1.0)
                                               │
                                               ▼
                                      NumPy CPUInterpreter
```

目前只有 CPU/NumPy 后端可以真正执行。`Architecture.SM90`、`Architecture.SM100` 和硬件资源描述已经进入公共接口，但 CUDA megakernel、设备端命令解释器、kernel 二进制分发和自动 codegen 尚未实现。因此，Phase 1 是后续 GPU 实现的语义 oracle 与 ABI 原型，不应被描述成已经得到 H20/B300 性能的 megakernel。

## 2. 四层 IR

代码中的编译入口是 `compile_graph()`。它对应四层稳定边界；其中第二层的“TaskGraph”是 `tuple[Task, ...] + TaskDependencies` 的逻辑称呼，当前没有名为 `TaskGraph` 的包装类。

### 2.1 第一层：Graph IR——后端无关的模型语义

`Graph` 由 `Node`、`Value`、`Storage` 和 `EffectResource` 构成。

- `GraphBuilder` 是当前 Python 前端。`input()`、`parameter()`、`state()` 创建外部绑定，`call()` 通过 `SchemaRegistry` 查找 `OpSchema`，完成输出 shape/dtype 推导后追加节点。
- `Node` 表示语义算子，只记录 op 名、输入/输出 value、快照化属性、显式 control dependency、effect use 和 `metadata_only` 标志，不记录 CUDA tile、warp 或 PTX 指令。schema 可以返回零个输出，以表达只修改 state/effect 的 side-effect-only op；整张 Graph 仍必须至少标记一个输出。
- `Value` 使用 SSA 风格：每次算子调用产生新的逻辑 value。`Storage` 表示实际存储身份；多个 value 可以共享同一个 `storage_id`，从而表达 view 和原地状态更新。
- `EffectResource` 表达无法或不适合用张量区域准确建模的顺序约束，例如某个 cache epoch、随机数状态或外部队列。
- `Symbol` 表示编译期特化维度。编译前必须通过 `CompileOptions.symbolic_bindings` 绑定为整数；它不是运行期 ragged 维度。

语义 schema 与 kernel registry 是两套独立注册表。新增 SM100 kernel 不需要修改建图逻辑；新增语义 op 时才需要增加 `OpSchema`。

### 2.2 第二层：Task IR——切分、资源访问与依赖

编译器先调用 `Graph.bind()` 创建独立快照（即使没有 Symbol），再冻结该快照；因此原始 builder 后续追加内容不会改变 `CompiledArtifact.graph`，attrs 的 mapping/标准容器也已被冻结或复制。随后按 Graph 拓扑顺序处理节点：

1. 用 `(node.op, target, constraint)` 从 `KernelRegistry` 选择一个 `KernelVariant`；优先级高者先选，同优先级依次按名字、backend、较新的 ABI version 和 stable id 确定性打破平局。
2. `metadata_only` node 不选择 variant；它必须没有 control/effect 且所有输出 alias 输入。普通 node 调用 variant 的 `lower(LoweringContext)`，产生一个或多个 `TaskDraft`。唯一合法的零 task 普通 node 是：至少有一个输出、所有输出 numel 都为 0、没有 effect，且所选 variant 显式声明 `empty_is_noop=True`。
3. 用 variant 声明的 `ParameterSchema` 把 task 参数编码成定长类型的 little-endian bytes，形成 `Task`。
4. 双向校验 operand 与 buffer access：operand 必须引用当前 node 的 input/output、buffer 必须等于 value 的 storage，且每个 buffer operand/access 都要有 mode 兼容的另一方。
5. 对非 base view 的 value-relative region 保守降级为 whole-resource access；plugin 只有在明确设置 `region_is_storage=True` 时才能声称 region 已经使用 storage 坐标。
6. 将 node 上的 effect use 转成额外的 task `Access`。
7. 校验该节点每个输出 storage 至少被一个 task 声明为 `WRITE`、`READ_WRITE` 或 `REDUCE`；零输出的 side-effect-only node 则依赖其 state/effect access。
8. 对 metadata/零元素等 taskless node，沿显式 control edge 和输入 producer 向前寻找最近的 taskful ancestors，确保穿过 taskless node 的 control dependency 不会丢失。
9. 基于所有 task 的资源访问推导 `TaskDependencies`：整资源访问走 last-writer/readers frontier，区域访问走保守历史匹配。

一个 task 包含：稳定的顺序 id、来源 node、kernel stable id、逻辑三维坐标、operand binding、精确或保守的 access、参数 bytes、代价估计和可选 guard。后端 lowering 决定 tile 粒度；通用编译器不理解 GEMM 的 M/N/K 或 attention 的 head/page 含义。

#### Region-aware dependency

`Access` 把一次访问描述为：

```text
ResourceRef(kind, id) × AccessMode × Region? × value_id?
```

`Region` 是各轴半开区间 `[begin, end)` 的直积。`None` 表示整资源或无法证明范围。对全为整资源的访问，依赖器维护 writer/readers frontier，连续 2000 个 whole-buffer write 只形成 1999 条链边，不先构造完全图；一旦出现区域访问，则按 task id 顺序匹配该资源的相关历史访问：

- 不同 resource 永不冲突。
- 两边都有 region 且可证明不重叠时不冲突。
- `READ`/`READ` 不冲突。
- 其余重叠访问形成 RAW、WAR 或 WAW 边。
- 两个 `REDUCE` 只有在都标记 `atomic=True` 且 `reduction_op` 相同的情况下才允许并行。
- 显式 `Node.control_deps` 会在前驱节点的所有 task 与后继节点的所有 task 之间建立边。

例如，producer 写 `[0:2, :]` 和 `[2:4, :]` 两个 tile，consumer 分别读这两个 tile，那么第二个 producer 完成前，第一个 consumer 就可以开始。这是跨 op 流水的基础。

不同 rank 的 alias/view 当前没有 byte-level 证明，`Region.overlaps()` 会保守地认为它们重叠。对于带 offset/非 base stride 的 view，compiler 默认在依赖生成前把 value-relative region 直接降级为 `None`；只有后端已经把区域正规化到 base storage 坐标并声明 `region_is_storage=True` 时才保留它。依赖生成结束后默认做 transitive reduction；它只删除已有的传递边，不改变可达关系。

#### State、effect 和 view 如何进入依赖

- 原地更新产生新的 SSA `Value`，但沿用状态输入的 `storage_id`。因此后续访问仍能通过同一个 buffer resource 捕获 RAW/WAR/WAW hazard。
- 一个 node 的 effect use 会附加到该 node 的每个 task。effect 没有 region，因此写 effect 会成为保守顺序屏障；例如四个 tile 都写同一 effect 时会形成串行链。
- 默认 `view` schema 将 node 标记为 `metadata_only`，只产生共享 storage 的 value 元数据。编译器按标志而不是硬编码 op 名跳过这类 node，不生成 runtime task，并拒绝其携带 control/effect edge或产生非 alias 输出；真正消费 view 的 kernel 必须通过 value descriptor 的 offset/stride 解释它。

### 2.3 第三层：ExecutionPlan——确定性的静态调度建议

`schedule_tasks()` 实现 critical-path list scheduling：

1. 反向计算每个 task 的 `estimated_cost + max(successor critical path)`。
2. 将入度为零的 task 放入 ready heap，优先取 critical path 更长者，同分时取较小 task id。
3. 在依赖均完成的前提下，选择最早可用的逻辑 worker，并计算模拟的 `start`/`finish`。
4. 输出 `dispatch_order`、每个 task 的 `worker_hint` 和估算 makespan。

这层的 `worker_hint` 只是 runtime 的放置提示，不表示“逻辑 worker i 就是物理 SM i”。CUDA 不保证普通 CTA 固定运行在指定 SM 上，未来 persistent runtime 也应允许 ready task 被任意驻留 worker 窃取。当前调度器也没有使用 occupancy、shared memory、cluster、TMEM、cache locality 或实测 latency，因此它是确定性的功能基线，不是硬件性能模型。

### 2.4 第四层：CommandProgram——可重定位 runtime 指令

`CommandProgram` 是 runtime-facing IR，序列化格式 magic 为 `MKCB`，当前 ABI 为 1.0。它把对象图展平成以下 section：

- buffer descriptor：物理 dtype/shape/stride、字节数、flags 和 quant spec id；
- value descriptor：逻辑 dtype/shape/stride、所属 buffer、byte offset 和 quant spec id；
- 64-byte task instruction：kernel id、worker hint、依赖/后继/operand/参数切片以及逻辑 xyz；
- operand descriptor；
- 扁平 predecessor、successor 表；
- 共享 i64 dimension 表；
- 每 task 对齐后的参数 blob。

命令中没有原始指针。`buffer_id` 只是 invocation binding table 的索引，host runtime 必须在每次调用时另行提供真实地址。这使同一命令可以重定位、缓存和用于不同输入地址。完整字段与验证规则见 [data_structures.md](data_structures.md)。

## 3. 一次编译如何工作

以当前 gated MLP 为例，流程是：

1. 前端构造 `rms_norm → matmul(up)` 与 `matmul(gate) → silu → mul → matmul(down) → residual add` 的 Graph DAG。
2. shape schema 检查 rank、K 维和 dtype，并为每个输出分配 temporary storage。
3. NumPy plugin 将每个 op 切成二维 tile；每个 tile 声明自己读取和写入的矩形 region。
4. 依赖器只连接真正重叠的 producer/consumer tile，而不是无条件建立整 op barrier。
5. 调度器按 critical path 和 worker 可用时间给出 dispatch order。
6. compiler 生成无指针的 MKCB 命令。
7. `CPUInterpreter` 依据同一 task DAG 执行 plugin executor，并与直接 NumPy 公式比较。

这种闭环先证明前端切分、别名 hazard、依赖和命令布局正确，再把相同 contract 交给 SM90/SM100 后端。

## 4. CPU oracle 的职责

`CPUInterpreter` 不是性能模拟器，而是语义与调度正确性的 oracle：

- 只接受 `Architecture.CPU` artifact；
- 在执行任何 task 前检查所有 storage 都是 row-major、无量化、逻辑/物理 dtype 与 shape 一致，并拒绝 value dtype reinterpretation；即使图只有 input/view、没有 compute task，也不会绕过该检查；
- 严格检查所有 input/constant/state 的名称、shape 和 NumPy dtype；
- 为 temporary storage 分配 NumPy 数组；
- 按 DAG ready 状态执行 task，使用 `dispatch_order` 作为确定性的 ready-task 排序；
- guard 为 false 时跳过 executor，但仍把 task 视为完成并释放 successors；
- 通过 registry stable id 找到同一 variant，解包 typed params 后调用 executor；
- 输出 view 时根据共享 base buffer、byte offset 和 element stride 创建 NumPy view。

当前 oracle 只支持 BOOL、INT32、INT64、UINT8、FP16 和 FP32 的 NumPy 映射，且要求外部 binding 为 C-contiguous；默认计算 plugin 又进一步约束为 row-major、无量化、逻辑/物理 shape 一致的 contiguous FP32。BF16/FP8/FP4、量化数值语义、GPU 异步 proxy 和真实 CTA 并发都不由它覆盖。GPU 后端仍需独立的 sanitizer、race 和端到端数值测试。

## 5. 前后端责任边界

前端/通用编译器负责：

- 模型语义、shape/dtype/layout/quant 元数据；
- value/storage alias、state/effect/control dependency；
- 选择 variant、收集后端声明的 task/access；
- 通用 hazard 推导、静态调度、ABI 序列化和结构验证。

后端 plugin/runtime 负责：

- tile 形状、kernel 参数、operand 顺序和精确 region；
- warp specialization、TMA/WGMMA/tcgen05、SMEM/TMEM pipeline；
- 资源合法性、目标架构约束和 kernel binary dispatch；
- persistent worker、ready queue、跨 CTA 可见性与完成协议；
- 真实代价数据、autotune 与性能回灌。

这条边界的核心约束是：后端可以改变“一个 op 被切成多少 task、每个 task 怎么算”，但不能偷偷改变 op 的语义或漏报读写范围；前端可以改进 DAG 和调度，却不应编码某条 PTX 指令或固定 warp 角色。

## 6. 相关文档

- [data_structures.md](data_structures.md)：核心数据结构、动态性描述与命令 ABI。
- [backend_contract.md](backend_contract.md)：kernel 注册流程、SM90/SM100 边界和设备端同步协议。
- [roadmap.md](roadmap.md)：当前限制及面向 DeepSeek-V4 的分阶段路线。
