# 后端与 kernel 插件契约

## 1. 契约的目的

后端开发者只需要回答四个问题：

1. 这个 variant 在什么目标、shape、dtype、layout 下合法？
2. 一个语义 node 应切成哪些 task？
3. 每个 task 的 operand、参数和真实资源访问区域是什么？
4. 该 task 在目标架构内部如何实现和同步？

通用编译器负责 variant 选择、跨 task hazard、调度和命令序列化。kernel 内部的 warp specialization、TMA/WGMMA/tcgen05 pipeline 不应泄漏到 Graph IR；相反，kernel 也不能依赖某个 task 固定落到某个物理 SM。

Phase 1 的完整可执行插件路径是 CPU executor。SM90/SM100 variant 已能参与选择、lowering 和命令生成，但项目尚无 GPU binary handle/dispatch table/device runtime，所以“注册 CUDA kernel 并执行”仍是后续工作。下文将当前接口和 GPU runtime 必须实现的协议明确分开。

## 2. 当前 kernel 开发者的完整注册流程

### 步骤 1：确认或注册语义 schema

kernel variant 实现已有语义时，只需复用 op 名。如果要增加新 op，先在 `SchemaRegistry` 注册 `OpSchema`。schema 只做后端无关的 arity、shape、dtype 和属性检查，并返回零个或多个 `OutputSpec`；零输出用于只更新 state/effect 的 op，整张 Graph 仍需标记输出。纯 view/reshape 元数据 op 可设置 `OpSchema.metadata_only=True`，但它的所有输出必须 alias 输入且不能带 control/effect，此类 node 不需要 kernel variant。

不要在 schema 中选择 tile、WGMMA shape 或 SM 数量。原地/alias 语义必须在这里用 `OutputSpec(alias_input=...)` 明确声明，不能由后端偷偷决定。

### 步骤 2：定义版本化参数 schema

使用 `ParameterSchema` 和 `ParamField` 声明 task 私有参数。只允许明确的 u32/i32/u64/i64/f32/f64。字段变化、含义变化或 operand 顺序不兼容时，应提高 `KernelVariant.abi_version`。

### 步骤 3：写 constraint

`constraint(LoweringContext) -> bool` 必须拒绝所有 kernel 无法正确处理的情况，包括：

- dtype、storage dtype、quant format；
- rank、shape、对齐与整除条件；
- contiguous/strided/opaque layout；
- byte offset 和 alias 形态；
- target 特性，例如 SM90 WGMMA 与 SM100 tcgen05/TMEM；
- tile 尾部是否有 mask 实现。

constraint 只能返回合法性，variant 间的性能排序当前由静态 `priority` 决定。后续 autotune 可以生成更精确的选择策略。

### 步骤 4：写 lowering

`lower(LoweringContext)` 逐个产生 `TaskDraft`。每个 draft 必须提供：

- `node_id`：等于当前 context node id；
- `coordinates`：稳定、非负的逻辑 xyz；
- `operands`：严格按 kernel ABI 顺序排列的 `OperandBinding`；
- `accesses`：覆盖所有真实 buffer/effect/worklist 读写，并尽量给出精确 `Region`；
- `params`：key 与参数 schema 完全一致；
- `estimated_cost`：正数，用于静态 critical-path scheduling；
- 可选 `Guard`：表达 bounded dynamic work。

少报 access 会造成竞态和错误结果；过度使用 whole-resource access 虽然安全，但会失去跨 op 流水。通用编译器会双向检查每个 buffer operand/access 的 value、storage 和 mode，并检查每个非空输出至少有一个写；它仍不能证明声明的 region 覆盖 kernel 实际触碰的每个地址。对非 base view，value-relative region 会自动降级为 whole-resource；只有后端已把范围正规化到 base storage 坐标时，才应设置 `Access.region_is_storage=True`。

普通 node 的 lowering 必须产生至少一个 task。唯一例外是：node 至少有一个输出、所有输出 numel 都为 0、没有 effect，且 variant 显式设置 `empty_is_noop=True`，此时空 draft 集表示合法 no-work program。只有 kernel 对所有允许的零维组合确实无需初始化/发布数据时才能打开该能力；side-effect-only node 即使没有输出也必须产生 task，否则副作用会丢失。改变这项语义能力时应连同 kernel contract 审查并提高 ABI version，避免同一 stable id 的 manifest 含义漂移。

### 步骤 5：声明资源与架构

`KernelResources` 记录：

- `threads_per_cta`；
- `dynamic_smem_bytes`；
- `registers_per_thread_hint`；
- `cluster_shape`；
- `pipeline_stages`；
- SM100 的 `tmem_columns`；
- `warp_roles`；
- `requires_cooperative_launch`。

这些是供未来 manifest/resource planning 使用的声明，不会自动创建 warp、barrier 或 cluster；当前 `KernelRegistry.manifest()` 还没有导出 resources。代码检查 threads 在 `[1, 1024]`、dynamic SMEM/register/TMEM hint 非负、pipeline stage 与 cluster 三维为正，并拒绝空、越界或相互重叠的 warp role。GPU toolchain 仍必须按目标设备校验真实资源上限、cluster launch 条件和可驻留 CTA 数。

### 步骤 6：提供执行实现

- CPU variant：设置 `executor(ExecutionContext)`，通过 `context.array(i)` 取 operand buffer，通过 `context.parameters` 取已解包参数。
- GPU variant：Phase 1 尚无可注册的 module/function/binary handle。可以先注册 lowering 以验证 Task DAG 和 MKCB 生成，但不能在本仓库 runtime 中执行。后续接口必须把 stable kernel id 映射到与 command ABI 匹配的设备端实现。

### 步骤 7：注册与选择

调用 `KernelRegistry.register(variant)`。stable id 由 `backend:name:abi_version` 计算；registry 会拒绝 stable-id 冲突和完全重复注册，但允许同一 op 下同名、不同 backend/ABI version 的实现并存。编译时仅考虑 target 在 `architectures` 且 constraint 为真的 variant，按 priority 降序、name、backend、ABI version 降序、stable id 确定性选择。

### 步骤 8：测试

每个 variant 至少应覆盖：

1. schema 正例和 shape/dtype/layout/quant 反例；
2. tile 边界和尾块；
3. lowering 的 operand 顺序、参数 round-trip 和 access region；
4. 同 storage alias 下的 RAW/WAR/WAW；
5. MKCB serialize/deserialize 确定性；
6. 与 CPU oracle 的数值比较；
7. GPU 上的 memcheck/racecheck、不同调度次序和多次 epoch 复用；
8. 性能基线及 nsys/ncu 数据。

## 3. 一个完整的 Phase 1 插件示例

下面的 `scaled_add(x, y, alpha)` 展示 schema、lowering、typed params、CPU executor 与注册。它可以放在独立包中，不需要修改 compiler 核心。

```python
from collections.abc import Iterable

from megakernel import (
    Access, AccessMode, Architecture, DType, KernelRegistry, Layout,
    KernelResources, KernelVariant, OpSchema, Region, ResourceRef,
)
from megakernel.errors import ShapeError
from megakernel.ir import OutputSpec
from megakernel.parameters import ParamField, ParameterSchema, ParamType
from megakernel.registry import LoweringContext
from megakernel.runtime.interpreter import ExecutionContext
from megakernel.task import OperandBinding, TaskDraft


def infer_scaled_add(inputs, attrs):
    if len(inputs) != 2 or inputs[0] != inputs[1]:
        raise ShapeError("scaled_add expects two identical tensor specs")
    float(attrs["alpha"])
    return (OutputSpec(inputs[0]),)


SCALED_ADD_PARAMS = ParameterSchema(
    ParamField("row_begin", ParamType.U32),
    ParamField("row_end", ParamType.U32),
    ParamField("col_begin", ParamType.U32),
    ParamField("col_end", ParamType.U32),
    ParamField("alpha", ParamType.F32),
)


def legal_cpu_fp32(context: LoweringContext) -> bool:
    graph, node = context.graph, context.node
    values = [graph.value(i) for i in (*node.inputs, *node.outputs)]
    return all(
        value.spec.dtype == DType.FP32
        and value.spec.storage_dtype == DType.FP32
        and value.spec.layout == Layout.ROW_MAJOR
        and value.spec.quant is None
        and value.spec.physical_shape == value.spec.shape
        and value.spec.rank == 2
        and value.byte_offset == 0
        and value.resolved_strides() == value.spec.contiguous_strides()
        and value.spec.shape == graph.storage(value.storage_id).spec.shape
        and graph.storage(value.storage_id).spec.storage_dtype == DType.FP32
        and graph.storage(value.storage_id).spec.quant is None
        for value in values
    )


def lower_scaled_add(context: LoweringContext) -> Iterable[TaskDraft]:
    graph, node = context.graph, context.node
    x, y = (graph.value(i) for i in node.inputs)
    out = graph.value(node.outputs[0])
    m, n = out.spec.concrete_shape()
    tile_m, tile_n = 2, 4
    for row in range(0, m, tile_m):
        for col in range(0, n, tile_n):
            row_end, col_end = min(row + tile_m, m), min(col + tile_n, n)
            region = Region(((row, row_end), (col, col_end)))
            yield TaskDraft(
                node_id=node.id,
                coordinates=(row // tile_m, col // tile_n, 0),
                operands=(
                    OperandBinding(x.id, x.storage_id, AccessMode.READ),
                    OperandBinding(y.id, y.storage_id, AccessMode.READ),
                    OperandBinding(out.id, out.storage_id, AccessMode.WRITE),
                ),
                accesses=(
                    Access(ResourceRef.buffer(x.storage_id), AccessMode.READ, region, x.id),
                    Access(ResourceRef.buffer(y.storage_id), AccessMode.READ, region, y.id),
                    Access(ResourceRef.buffer(out.storage_id), AccessMode.WRITE, region, out.id),
                ),
                params={
                    "row_begin": row,
                    "row_end": row_end,
                    "col_begin": col,
                    "col_end": col_end,
                    "alpha": float(node.attrs["alpha"]),
                },
                estimated_cost=max(1.0, 2.0 * (row_end - row) * (col_end - col)),
            )


def execute_scaled_add(context: ExecutionContext) -> None:
    p = context.parameters
    tile = (slice(p["row_begin"], p["row_end"]),
            slice(p["col_begin"], p["col_end"]))
    context.array(2)[tile] = context.array(0)[tile] + p["alpha"] * context.array(1)[tile]


def register_scaled_add(schemas, kernels: KernelRegistry) -> None:
    schemas.register(OpSchema("scaled_add", infer_scaled_add, defaults={"alpha": 1.0}))
    kernels.register(KernelVariant(
        name="example.scaled_add.tile2d",
        op="scaled_add",
        backend="numpy-plugin",
        architectures=frozenset({Architecture.CPU}),
        lower=lower_scaled_add,
        parameters=SCALED_ADD_PARAMS,
        executor=execute_scaled_add,
        resources=KernelResources(threads_per_cta=1),
        constraint=legal_cpu_fp32,
        priority=10,
        abi_version=1,
        tags=frozenset({"reference", "tile2d"}),
        empty_is_noop=True,
    ))
```

实际使用时，把同一个 `schemas` 传给 `GraphBuilder(schemas=schemas)`，把 `kernels` 传给 `compile_graph()`。如果后续把 operand 顺序改成 `(out, x, y)` 或更改参数布局，应将 `abi_version` 升为 2。

## 4. SM90/H20 与 SM100/B300 的后端边界

Graph、Task DAG、region dependency 和 MKCB 应保持架构无关。架构差异集中在 variant constraint/resources、kernel 内部实现和 device runtime：

| 方面 | Hopper H20 / SM90 | Blackwell B300 / SM100 |
|---|---|---|
| Tensor Core 主路径 | WGMMA，warpgroup 参与，累加器主要占寄存器 | tcgen05，显式 TMEM 生命周期，支持 CTA/2-CTA cooperative 形态 |
| 异步搬运 | TMA + mbarrier | TMA + mbarrier，并结合 tcgen05/TMEM |
| task worker 调度 | 软件 persistent ready queue/工作窃取；不能依赖 CLC | 基线仍可用软件 ready queue；architecture-local tile loop 可评估 CLC |
| 累加器资源 | register pressure 是关键约束 | `tmem_columns`、TMEM alloc/dealloc 与 epilogue handoff 是关键约束 |
| cluster | 可用于 SM90 cluster/TMA multicast 等，但不能假定所有 task 在同 cluster | 可用于 tcgen05 `cta_group::2` 等 2-SM cooperative variant |
| kernel 内完成 | 等待 WGMMA group、TMA/mbarrier phase 后才能发布 task 完成 | 完成 tcgen05 fence/TMEM handoff、TMA/mbarrier phase、epilogue store 后才能发布 |

以下内容必须由后端私有实现，不进入通用前端：

- warp role 的具体指令流和寄存器分配；
- TMA descriptor、SMEM swizzle、mbarrier 数量与 phase；
- WGMMA commit/wait 或 tcgen05/TMEM fence、alloc/dealloc；
- SM100 CLC、2-SM cooperative 和 architecture-specific tile scheduler；
- NVFP4/FP8 packed layout 与 scale load pipeline。

`KernelResources.warp_roles` 等字段用于 manifest、合法性检查和调试，不意味着通用 runtime 能把某个 Python task 拆给这些 warps。一个 task 被某个 resident worker 领取后，整个 variant 必须自行完成内部 warp/CTA 协作。

CLC 尤其不能直接替代通用 Task DAG：CLC 适合 architecture-local 的 tile 获取/持久化循环，而 Task DAG 还包含不同 op、state/effect hazard、guard 和不同 kernel id。是否把一组同构 task 交给 CLC 是 SM100 codegen/runtime 优化，不能改变 MKCB 的依赖语义。

### 单次 megakernel launch 的资源包络

CUDA 的一个 grid 只有一个固定 `blockDim`、dynamic shared-memory 大小和 cluster launch 形态。Phase 1 虽然允许每个 variant 声明不同 `KernelResources`，但未来 device interpreter 不能在同一次 launch 中按 task 任意切换这些 launch 属性。GPU linker/codegen 必须在编译 command program 时执行 compatibility 检查：

- `threads_per_cta` 采用共同 envelope；线程较少的 variant 只能安全地使用子集，且不能让部分线程绕过后续 CTA barrier；
- shared-memory/register 的静态包络由同一 megakernel 中最坏路径决定，会直接影响 persistent occupancy；
- 普通 1-CTA task 与要求 `cluster_shape != (1,1,1)` 或 2-SM cooperative 的 task 不能假定拥有相同 worker 单位；runtime 必须采用统一 cluster envelope 和 cluster-aware dispatch，或把不兼容部分切成另一个 launch；
- `requires_cooperative_launch` 只要被任一可达 variant 要求，就会提升整个 launch 的约束；
- SM100 的 TMEM 可以在 task 内按 variant 分配，但必须在所有退出/guard/error 路径正确 dealloc，并服从共同驻留预算。

首个 GPU backend 应优先定义一个窄而统一的 CTA resource class。如果 variant 不兼容，允许显式生成多个 megakernel segment 和普通 kernel fallback；不能静默忽略 resources 后仍宣称是一个正确的单 launch megakernel。后续 profile-guided partition 再权衡“更大融合范围”和“最坏资源包络导致的低 occupancy”。

## 5. 设备端 command runtime 的最低契约

### 5.1 Host launch 前

host runtime 至少需要：

1. 解析并完整验证 MKCB，校验 target 与当前 device；
2. 用随 binary 发布的 kernel manifest 验证 `kernel_id`、ABI version、operand/params 和目标架构；
3. 建立 invocation-local buffer binding table，校验地址、容量、对齐和可访问性；
4. 为 temporary storage 做 liveness-aware 或保守 arena 分配；
5. 分配 runtime-owned task state，例如 completion epoch/ready queue；这些可变状态不能写入可缓存的只读 command blob；
6. 选择保证可驻留、不会因 dependency spin 造成 starvation 的 persistent grid/cluster 配置。

### 5.2 Worker 领取 task

正确性不能依赖 `worker_hint`。推荐的最小调度基线是只把 ready candidate 放入队列，避免 CTA 因等待未调度的 predecessor 而占满 GPU；reference runtime 在出队后仍对每个 predecessor 做 device-scope acquire epoch 检查。只有在 ready-queue publication/dequeue 本身已被严格证明建立等价的 fan-in happens-before 时，才可以省掉逐 predecessor acquire。worker 需要：

1. 求值 guard；false 时走“无数据写入但完成”的路径，并依赖 frontend/plugin 保证 alias/预初始化输出仍有效或 consumer 同样受保护；
2. 按 operand/value descriptor 解析 base pointer、byte offset、shape 和 stride；
3. dispatch stable kernel id 对应的实现；
4. 等待该实现所有异步工作真正结束；
5. 发布 task completion，唤醒或使 successor 可见。

不能让所有 CTA 先领取尚未 ready 的 task 后原地自旋，否则驻留 CTA 可能占满 GPU，使其 predecessor 永远无法被调度。

## 6. Release/acquire/epoch：跨 CTA 正确性的硬要求

### 6.1 为什么普通 flag 不够

producer task 对 output 的普通 global store 与“task 已完成”flag 是两个不同地址。没有内存序时，另一个 SM 上的 consumer 即使看见 flag，也不保证看见 output。`volatile` 不是跨 CTA happens-before；CTA-scope或 cluster-scope同步也不能覆盖任意 SM 上的 persistent worker。

在本项目的单 GPU边界内，task completion 必须使用覆盖整个 device 的 scope：CUDA C++ 的 `cuda::thread_scope_device`，或 PTX 的 `.gpu`。只有当 producer/consumer 被严格限定在同一 CTA/cluster 时才可缩小 scope；通用 Task DAG 不具备这个保证。未来扩展 host 或多 GPU 观察者时才需要 `.sys`/system scope。

### 6.2 推荐的 reference protocol

为每个 invocation 分配一个不会与仍在运行的旧 invocation 重复的 64-bit `epoch`，runtime-owned 数组初始化为非当前 epoch：

```text
completion_epoch[task_id] : atomic<u64, device scope>
```

consumer：

```cuda
for (uint32_t pred : predecessors(task)) {
    while (completion_epoch[pred].load(cuda::memory_order_acquire) != epoch) {
        backoff_or_help_ready_work();
    }
}
// acquire 之后才能读取 predecessor 写出的 tensor/state。
```

producer：

```cuda
run_variant_and_wait_for_all_async_work();
completion_epoch[task_id].store(epoch, cuda::memory_order_release);
```

release 保证该 task 在 program order 中更早的数据写入先于 completion 发布；读取到同一 epoch 的 acquire 保证 consumer 后续读取看见这些写入。一个普通 `__threadfence()` 加非原子 flag 容易引入竞态，优先使用 scoped atomic 的 release store/acquire load，或经过同等验证的 PTX sequence。

### 6.3 epoch 解决 ABA 与复用

只用布尔 `done` 时，复用 completion table 可能出现 ABA：新 invocation 的 worker 读到旧 invocation 遗留的 `true`，提前消费旧数据。epoch 必须满足：

- 每个同时在途或可能被旧 worker 观察到的 invocation 唯一；
- completion slot 只有等于当前 epoch 才算完成；
- host 不得在旧 megakernel 仍运行时复用相同 epoch/buffer state；
- 64-bit wraparound 前必须 drain 所有 invocation 并重置表，不能静默回绕；
- state buffer 的 request/sequence ownership 与 task completion epoch 分开管理。

### 6.4 ready counter 优化的陷阱

successor indegree 原子递减可以减少 predecessor polling，但不能把“计数到零”自动当成已建立所有数据的 acquire。PTX atomic reduction `red` 不形成 acquire pattern；错误地使用 `red.add`/`red.inc` 后接 acquire fence仍可能读到旧 output。

首个 GPU runtime 应先实现上面的 completion-epoch reference protocol。若优化成 predecessor fan-in counter，需要用经过 CUDA memory model证明的 scoped atomic RMW/release sequence，并为多 producer、最后一个 consumer、不同 memory synchronization domain 写 litmus/race tests；在证明完成前不能仅凭“atomic”二字替换 acquire。

### 6.5 异步 proxy 与 task 完成

外层 release 只能发布当前线程已经正确观察到的完成。variant 内部使用 TMA、WGMMA 或 tcgen05 时，必须先按对应指令模型完成 async-proxy 同步：

- TMA stage 使用正确初始化、arrival count、transaction bytes 和 parity 的 mbarrier；stage slot 重用时翻转 phase；
- SM90 在读取/写回 WGMMA 结果前完成所需 commit/wait/fence；
- SM100 在读取 TMEM、执行 epilogue和 dealloc 前完成 tcgen05/TMEM 所需 fence 与 CTA/cluster 同步；
- async engine 的最终 global store 若涉及不同 proxy，执行对应 proxy fence/完成等待；
- 若 task 由多个 warp 或 cluster 内多个 CTA 协作，指定 publisher 必须先通过正确作用域的 barrier/acquire 汇合所有参与者的写入；
- 只有最终 output/state store 已经进入可由 device-scope release发布的顺序后，指定 publisher 才能写 completion epoch。

mbarrier 的 phase 是 kernel 内 pipeline stage 的局部概念；task completion epoch 是跨 CTA、跨 op、跨 invocation 的外层概念。二者不能混用。

## 7. GPU backend 验收清单

一个 SM90/SM100 variant 与 runtime 合入前，应满足：

- 同一 Graph 在 CPU oracle 与 GPU 上数值一致，覆盖尾 tile、alias、state 和 guard；
- 打乱 ready task 的 worker 归属仍正确，`worker_hint` 改变不影响结果；
- 连续运行大量 invocation，epoch table 复用无 ABA；
- 两个独立 request/state 并发时不串扰；
- compute-sanitizer memcheck/racecheck 无错误；
- command fuzz/损坏输入在 host trust boundary 被拒绝；
- SM90、SM100 binary manifest 不会错误接收另一架构 kernel id；
- nsys 验证单次 persistent launch和跨 task overlap，ncu 验证预期的 TMA/Tensor Core/occupancy；
- 每项性能结论都记录 GPU、dtype、shape、batch、prefill/decode 模式和对照基线。
