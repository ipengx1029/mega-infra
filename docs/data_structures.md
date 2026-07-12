# Phase 1 数据结构与命令 ABI

本文记录当前代码中的真实数据模型。带“占位”标记的结构已经定义，但尚未接入完整编译/runtime 链路；这类结构不能视为已支持的功能。

## 1. 稳定枚举

这些枚举值会进入 ABI 或参与跨层转换，修改数值需要按 ABI 变更处理。

| 枚举 | 当前值与含义 |
|---|---|
| `DType` | `BOOL=1`、`INT32=2`、`INT64=3`、`UINT8=4`、`FP16=10`、`BF16=11`、`FP32=12`、`FP8_E4M3=20`、`FP8_E5M2=21`、`FP8_E8M0=22`、`FP4_E2M1=30` |
| `Layout` | `ROW_MAJOR=1`、`COLUMN_MAJOR=2`、`OPAQUE=255` |
| `StorageKind` | `INPUT=1`、`CONSTANT=2`、`TEMPORARY=3`、`STATE=4` |
| `EffectMode` | `READ=1`、`WRITE=2`、`READ_WRITE=3` |
| `AccessMode` | `READ=1`、`WRITE=2`、`READ_WRITE=3`、`REDUCE=4` |
| `ResourceKind` | `BUFFER=1`、`EFFECT=2`、`WORKLIST=3` |
| `Architecture` | `CPU=0`、`SM90=90`、`SM100=100` |
| `GuardOp` | `EQ=1`、`NE=2`、`LT=3`、`LE=4`、`GT=5`、`GE=6` |

`EffectMode` 目前通过相同整数值转换为前三种 `AccessMode`。如果将来扩展任一枚举，不能再默认这种数值对应关系。

## 2. Graph IR 数据结构

### 2.1 Shape、dtype 与物理存储

`TensorSpec` 同时保存逻辑视图和物理存储信息：

| 字段 | 含义 |
|---|---|
| `shape` | 逻辑 shape，元素是非负整数或 `Symbol`，最大 rank 为 8 |
| `dtype` | 计算/逻辑 dtype |
| `layout` | row-major、column-major 或 backend-owned opaque layout |
| `storage_dtype` | 实际打包数据的 dtype；未显式指定时等于 `dtype` |
| `physical_shape` | 计算存储字节数所用的物理 shape；未指定时等于逻辑 shape |
| `quant` | 可选 `QuantSpec` |

`numel()` 基于逻辑 shape；`nbytes()` 基于 `physical_shape × storage_dtype.bits`，最后向上取整到字节。逻辑 stride 由逻辑 shape/layout 推导，physical stride 则由物理 shape、storage dtype 和 layout 推导。`OPAQUE` 的自动 stride 全为 0，表示通用层不解释物理布局。

`Symbol(name, minimum, maximum)` 只支持 compile-time specialization。`Graph.bind()` 会克隆整张图并将逻辑/物理 shape 都绑定为整数，同时保留 alias、control dependency 和 effect。

### 2.2 QuantSpec

`QuantSpec` 描述一种可复用的逻辑到 packed storage 约定：

```python
QuantSpec(
    format="nvfp4-e2m1x2-block32",
    storage_dtype=DType.UINT8,       # 两个 E2M1 打包到一个 byte
    block_shape=(1, 32),
    scale_dtype=DType.FP8_E8M0,
    accumulator_dtype=DType.FP32,
)
```

一个与当前 ABI span 规则一致的 packed E2M1 tensor 可以写成：

```python
quant = QuantSpec(
    "nvfp4-e2m1x2-block32", DType.UINT8, (1, 32), DType.FP8_E8M0
)
spec = TensorSpec(
    (8, 64),
    DType.FP4_E2M1,
    storage_dtype=DType.UINT8,
    physical_shape=(8, 32),
    quant=quant,
)
```

它的 `stable_id` 是对 format、storage dtype、block shape、scale dtype 和 accumulator dtype 做 FNV-1a 得到的 32-bit id。该 id 会进入 buffer/value descriptor。scale tensor 本身应建模为普通 `Value` 并作为 op operand 绑定，而不是藏在 `QuantSpec` 中。

当前边界：

- 命令缓冲只携带 `quant_spec_id`，不会序列化完整 `QuantSpec` manifest；host/runtime 必须与编译端共享相同定义。
- 默认 schema 没有量化算子，CPU plugin/解释器也没有 FP8/FP4 数值实现。
- `QuantSpec` 严格校验 format 为非空字符串、block shape 为非空正整数元组、各 dtype 枚举有效，并要求 tensor storage dtype 与 quant storage dtype 一致；scale shape、对齐、block 可整除性及硬件格式合法性仍应由 schema/variant constraint 校验。

### 2.3 Storage、Value、OutputSpec

`Storage(id, name, spec, kind)` 是内存身份。`Value(id, name, spec, storage_id, producer, output_index, byte_offset, strides, alias_input)` 是 SSA 逻辑值。显式保存 `alias_input` 可以在同一 node 的多个输入恰好共享 storage 时，仍准确知道输出选择了哪个输入的 view 元数据；`Graph.bind()` 会原样保留这个选择。

普通输出的 `OutputSpec.alias_input=None` 会创建新的 `TEMPORARY` storage；`alias_input=i` 会复用第 i 个输入的 storage。`inherit_view=True` 仅允许用于 alias 输出，并继承该输入的显式 stride；默认 `copy_inplace` 使用它保留 view destination。这样能同时满足：

- 数据流仍然是 SSA：原地更新前后的逻辑 value 不同；
- hazard 分析仍然正确：它们具有相同 `storage_id`；
- view 可以有独立 shape/stride/offset，而无需创建或复制 buffer。

`OutputSpec` 本身检查 offset 非负、stride 非负且 rank 相同，以及 `inherit_view` 必须伴随 alias。具体 shape 绑定后，`CommandProgram.validate()` 会根据 value offset/shape/stride/逻辑 element bits 计算最大 span，拒绝超过 base buffer `nbytes` 的 view；Graph 构造阶段尚不能对含 Symbol 的 view 做完整 bounds 证明。

### 2.4 Node、Graph 与 schema

`Node` 字段如下：

| 字段 | 含义 |
|---|---|
| `id/name/op` | 连续 node id、唯一名字和语义 op 名 |
| `inputs/outputs` | value id 元组 |
| `attrs` | schema 合并默认值后的属性 |
| `control_deps` | 必须指向更早 node 的显式控制边 |
| `effects` | `EffectUse(resource_id, mode)` 元组 |
| `metadata_only` | 是否只产生 alias 元数据、无需 runtime task |

`OpSchema.metadata_only` 会传到 `Node.metadata_only`。metadata-only node 的所有输出必须 alias 输入，且不能携带 control/effect；默认 `view` 通过这个通用标志实现，compiler 不再依赖 op 字符串。一个 schema 也可以返回空 `OutputSpec` 元组来表达 side-effect-only op；`GraphBuilder.call()` 此时返回 `()`，但整张图仍必须标记至少一个 graph output（通常是被修改的 state）。

`Graph` 是 append-only 容器，所有 value/storage/node/effect id 均按插入顺序连续。`add_node()` 是事务式追加：归属、control/effect 或 alias 校验失败时会回滚本次新增的 value/storage/node/name。node attrs 在追加时递归转换成只读 mapping、tuple/frozenset，并 deepcopy 其他值，避免调用方后续修改 attrs 破坏编译确定性。

`validate()` 检查至少一个 graph output、输入/输出 id、所有连续 id、value→storage/producer owner、node input/output owner、alias-input/storage 一致、control edge 方向、effect 引用和 metadata-only contract。`freeze()` 在验证后禁止继续添加 input/effect/node/output。`compile_graph()` 总是通过 `bind()` 创建并冻结一个独立快照，所以编译后修改原始 builder graph 不会改变 artifact。具体 view span 在 command 层检查；量化格式和 op-specific 合法性仍主要由 `OpSchema.infer` 和 backend constraint 负责。

## 3. State、effect 与 view

### 3.1 State

`GraphBuilder.state()` 创建 `StorageKind.STATE` 的外部输入。命令中的 buffer flags 会包含 `INPUT | STATE`；如果同一 storage 的某个 value 被标记为 graph output，还会包含 `OUTPUT`。

当前的 `copy_inplace(destination, source)` 是状态更新参考 op。其输出 alias destination，因此写 region 会和此前/以后对同一 state storage 的访问形成正确 hazard。

需要特别注意：`STATE` 只表达“此 binding 由调用方持有并可能被原地修改”。Phase 1 没有 state allocator、request/sequence 生命周期管理、跨 invocation epoch、回滚或并发请求隔离。host 必须绑定正确数组，并自行保证同一 state 不被不安全地并发调用。

### 3.2 Effect

`EffectResource(id, name, lifetime)` 是非张量资源。lifetime 目前可取：

- `invocation`：单次调用；
- `sequence`：生成序列；
- `request`：请求；
- `model`：模型实例。

`EffectUse` 声明 `READ`、`WRITE` 或 `READ_WRITE`。lowering 时，每个 node task 都会附加一个 `ResourceKind.EFFECT` 的 whole-resource `Access`。因此 effect 可以表达 cache metadata、opaque allocator、随机状态等保守顺序，但它不能替代可精确切分的 buffer region；给 tiled op 添加一个 WRITE effect 会串行化该 op 的 tiles。

lifetime 当前只是 Graph 元数据，不会进入 MKCB，也没有 runtime 自动管理。effect 的作用已经在编译期折叠进 predecessor/successor 表。

### 3.3 View

`view` 是 metadata-only op：

- 新 value 继承 base 的 `storage_id`；
- `byte_offset` 在 alias 链上累加，alias op 还可用 `inherit_view` 保留输入 stride；
- 可保存显式 element stride；
- 默认 view schema 设置 `metadata_only=True`；编译器按该标志跳过节点、不选择 kernel、不生成 task，并拒绝其携带 control dependency/effect 或非 alias 输出；
- ABI 通过 `ValueDescriptor(buffer_id, byte_offset, shape, strides)` 传给 runtime。

后端 kernel 必须按 operand 的 value descriptor 取地址和 stride，而不能假定 operand 总从 buffer 起点连续访问。当前 NumPy plugin 的 constraint 只接受 row-major、无量化、offset 为 0、连续且占满 storage 的 FP32 value，因此 view 仅在 metadata-only 输出测试中执行；尚无默认计算 kernel 消费非连续 view。

## 4. Task IR 数据结构

### 4.1 Resource 与 region

`ResourceRef(kind, id)` 给 buffer、effect 和未来 worklist 使用统一命名空间。`ResourceRef.buffer(id)` 是常用构造器。

`Interval(begin, end)` 是半开区间，允许空区间。`Region(axes)` 是矩形逻辑区域。`Region.full(shape)` 生成每一轴的 `[0, extent)`。

`Access` 字段：

| 字段 | 含义 |
|---|---|
| `resource` | 被访问资源 |
| `mode` | READ/WRITE/READ_WRITE/REDUCE |
| `region` | 矩形区域；`None` 表示 whole/unknown |
| `value_id` | 依赖算法以 resource id 为准；但经 `compile_graph()` 的 buffer access 必须填写当前 node input/output value id，非 buffer access 可省略 |
| `reduction_op` | reduce 的语义名，如 `add` |
| `atomic` | 多个同构 reduce 是否可无边并发 |
| `region_is_storage` | region 是否已经是 base storage 坐标；默认 false，表示 value-relative |

`REDUCE` 必须提供非空 `reduction_op`；非 REDUCE access 禁止携带 reduction metadata/atomic flag。region 默认使用 value 逻辑坐标。compiler 检测到 value 不是 offset=0、shape/stride 等于 storage 的 base view 时，会把带 region 的 access 降级成 whole-resource，防止 strided alias 漏 hazard。只有 plugin 已自行把范围正规化到 base storage 坐标时，才可设置 `region_is_storage=True` 保留该 region；不同 rank 的 storage-region 比较仍保守地视为重叠。

### 4.2 OperandBinding 与 Access 不能互相替代

`OperandBinding(value_id, buffer_id, mode)` 定义 kernel 调用时第几个 operand 指向哪个 value/buffer。`Access` 定义依赖分析所需的资源范围。二者通常成对出现，但职责不同：

- operand 没有 region，不能用于精确依赖；
- effect 或 worklist access 可能没有 kernel tensor operand；
- compiler 会双向校验：每个 operand 必须有同 buffer、mode 足够强的 access；每个 buffer access 必须命名当前 node input/output、storage 一致，并有 mode 足够强的 operand。effect/worklist access不要求 tensor operand；每个非空输出仍必须有写 access。

后端必须同时正确声明两者。

### 4.3 TaskDraft、Task 与依赖表

`TaskDraft` 是 lowering 输出，参数还是 Python mapping；`Task` 是参数 schema 编码后的不可变记录，并获得连续 id 与 kernel stable id。`coordinates=(x, y, z)` 是后端定义的逻辑坐标，不等于 CUDA `blockIdx`，三个分量必须是 uint32 整数；`estimated_cost` 必须是有限正数。普通 lowering 原则上必须产生 task；空 draft 集只有在 node 至少有输出、所有输出 numel 为 0、没有 effect，且 variant 显式声明 `empty_is_noop=True` 时才合法。metadata-only node 不调用 lowering。

`TaskDependencies` 同时保存逐 task 的 predecessor 和 successor 元组。所有 id 排序、去重；`edge_count` 是 predecessor 总长度。双向表在 ABI 中均被保留，方便 polling predecessor 或完成后唤醒 successor 两类 runtime 实现。若显式 control dependency 指向 metadata/零元素等 taskless node，compiler 会沿其 control edge 和输入 producer 追溯最近的 taskful ancestors，再建立有效控制边。

### 4.4 ParameterSchema

每个 `KernelVariant` 必须声明自己的 typed parameter schema。支持：

| `ParamType` | 编码 |
|---|---|
| `U32` / `I32` | little-endian 32-bit integer |
| `U64` / `I64` | little-endian 64-bit integer |
| `F32` / `F64` | IEEE little-endian float/double |

`pack()` 要求 mapping 的 key 与 schema 完全相同，缺少或多余字段都会失败。ABI version 变化时，kernel 开发者必须显式升级 `KernelVariant.abi_version`，从而改变 stable kernel id；不能静默重解释旧参数 bytes。

## 5. Guard 与 worklist

### 5.1 Guard：已接入的有界动态谓词

`Guard(scalar, op, value)` 表示一个 runtime scalar 与常量的单次比较，例如：

```python
Guard("compress_ready", GuardOp.NE, 0)
```

scalar 名通过 FNV-1a 得到 32-bit id。编译器会检测同一 program 中 guard scalar 的 id 碰撞。带 guard 的 task 设置 `TaskFlags.HAS_GUARD`，并在该 task 的参数 slice 开头写入固定 16-byte `GUARD_STRUCT = <scalar_id:u32, op:u32, rhs:i64>`，随后才是 kernel 参数。

CPU artifact 同时保留原始 `Task.guard` 和 `runtime_scalars: id → name`。CPU 解释器直接用字符串 map 求值，executor 解包的 `task.params` 不含 guard header；未来设备解释器则应从 command parameter slice 读取 guard header，并把剩余部分传给 kernel。

guard 为 false 时 task 被视为成功跳过，successor 仍可继续。因而 skip path 必须有明确语义：输出应 alias 已有有效数据/已经预初始化，或所有可能读取未写 temporary 的 consumer 也必须被等价条件保护；compiler 当前不会证明这一点。依赖在编译期按“task 可能执行”保守生成，不会因 guard 而删除。当前只支持一个“scalar 与常量比较”，没有布尔组合、两个 scalar 比较或 shape-dependent region。

### 5.2 WorklistSpec：数据模型占位，尚未接线

`WorklistSpec` 为 sparse attention 和 MoE 的 bounded ragged/indirect domain 预留：

| 字段 | 含义 |
|---|---|
| `name` | worklist 名 |
| `indices_buffer` | 间接索引 buffer |
| `count_scalar` | 当前有效项数的 runtime scalar |
| `capacity` | 编译期上界 |
| `index_dtype` | `INT32` 或 `INT64`，默认 `INT32` |
| `indptr_buffer` | 可选分段 offset/CSR indptr |
| `weights_buffer` | 可选 route weight |
| `sentinel` | 无效索引，默认 -1 |
| `logical_index_space` | 逻辑索引空间标签，默认 `flat` |

`ResourceKind.WORKLIST` 也已定义，worklist resource access 已可参与编译期 dependency derivation。但 Phase 1 的 `Graph`、`CompiledArtifact`、`CommandProgram` 和 `CPUInterpreter` 都没有保存或执行 `WorklistSpec`；它尚不能自动展开 task，也没有 ABI descriptor。当前结构只校验名称/count scalar、capacity、buffer id 和 index dtype 等局部不变量。可以手工生成固定容量 task 并用 guard 跳过一部分，但这不等价于完整的 ragged worklist runtime。

## 6. Kernel 注册数据结构

`KernelVariant` 是 op 实现插件：

| 字段 | 作用 |
|---|---|
| `name/op/backend` | 实现名、语义 op 和 backend 名；稳定身份还包含 ABI version |
| `architectures` | 可运行目标集合 |
| `lower` | `LoweringContext → Iterable[TaskDraft]` |
| `parameters` | typed parameter schema |
| `executor` | 当前仅 CPU oracle 使用的回调；GPU binary dispatch 尚未定义 |
| `resources` | threads、dynamic SMEM、register hint、cluster、pipeline、TMEM、warp role、cooperative launch |
| `constraint` | 基于 graph/node/target 的合法性判定 |
| `priority` | variant 选择优先级 |
| `abi_version` | kernel 私有 ABI 版本 |
| `tags` | 自由标签，当前选择器不使用 |
| `empty_is_noop` | lowering 面对全零元素输出时，空 task 集是否语义安全；默认 false |

stable kernel id 是 `FNV1a32(f"{backend}:{name}:v{abi_version}")`。registry 会拒绝 stable-id 冲突和完全重复注册；同一 op 下允许同名但 backend 或 ABI version 不同的 variant，以支持多版本并存。选择器的确定性顺序是 priority 降序、name、backend、ABI version 降序、stable id。manifest 可导出 id、名字、op、backend、ABI version、architecture 与 `empty_is_noop`，但 MKCB 当前不内嵌 manifest，resources 也尚未导出。

`KernelResources` 是声明而非自动执行策略。当前检查 threads 范围、非负 register/TMEM hint、正 pipeline/cluster、warp role 非空且不重叠并位于 CTA warp 范围内；SM90/SM100 特定的 shared-memory/TMEM 上限、cluster launch 合法性和实际 occupancy 仍必须由后续 GPU backend/toolchain 检查。

## 7. MKCB 1.0 命令 ABI

### 7.1 总体原则

- magic：`MKCB`；major/minor：1/0；
- little-endian；所有 section 起始 offset 16-byte 对齐；
- CRC32 覆盖整个 command buffer；计算时只把 header 内的 CRC 字段暂置为 0，因此 header 和 body 的损坏都会被检测；
- 不包含 raw pointer、Python 对象地址或进程内函数指针；
- id 和表索引为 32-bit，shape/stride dimension 为有符号 64-bit；
- record 使用固定大小，变长数据通过 `start + count` 切片引用。

### 7.2 固定 record

| Record | 大小 | 关键内容 |
|---|---:|---|
| Header | 128 B | magic/version、target、总大小、CRC、各 section count/offset、reserved |
| Buffer | 64 B | id、flags、物理 dtype/rank、shape/stride table offset、nbytes、quant id |
| Value | 64 B | id、buffer id、flags、逻辑 dtype/rank、shape/stride offset、quant id、byte offset |
| Task | 64 B | 16 个 u32，见下节 |
| Operand | 16 B | buffer id、value id、access mode、flags |
| Guard prefix | 16 B | scalar id、comparison opcode、i64 rhs |

Task 的 16 个 word 依次为：

```text
task_id, kernel_id, flags, worker_hint,
dependency_start, dependency_count,
successor_start, successor_count,
operand_start, operand_count,
param_offset, param_size,
logical_x, logical_y, logical_z, reserved
```

`worker_hint` 来自 `ExecutionPlan`。`UINT32_MAX` 被保留为“无 hint”的编码；当前正常 `compile_graph()` 要求 `num_workers > 0`，因此会写入合法 worker id。

### 7.3 Buffer 与 value 分离

buffer descriptor 使用 storage 的 `storage_dtype`、`physical_shape`、physical stride 和 `nbytes`；value descriptor 使用 value 的逻辑 `dtype`、逻辑 shape/stride 和 byte offset。量化 packed buffer、alias 和 view 都依赖这一区分。

buffer flags 的生成规则：

- INPUT → `INPUT`；
- CONSTANT → `INPUT | CONSTANT`；
- STATE → `INPUT | STATE`；
- TEMPORARY → `TEMPORARY`；
- 任一 graph output 指向该 storage → 追加 `OUTPUT`；
- physical layout 为 opaque → 追加 `OPAQUE_LAYOUT`。

### 7.4 参数布局

每个 task 开始写参数前，compiler 将共享 params blob 对齐到 16 bytes。若有 guard，先写 guard prefix，再写 variant 的 packed kernel params；`TaskInstruction.param_size` 包括两者。无参数 task 可以有 size 0，但 offset 仍指向合法 params blob 位置。

### 7.5 反序列化验证

`CommandProgram.from_bytes()` 与 `validate()` 当前检查：

- magic、major/minor 兼容性、header/total size 与覆盖全缓冲的 CRC；
- target/dtype/task flag/access mode 枚举值；
- section offset 不早于 header、16-byte 对齐、顺序且不重叠，并恰好解释到 total size；
- section、dimension slice、task 子表和参数 slice 不越界，guard prefix 不截断、comparison opcode 有效且参数 offset 对齐；
- rank 不超过 8，buffer/value/task id 连续，已知 flags 与 task reserved=0；
- buffer physical stride span 不超过 `nbytes`，value view span 不超过所属 buffer；
- dependency、operand、value 引用有效，operand buffer 与 value storage descriptor 一致；
- predecessor/successor 无重复、全部向前拓扑，并且双表描述完全相同的边。

它尚未验证 header/buffer/value record 中所有保留 word 必须为零、operand flags、guard scalar id 是否存在于 invocation binding、kernel 私有 payload、kernel id 存在于本地 binary manifest、参数 size 与 kernel schema 一致、quant id 对应已知 manifest，或 invocation binding 的真实地址/对齐/容量。GPU host runtime 必须补齐这些 trust-boundary 校验。

## 8. 关键不变量清单

开发新 frontend/backend 时应保持：

1. Graph、storage、value、node、task id 连续且确定性生成。
2. value 是 SSA，alias 通过 storage id 表达，不能复用旧 value 冒充新结果。
3. 每个会执行的输出必须有写 access；所有真实读写 region 必须完整声明。
4. 参数只能通过版本化 `ParameterSchema` 编码，不能传 Python tuple/pickle/raw pointer。
5. `worker_hint` 不得成为正确性条件。
6. guard 跳过必须等价于该 task 对后续依赖而言已经完成。
7. `quant_spec_id` 和 kernel stable id 的定义必须在编译端与 runtime/binary manifest 一致。
8. 设备端发布完成前，必须满足 [backend_contract.md](backend_contract.md) 中的 async completion 与 release/acquire/epoch 协议。
