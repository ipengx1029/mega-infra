# 路线图：从 Phase 1 到单卡 DeepSeek-V4 megakernel

## 1. Phase 1 已交付的基线

Phase 1 的目标不是追求 GPU 性能，而是固定前后端解耦所需的最小语义与 ABI：

- Python `GraphBuilder` 与可扩展 `OpSchema`；
- value SSA + storage alias 的 Graph IR；
- input/constant/temporary/state、显式 effect 与 control dependency；
- compile-time symbol specialization；
- view、逻辑/物理 dtype/shape、`QuantSpec` 元数据；
- 精确 alias-input/inherit-view 元数据、side-effect-only schema 和事务式 node 追加；
- 通用 metadata-only op contract、显式 `KernelVariant.empty_is_noop` 的零元素 no-work program、穿过 taskless node 的 control-edge 保留，以及冻结的编译快照；
- backend `KernelVariant` 注册、constraint、typed params 和 task lowering；
- operand/access value-storage-mode 双向校验，以及非 base view region 的保守 whole-storage fallback；
- region-aware RAW/WAR/WAW/REDUCE dependency 与 transitive reduction；
- 确定性的 critical-path list scheduler；
- 无 raw pointer、可 round-trip 的 MKCB 1.0 command buffer，包含全缓冲 CRC、section/span/topology/双依赖表校验；
- guard 的编译与 CPU 执行；
- `WorklistSpec`/`ResourceKind.WORKLIST` 的数据模型占位；
- NumPy tiled backend 和 CPU DAG interpreter；
- gated MLP、状态原地更新、view、effect、guard、registry、scheduler 与 ABI 测试。

这套基线优先保证“语义、切分、依赖、命令”可以被独立测试。GPU 后端接入时应复用这些测试，而不是另起一套模型表示。

## 2. 当前限制

### 2.1 Frontend 与 Graph IR

- 当前只有手写 Python builder，没有 `torch.export`/FX importer，也不能自动捕获任意 PyTorch module。
- 默认 op 只有 `add`、`mul`、`silu`、rank-2 `matmul`、rank-2 `rms_norm`、`copy_inplace` 和 `view`。
- 不支持条件/循环/子图、tuple/list 容器语义、可变 arity、复杂 broadcast 或自动 dtype promotion。
- `Symbol` 必须在编译时绑定；运行期 sequence length、expert token count 和 sparse top-k 尚不能直接改变 task domain。
- concrete MKCB 已检查 view span 不超过 buffer，但 Graph 阶段还没有完整的 symbolic bounds 证明；region 对非 base view 只能 whole-storage fallback，尚未把 offset/stride/reshape 自动映射为统一 byte/affine region。
- state lifetime 和 effect lifetime 只是元数据，没有 request/sequence manager。

### 2.2 Task、依赖与调度

- 一个 node 只选择一个 variant，尚无跨 node fusion、复合 kernel pattern、producer/consumer 联合切 tile 或 cost-based re-partition。
- whole-resource 访问已有 last-writer/readers frontier，长写链保持线性边数；带静态矩形 region 的资源仍需扫描相关历史并做传递边化简，大 task 数下需要区间索引/bitset 等进一步优化。
- 同 rank 的复杂 strided alias 也没有严格 byte overlap 证明，后端必须保守申报。
- atomic reduce 的并行许可依赖 plugin 正确填写 `atomic/reduction_op`，没有验证实际 kernel 原子语义。
- scheduler 只使用人工 `estimated_cost`，不了解 SMEM/register/TMEM/cluster occupancy、HBM/L2 locality、kernel 切换代价或实测时间。
- `worker_hint` 只是提示；没有设备端 ready queue、工作窃取、背压或公平性实现。

### 2.3 ABI 与 runtime

- 只有 CPU interpreter；SM90/SM100 没有 host launcher、persistent megakernel、device command interpreter 或 binary dispatch table。
- MKCB 不携带真实地址是刻意设计，但当前也没有正式的 buffer binding ABI、temporary arena planner 或 alignment contract。
- worklist 没有 command descriptor；effect/lifetime/region 不序列化，因为当前只在编译时折叠为依赖。
- command parser 已检查 CRC、section 顺序/重叠、view span、拓扑和 predecessor/successor 镜像，但尚未校验所有 reserved/operand flags、kernel/quant manifest、私有参数 schema 和 invocation 真实 binding。
- 没有跨 CTA release/acquire completion protocol、invocation epoch 或 state 并发隔离。
- 当前 CPU dtype 映射包含 BOOL/整数/UINT8/FP16/FP32，但不含 BF16/FP8/FP4；默认计算 plugin 仅支持 C-contiguous binding 上的 row-major、无量化 FP32。

### 2.4 模型与系统范围

- 未实现 attention、KV cache paging、RoPE、softmax、top-k、gather/scatter、MoE grouped GEMM 或 sampling。
- 未实现预填充/解码的不同 specialization。
- 当前项目范围明确为单 GPU；tensor/expert/pipeline parallel 和通信不进入近期 runtime。
- 未做 autotune、profile-guided cost model、code cache、模型级内存规划和真实性能报告。

## 3. DeepSeek-V4 对 IR 的压力

仓库中的 `deepseekv4/model.py` 和 `config.json` 展示了比普通 dense Transformer 更复杂的单卡语义：

- 43 层模型，attention 层按配置采用纯 sliding window 或不同 compression ratio；
- 低秩 Q 投影、MLA 风格 head、RoPE/non-RoPE 分量和 grouped low-rank output projection；
- Compressor 维护 `kv_state`、`score_state` 和 compressed KV cache，decode 时按位置周期性地产生压缩项；ratio=4 还有 overlap window；
- Indexer 维护独立 compressed cache，通过 scoring/top-k 选择 sparse KV position；
- sparse attention 同时消费 sliding-window index 与 compressed top-k index，并允许 sentinel `-1`；
- MoE 包含 hash routing 与 score/top-k routing、256 routed experts、每 token 激活 6 个 expert及 shared expert；
- 权重/activation 涉及 FP8、FP4 与 block scale；
- Hyper-Connections 使用多份 hidden state、Sinkhorn mixing、pre/post/comb 分支；
- 还有 MTP block 和共享 embedding/head。

这些结构可映射到当前抽象，但不能靠添加几个固定 op 就一次完成：

| DeepSeek-V4 语义 | 目标抽象 |
|---|---|
| KV cache、compressor `kv_state/score_state` | `STATE` storage + position region + request/sequence lifetime |
| cache metadata/epoch/allocator | `EffectResource`，必要时拆成可精确访问的 metadata buffer |
| metadata reshape/slice/ring-buffer view | `Value`/`Storage` alias + affine/byte region |
| FP8/FP4 data 与 scale | `TensorSpec` logical/physical split + `QuantSpec` + 显式 scale `Value` |
| `should_compress` 等周期条件 | compile specialization + runtime scalar `Guard` |
| index top-k、sparse KV 列表 | `WorklistSpec(indices/count/indptr/sentinel)` |
| MoE token→expert route/weight | 分段 worklist + count/scan/dispatch + grouped GEMM |
| HC/Sinkhorn 与低秩投影 | 普通 dense/reduction op + 可融合 composite pattern |

关键原则是先保证这些语义在 CPU oracle 上可验证，再选择 SM90/SM100 kernel。不能为了一个快速 sparse attention kernel，把 cache 更新、top-k sentinel 或量化 scale 语义藏进不可见的 backend side effect。

## 4. 分阶段实施计划

每个阶段都必须同时交付：代码、数据结构说明、工作原理文档、单元/端到端测试和可复现的验证命令。涉及 GPU 性能的阶段还必须保存硬件、shape、dtype、模式和 profiler 数据。

### Phase 2：完善 frontend、affine alias 与 host binding 契约

目标：让后续动态 attention 不建立在含糊的 alias 与 runtime 边界上。

实现：

- 在现有 concrete span check 和 whole-storage fallback 之上，增加 symbolic view bounds 与 affine/byte-region 正规化，正确处理 slice/reshape/transpose/ring-buffer wrap；
- 增加 temporary liveness、alignment 和 arena plan；
- 定义独立 invocation binding ABI 与 kernel manifest/schema 校验；
- 继续强化 MKCB parser：所有 reserved/flags、kernel/quant manifest、整数 overflow 和 fuzz；
- 明确 state/effect lifetime owner 和 invocation/request id；
- 为依赖构造建立区间索引，避免大 task 数时无界的历史扫描；
- 增加最小 `torch.export`/FX importer，将已支持的 dense op、参数、alias 和显式 state 映射到 Graph，并对不支持节点给出可定位诊断；保留 GraphBuilder 作为 escape hatch；
- 扩充 op schema：broadcast elementwise、cast、reduce、softmax、concat/split、slice、transpose、RoPE、gather/scatter。

验收：

- 随机 strided/alias graph 的 hazard 与串行 reference 一致；
- command fuzz 不崩溃，现有 section/span/edge 校验保持覆盖，并拒绝新增 manifest/flags/overflow 不一致；
- arena 中不同 live range 不覆盖，同 storage alias 保持身份；
- 文档给出 ABI compatibility 与 migration 规则。

### Phase 3：SM90/H20 最小可执行 megakernel runtime

目标：在单个持久化 CUDA launch 内执行 Phase 1 的基础 op DAG。

实现：

- host launcher、device-visible command/binding table、kernel stable-id dispatch；
- 定义统一 CTA resource class，并在编译期拒绝或切分 blockDim/SMEM/cluster/cooperative-launch 包络不兼容的 variant；
- resident ready queue/工作窃取，`worker_hint` 仅作 locality hint；
- device-scope release/acquire + 64-bit invocation epoch reference protocol；
- guarded task 的设备端解析和 skip-complete；
- SM90 contiguous BF16/FP16 的 elementwise、RMSNorm、GEMM plugin；
- TMA/mbarrier 与 WGMMA 的 architecture-local warp specialization；
- 资源/occupancy 检查，避免 persistent CTA starvation。

验收：

- gated MLP 和 state update 在 H20 上与 CPU oracle 一致；
- 随机 ready 顺序、不同 worker 数、连续多 epoch 和并发独立 state 均正确；
- compute-sanitizer memcheck/racecheck 通过；
- nsys 证明单个 megakernel launch，ncu 建立各 op baseline；
- release/acquire/epoch 协议有专门 litmus tests。

### Phase 4：SM100/B300 backend 与量化路径

目标：保持同一 Task/ABI，增加 Blackwell-native variant。

实现：

- tcgen05/TMEM GEMM、TMEM alloc/dealloc 与 epilogue handoff；
- SM100 TMA/mbarrier pipeline、必要的 proxy/tcgen05 fence；
- 1-CTA 基线后再增加 `cta_group::2`/2-SM cooperative variant；
- 对同构 tile domain 评估 CLC，仍保留通用 Task DAG ready 语义；
- FP8 与 NVFP4 packed buffer、block-scale manifest、scale operand 和 FP32 accumulate；
- 按 target/dtype/shape/resources 的 variant selection 与 autotune cache 雏形。

验收：

- 同一 graph 无前端修改即可分别编译到 SM90/SM100；
- FP8/FP4 与高精 reference 的误差阈值明确；
- TMEM 生命周期、cluster 尾部和量化 scale 尾块测试通过；
- ncu 验证 tcgen05/TMEM/TMA 使用，性能报告不混用 B200/B300 或不同 shape 数据。

### Phase 5：真正的动态 worklist runtime

目标：让 ragged/sparse work 由数据驱动，而非为 capacity 全量静态展开。

实现：

- 将 `WorklistSpec` 接入 artifact 和新版本 command ABI；
- 显式 indices/count/capacity/indptr/weights/sentinel descriptor；
- worklist producer、count/scan、发布 epoch 和 consumer acquire；
- bounded overflow policy：报错、spill 或 fallback，不能静默截断；
- 动态 task expansion 或 persistent indirect loop，并保留确定性 CPU oracle；
- guard 支持组合 predicate 或由小型 predicate bytecode替代，同时保持可验证上界。

验收：

- 0、1、capacity、overflow、重复 index、sentinel 和极端倾斜分布；
- worklist producer/consumer 在不同 CTA/SM 上的 race tests；
- 同一输入在 CPU/GPU 得到相同有效项和归并结果；
- command ABI major/minor 兼容策略落文档。

### Phase 6：单卡 MoE

目标：先完成单卡 routing 与 grouped experts，不引入 expert parallel 通信。

实现：

- score/hash gate、top-k、weight gather/normalize；
- token count、prefix sum、dispatch worklist、inverse gather；
- shared expert 与 routed expert 并行；
- grouped GEMM / grouped gated-dual-GEMM plugin；
- SM100 FP4 expert weight和 block scale，SM90 fallback；
- load imbalance-aware scheduling、capacity/overflow 和 reduction correctness。

验收：

- 0-token expert、全部 token 命中一个 expert、重复 route、hash layer和 score layer；
- scatter/reduce 无丢失、无重复，和 eager reference 一致；
- 256 expert/每 token top-6 的配置级压力测试；
- 单卡边界明确忽略 `world_size > 1` 的 all-reduce/分片路径。

### Phase 7：DeepSeek-V4 sparse/compressed attention

目标：按可验证的小步实现本地模型中的 attention，而不是一开始手写整层超级 kernel。

建议子阶段：

1. dense/sliding-window attention：Q/KV projection、norm、RoPE、ring KV state、masked softmax；
2. compressor：prefill 分组与 decode remainder state，先 ratio=128，再做 ratio=4 overlap；
3. deterministic compressed index：静态 `get_compress_topk_idxs`；
4. learned Indexer：独立 compressed KV、score、top-k/sentinel worklist；
5. sparse attention：window + compressed index 合并、sink 和 grouped output projection；
6. prefill/decode 分离 specialization 与跨层 cache ownership。

每一步都用 state region 表达准确位置：ring-buffer wrap 应拆成两个 region，而不是谎报一个连续区间；无法证明时使用 whole storage access保证正确。`should_compress` 可先用 runtime scalar guard，随后由 worklist producer直接发布有效压缩项。

验收：

- start_pos=0 prefill、单 token decode、window wrap、compression boundary 前后；
- ratio 0/4/128、overlap remainder、top-k 少于上限和 sentinel；
- KV/cache/state 多序列隔离与跨 invocation epoch；
- BF16/FP8/FP4 各路径的数值误差和 cache 一致性；
- 与 `deepseekv4/model.py` 的中间张量逐层比对。

### Phase 8：完整 DeepSeek-V4 block、MTP 与全图优化

目标：完成单卡模型语义后，再进行跨 op/cross-layer fusion。

实现：

- Hyper-Connections 的 hc_pre/Sinkhorn/hc_post；
- attention + HC、MoE + HC 的复合 pattern；
- MTP block、共享 embed/head 与 logits；
- 扩展 Phase 2 importer 或增加显式模块 adapter，覆盖完整模型的 stateful/dynamic pattern，避免人工逐 node 建图；
- pattern-based fusion、producer/consumer tile co-design、memory reuse；
- profile-guided cost model、variant autotune、command/code cache；
- 以 decode batch=1 为首要场景，同时保留小 batch 与 prefill correctness。

验收：

- 单层、数层和 43 层端到端 logits/reference；
- state/cache 长序列回归与 MTP 分支；
- 每次 fusion 都有 unfused oracle 和独立回退路径；
- H20/B300 分别给出 latency、tokens/s、HBM、occupancy 和对照实现；
- 只有在正确性、race 和 ABI 测试通过后才接受性能优化。

## 5. 推荐的实现顺序与停止条件

近期最有价值的顺序是：Phase 2 frontend/host contract → Phase 3 SM90 reference runtime → Phase 4 SM100/quant → Phase 5 worklist。不要直接从 Phase 1 跳到完整 sparse attention：没有 epoch 的 state、没有 acquire 的 worklist和不准确的 alias region，即使短测试偶尔得到正确结果，也无法成为可扩展框架。

每个阶段满足以下条件才能进入下一阶段：

- public data structure 与 ABI 变化已文档化；
- CPU oracle 或明确的高精 reference 覆盖新语义；
- 所有新增结构有 round-trip、边界和失败测试；
- GPU 阶段通过 memory/race sanitizer；
- 性能数据可复现且不牺牲 fallback correctness；
- 已知限制更新在本文件，而不是只留在 issue 或代码注释中。
