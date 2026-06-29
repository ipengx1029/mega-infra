# 背景
这是一个megakernel的专题repo，目录下我放了3个当前主流的megakernel的项目实现供你参考
- MegaKernel：是通过纯手动编写DAG图、算子依赖关系、手搓megakernel算子来实现将一个llama-1b模型变成一个超级kernel，在batch=1下的decode中取得较好效果
- mirage：mirage这个仓库包含2块，一块是mirage超级优化器用来自动生成kernel，还有一块是mpk，在大模型推理的megakernel中，当前的算子大都是预定好的，mpk通过ai编译器方式自动构图以及依赖关系，然后通过codegen的方式生成一个megakernel
- Triton-distributed：这个repo中有利用triton进行megakernel的实现，主要是利用了当前仓库中开发的triton的dist特性进行多卡的mega

# 讨论
读了这3个megakernel的代码和论文，发现形成一个megakernel当前主要是2部分：
1. 前端对模型组网进行分析，对每个op中的计算切分任务，构造不同任务之间的依赖关系以及设定在计算中如何同步，同时如何将task在计算时分配给每个SM（BLOCK）
2. 用cuda（或其他DSL）编写代码，通过warp-specialization完成计算和加载的overlap
---
我认为，一个兼容性强，设计优秀的前端分割、依赖构造框架，能为后续适配不同模型带来极大方便

# 硬件背景
假设当前是在Hopper（H20） / BlackWell（B300）硬件上实现megakernel

# 需求
总体项目：实现一个megakernel，开发者可以在前端通过python（或者pytorch等等都行）构建模型组网，然后框架自动根据组网的输入输出定义、数据流的DAG分割每个op的任务并且构建task之前的依赖关系，也要形成任务指令可供runtime时kernel解析（让kernel知道当前执行什么指令、取哪儿的数据等等），通过codegen等方式形成kernel（也可以像MegaKernels一样不用codegen，哪种灵活你选择哪种）。后端kernel的编写尽量与前端结耦，后端kernel只要遵循固定的开发规范格式，就可以方便地将开发的kernel注册进来供选择使用。 这样的设计方式可以支持前后端解耦，负责框架的人需要想办法让前端框架更易用地适配模型，更好地划分task使得调度更好；写kernel的人则专心在kernel的优化上。
---
上述是个宏大的目标，我们一步一步来：
1. 假设当前不要求支持多卡
2. 可以分步骤分阶段实现
3. 每一步实现都要有测试和数据结构设计以及如何工作的解释文档
4. 要注意，设计时要考虑，尽量后续放deepseekv4这种复杂的模型容易接入（deepseekv4有复杂的attention）
