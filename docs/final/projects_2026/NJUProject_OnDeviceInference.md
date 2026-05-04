# 端侧大模型推理框架

## 一、任务说明

### 1.1 背景

近年来，端侧（on-device）大模型推理成为一个重要方向：以 llama.cpp、Ollama、LM Studio、MLC-LLM 为代表的项目，让用户能够在自己的消费级硬件（如笔记本、游戏显卡）上运行原本需要数据中心 GPU 才能跑起来的大模型。与云端部署相比，端侧推理有更好的隐私性、离线可用性和零边际成本，但也面临一个核心矛盾——**消费级显卡显存有限（通常 8~16 GB），而用户希望运行的模型参数规模往往在 7B~14B 甚至更大**，仅权重就放不进显存。

解决这一矛盾的主流思路是 **权重 offloading（卸载）**：模型权重常驻 CPU 内存（或磁盘），在推理时按需分批次加载到 GPU 显存中参与计算，计算完成后释放或缓存。由于 Transformer 具有良好的分层结构，每一层的权重可以被独立加载、独立计算，因此非常适合进行"逐层 offloading"。进一步，通过 CUDA 多流（stream）机制，可以把"下一层权重的搬运"与"当前层的计算"重叠起来，从而把 PCIe 传输的开销尽量隐藏在计算之后。

本项目希望你从零实现一个最小但可用的端侧大模型推理框架，在 NVIDIA RTX 3070（8 GB 显存）这样的消费级显卡上跑通原本装不下的模型（Qwen3-8B），并与业界主流推理框架（vLLM / SGLang）进行性能对比。

### 1.2 端侧推理与 Offloading

一个典型的 Offloading 推理系统至少需要考虑以下几件事：

1. **权重驻留位置**：权重完整保存在 CPU pinned memory 中（这是 H2D 传输的前置条件）；GPU 上只保留若干"正在被使用"或"即将被使用"的层。
2. **加载调度**：在进入第 i 层计算之前，该层权重必须已经在 GPU 上；计算完成后，该层权重可以被释放或保留在 GPU 上作为缓存。
3. **计算-传输重叠**：使用独立于默认计算流的 CUDA stream 来做 H2D 拷贝，使得第 i+1 层的权重搬运可以和第 i 层的计算并行进行。
4. **KV Cache 管理**：与训练不同，推理时每一步 decode 都会向 KV Cache 追加新的 token，KV Cache 的体积会随着生成长度线性增长，需要在 offloading 策略中一并考虑。

本项目将以此为主线，引导你从一个 "只能跑小模型的最小框架" 出发，逐步扩展到 "能在 8 GB 显存上跑通 Qwen3-8B 的完整 offloading 推理系统"。

---

## 二、方法设计规范

### 2.1 最小推理框架实现（必选）

你需要基于 **PyTorch** 从零实现一个最小的大模型推理框架，并在一个**单卡可装下**的模型上跑通。

#### 硬性约束

- **仅允许使用 PyTorch 生态**，包括：
  - `torch`、`torch.nn`（仅限 `Linear`、`Embedding`、`ModuleList` 等基础模块）
  - `torch.nn.functional`（允许使用 `scaled_dot_product_attention`、`silu`、`rms_norm` 等）
  - `safetensors` / `torch` 原生权重加载接口
- **禁止使用 `transformers` 库中任何模型相关的 `nn.Module`**，特别是 `Qwen3ForCausalLM`、`Qwen3Model`、`Qwen3DecoderLayer`、`Qwen3Attention` 等；`transformers` 仅可用于以下三个目的：
  1. 下载模型权重文件（`snapshot_download` / `AutoModel.from_pretrained` 下载后仅使用其 `.safetensors` 文件）
  2. 加载 tokenizer（`AutoTokenizer`）
  3. 作为正确性对比的**参考实现**（不得作为最终产物的一部分）
- **禁止直接调用 `model.generate()`**；推理主循环必须自己实现。

#### 你至少需要自己实现的组件

- RMSNorm
- Rotary Position Embedding（RoPE）的构造与应用
- QKV 投影与 Grouped-Query Attention（GQA）
  - 提示：Qwen3 使用 GQA，Q 头数与 KV 头数不同，可以使用 `F.scaled_dot_product_attention(..., enable_gqa=True)`（PyTorch ≥ 2.5），也可以手动 `repeat_interleave` 扩展 KV 头
- SwiGLU 结构的 MLP（`gate_proj` / `up_proj` / `down_proj` + SiLU）
- Decoder Layer 的组装（含残差连接与两处 LayerNorm）
- 权重加载：从 HuggingFace 的 `.safetensors` 文件读出权重，映射到你自己的模块参数
- KV Cache 的数据结构与拼接/追加逻辑
- 完整的推理主循环：prefill 阶段 + decode 阶段

#### 本阶段实验要求

- **目标模型**：Qwen3-1.7B（FP16 / BF16 精度）
- **正确性验证**：给定相同的 prompt 和采样参数（greedy，`do_sample=False`），你的实现生成的前若干 token 必须与 HuggingFace `transformers` 的参考实现**逐 token 对齐**
- **性能对比**：在相同 prompt 集上，对比
  - 你自己的实现
  - **vLLM 或 SGLang 中任选一个**工业级框架

  两者的 **prefill 延迟**、**decode tokens/s**、**TTFT（Time To First Token）**；并在报告中分析性能差距的可能来源（例如 kernel 融合、PagedAttention、Python 侧调度开销等，不要求自己补齐，只要求分析）

### 2.2 Offloading 推理（必选，本项目核心）

在 2.1 基础上扩展你的框架，使其能够在 **8 GB 显存的 RTX 3070** 上跑通**单卡装不下**的模型。

#### 目标

- **目标模型**：Qwen3-8B（FP16，权重约 16 GB，3070 无法整体装入显存）
- **必须实现**：逐层权重 offloading + 基于独立 CUDA stream 的权重预取（prefetch）

#### 实现要求

1. **权重驻留**：所有层权重在启动时加载到 **CPU pinned memory**（`torch.Tensor.pin_memory()`），这是高效 H2D 拷贝的前置条件；embedding 和 lm_head 可以常驻 GPU，也可以参与 offloading，由你自行决定并在报告中说明。
2. **显存预算**：暴露一个命令行参数 `--gpu-layer-budget N`（或等价的显存预算参数），表示 GPU 上最多保留 N 层权重。预算 ≥ 总层数时退化为"全部常驻 GPU"；预算越小，offloading 越激进。
3. **三种实现**（必须在同一代码仓库中都能跑）：
   - **(a) 朴素同步 offload**：进入第 i 层时同步从 CPU 拷贝权重到 GPU，计算完释放；没有预取，没有多流
   - **(b) Prefetch 优化**：使用独立于默认计算流的 CUDA stream，在第 i 层计算的同时，异步把第 i+1 层权重从 CPU 预取到 GPU；注意流间同步（`stream.wait_event()` / `torch.cuda.Event`），确保第 i+1 层计算开始前其权重已经就绪
   - **(c) 工业框架对照**：使用 vLLM 的 `--cpu-offload-gb` 参数（或 SGLang 对应功能）在同一模型上跑，作为参照

#### 本阶段实验要求

- 在 Qwen3-8B 上跑通上述三种实现，固定 prompt 集、固定 `max_new_tokens=256`、固定 greedy 采样
- 报告 **prefill latency**、**decode tokens/s**、**TTFT**、**峰值显存占用** 四项指标
- **显存预算扫描**：固定 prefetch 实现，扫描 `--gpu-layer-budget` ∈ {1, 2, 4, 8, 16, 32}（或你自选的合理集合），绘制"显存预算 vs decode tokens/s"曲线，分析拐点

### 2.3 可选优化（选做，可叠加加分）

在完成 2.1 和 2.2 的基础上，以下选做项可任选若干完成：

#### 选做一：更大的 Dense 模型

- **目标模型**：Qwen3-14B（FP16，权重约 28 GB）
- 直接复用 2.2 的 offloading 框架跑通
- 报告显存预算、tokens/s 与 8B 的对比，分析"单卡显存不变、模型变大"对各项指标的影响

#### 选做二：权重量化以减小传输量

- 使用 INT8 或 INT4 量化权重（允许调用 `bitsandbytes`、GPTQ、AWQ 等已有工具，不要求自己实现量化算法）
- 核心观察点：**量化后 H2D 传输量下降，offloading 的 decode tokens/s 是否改善？** 传输时间与计算时间的比值如何变化？
- 在报告中分析量化对端侧 offloading 场景的价值

#### 选做三：KV Cache Offloading

- 面向长 context 场景（建议输入 prompt ≥ 4K tokens，生成 ≥ 1K tokens）
- 把不活跃的 KV Cache 块换出到 CPU，需要时再换回
- 设计驱逐策略（LRU / FIFO / 按 layer / 按 token 段 等）
- 报告长 context 下的 decode tokens/s 与峰值显存

#### 选做四：MoE 模型 Offloading（高阶挑战）

- **目标模型**：Qwen3-30B-A3B（MoE，128 个 expert / 每 token 激活 8 个）
- 与 Dense 模型的 offloading 不同，MoE 的 offloading 调度单位是 **expert** 而非 **layer**：每个 token 只激活少量 expert，应当只把被激活的 expert 加载到 GPU
- 至少实现：
  - Expert 粒度的 on-demand 加载
  - 一种 expert 缓存策略（例如 LRU / LFU / 按历史激活频次）
- 报告：不同 batch size、不同 expert 缓存容量下的 decode tokens/s
- **说明**：本选做工作量接近 2.2 本身，但能获得相应较高的加分

---

## 三、实验要求

### 3.1 评估指标

| 指标 | 说明 |
|---|---|
| **Prefill Latency** | 从输入 prompt 到生成第一个 token 的时间 |
| **TTFT** | Time To First Token，与 Prefill Latency 定义一致 |
| **Decode Tokens/s** | 生成阶段的稳态吞吐量，(生成 token 数 − 1) / (总生成时间 − TTFT) |
| **Speedup** | 对照 baseline（朴素同步 offload）的 decode tokens/s 加速比 |
| **峰值显存** | `torch.cuda.max_memory_allocated()`，单位 GB |

### 3.2 必做实验

1. **正确性验证（2.1）**：自己实现的 Qwen3-1.7B 与 HuggingFace 参考实现在 greedy 下逐 token 对齐
2. **Baseline 对比（2.1）**：Qwen3-1.7B 上，自己的实现 vs vLLM/SGLang 的 prefill latency、decode tokens/s、TTFT 对比，并分析差距来源
3. **Offloading 三方对比（2.2）**：Qwen3-8B 上，朴素 offload / prefetch / vLLM-offload 三种实现的指标对比
4. **显存预算扫描（2.2）**：Qwen3-8B 上，在 prefetch 实现下扫描 GPU 层数预算，画出 tokens/s 曲线

### 3.3 实验配置统一要求

为保证结果可比，所有实验必须：

- 使用相同的 prompt 集（建议从 MT-Bench、Alpaca-Eval 等公开集合中选 10~20 条长短混合的 prompt）
- 使用相同的 `max_new_tokens`（建议 256）
- 使用相同的采样设置（建议 greedy，`do_sample=False`）
- 使用相同的硬件（单张 RTX 3070 8GB）与软件环境（固定 PyTorch / CUDA 版本）
- 使用一致的计时方式（GPU 计时请使用 `torch.cuda.synchronize()` 或 `torch.cuda.Event`）

---

## 四、提交内容

### 4.1 代码

- 代码结构不做严格要求，但建议包含以下模块：
  - 模型实现（Qwen3 架构的 PyTorch 重写）
  - 权重加载
  - Offloading 调度器（朴素版 + prefetch 版）
  - 推理主循环（prefill + decode）
  - 评测脚本
- 要给出能够运行的 python 环境（`requirements.txt`）
- 要给出能成功运行各实验的脚本（例如 `run_baseline.sh`、`run_offload_naive.sh`、`run_offload_prefetch.sh`、`run_vllm_compare.sh` 等）

### 4.2 报告

报告至少应包含：

- **实现方法**：详细描述你的推理框架架构、offloading 调度策略、CUDA stream 的使用方式，以及遇到的关键设计决策
- **实验结果**：
  - Qwen3-1.7B 正确性验证结果
  - Qwen3-1.7B 与 vLLM/SGLang 的性能对比表
  - Qwen3-8B 三种 offloading 实现的性能对比表
  - 显存预算扫描曲线
  - （选做部分的相应结果）
- **结果分析**：
  - 自己的实现与工业框架的差距来源分析
  - Prefetch 相对朴素实现的加速来源分析（建议结合 Nsight Systems 或 `torch.profiler` 的 timeline 截图）
  - 显存预算扫描曲线的拐点解释
  - 实验过程中遇到的问题与解决过程

---

## 五、注意事项

**参考文献**：如果你在实验和报告中参考了已发表的文献、博客、开源实现或官方文档，请在报告末尾列出。

**硬件说明**：本题以 RTX 3070 (8GB) 为目标硬件，每个小组均配备一张 RTX 3070，请以该硬件为基准完成所有必做实验。

**关于 "自己实现" 的边界**：本项目的教学目标是让你理解大模型推理的完整流程与 offloading 系统的设计，因此 2.1 中对 `transformers` 的 `nn.Module` 有严格禁用。阅读 `transformers` 的源码作为学习参考是鼓励的，但**最终提交的代码中不得 import 任何 `transformers` 中的模型 `nn.Module`**。

如有疑问，请联系 fuliangliu@smail.nju.edu.cn。

### 参考资料

- [1] Qwen3 Technical Report. https://qwenlm.github.io/
- [2] vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention. https://github.com/vllm-project/vllm
- [3] SGLang. https://github.com/sgl-project/sglang
- [4] llama.cpp. https://github.com/ggerganov/llama.cpp
- [5] Eliseev, A., & Mazur, D. (2023). Fast Inference of Mixture-of-Experts Language Models with Offloading. https://github.com/dvmazur/mixtral-offloading
- [6] Kamahori, K., et al. (2025). Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models. ICLR 2025. https://github.com/efeslab/fiddler
- [7] Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need.
- [8] Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
