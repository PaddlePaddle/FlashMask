<div align="center">

# ⚡ FlashMask

**灵活且高效的列式稀疏掩码注意力**

[![Paper](https://img.shields.io/badge/FlashMask-ICLR%202025-b31b1b.svg)](https://arxiv.org/abs/2410.01359)
[![Paper](https://img.shields.io/badge/FlashMask--3-NeurIPS%202026-b31b1b.svg)](https://arxiv.org/abs/2610.17901)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-supported-orange.svg)](https://www.paddlepaddle.org.cn/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c.svg)](https://pytorch.org/)

[English](README.md) | 简体中文

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ba2efae-48ea-4981-82f4-e1c3120c34f8" width="85%" alt="FlashMask 总览"/>
</p>
<p><i>图 1：FlashMask 总览。(a) 支持的掩码类型；(b) 列式稀疏表示；(c) 高效实现</i></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f623d2aa-b947-496b-95bc-d7e7fad92579" width="85%" alt="FlashMask-3 执行栈总览"/>
</p>
<p><i>图 2：FlashMask-3 在执行栈四个层次上的总览</i></p>

</div>


## 📋 目录

- [最新动态](#-最新动态)
- [概览](#-概览)
- [核心特性](#-核心特性)
- [性能](#-性能)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [引用](#-引用)
- [致谢](#-致谢)
- [许可证](#-许可证)

---

## 📢 最新动态
- **2026-09-25 FlashMask-4**：支持 Hopper / Blackwell 架构；基于 CuTe DSL 运行时 JIT 编译，编译速度快；在全部支持的 headdim 下提供灵活掩码；完整支持 sink bias 与确定性计算；支持大 headdim；同时提供 PaddlePaddle 与 PyTorch 接口。
- **2026-09-25 FlashMask-3**：支持 Hopper 架构，提出掩码感知的计算通信 overlap 及负载均衡的分布式计算。
- **2026-09-25 FlashMask**：支持 Ampere 架构，提出列式稀疏掩码表示，将掩码存储复杂度从 $O(N^2)$ 降至 $O(N)$，并利用掩码稀疏性跳过无效计算。

---

## 🎯 概览

FlashMask 是一个面向灵活稀疏掩码的**高效注意力内核库**。它采用列式稀疏表示，将二维稠密掩码矩阵编码为一维的列索引区间，并利用掩码稀疏性跳过无效计算。

本仓库从 [PaddlePaddle/flash-attention](https://github.com/PaddlePaddle/flash-attention) 演进而来，现作为 FlashMask 独立仓库维护。本仓库基于 **CuTe DSL** 实现了面向 **Hopper（SM90）与 Blackwell（SM100）** 的前向和反向 Kernel，同时提供 **PaddlePaddle 与 PyTorch** 接口。核心代码基于 CuTe DSL 运行时 JIT 编译，编译速度快。除灵活掩码外，还支持掩码感知的计算通信 overlap 及负载均衡的分布式计算，并在 SM100 上新增大 headdim 支持。

### 核心公式

$$
O = \mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d}} + M\right)V
$$

其中：

- $Q,\ K,\ V$ 为注意力模块的输入张量；
- $d$ 为 $Q$ 最后一维的大小（headdim）；
- $M$ 为 FlashMask 引入的列式稀疏掩码。

---

## ✨ 核心特性

FlashMask-4 提供三项能力：
- 低显存、灵活、高效的掩码 Kernel 实现；
- 掩码感知的计算通信 overlap 及负载均衡的分布式计算；
- 大 headdim 支持；

三者由同一个 `flashmask` 包提供，核心代码基于 CuTe DSL 运行时 JIT 编译，编译速度快，**PaddlePaddle 与 PyTorch 前端共享同一套内核实现**。


### 低显存、灵活、高效的掩码 Kernel 实现

下表对比 **FlashMask-4** 与 **FlashAttention-4 的 `mask_mod` 场景**，并非 FlashAttention-4 全部接口的能力。`d` 为 Q/K 的 headdim，`dv` 为 V 的 headdim。

| d | dv | FlashMask SM90 | FlashMask SM100 | FlashAttention `mask_mod` SM90 | FlashAttention `mask_mod` SM100 |
|---:|---:|:---:|:---:|:---:|:---:|
| 32 | 32 | ✅ | ✅ | ✅ | ✅ |
| 64 | 64 | ✅ | ✅ | ✅ | ✅ |
| 80 | 80 | ✅ | ✅ | ✅ | ✅ |
| 128 | 128 | ✅ | ✅ | ✅ | ✅ |
| **192** | **128** | ✅ | ✅ | ⚠️ | ✅ |
| **256** | **256** | ✅ | ✅ | ✅ | ❌ |
| **512** | **512** | ❌ | ✅ | ❌ | ❌ |
| **576** | **512** | ❌ | ✅ | ❌ | ❌ |

> ✅ 支持，⚠️ 部分支持，❌ 不支持。FlashAttention SM90 在 `d != dv` 时反向不支持 GQA/MQA（`qhead_per_kvhead > 1`），因此 `192/128` 仅支持 MHA；FlashMask SM90 在该配置下支持 GQA/MQA。
>
> 表中 FlashAttention 两列基于 commit `edb5c7` 于 2026-09-23 测试得到，与下文性能测试的 baseline 一致。

**FlashMask-4 的优势不只是支持更大的 headdim，而是在这些维度下仍保留灵活掩码、sink bias 与确定性计算的完整能力：**

- **线性掩码存储**：用少量边界向量描述掩码，存储复杂度从 $O(N^2)$ 降至 $O(N)$。
- **稀疏计算**：将计算块分为全掩码、部分掩码与无掩码三类，跳过全掩码块，仅对部分掩码块执行逐元素掩码。
- **灵活掩码广泛支持**：SM90 上 `192/128` 支持 GQA/MQA；SM100 上 `256/256`、`512/512`、`576/512` 均可使用自定义掩码。FlashAttention-4 `mask_mod` 在这些场景下不可用。
- **配套能力完整**：上表所有配置均支持 sink bias 与确定性计算（deterministic）。

内核位于 [`flashmask/cute`](flashmask/cute/)，用 CuTe DSL 分别实现 SM90 与 SM100 的前向、反向，并按 GPU 与输入配置在运行时 JIT 编译。

### 计算通信 overlap 及负载均衡的分布式计算

FlashMask-4 将掩码信息用于上下文并行（CP）的计算分配与通信调度：**CPB（context-parallel balance）减少各卡之间的负载不均，Overlap 减少计算过程中的通信等待**。两者可独立使用，也可组合使用。

#### CPB：掩码感知的负载均衡（正在开源中）

在稀疏掩码下，长度相同的序列块可能对应差异很大的计算量。CPB 按实际有效的注意力计算量分配任务，通过 **HCC-IPO**（Hierarchical Compute-Communication Iterative Pairwise Optimal）优化负载，减轻最慢 rank 对整体吞吐的拖累；同时以**无需额外掩码通信的本地化变换**构造各 rank 的局部掩码。

实现位于 [`flashmask/cp_balance`](flashmask/cp_balance/)，**支持 SM90 与 SM100**，不依赖 Overlap。

#### Overlap：掩码感知的通信计算重叠

利用掩码识别无需传输的 KV 块以减少无效通信，并将 KV all-gather、注意力计算与梯度 reduce-scatter 组织为前向、反向流水，尽可能隐藏通信开销。

- **机内 / 机间分层通信**：区分机内（NVLink）与机间（RoCE / IB）链路，按拓扑分级调度 KV 传输，进一步降低跨节点的暴露延迟。
- **与 DeepEP 兼容**：通信实现与 DeepEP 互不冲突，可在同一训练进程中同时使用。

实现位于 [`flashmask/overlap`](flashmask/overlap/)，基于 **NCCL GIN** 通信并与 CuTe DSL 内核集成，**目前仅支持 SM100**。

<details>
<summary>在 SM90（Hopper）上使用 Overlap</summary>

本仓库的 CuTe DSL 版 Overlap 仅面向 SM100。如需在 SM90 上使用，可改用 C++/CUDA 实现的 FlashMask-3：基于 PaddlePaddle flash-attention 的 [PR #166](https://github.com/PaddlePaddle/flash-attention/pull/166)，配合新版 PaddlePaddle，调用 `_C_ops.flashmask_attention_v2` 相关 API。

</details>

### 大 headdim

FlashMask-4 在 SM100 上将**灵活掩码、sink bias 与确定性计算扩展到大 headdim**，并针对 KV 共享存储做了专门优化。

- **大维度前向与反向**：支持 `d/dv` 为 **`512/512`** 与 **`576/512`** 两组配置。
- **K/V 分离与 KV 共享**：两组配置均提供 K/V 分离与 KV 共享两个版本。
- **完整掩码能力**：大维度下使用相同的列式掩码接口，同样支持 sink bias 与确定性计算。

以上大维度配置目前仅由 SM100 CuTe DSL 路径提供，SM90 暂不支持。

---

## 🚀 性能

FlashMask、FlashMask-3 以及计算通信 overlap 及负载均衡的分布式计算的性能数据见[论文](#-引用)，这里给出 FlashMask-4 的数据。

### FlashMask-4 vs FlashAttention-4 `mask_mod`

Baseline 为 FlashAttention-4 `mask_mod`（commit `edb5c7`）。性能指标为 TFLOPs/s，越高越好；正文中的提升范围取自总计（前向 + 反向）。

雷达图的读法：三行分别为前向（FWD）、反向（BWD）、总计（TOTAL），每列为一个序列长度。

#### Hopper

headdim 128/128、序列长度 4K ～ 128K 下，FlashMask-4 相对 baseline 的总计性能变化为 -13.1% ～ +49.0%。

<div align="center">
  <img src="docs/assets/fa4_mask_mod_vs_flashmaskv4_sm90_bf16_128_128_fwd_bwd_total_radar.png" width="80%" alt="FlashMask-4 与 FlashAttention-4 mask_mod 性能对比（Hopper，headdim 128/128）"/>
  <br/><sub>SM90 FlashMask-4 vs FlashAttention-4 mask_mod（d=128, dv=128）</sub>
</div>

headdim 192/128、序列长度 4K ～ 128K 下，FlashMask-4 相对 baseline 的总计性能变化为 -7.3% ～ +23.8%。

<div align="center">
  <img src="docs/assets/fa4_mask_mod_vs_flashmaskv4_sm90_bf16_192_128_fwd_bwd_total_radar.png" width="80%" alt="FlashMask-4 与 FlashAttention-4 mask_mod 性能对比（Hopper，headdim 192/128）"/>
  <br/><sub>SM90 FlashMask-4 vs FlashAttention-4 mask_mod（d=192, dv=128）</sub>
</div>

headdim 256/256、序列长度 4K ～ 128K 下，FlashMask-4 相对 baseline 的总计性能变化为 +3.1% ～ +155.8%。

<div align="center">
  <img src="docs/assets/fa4_mask_mod_vs_flashmaskv4_sm90_bf16_256_256_fwd_bwd_total_radar.png" width="80%" alt="FlashMask-4 与 FlashAttention-4 mask_mod 性能对比（Hopper，headdim 256/256）"/>
  <br/><sub>SM90 FlashMask-4 vs FlashAttention-4 mask_mod（d=256, dv=256）</sub>
</div>

#### Blackwell

headdim 128/128、序列长度 4K ～ 128K 下，FlashMask-4 相对 baseline 的总计性能变化为 -6.2% ～ +101.3%。

<div align="center">
  <img src="docs/assets/fa4_mask_mod_vs_flashmaskv4_sm100_bf16_128_128_fwd_bwd_total_radar.png" width="100%" alt="FlashMask-4 与 FlashAttention-4 mask_mod 性能对比（Blackwell，headdim 128/128）"/>
  <br/><sub>SM100 FlashMask-4 vs FlashAttention-4 mask_mod（d=128, dv=128）</sub>
</div>

headdim 192/128、序列长度 4K ～ 128K 下，FlashMask-4 相对 baseline 的总计性能变化为 -9.4% ～ +76.8%。

<div align="center">
  <img src="docs/assets/fa4_mask_mod_vs_flashmaskv4_sm100_bf16_192_128_fwd_bwd_total_radar.png" width="100%" alt="FlashMask-4 与 FlashAttention-4 mask_mod 性能对比（Blackwell，headdim 192/128）"/>
  <br/><sub>SM100 FlashMask-4 vs FlashAttention-4 mask_mod（d=192, dv=128）</sub>
</div>

### 不同 headdim 的耗时（Blackwell SM100）

FlashAttention-4 `mask_mod` 在 SM100 上不支持 `d=256, dv=256`，因此这里只比较 FlashMask-4 自身在 Blackwell (SM100) 上不同 headdim 的前向 + 反向总耗时，以 `d=128, dv=128` 为基准（`1.00x`）。

| 序列长度 | `d=128, dv=128` | `d=192, dv=128` | `d=256, dv=256` |
|---:|---:|---:|---:|
| 理论计算量 | 1.00x | 1.29x | 2.00x |
| 4K | 1.00x | 1.36x | 2.75x |
| 8K | 1.00x | 1.36x | 2.80x |
| 32K | 1.00x | 1.35x | 2.85x |
| 64K | 1.00x | 1.34x | 2.88x |
| 128K | 1.00x | 1.34x | 2.90x |

- 理论计算量只统计核心 MMA 运算，不含 softmax 等其他计算。
- 每个序列长度的实测倍率，为该长度下所有掩码类型总耗时倍率的几何平均。
- 三种 headdim 均在相同 head 数（H=16）下测得，可直接与理论计算量对比。

### 大 headdim：KV 共享 vs K/V 分离

`d/dv` 为 576/512、序列长度 4K ～ 128K 下，FlashMask-4 KV 共享版本相对 K/V 分离版本的总计性能提升 +35.3% ～ +81.7%。

<div align="center">
  <img src="docs/assets/flashmaskv4_sm100_vs_flashmaskv4kvshared_sm100_fwd_bwd_total_radar.png" width="100%" alt="FlashMask-4 KV 共享与 K/V 分离性能对比"/>
  <br/><sub>SM100 FlashMask-4 KV 共享 vs K/V 分离</sub>
</div>

---

## 📦 安装

### 环境要求

- **GPU**：Hopper（SM90）或 Blackwell（SM100）。
- **深度学习框架**：已启用 CUDA 的 PaddlePaddle 或 PyTorch（二者均不会被自动安装）。
- **构建工具**：`setuptools>=64`。
- **自动安装的依赖**：`nvidia-cutlass==4.2.0.0`、`nvidia-cutlass-dsl[cu13]==4.5.2`、`apache-tvm-ffi>=0.1.5,<0.2.0`。它们由 `setup.py` 在构建时安装，因此安装命令需要加 `--no-build-isolation`。
- **Overlap（`ovl`）额外要求**：NCCL >= 2.30.5（如 `nvidia-nccl-cu13`，或通过 `NCCL_HOME` 指定）。缺少 NCCL 时会跳过该组件，核心内核不受影响。

### 快速安装

```bash
git clone https://github.com/FlashMask/flashmask.git
cd flashmask

# 仅安装 CuTe DSL 核心内核，足以运行单卡 FlashMask
FLASHMASK_BUILD=core pip install -e . --no-build-isolation
```

### 构建选项

构建行为由两个环境变量控制，默认值为 `FLASHMASK_BUILD=all`、`FLASHMASK_FRAMEWORK=paddle`。取值非法时，安装会报错并列出允许的取值。

#### `FLASHMASK_BUILD`：选择组件

可选值为 `core`、`cpb`、`ovl`、`all`，可用 `+` 或 `,` 组合（如 `core+cpb`）。`core` 始终构建，无需显式指定。

| 组件 | 用途 |
|---|---|
| `core` | CuTe DSL FlashMask 内核（始终构建） |
| `cpb` | 上下文并行负载均衡 |
| `ovl` | 基于 NCCL GIN 的通信计算重叠（仅 SM100） |
| `all` | 以上全部（默认） |

#### `FLASHMASK_FRAMEWORK`：选择框架

默认面向 PaddlePaddle 构建，**PyTorch 环境请设置 `FLASHMASK_FRAMEWORK=torch`**。`core`、`cpb`、`ovl` 由两个框架共享同一套实现。

| 取值 | 说明 |
|---|---|
| `paddle` | 面向 PaddlePaddle 构建（默认） |
| `torch` | 面向 PyTorch 构建 |
| `both` | 同时面向两个框架构建 |

```bash
# 核心内核 + 负载均衡
FLASHMASK_BUILD=core+cpb pip install -e . --no-build-isolation

# PyTorch 环境下构建全部组件
FLASHMASK_FRAMEWORK=torch FLASHMASK_BUILD=all pip install -e . --no-build-isolation
```

---

## 🏁 快速开始

### 功能测试

```bash
# PaddlePaddle
python3 -m pytest tests/paddle/test_flashmask.py -v -x
# PyTorch
python3 -m pytest tests/torch/test_flashmask.py -v -x
```

### 性能测试

```bash
# PaddlePaddle
cd tests/paddle && bash run_perf.sh
# PyTorch
cd tests/torch && bash run_perf.sh
```

> 脚本按 GPU 架构自动选择 headdim（SM90：128/192/256；SM100：额外测试 576 的 split 与 shared KV），每个 headdim 占一张卡并行运行，默认从 GPU 0 开始。可通过 `DEVICES=2,3,4` 指定使用的卡。

### 计算通信 overlap 及负载均衡的分布式计算

以下命令在仓库根目录执行（脚本会自行切换到所在目录）。PaddlePaddle 与 PyTorch 的脚本同名，分别位于 `tests/paddle/distributed/` 与 `tests/torch/distributed/`：

```bash
# PaddlePaddle
cd tests/paddle/distributed && bash run_cp_balance.sh        # CPB 负载均衡（SM90 / SM100）
cd tests/paddle/distributed && bash run_cp_overlap_sweep.sh  # Overlap 通信计算重叠（仅 SM100）

# PyTorch
cd tests/torch/distributed && bash run_cp_balance.sh         # CPB 负载均衡（SM90 / SM100）
cd tests/torch/distributed && bash run_cp_overlap_sweep.sh   # Overlap 通信计算重叠（仅 SM100）
```

> 默认单机运行，CPB 使用 GPU 0～3（CP=4）。可通过 `DEVICES` 指定每台机器使用的 GPU，通过 `TARGET_RANKS` 指定参与的节点实现多机运行，详见各脚本头部注释。

更完整的基准测试见外部仓库 [test_flashmask](https://github.com/umiswing/test_flashmask)。

---

## 📚 引用

如果 FlashMask 对你的研究有帮助，请引用：

- **FlashMask**（ICLR 2025）：[FlashMask: Efficient and Rich Mask Extension of FlashAttention](https://arxiv.org/abs/2410.01359)
- **FlashMask-3** (NeurIPS 2026)：[FlashMask-3: Efficient & Expressive Mask-Aware Distributed Attention](https://arxiv.org/abs/2610.17901)

```bibtex
@inproceedings{ICLR2025_wang2025flashmask,
  author    = {Wang, Guoxia and Zeng, Jinle and Xiao, Xiyuan and Wu, Siming and Yang, Jiabin and
               Zheng, Lujing and Chen, Zeyu and Bian, Jiang and Yu, Dianhai and Wang, Haifeng},
  booktitle = {International Conference on Learning Representations},
  title     = {FlashMask: Efficient and Rich Mask Extension of FlashAttention},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/file/44c624d8f4e802453ad12593b2aa8e38-Paper-Conference.pdf},
  year      = {2025}
}

@inproceedings{wang2026flashmask3,
  author    = {Wang, Guoxia and He, Qianyue and Wu, Siming and Xie, Haoyang and
               Bao, Qiwen and Zeng, Jinle and Yang, Jiabin and Wu, Tian and Yu, Dianhai},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},

  title     = {FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention},
  url       = {https://arxiv.org/pdf/2610.17901},
  year      = {2026}
}
```

---

## 🙏 致谢

本仓库借鉴了 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的代码，感谢 [Tri Dao](https://github.com/tridao) 及其团队的出色工作。

---

## ⚖️ 许可证

FlashMask 采用 [BSD 3-Clause 许可证](LICENSE)。派生自上游项目的文件保留相应的版权与许可证声明；适用 Apache-2.0 的文件见 [LICENSE.Apache-2.0](LICENSE.Apache-2.0)。
