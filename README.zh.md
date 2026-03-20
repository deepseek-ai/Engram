<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<p align="center">
  <a href="README.md">English</a> ·
  <strong>中文</strong>
</p>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/"><img alt="Homepage"
    src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true"/></a>
  <a href="https://chat.deepseek.com/"><img alt="Chat"
    src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white"/></a>
  <a href="https://huggingface.co/deepseek-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/Tc7c45Zzu5"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true"><img alt="Wechat"
    src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://twitter.com/deepseek_ai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <br>
</div>

## 1. 简介 (Introduction)

本仓库包含论文的官方实现：**[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](Engram_paper.pdf)**。

> **摘要：** 虽然混合专家模型 (MoE) 通过条件计算扩展了容量，但 Transformer 缺乏原生的知识查找原语。为了解决这个问题，我们探索了**条件记忆 (conditional memory)** 作为一种补充的稀疏轴，并通过 **Engram** 模块实现了它，该模块现代化的经典 $N$-gram 嵌入，实现了 $\mathcal{O}(1)$ 的查找。

**关键贡献：**
- **稀疏分配：** 我们制定了神经计算 (MoE) 和静态记忆 (Engram) 之间的权衡，确定了引导最佳容量分配的 U 型缩放法则。
- **实证验证：** 在严格的等参数和等 FLOPs 约束下，Engram-27B 模型在知识、推理、代码和数学领域表现出优于 MoE 基线的持续改进。
- **机理分析：** 我们的分析表明，Engram 将早期层从静态模式重构中解放出来，从而可能为复杂的推理保留了有效深度。
- **系统效率：** 该模块采用确定性寻址，能够将海量嵌入表卸载到主机内存，且推理开销极小。


## 2. 架构 (Architecture)

Engram 模块通过检索静态 $N$-gram 记忆并将其与动态隐藏状态融合来增强主干网络。架构如下图所示（提供 [drawio](drawio/Engram.drawio)）：

<p align="center">
  <img width="75%" src="figures/arch.png" alt="Engram Architecture">
</p>

## 3. 评估 (Evaluation)

### 缩放法则 (Scaling Law)
<p align="center">
  <img width="90%" src="figures/scaling_law.png" alt="Scaling Law">
</p>

---

### 大规模预训练 (Large Scale Pre-training)
<p align="center">
  <img width="80%" src="figures/27b_exp_results.png" alt="Pre-training Results">
</p>

---

### 长文本训练 (Long-context Training)
<p align="center">
  <img width="80%" src="figures/long_context_results.png" alt="Long Context Results">
</p>


## 4. Engram 案例研究 (Case Study of Engram)
<p align="center">
  <img width="80%" src="figures/case.png" alt="Long Context Results">
</p>

## 5. 快速开始 (Quick Start)

推荐使用 Python 3.8+ 和 PyTorch。
```bash
pip install torch numpy transformers sympy
```
我们提供了一个独立实现来演示 Engram 模块的核心逻辑：
```bash
python engram_demo_v1.py
```

> ⚠️ **注意：** 提供的代码是一个演示版本，旨在说明数据流。它模拟了标准组件（如 Attention/MoE/mHC），以便专注于 Engram 模块。


## 6. 许可证 (License)
Engram 模型的使用受 [Model License](LICENSE) 约束。

## 7. 联系方式 (Contact)

如果您有任何问题，请提出 issue 或通过 [service@deepseek.com](mailto:service@deepseek.com) 联系我们。