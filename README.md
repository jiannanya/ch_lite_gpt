# ch_lite_gpt

一个可以在 **CPU** 上完整跑通训练与推理的中文短文阅读理解 lite GPT（中文指令风格）。

## 目录

- `litegpt/`：核心库（模型、数据、训练、采样）
- `runner/`：命令入口（造数据、造分词器、训练、推理）
- `corpus/`：中文训练/验证数据（jsonl）
- `artifacts/`：tokenizer 与 checkpoint 输出目录

### 价值

`ch_lite_gpt` 的价值不在于“跑出一个很强的模型”，而在于它把 LLM 工程最核心的一条闭环做到了：

**数据 → 分词 → 模型 → 训练 → 推理 → 验收**。

当这条链路稳定之后，你再去扩数据、扩参数、换任务、做优化，都会容易很多。

> 一句话总结：这是一个“能在普通 CPU 机器上跑通闭环”的中文小模型工程，覆盖 **数据生成 → 分词器 → 训练 → 推理 → 验收**，并把任务聚焦在“短文阅读理解（短答案）”这一容易验收、容易对齐的方向。

向希望快速搭建一个可训练、可推理、可验证的 lite GPT 工程的同学：不依赖外部数据集下载，不需要复杂分布式环境，在 Windows + CPU 也能完成一个可用的技术闭环。

---

## 快速开始（PowerShell）

在本目录执行：

```powershell
python -m pip install -r requirements.txt
```


计算参数规模

```powershell
python -m runner.count_params --hparams hparams_100m.yaml
```

输出：

```powershell
params: 95,469,312
weights fp32: 364.19MB
weights bf16/fp16: 182.09MB
weights int8 (rough): 91.05MB
adam states fp32 (rough): 728.37MB  # m + v
```

生成“中等规模”中文数据集

```powershell
python -m runner.make_corpus --hparams hparams_100m.yaml
```

训练 ByteLevel BPE 分词器

```powershell
python -m runner.make_tokenizer --hparams hparams_100m.yaml
```

CPU 训练

```powershell
python -m runner.train_cpu --device cpu --hparams hparams_100m.yaml
```

训练完成后再验证（推荐：不会提前开始验证）：

```powershell
python -m runner.train_and_verify --device cpu --hparams hparams_100m.yaml
```

推理验证

```powershell
python -m runner.query_model --ckpt artifacts/last.pt --prompt "阅读下面短文：\n...\n\n问题：..." --device cpu
```

默认产物是 fp32 checkpoint：`artifacts/last.pt`。

（可选）如需导出 int8 动态量化权重：把 `hparams_100m.yaml` 里的 `export.int8` 改为 `true`，训练结束会额外生成 `artifacts/int8.pt`。

数据格式

`corpus/*.jsonl` 每行：

```json
{"query": "...", "answer": "..."}
```

## Part One 详细介绍

### 1. 项目目标与设计取舍

#### 目标

- **CPU 可训练**：不依赖 GPU 也能完成可观测的训练过程。
- **推理可演示**：训练完成后可用命令行进行生成式推理。
- **验收可复现**：用统一的验证集抽样验收，避免“看着像对了但不稳定”的主观评估。
- **任务聚焦**：训练与验证同域，统一为“短文阅读理解/信息抽取/判断正误”等模板。

#### 关键取舍

- 模型采用 **Decoder-only Transformer**（最直接的 GPT 路线）。
- 训练目标采用 **Prefix Masking**：只对答案段计算 loss，让模型把容量集中在“根据问题生成答案”。
- 数据与验收都偏向“短答案”，便于 stop 截断与自动比对。

---

### 2. 代码结构：职责清晰的 3 层分工

工程目录大致分为三层：

- `litegpt/`：核心库（模型、训练循环、数据与 batch、采样与解码）
- `runner/`：命令入口（造数据、造分词器、训练、推理、验收）
- `corpus/` 与 `artifacts/`：数据与产物

这种分层的好处是：

- 核心逻辑在 `litegpt/`，便于复用与单测；
- CLI 在 `runner/`，便于组合成脚本链路；
- 数据与权重输出集中管理，不会散落在根目录。

---

### 3. 数据：合成“短文阅读理解”来保证同域闭环

`runner/make_corpus.py` 会生成 JSONL：每行是一个样本。

```json
{"query": "...", "answer": "..."}
```

训练时会拼成一条序列：

```text
用户:{query}
助手:{answer}
```

#### 为什么选择合成阅读理解数据

- **答案确定**：便于用规则做 EXACT/CONTAINS 评估。
- **同域一致**：train/valid 的样式一致，减少验证偏差。
- **训练更像“指令对齐”**：query 是任务描述 + 短文，answer 是短答案。

数据生成里包含多种阅读理解变体（示意）：

- 会议纪要/公告/通知：抽取主持人、渠道、地点、时间
- 信息抽取 JSON：要求“只输出 JSON”
- 判断正误：只输出“对/错”

这些任务共同点是：输出短、可控、适合 stop 截断。

---

### 4. 分词器：ByteLevel BPE，兼顾中文与工程可控性

工程支持训练 ByteLevel BPE 分词器（典型流程是先生成语料，再训练 tokenizer）。

为什么是 ByteLevel BPE：

- **不用中文分词**：中文天然没有空格，byte-level 能覆盖任意字符。
- **词表可控**：小模型/CPU 训练时，控制词表规模有利于降低 embedding 与 lm head 的参数量。
- **工程稳定**：避免奇怪字符导致无法编码/解码。

---

### 5. 模型：Decoder-only Transformer（RoPE + ScaleNorm + SwiGLU + 权重共享）

核心模型实现在 `litegpt/model.py`，结构是一个标准的 decoder-only Transformer：

- Token Embedding
- 堆叠若干层 Block
- Final Norm
- LM Head（并与 embedding 权重共享）

#### 5.1 因果自注意力（Causal Self-Attention）

每个位置只能看历史：用下三角 mask 保证自回归。

注意力的核心计算：

- 线性得到 Q/K/V
- 计算分数 $QK^\top / \sqrt{d_h}$
- 加因果 mask
- softmax 得到权重
- 加权求和得到输出

#### 5.2 RoPE：把位置信息注入到 Q/K

RoPE 不把“位置向量”加到 embedding 上，而是对 Q/K 做旋转变换，从而让注意力分数具备相对位置信息。

工程实现上会缓存 RoPE 的 cos/sin 表，避免每次 forward 重算。

#### 5.3 ScaleNorm：轻量稳定的归一化

每层使用 ScaleNorm 做预归一化（Pre-Norm），在小模型训练中通常更稳定，也更省。

#### 5.4 SwiGLU：门控前馈

FFN 使用 SwiGLU（SiLU + gate），兼顾表达力与收敛性。

#### 5.5 权重共享（Weight Tying）

把 `lm_head.weight` 直接绑定到 `embedding.weight`，减少参数量并让输入输出共享同一个词向量空间。

---

### 6. 训练：只对答案段算 loss（Prefix Masking）

训练目标仍然是 next-token prediction，但对 query 前缀部分的标签设为 ignore（不参与 loss）。

直觉：

- query 是“题目”，answer 是“答案”；
- 我们更关心模型“如何答题”，而不是“如何复读题干”。

这在指令数据、阅读理解数据上特别有效。

训练入口示例：

```powershell
python -m runner.train_cpu --device cpu --hparams hparams_100m.yaml
```

训练循环通常包含：

- AdamW
- warmup + cosine 学习率
- 梯度累积（用小 micro-batch 模拟大 batch）
- 梯度裁剪
- 定期评估与保存 checkpoint

---

### 7. 推理：自回归生成 + stop 控制“短答案”

推理入口示例：

```powershell
python -m runner.query_model --ckpt artifacts/last.pt --prompt "阅读下面短文：\n...\n\n问题：..." --device cpu
```

阅读理解的输出通常是短答案，因此推理侧最关键的是“可控输出”：

- `max_new_tokens` 不要太大
- `stop` 建议用换行 `\n` 或句号等
- 温度/top-k/top-p 可以更保守（很多场景 temperature=0 或很小就够）
- 重复惩罚、trigram blocking 用来抑制尾部重复

目标是让模型“答完就停”，而不是继续胡写时间戳或拼接杂讯。

---

### 8. 验收：从 valid 抽样，做自动对比

工程提供 `runner/verify.py` 风格的验收：

- 从 `corpus/valid.jsonl` 抽样若干条
- 打印 query/expected
- 调用推理脚本拿到 model 输出
- 做 EXACT / CONTAINS / MISS 的统计

这种验收方式的价值是：

- 不靠主观“看着不错”
- 可固定 seed 做回归
- 适合迭代 prompt 模板、stop 策略、数据生成规则

---

### 9. 一条推荐的“最小闭环”命令链

在本目录下：

```powershell
python -m pip install -r requirements.txt
python -m runner.make_corpus --hparams hparams_100m.yaml
python -m runner.make_tokenizer --hparams hparams_100m.yaml
python -m runner.train_cpu --device cpu --hparams hparams_100m.yaml
python -m runner.train_and_verify --device cpu --hparams hparams_100m.yaml
```

你会得到：

- `corpus/`：训练/验证数据
- `artifacts/`：tokenizer 与 checkpoint（如 `last.pt`）
- verify 的抽样验收输出

---

### 10. 经验与可扩展方向

如果你准备继续把它“从能跑”推进到“更好用”，建议优先级如下：

1) **更严格的输出约束**：短答案任务强依赖 stop 与输出清洗；必要时做“答案抽取”而非直接对比整段生成。
2) **更丰富的阅读理解分布**：加入更多实体类型（人名/部门/日期/渠道）、更多否定/干扰项，提高泛化。
3) **更强的训练监控**：记录 train/val loss、生成样例、训练耗时，方便 CPU 侧做效率对比。
4) **更高效推理**：KV cache、批量推理、量化/编译优化等。

---

## Part Two 原理与实际训练

## 神经网络结构

1) 堆叠 $L$ 个 **Transformer Block**：

$$X \leftarrow \mathrm{Block}_1(X) \leftarrow \cdots \leftarrow \mathrm{Block}_L(X)$$

1) **Final Norm**（ScaleNorm）后接 **LM Head** 得到 logits：

$$\mathrm{logits} = X W_{\text{lm}} \in \mathbb{R}^{B\times T\times V}$$

1) **权重共享（Weight tying）**：通常令 $W_{\text{lm}} = W_{\text{embed}}^\top$，减少参数、也让输入/输出词向量空间一致。

### Block 结构（ScaleNorm + RoPE Attention + SwiGLU FFN）

每个 Block 是典型的 **Pre-Norm 残差结构**：

$$\begin{aligned}
Y &= X + \mathrm{Attn}(\mathrm{ScaleNorm}(X))\\
Z &= Y + \mathrm{FFN}(\mathrm{ScaleNorm}(Y))
\end{aligned}$$

其中包含三块关键组件：

#### 1) ScaleNorm（归一化）

ScaleNorm 的直觉：把向量长度缩放到一个稳定范围（并带一个可学习缩放参数 $g$），让训练更稳定。

一种常见写法：

$$\mathrm{ScaleNorm}(x) = \frac{g}{\lVert x \rVert_2 + \epsilon} \cdot x$$

与 RMSNorm/LayerNorm 相比，它更“轻量”（不做均值去除），对小模型与 CPU 训练也更友好。

#### 2) 因果多头自注意力（Causal Multi-Head Self-Attention）

对输入 $X \in \mathbb{R}^{B\times T\times d}$，线性映射得到：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

按头拆分为 $h$ 个 head 后，单头注意力为：

$$\mathrm{Attn}(Q,K,V)=\mathrm{Softmax}\Big(\frac{QK^\top}{\sqrt{d_h}} + M\Big)V$$

$M$ 是 **因果 mask**（下三角）：禁止当前位置 $t$ 看到未来位置 $>t$，保证自回归。

多头输出拼接后做一次输出投影 $W_O$：

$$\mathrm{MHSA}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)W_O$$

#### 3) RoPE（旋转位置编码，注入到 Q/K）

RoPE 的做法不是把“位置向量”加到 embedding 上，而是把位置信息以“旋转”的方式注入到 $Q/K$。

对向量按 2 维成对（$(x_{2i},x_{2i+1})$）做旋转：

$$\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}=
\begin{bmatrix}\cos\theta_i(t) & -\sin\theta_i(t)\\\sin\theta_i(t) & \cos\theta_i(t)\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}$$

然后用 $Q',K'$ 去计算注意力。好处是注意力分数隐式包含相对位置信息，长序列泛化更好。

#### 4) SwiGLU 前馈网络（FFN）

前馈层使用 **SwiGLU**（门控 + SiLU），在同等参数/计算下通常比纯 ReLU/GeLU 更强：

$$\mathrm{FFN}(x) = (\mathrm{SiLU}(xW_1)\odot (xW_2))W_3$$

其中 $\odot$ 是逐元素乘法，$W_1,W_2$ 负责“生成特征”和“生成门控”，$W_3$ 投影回 $d$。

### 训练目标（只对答案段算损失：Prefix Masking）

样本被拼成一条序列（示意）：

`用户: {query}\n助手: {answer}`

训练时做标准的 next-token 预测：

- 输入：$x_{0:T-2}$
- 目标：$x_{1:T-1}$

损失用交叉熵：

$$\mathcal{L} = -\sum_{t=0}^{T-2} w_t\log p_\theta(x_{t+1}\mid x_{\le t})$$

其中 $w_t \in \{0,1\}$：

- **query / 前缀**部分：$w_t=0$（忽略，不反向）
- **answer / 答案**部分：$w_t=1$（参与训练）

这会强制模型把容量集中在“根据问题生成答案”上，而不是去复读提示词。

### 计算量与为何能在 CPU 上训练

- 注意力的主要复杂度：$\mathcal{O}(B\cdot h\cdot T^2\cdot d_h)=\mathcal{O}(B\cdot T^2\cdot d)$
- 前馈主要复杂度：$\mathcal{O}(B\cdot T\cdot d\cdot d_{ff})$

因此 CPU 训练最关键的杠杆是 **控制 $T$（seq_len）与 $d$、层数 $L$**，并用梯度累积模拟更大 batch。

### 推理原理（自回归采样）

推理时从 prompt 开始逐 token 生成：

1) 得到最后一位 logits
2) 按 temperature / top-k / top-p / repeat penalty 等策略采样下一个 token
3) 追加到序列并重复，直到遇到 stop 条件或达到最大长度

阅读理解这种“短答案”任务通常推荐：较小的 `max_new_tokens` + `stop='\n'`，尽量让输出可控（避免尾巴噪声影响验收）。

### 训练过程示例：

```
step 10 loss 410.9045 lr 0.000010 accum 8 +23.4s elapsed 23.4s eta 11700.8s
step 20 loss 248.8220 lr 0.000020 accum 8 +23.9s elapsed 47.4s eta 11791.4s
step 30 loss 87.5851 lr 0.000030 accum 8 +23.4s elapsed 70.7s eta 11720.3s
step 40 loss 130.9726 lr 0.000040 accum 8 +28.1s elapsed 98.8s eta 12256.1s
step 50 loss 14.3681 lr 0.000050 accum 8 +23.4s elapsed 122.2s eta 12099.9s
step 60 loss 14.9123 lr 0.000060 accum 8 +24.3s elapsed 146.6s eta 12066.0s
step 70 loss 6.1132 lr 0.000070 accum 8 +22.9s elapsed 169.5s eta 11934.8s
step 80 loss 48.7968 lr 0.000080 accum 8 +38.4s elapsed 207.9s eta 12784.6s
step 90 loss 19.1148 lr 0.000090 accum 8 +25.5s elapsed 233.4s eta 12734.1s
step 100 loss 6.6947 lr 0.000100 accum 8 +23.1s elapsed 256.5s eta 12569.3s
step 110 loss 0.0742 lr 0.000110 accum 8 +23.9s elapsed 280.4s eta 12464.3s
step 120 loss 10.2751 lr 0.000120 accum 8 +26.4s elapsed 306.8s eta 12477.7s
step 130 loss 3.1067 lr 0.000130 accum 8 +24.0s elapsed 330.8s eta 12391.7s
step 140 loss 3.7960 lr 0.000140 accum 8 +23.6s elapsed 354.4s eta 12302.7s
step 150 loss 1.7859 lr 0.000150 accum 8 +35.8s elapsed 390.2s eta 12616.5s
step 160 loss 3.2165 lr 0.000160 accum 8 +23.4s elapsed 413.6s eta 12511.9s
step 170 loss 6.2035 lr 0.000170 accum 8 +24.9s elapsed 438.5s eta 12457.8s
early stop: train loss 0.000003 <= 0.0001
done: out_dir=artifacts elapsed=454.13s
```

### 训练平台：

```
- os: Windows 10.0.26100
- cpu: 12th Gen Intel(R) Core(TM) i9-12900K
- logical_cores: 24
- ram_total_gib: 31.7474
```

### 测试结果示例:

```
================================================================================   
[sample 1/3] prompt:
阅读下面短文：
11月13日，运维团队对线上延迟波动进行了复盘。会议由赵婷主持，孙浩负责记录，时长约30 分钟。

复盘结论指出：本次波动的直接原因并非代码逻辑错误，而是依赖服务偶发超时导致连锁反应 。为避免再次发生，短期措施是做回滚开关，并在关键链路补充监控与告警。

行动项包括：本周内完成改动并灰度验证；同时更新应急预案。如需协助，请通过邮件联系赵 婷。

问题：如果需要协助，应通过什么渠道联系？只输出渠道名称。

expected:
邮件

model:
邮件
match: EXACT

================================================================================   
[sample 2/3] prompt:
阅读下面短文：
1月8日，数据平台团队对线上吞吐波动进行了复盘。会议由王敏主持，林然负责记录，时长约30分钟。

复盘结论指出：本次波动的直接原因并非代码逻辑错误，而是依赖服务偶发超时导致连锁反应 。为避免再次发生，短期措施是加缓存，并在关键链路补充监控与告警。

行动项包括：本周内完成改动并灰度验证；同时更新应急预案。如需协助，请通过企业微信联 系王敏。

问题：会议由谁主持？只输出人名。

expected:
王敏

model:
李雷
match: MISS

================================================================================   
[sample 3/3] prompt:
阅读下面短文：
10月27日，搜索团队对线上命中率波动进行了复盘。会议由李雷主持，吴桐负责记录，时长约45分钟。

复盘结论指出：本次波动的直接原因并非代码逻辑错误，而是依赖服务偶发超时导致连锁反应 。为避免再次发生，短期措施是做回滚开关，并在关键链路补充监控与告警。

行动项包括：本周内完成改动并灰度验证；同时更新应急预案。如需协助，请通过邮件联系李 雷。

问题：短期措施是什么？只输出原文中的那一个动作。

expected:
做回滚开关

model:
做降级
match: MISS

--------------------------------------------------------------------------------   
summary: exact=1/3 (33.3%)

```
