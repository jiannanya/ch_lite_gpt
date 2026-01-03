from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from litegpt.config import read_yaml
from litegpt.file_io import mkdir


@dataclass(frozen=True)
class QA:
    q: str
    a: str


_FORBIDDEN: tuple[str, ...] = (
    "色情",
    "暴力",
)


def _choice(rng: random.Random, xs: list[str]) -> str:
    return xs[rng.randrange(len(xs))]


def _mk_reading_qa(rng: random.Random) -> QA:
    # Synthetic reading comprehension with deterministic answers.
    topics = [
        (
            "团队周会纪要",
            [
                ("目标", "本周完成登录流程联调并补齐单元测试"),
                ("风险", "第三方接口偶发超时，需要加重试与降级"),
                ("安排", "周三完成联调，周五做回归并发布"),
            ],
            "请根据文本回答：本周最主要的风险是什么？只输出一句话。",
            "本周主要风险是第三方接口偶发超时，需要增加重试与降级。",
        ),
        (
            "产品更新公告",
            [
                ("更新内容", "新增批量导入与导出功能"),
                ("影响范围", "仅影响管理端，不影响移动端"),
                ("回滚方案", "出现异常可一键回退到上一版本"),
            ],
            "请根据文本回答：这次更新影响哪些端？只输出一句话。",
            "这次更新仅影响管理端，不影响移动端。",
        ),
        (
            "课程通知",
            [
                ("时间", "下周一晚 7 点"),
                ("主题", "向量与矩阵的直观理解"),
                ("作业", "提交两道矩阵乘法维度题"),
            ],
            "请根据文本回答：作业要求是什么？只输出一句话。",
            "作业要求是提交两道矩阵乘法维度题。",
        ),
    ]
    title, kvs, q, a = _choice(rng, topics)
    lines = [f"【{title}】"]
    for k, v in kvs:
        lines.append(f"- {k}：{v}")
    text = "\n".join(lines)
    # Keep phrasing aligned with training/validation style.
    prompt = f"阅读下面短文：\n{text}\n\n问题：{q}"
    return QA(prompt, a)


def _mk_reading_long(rng: random.Random) -> QA:
    """Longer reading comprehension with deterministic answers.

    Design goals:
    - Longer passages (5-10 lines) with some distractors.
    - Clear, unambiguous questions.
    - Answers are deterministic and short (often a single sentence / fixed JSON).
    """

    orgs = ["数据平台组", "研发团队", "教务处", "社区图书馆", "产品运营部", "课程助教"]
    persons = ["张伟", "李雷", "王敏", "陈晨", "赵婷", "周航"]
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
    channels = ["企业微信", "邮件", "工单系统", "公告栏"]
    org = _choice(rng, orgs)
    owner = _choice(rng, persons)
    city = _choice(rng, cities)
    channel = _choice(rng, channels)

    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.choice([9, 10, 14, 15, 19])
    minute = rng.choice([0, 10, 30, 40])
    date = f"{month}月{day}日"
    time_str = f"{hour:02d}:{minute:02d}"
    room = f"{rng.randint(2, 12)}号会议室"
    duration = rng.choice([30, 45, 60])
    limit = rng.choice([50, 80, 120])

    title = rng.choice(["系统维护通知", "课程安排调整", "周会纪要", "发布计划变更"])
    bullets = [
        f"负责人：{owner}",
        f"时间：{date} {time_str}（预计 {duration} 分钟）",
        f"地点：{city}·{room}",
        f"影响：仅影响部分用户，预计峰值并发上限 {limit}",
        f"反馈渠道：如有问题请通过{channel}联系",
    ]
    distract = rng.choice(
        [
            "提醒：请提前 10 分钟到场签到。",
            "备注：如无法参加，请在群内说明原因。",
            "说明：此次变更不涉及账号权限调整。",
            "提示：请勿在会议中分享敏感信息。",
        ]
    )
    passage = "\n".join(
        [
            f"【{org}】{title}",
            f"为保证服务稳定，{org}将进行例行安排。以下为关键信息：",
            *[f"- {x}" for x in bullets],
            distract,
            "如果你需要转发，请保持信息完整。",
        ]
    )

    qtype = rng.randrange(4)
    if qtype == 0:
        q = "请根据文本回答：时间是什么？只输出一句话。"
        a = f"时间是{date} {time_str}。"
    elif qtype == 1:
        q = "请根据文本回答：地点在哪里？只输出一句话。"
        a = f"地点在{city}·{room}。"
    elif qtype == 2:
        q = "请从文本中抽取信息并输出 JSON（只输出 JSON），键为 date/time/room/channel："
        a = json.dumps({"date": date, "time": time_str, "room": f"{city}·{room}", "channel": channel}, ensure_ascii=False)
    else:
        stmt = rng.choice(
            [
                (f"这次安排会影响所有用户。", "错"),
                (f"预计峰值并发上限是{limit}。", "对"),
                (f"反馈渠道是{channel}。", "对"),
            ]
        )
        s, ans = stmt
        q = f"判断正误：{s}（只输出 对/错）"
        a = ans

    prompt = f"阅读下面短文：\n{passage}\n\n{q}"
    return QA(prompt, a)


def _mk_reading_article(rng: random.Random) -> QA:
    """More realistic multi-paragraph reading comprehension.

    Compared to _mk_reading_long, this tries to look less like a template:
    - 2-3 paragraphs with numbers, dates, and roles
    - questions require locating a specific fact, not summarizing freely
    """

    teams = ["数据平台", "推荐系统", "客户端", "搜索", "基础架构", "运维"]
    team = rng.choice(teams)
    owner = rng.choice(["张伟", "李雷", "王敏", "陈晨", "赵婷", "周航"]) 
    reviewer = rng.choice(["孙浩", "吴桐", "林然", "许佳"]) 

    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date = f"{month}月{day}日"
    duration = rng.choice(["30分钟", "45分钟", "1小时"])
    metric = rng.choice(["延迟", "错误率", "吞吐", "命中率"])
    metric_val = rng.choice(["下降了 12%", "上升了 0.3%", "提升了 18%", "波动在 1% 以内"])
    action = rng.choice(["加缓存", "做降级", "补监控", "加重试", "做回滚开关"])
    channel = rng.choice(["企业微信", "邮件", "工单系统"])

    p1 = (
        f"{date}，{team}团队对线上{metric}波动进行了复盘。"
        f"会议由{owner}主持，{reviewer}负责记录，时长约{duration}。"
    )
    p2 = (
        f"复盘结论指出：本次波动的直接原因并非代码逻辑错误，而是依赖服务偶发超时导致连锁反应。"
        f"为避免再次发生，短期措施是{action}，并在关键链路补充监控与告警。"
    )
    p3 = (
        f"行动项包括：本周内完成改动并灰度验证；同时更新应急预案。"
        f"如需协助，请通过{channel}联系{owner}。"
    )
    passage = "\n\n".join([p1, p2, p3])

    qtype = rng.randrange(4)
    if qtype == 0:
        q = "问题：会议由谁主持？只输出人名。"
        a = owner
    elif qtype == 1:
        q = f"问题：短期措施是什么？只输出原文中的那一个动作。"
        a = action
    elif qtype == 2:
        q = "问题：记录由谁负责？只输出人名。"
        a = reviewer
    else:
        q = "问题：如果需要协助，应通过什么渠道联系？只输出渠道名称。"
        a = channel

    prompt = f"阅读下面短文：\n{passage}\n\n{q}"
    return QA(prompt, a)


def _mk_formatting(rng: random.Random) -> QA:
    # Convert between bullet list and JSON deterministically.
    items = [
        ("计划", ["确定目标", "拆分任务", "安排时间", "执行与复盘"]),
        ("复盘", ["回顾目标", "列出事实", "分析原因", "提出改进"]),
        ("写作", ["先给结论", "再给理由", "最后给例子", "补充边界"]),
    ]
    title, xs = _choice(rng, items)
    if rng.random() < 0.5:
        prompt = f"把下面要点转换成 JSON 数组（只输出 JSON）：\n" + "\n".join([f"- {x}" for x in xs])
        completion = json.dumps(xs, ensure_ascii=False)
    else:
        payload = {"title": title, "items": xs}
        prompt = f"把下面 JSON 改写成分点列表（每行以 - 开头，只输出列表）：\n{json.dumps(payload, ensure_ascii=False)}"
        completion = "\n".join([f"- {x}" for x in xs])
    return QA(prompt, completion)


def _mk_debug(rng: random.Random) -> QA:
    # Small, deterministic debugging guidance.
    cases = [
        (
            "Python 运行时报错：ModuleNotFoundError: No module named 'tokenizers'。我该怎么办？",
            "先确认你激活的是正确的虚拟环境；然后用 pip 安装缺失依赖（例如 pip install tokenizers）；最后重新运行脚本验证导入是否正常。",
        ),
        (
            "训练时 loss 变成 NaN，可能原因和排查步骤是什么？",
            "常见原因包括学习率过大、数值溢出或数据异常；可以先降低学习率、开启梯度裁剪、检查输入是否含非法值，并打印梯度/激活的最大最小值定位溢出点。",
        ),
        (
            "模型输出总是重复同一句话，怎么改善？",
            "可以提高重复惩罚、使用 top-p/top-k 采样并降低温度，同时在训练数据中增加多样化答案，必要时加入 n-gram 限制来抑制循环重复。",
        ),
    ]
    q, a = _choice(rng, cases)
    prompt = f"请给出简洁、可执行的排查/解决建议：\n{q}"
    return QA(prompt, a)


def _contains_forbidden(text: str) -> bool:
    return any(x in text for x in _FORBIDDEN)


def _filter_forbidden(items: Iterable[QA]) -> list[QA]:
    out: list[QA] = []
    for qa in items:
        if _contains_forbidden(qa.q) or _contains_forbidden(qa.a):
            continue
        out.append(qa)
    return out


def _mk_facts() -> list[QA]:
    return [
        QA("什么是注意力机制？", "注意力机制会为不同位置分配权重，让模型把注意力集中在与当前任务最相关的信息上。"),
        QA("用一句话解释什么是注意力机制。", "注意力机制通过对不同位置赋予不同权重，让模型更关注关键信息。"),
        QA("什么是 Transformer？", "Transformer 是以自注意力为核心的序列建模架构，能够并行计算并建模长距离依赖。"),
        QA("什么是正则化？", "正则化是在训练目标或模型上加入约束/惩罚项，减少过拟合并提升泛化能力。"),
        QA("一句话解释正则化。", "正则化通过限制模型复杂度来缓解过拟合。"),
        QA("正则化是为了解决什么问题？", "主要用来缓解过拟合，让模型在新数据上表现更稳定。"),
        QA("正则化和学习率调度有什么区别？", "正则化是控制模型复杂度/约束参数；学习率调度是调整优化步长，两者作用不同。"),
        QA("什么是过拟合？", "过拟合是模型在训练集上表现很好，但在未见数据上泛化变差的现象。"),
        QA("什么是学习率？", "学习率是每一步参数更新的步幅大小，过大可能发散，过小会收敛很慢。"),
        QA("解释什么是梯度。", "梯度是损失函数对参数的偏导数方向，用于指导参数如何更新以降低损失。"),
        QA("一句话解释什么是量化。", "量化是用更低精度（如 int8）表示权重/激活，从而减少计算与内存并加速推理。"),
        QA("什么是位置编码？", "位置编码用于告诉模型序列中 token 的顺序信息，否则注意力机制本身不区分位置。"),
        QA("什么是 RoPE？", "RoPE 是一种旋转位置编码，把位置信息以旋转方式注入注意力的 Q/K 中。"),
        QA("RMSNorm 与 LayerNorm 有何不同？", "RMSNorm 只做均方根缩放而不做均值中心化，结构更简单、计算更省。"),
        QA("解释 top-k 采样。", "top-k 采样只在概率最高的 k 个候选中随机采样，以控制输出质量与多样性。"),
        QA("解释 top-p 采样。", "top-p 采样会选择累计概率达到阈值 p 的最小候选集合，再在其中采样。"),
        QA("什么是重复惩罚？", "重复惩罚会降低已生成 token 的概率，减少输出反复重复同一句话的情况。"),
        QA("如何写出清晰的回答？", "先给结论，再给关键理由，最后补充一个例子或边界条件，整体会更清晰。"),
        QA("一句话说明什么是梯度裁剪。", "梯度裁剪通过限制梯度范数上限，避免训练中梯度爆炸导致不稳定。"),
    ]


def _mk_explain(rng: random.Random) -> QA:
    topics = [
        ("什么是监督学习？", "监督学习利用带标签的数据学习输入到输出的映射，以便对新样本进行预测。"),
        ("什么是无监督学习？", "无监督学习在没有标签的情况下从数据中发现结构，例如聚类或降维。"),
        ("解释一下交叉熵损失。", "交叉熵用来衡量预测分布与真实分布的差异，分类任务中越小表示预测越接近真实标签。"),
        ("交叉熵损失为什么常用于分类？", "因为它直接比较预测概率与真实标签分布，能稳定推动正确类别概率变大。"),
        ("什么是 batch size？", "batch size 是每次参数更新使用的样本数量，影响训练稳定性与速度。"),
        ("batch size 大小如何影响训练？", "更大 batch 往往更稳定但更吃内存；更小 batch 噪声更大但可能更快迭代。"),
        ("解释一下什么是推理。", "推理是模型在训练完成后，根据输入生成输出的过程，也就是实际使用阶段。"),
        ("什么是微调？", "微调是在预训练模型基础上，用特定任务数据继续训练，使其更适合目标任务。"),
        ("什么是学习率调度？", "学习率调度是在训练过程中按策略调整学习率，比如先 warmup 再衰减。"),
        ("什么是 warmup？", "warmup 是训练初期逐步增大学习率，减少不稳定和梯度爆炸风险。"),
        ("什么是梯度爆炸？", "梯度爆炸是梯度过大导致参数更新失控，训练不稳定甚至发散。"),
        ("什么是梯度消失？", "梯度消失是梯度过小导致学习停滞，深层网络难以更新。"),
        ("解释一下什么是正则化。", "正则化通过惩罚复杂模型或加入约束，降低过拟合并提高泛化能力。"),
        ("正则化和学习率调度分别做什么？", "正则化是减少过拟合（约束模型复杂度）；学习率调度是改变更新步长（影响优化过程）。"),
        ("举例说明常见正则化方法。", "常见方法包括 L2 正则化（权重衰减）、dropout、早停（early stopping）等。"),
        ("为什么正则化能减少过拟合？", "它会限制模型过度拟合训练集细节，促使模型学习更通用的规律。"),
        ("L1 正则化是什么？", "L1 正则化是在损失中加入权重绝对值惩罚，倾向产生稀疏参数。"),
        ("L2 正则化是什么？", "L2 正则化是在损失中加入权重平方和惩罚项，鼓励权重变小更平滑。"),
        ("dropout 的作用是什么？", "dropout 训练时随机丢弃部分神经元，减少共适应以缓解过拟合。"),
        ("什么是注意力机制？", "注意力机制为不同位置分配权重，让模型聚焦于与当前预测相关的信息。"),
        ("自注意力和注意力有什么区别？", "自注意力是注意力的一种，查询、键、值都来自同一序列。"),
        ("什么是 token？", "token 是模型处理文本的基本单位，可以是字、子词或字节片段。"),
        ("什么是 BPE？", "BPE 是一种子词分词方法，通过合并高频符号对来构造词表。"),
        ("为什么需要验证集？", "验证集用于评估泛化能力，帮助发现过拟合并比较不同训练设置。"),
        ("什么是困惑度（perplexity）？", "困惑度是语言模型对序列不确定性的度量，越低通常表示预测越好。"),
        ("什么是 top-k 采样？", "top-k 采样只在概率最高的 k 个候选中采样，控制输出质量与多样性。"),
        ("什么是 top-p 采样？", "top-p 采样在累计概率达到阈值 p 的最小候选集合中采样。"),
        ("重复惩罚是做什么的？", "重复惩罚降低已生成 token 的概率，减少输出反复重复。"),
        ("如何让回答更清晰？", "先给结论，再给理由，最后给例子或边界条件。"),
    ]
    q, a = _choice(rng, topics)
    return QA(q, a)


def _mk_classify(rng: random.Random) -> QA:
    texts = [
        ("我今天特别开心，工作进展很顺利。", "积极"),
        ("这件事让我很失望，感觉没什么希望。", "消极"),
        ("天气一般，没什么特别的。", "中性"),
    ]
    t, label = _choice(rng, texts)
    prompt = f"情感分类：请判断下面句子是 积极/消极/中性 哪一种，只输出标签。\n句子：{t}"
    return QA(prompt, label)


def _mk_extract(rng: random.Random) -> QA:
    names = ["小王", "小李", "小张", "小陈"]
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
    name = _choice(rng, names)
    city = _choice(rng, cities)
    age = rng.randint(18, 55)

    text = f"{name}今年{age}岁，住在{city}，平时喜欢跑步和读书。"
    prompt = f"从文本中抽取信息，输出 JSON，键为 name/age/city：\n{text}"
    completion = json.dumps({"name": name, "age": age, "city": city}, ensure_ascii=False)
    return QA(prompt, completion)


def _gen_one(rng: random.Random) -> QA:
    # Requirement: train dataset should be short-article reading comprehension like valid.
    # Keep style consistent while preserving variety via different passage templates.
    r = rng.random()
    if r < 0.70:
        return _mk_reading_article(rng)
    if r < 0.95:
        return _mk_reading_long(rng)
    return _mk_reading_qa(rng)


def _gen_valid(rng: random.Random) -> QA:
    # Keep validation prompts aligned with training prompt style.
    return _gen_one(rng)


def _key(text: str) -> bytes:
    # Stable 64-bit key to keep memory low.
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _write_stream(
    path: str,
    *,
    rng: random.Random,
    n: int,
    train_keys: set[bytes],
    gen: Callable[[random.Random], QA],
    label: str,
) -> int:
    p = Path(path)
    mkdir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        written = 0
        seen_valid: set[bytes] = set()
        t0 = time.time()
        while written < n:
            qa = gen(rng)
            if _contains_forbidden(qa.q) or _contains_forbidden(qa.a):
                continue
            q = qa.q.strip()
            a = qa.a.strip()
            if not q or not a:
                continue
            k = _key(q)
            if k in train_keys or k in seen_valid:
                continue
            seen_valid.add(k)
            f.write(json.dumps({"query": q, "answer": a}, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                dt = time.time() - t0
                print(f"{label} {written}/{n} elapsed {dt:.1f}s")
    return written


def _valid_suite(rng: random.Random, k: int) -> list[QA]:
    """Small curated suite for valid to stabilize evaluation.

    Kept deterministic via rng; answers are unambiguous.
    """
    out: list[QA] = []
    for _ in range(k):
        # Force a mix of the two reading types.
        out.append(_mk_reading_article(rng) if rng.random() < 0.6 else _mk_reading_long(rng))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", type=str, default="hparams_100m.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.hparams)
    seed = int(cfg.get("seed", 123))
    train_path = str(cfg["paths"]["train_jsonl"])
    valid_path = str(cfg["paths"]["valid_jsonl"])

    # A larger, more general dataset to learn instruction patterns on CPU.
    # Configurable to support 100k+ scale.
    corpus_cfg = cfg.get("corpus", {})
    train_n = int(corpus_cfg.get("train_n", 100000))
    valid_n = int(corpus_cfg.get("valid_n", 5000))

    # Train/valid are generated separately to reduce overlap, making valid loss meaningful.
    train_rng = random.Random(seed)
    valid_rng = random.Random(seed + 1)

    train_keys: set[bytes] = set()
    p_train = Path(train_path)
    mkdir(p_train.parent)
    with p_train.open("w", encoding="utf-8") as f:
        t0 = time.time()

        # Fill train with reading comprehension only.
        wrote = 0
        while wrote < train_n:
            qa = _gen_one(train_rng)
            if _contains_forbidden(qa.q) or _contains_forbidden(qa.a):
                continue
            q = qa.q.strip()
            a = qa.a.strip()
            if not q or not a:
                continue
            k = _key(q)
            # Reading prompts have high entropy; safe to keep train unique.
            if k in train_keys:
                continue
            train_keys.add(k)
            f.write(json.dumps({"query": q, "answer": a}, ensure_ascii=False) + "\n")
            wrote += 1
            if wrote % 5000 == 0:
                dt = time.time() - t0
                print(f"train {wrote}/{train_n} elapsed {dt:.1f}s")

    # Valid: higher-quality distribution + a small curated suite; avoid overlap with train prompts.
    p_valid = Path(valid_path)
    mkdir(p_valid.parent)
    wrote_valid = 0
    seen_valid: set[bytes] = set()
    t0v = time.time()
    with p_valid.open("w", encoding="utf-8") as f:
        suite_rng = random.Random(seed + 999)
        suite_n = min(500, valid_n)
        for qa in _valid_suite(suite_rng, suite_n):
            if _contains_forbidden(qa.q) or _contains_forbidden(qa.a):
                continue
            q = qa.q.strip()
            a = qa.a.strip()
            if not q or not a:
                continue
            k = _key(q)
            if k in train_keys or k in seen_valid:
                continue
            seen_valid.add(k)
            f.write(json.dumps({"query": q, "answer": a}, ensure_ascii=False) + "\n")
            wrote_valid += 1

        # Fill the rest with the higher-quality valid generator.
        while wrote_valid < valid_n:
            qa = _gen_valid(valid_rng)
            if _contains_forbidden(qa.q) or _contains_forbidden(qa.a):
                continue
            q = qa.q.strip()
            a = qa.a.strip()
            if not q or not a:
                continue
            k = _key(q)
            if k in train_keys or k in seen_valid:
                continue
            seen_valid.add(k)
            f.write(json.dumps({"query": q, "answer": a}, ensure_ascii=False) + "\n")
            wrote_valid += 1
            if wrote_valid % 1000 == 0:
                dt = time.time() - t0v
                print(f"valid {wrote_valid}/{valid_n} elapsed {dt:.1f}s")

    print(f"wrote: {train_path} ({train_n})")
    print(f"wrote: {valid_path} ({wrote_valid})")


if __name__ == "__main__":
    main()
