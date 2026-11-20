# Qwen ACL E1: full version experiment, three repetitions

已完成的 E1 数据：**1,379 篇论文、3,627 个可获取正文的版本、10,881 条有效评分**。
每个可用版本有三次独立生成（`rep=0,1,2`）；1,377 篇有完整版本链。
原计划为 3,629 个版本、10,887 条；`2402.14228v3` 和 `2502.12509v4`
是无正文的撤稿记录，缺少相应六条评分，保留这两篇的其他历史版本。

## Mac 获取与校验

已有仓库时，在仓库目录执行：

```bash
git pull --ff-only origin main
python3 results/acl_qwen/e1/verify.py
```

首次获取：

```bash
git clone https://github.com/Candy26i/review_research.git
cd review_research
python3 results/acl_qwen/e1/verify.py
```

只需 Python 3 标准库，无需 GPU、模型或 Git LFS。
校验覆盖文件 SHA256、唯一键、版本/重复次数、原始响应、评分字段、输入标识和缺失项。
正文没有打包，因此本地校验输入哈希的对应关系；导出时另核对了全部 3,627 份实际正文。

## 直接读数据

在仓库目录运行以下 Python。`reviews.jsonl.gz` 同时包含数值评分和完整评语：

```python
import gzip
import json
from pathlib import Path

base = Path('results/acl_qwen/e1')
with gzip.open(base / 'reviews.jsonl.gz', 'rt', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]
print(len(rows))  # 10881
print(rows[0]['arxiv_id'], rows[0]['v'], rows[0]['rep'], rows[0]['rating'])
print(rows[0]['review']['summary'])
# 如已安装 pandas：df = pandas.DataFrame(rows)
```

唯一键为 `(arxiv_id, v, rep)`，ID 应保留字符串类型。`rating` 是 ICLR 的
`1,3,5,6,8,10` 分制；`soundness/presentation/contribution` 为 1–4，`confidence` 为 1–5。
`accept` 是模型输出的布尔建议，与按 rating 阈值推导的建议可能不同。
模型使用同一份 ICLR 评审提示评估 ACL 论文，原提示见 `review_prompt.md`。

| 文件 | 内容 |
| --- | --- |
| `reviews.jsonl.gz` | 10,881 条成功评分及解析后的完整 `review`；原字段保留，路径改为相对路径 |
| `raw_responses.tar.gz` | 10,881 份逐字节保留的 API 原始响应，含原始评语、usage、timings |
| `manifest.jsonl.gz` | 原始完整计划，含 1,379 篇论文标题、版本与公开标识 |
| `input_versions.jsonl.gz` | 3,627 个输入版本的 SHA256、词数和 regenerated 标记 |
| `run_configs.json` | 四种实际运行配置和原配置文件 SHA256 |
| `provenance.json` | 原始审计、来源哈希、撤稿证据、输入修复记录和导出转换说明 |
| `analysis.json`, `per_paper.jsonl`, `overview.png` | 已完成的 E1 描述统计、逐论文统计及图 |
| `SHA256SUMS`, `verify.py` | 校验清单和跨平台完整性校验脚本 |

提取逐字节原始响应（可选）：

```bash
cd results/acl_qwen/e1
tar -xzf raw_responses.tar.gz
```

评分行的 `raw_path` 相对于本目录，对应压缩包成员；`raw_sha256` 校验原始字节。
`source_log` 和审计中的 `data/...` 路径仅标识来源，源日志并未一起上传。
本包包括成功结果；原失败尝试排除，审计保留其计数。正文、模型权重以及后续 E2/E4 数据不在本包。

## 实验设置与解释边界

- 官方 `Qwen/Qwen3-30B-A3B-GGUF`，Q4_K_M，固定 revision
  `e4d4bafdfb96a411a163846265362aceb0b9c63a`，llama.cpp `b12428`。
- Temperature 1，关闭思考，输出上限 2,000 token，保持全文、不截断正文。
- 原始 32K 基线 10,110 条；补跑 32K 96 条、64K YaRN2 609 条、128K YaRN4 66 条。
  128K 将前八层 MoE 专家放 CPU；各配置均保留来源，不能视作完全相同的运行环境。
- 输入为 **regenerated**。历史脱敏正文和作者元数据缺失，不能称为与历史 DeepSeek
  输入逐字节一致的复现。统计文件中的历史对比仅为描述性对比，不代表公平模型排名。
- ACL 样本均为录用论文，没有人工评分或拒稿对照；“后一个版本”只是修订代理，
  不能由分数趋势直接推导评审准确率。不同正文提取方式也在审计中保留。
- `2502.11364v2 rep1` 原回答把 `strengths` 拼成 `strengthses`。原文和数值均保留，
  只有此已知叙述字段警告。两条基线记录曾进行 JSON 解析重试，详见原始审计。

导出前的原始 combined log SHA256：
`ce22eafab7b7ca053f7c3194b99ff8a40096e2851516059176c53ba7a0a393ec`。
可从具有本机原始数据的工作区运行 `python src/analysis/export_qwen_acl_e1.py` 重建数据包；
随后运行本目录 `verify.py`。导出程序不修改输入、源日志或实验配置。
