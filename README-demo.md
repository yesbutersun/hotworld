# ASR 热词纠错 Demo

这个 demo 面向 **ASR 识别后的专有名词纠错**。生产环境假设只有：

- 一份热词表：`hot-world.txt`
- 一段 ASR 输出文本

`standard.txt` 不参与纠错，只用于离线评估当前 demo 的修复比例。

## 当前流程

```text
ASR text
  -> 按标点和长度切块
  -> 块内窗口扫描
  -> 窗口内部归一化：允许删除空白，不跨标点替换
  -> 多音字拼音变体生成
  -> e/E 和 一/易 发音等价扩展
  -> 拼音 2-gram 倒排索引召回热词候选
  -> 声母/韵母级加权编辑距离打分
  -> 字符 LCS、首字母、长度、2-gram 覆盖率联合打分
  -> 硬过滤和短词保守过滤
  -> 长热词优先的非重叠最高分替换
```

典型修复：

```text
飞马应用 -> 飞码应用
元芳平台 -> 圆方平台
云溪大模型 -> 云犀大模型
玄机大模型 -> 璇玑大模型
招银易报 / 招银 e 报 -> 招银e报
各贷预约 -> 个贷预约
```

## 文件

- `asr_hotword_corrector_demo.py`: 主纠错脚本。
- `hot-world.txt`: 热词表，一行一个热词。
- `asr-result-online-tts.txt`: online ASR 结果。
- `asr-result-offline-tts.txt`: offline ASR 结果。
- `corrected-*.txt`: 纠错后的文本。
- `corrections-*.jsonl`: 每条替换的详细打分报告。
- `evaluate_hotword_correction.py`: 基于 `standard.txt` 的离线命中率评估。
- `inspect_missed_hotwords.py`: 基于 `standard.txt` 分析仍未恢复的热词。
- `standard.txt`: 只作为 demo 评估答案，不作为生产纠错输入。

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

Online ASR:

```bash
.venv/bin/python asr_hotword_corrector_demo.py \
  --input asr-result-online-tts.txt \
  --output corrected-online.txt \
  --report corrections-online.jsonl
```

Offline ASR:

```bash
.venv/bin/python asr_hotword_corrector_demo.py \
  --input asr-result-offline-tts.txt \
  --output corrected-offline.txt \
  --report corrections-offline.jsonl
```

Direct text:

```bash
.venv/bin/python asr_hotword_corrector_demo.py \
  --text "在客户经营与服务领域，招财号 招呼群、云溪大模型等构建了多元化的客户互动阵地。"
```

`--text` 输入默认只输出纠错后的文本，不生成 txt 和 jsonl 文件。如果仍需要落盘，可以显式传 `--output corrected-direct.txt --report corrections-direct.jsonl`。

切块长度可配置，默认先按 `。！？!?；;` 和换行切块，再把长块切到最多 200 字符：

```bash
.venv/bin/python asr_hotword_corrector_demo.py \
  --input asr-result-online-tts.txt \
  --chunk-max-len 120
```

如果需要关闭某类切块，可以传 `--chunk-max-len 0` 关闭长度切块，或传 `--chunk-separators ""` 关闭标点切块。

Mock LLM rerank:

```bash
.venv/bin/python asr_hotword_corrector_demo.py \
  --input asr-result-online-tts.txt \
  --output corrected-online-mock-llm.txt \
  --report corrections-online-mock-llm.jsonl \
  --mock-llm
```

`--mock-llm` 会使用 `standard.txt` 做本地 demo 的答案裁判，只用于验证 LLM 接入链路和效果；生产环境应替换成真实 LLM，不依赖 `standard.txt`。

`--text` 和 `--input` 同时存在时，优先使用 `--text`。`--input` 文件输入默认输出 `corrected.txt` 和 `corrections.jsonl`；`--text` 文本输入默认输出纯文本到终端。

Real LLM rerank:

```bash
export ASR_LLM_API_KEY="..."
export ASR_LLM_MODEL="..."

.venv/bin/python asr_hotword_corrector_demo.py \
  --input asr-result-online-tts.txt \
  --output corrected-online-llm.txt \
  --report corrections-online-llm.jsonl \
  --llm
```

`--llm` 会把规则层召回的少量候选交给 OpenAI-compatible Chat Completions 接口做二阶段裁决。默认接口地址来自 `ASR_LLM_API_URL` / `OPENAI_BASE_URL`，未设置时使用 `https://api.openai.com/v1/chat/completions`。模型只判断候选是否替换，不允许自由生成新热词。

热词表可选增加类别，格式为：

```text
云犀大模型	SYSTEM
招银e报	APP
```

不写类别时默认 `GENERAL`。

## 技术细节

### 1. 文本归一化

候选窗口保留原始 span，但打分时会去掉热词内部空格和轻标点：

```text
原始 span: 个贷 预约
归一 source: 个贷预约
替换结果: 个贷预约
```

这能处理：

```text
人 行监管
对公 线上作业
招银 e 报
```

连续可跳过字符最多 2 个，避免跨越太远导致误召回。

### 2. 多音字拼音变体

使用 `pypinyin` 生成多音字读音：

```python
pinyin(text, style=Style.NORMAL, heteronym=True)
```

例如：

```text
行 -> xing / hang
重 -> zhong / chong
```

每个片段最多保留 24 个拼音变体，避免组合爆炸。

### 3. e/E 与 一/易 等价

ASR 经常把英文 `e/E` 识别成 `一` 或 `易`。脚本在拼音层把：

```text
e <-> yi
```

视作等价音节，因此：

```text
招银易报 -> 招银e报
招银 e 报 -> 招银e报
一餐通 -> E餐通
```

可以进入候选。

### 4. 拼音 2-gram 倒排索引

热词先转成拼音，再切成 2-gram。

例如：

```text
招乎开放平台 -> zhao hu kai fang ping tai
```

生成：

```text
zhao_hu
hu_kai
kai_fang
fang_ping
ping_tai
```

倒排索引可以理解为：

```text
zhao_hu -> 招乎, 招乎群, 招乎开放平台, ...
hu_kai  -> 招乎开放平台
```

扫描 ASR 片段时，先用片段的拼音 2-gram 找候选热词，再进入精排，不需要和全量热词逐个比较。

多音字会产生多个拼音变体。`gram_coverage` 不使用所有变体 2-gram 的并集做分母，而是对每个变体单独算覆盖率后取最大值，避免多音字把覆盖率稀释。

这对长热词很重要。例如：

```text
元芳平台 -> 圆方平台
```

两者拼音都是：

```text
yuan fang ping tai
```

按单个变体计算覆盖率后可以稳定召回。

### 5. ASR confusion 规则

脚本内置常见近音混淆：

```python
n <-> l
zh <-> z
ch <-> c
sh <-> s
in <-> ing
en <-> eng
an <-> ang
f <-> h
r -> l / y
```

这些规则不直接替换文本，只降低拼音编辑距离中的替换成本。

例如：

```text
宁静系统 -> 灵境系统
```

`n/l` 是低成本混淆，比普通替换更容易通过。

### 6. 加权音节编辑距离

普通编辑距离只知道字符是否相同。这里先把拼音音节拆成声母和韵母，再计算替换成本：

```text
完全相同: 0.0
e 和 yi: 0.0
confusion 内混淆: 0.25
普通声母/韵母替换: 1.0
插入/删除一个音节: 1.0
```

相似度：

```text
phonetic_similarity =
  1 - weighted_edit_distance / max(source_len, target_len)
```

### 7. 综合打分

每个候选替换计算这些特征：

```text
phonetic_similarity       拼音/音素相似度
char_similarity           字符 LCS 相似度
first_initial_similarity  首音节声母相似度
gram_coverage             拼音 2-gram 覆盖率
length_score              长度相似度
```

当前无上下文版本的最终分数：

```text
score =
  0.50 * phonetic_similarity
+ 0.20 * char_similarity
+ 0.10 * first_initial_similarity
+ 0.10 * gram_coverage
+ 0.10 * length_score
```

默认阈值：

```text
score >= 0.82
phonetic_similarity >= 0.72
gram_coverage >= 0.34
source 与 target 归一化长度必须相同
source 不能已经是热词表中的正确热词
```

### 8. 短词保守过滤

去掉上下文后，短词同音误改风险最高。例如：

```text
思想家 -> 私享家
宜事通 -> 移事通
马云 -> 码云
飞马 -> 飞码
```

如果只看拼音，它们分数会很高，但生产环境里不能确认是否应该替换。

因此当前规则是：

```text
3 字及以下不做规则层自动替换
纯空白删除后精确命中热词的格式修正除外
```

允许自动修正的例子：

```text
人 行 -> 人行
招银 e 报 -> 招银e报
对公 线上作业 -> 对公线上作业
```

不再维护短词白名单。短词候选应进入 LLM 二阶段裁决，或只输出建议，不由规则层直接改。

### 9. 长热词优先

候选 proposal 按以下字段降序排序：

```text
score
target 热词长度
span 长度
char_similarity
```

然后贪心选择不重叠的最高分替换。

这能解决短热词和长热词重叠时的竞争问题：

```text
元芳平台 -> 圆方平台
```

应该优先替换完整长词，而不是只替换：

```text
元芳 -> 圆方
```

### 10. 已命中热词保护

如果 ASR 原文中已经完整出现某个热词，脚本会保护这个 span，不允许内部短片段被替换。

例如：

```text
联合贷
```

中间的：

```text
合贷
```

和：

```text
个贷
```

拼音相同。没有保护时可能误改成：

```text
联个贷
```

当前保护会保留 `联合贷`。

## 输出报告

`corrections-online.jsonl` 和 `corrections-offline.jsonl` 每行是一条替换：

```json
{
  "start": 0,
  "end": 4,
  "source": "元芳平台",
  "target": "圆方平台",
  "score": 0.9,
  "phonetic_similarity": 1.0,
  "char_similarity": 0.5,
  "first_initial_similarity": 1.0,
  "gram_coverage": 1.0,
  "decision_source": "rule",
  "llm_reason": "",
  "hotword_category": "GENERAL",
  "llm_confidence": 0.0
}
```

这个文件适合人工抽检，也适合后续接 LLM 做二阶段裁决。

## Evaluation

评估命令：

```bash
.venv/bin/python evaluate_hotword_correction.py \
  --before asr-result-online-tts.txt \
  --after corrected-online.txt
```

当前无上下文版本结果：

```text
online:
纠错前: 347 / 537 = 64.62%
纠错后: 481 / 537 = 89.57%
漏词修复率: 70.53%
误伤已命中热词: 0

offline:
纠错前: 340 / 537 = 63.31%
纠错后: 474 / 537 = 88.27%
漏词修复率: 68.02%
误伤已命中热词: 0
```

Mock LLM rerank 结果：

```text
online:
纠错前: 347 / 537 = 64.62%
纠错后: 523 / 537 = 97.39%
漏词修复率: 92.63%
误伤已命中热词: 0

offline:
纠错前: 340 / 537 = 63.31%
纠错后: 516 / 537 = 96.09%
漏词修复率: 89.34%
误伤已命中热词: 0
```

这个评估口径是：`standard.txt` 中应出现的热词，在纠错文本中是否精确出现。它衡量的是热词恢复率，不等同于人工标注的 precision/recall。

生产环境没有 `standard.txt` 时，建议保留同样的 `corrections-*.jsonl` 报告，对线上样本做人工抽检，持续维护阈值、热词类别和 LLM 裁决策略。

## 为什么去掉上下文后召回会下降

以前有标准文本上下文时，可以根据参考句子判断某个同音词是否应该替换。去掉这层后，只剩下热词表、拼音和字符相似度。

因此一些词会被保守放弃：

```text
思想家 -> 私享家
李沈毅 -> 离审易
金建 -> 津谏
疾风 -> 极风
```

这些词要么很短，要么拼音高度相似但字符差异大。没有上下文时强行替换，容易误伤真实文本。

## Important Parameters

- `--threshold`: 替换置信度阈值，默认 `0.82`。越高越保守。
- `--max-extra-len`: 原始窗口允许比最长热词多扫描多少字符，默认 `1`。只影响扫描窗口，不允许最终 source/target 长度不同。

## LLM Reranking

当前已经预留了 LLM 二阶段裁决入口：

```text
collect_proposals()
  -> mock_llm_rerank() / real_llm_rerank()
  -> select_non_overlapping()
```

LLM 不应该直接搜索全量热词表，而应该只裁决规则层已经召回的少量 proposal。当前实现里，3 字及以下短词不会由规则层自动替换，但会在 `--mock-llm` 打开时作为 `llm_candidate` 送入 mock LLM。

建议给 LLM 的输入：

```text
当前句子上下文
ASR 片段
候选热词
规则打分
热词类别，例如 APP / ORG / CONTACT / GENERAL
```

建议输出结构化结果：

```json
{"decision": "replace", "target": "码云", "reason": "代码库上下文中码云更合理"}
```

当前 mock LLM 的行为：

```text
候选替换后的局部上下文
  -> 与 standard.txt 中 target 热词的局部上下文做相似度比较
  -> 相似度达到阈值则模拟 LLM 返回 replace
```

这不是生产逻辑，只是为了在没有真实 LLM 时验证链路。生产替换点是 `real_llm_rerank()`，真实 LLM 应基于当前句子上下文、候选热词、规则分数、热词类别做结构化裁决。

真实 LLM 接受的返回格式为：

```json
{"decision": "replace", "target": "云犀大模型", "confidence": 0.91, "reason": "上下文描述大模型产品"}
```

## 是否需要 RAG

当前纯规则版本不需要 RAG。

如果生产环境有大量历史语料、FAQ、系统说明、热词使用样例，可以把 RAG 放在 LLM 裁决阶段，而不是放在第一层召回：

```text
拼音倒排索引召回候选热词
  -> 对高风险候选检索历史相似句
  -> LLM 结合当前句子和历史样例裁决
```

不建议用联网搜索做生产纠错依据。内部系统名、产品名和组织名通常不在公网稳定存在，联网搜索会带来延迟、隐私和不可复现问题。

## Current Design Tradeoffs

当前 demo 偏保守：

- 只依赖热词表和 ASR 文本，不依赖 `standard.txt`。
- 不处理长度不一致的复杂替换。
- 3 字及以下短词不做规则层自动替换。
- 已正确出现的热词不会被替换成另一个同音热词。
- 默认无 LLM、无联网，结果可离线复现。
- `--mock-llm` 仅用于本地 demo，不是生产逻辑。

继续提升召回率的优先方向：

1. 为热词增加类别，按 APP/ORG/CONTACT/GENERAL 设置不同阈值。
2. 对短词和高风险候选接入 LLM/BERT 二阶段裁决。
3. 有内部语料时，引入 RAG 给 LLM 提供热词使用样例。
4. 对长度不一致候选只输出建议，避免规则层直接替换。
