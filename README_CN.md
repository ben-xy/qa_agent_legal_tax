# 新加坡法律与税务问答代理

[EN](README.md) | [中文](README_CN.md)

## 概览

本项目是一个面向新加坡法律、税务与合规场景的智能问答代理。系统采用 RAG（检索增强生成）架构，结合关键词检索与向量检索的混合检索方案。

### 数据流流程

```
用户问题
    ↓
问题分类（税务/财务/合规/通用）
    ↓
文档检索（混合检索：BM25 + 向量检索 + Rerank）
    ↓
上下文整理
    ↓
LLM 生成答案（Gemini）
    ↓
答案校验与法律引用提取
    ↓
置信度评分
    ↓
结果格式化
    ↓
展示给用户
```

### 核心组件

1. **QAAgent**：主流程编排器
2. **HybridRetriever**：从 acts_chunked/ 目录检索文档
3. **LLMService**：LLM 服务集成（Gemini 和 ChatGPT）
4. **LegalValidator**：答案校验与法律引用提取
5. **Configuration System**：基于环境变量的配置管理

## 功能特性

- **法律与税务问答**：回答新加坡法案、监管与税务相关问题
- **混合检索**：融合 BM25 关键词检索与向量语义检索
- **双模型提供方**：同时支持 OpenAI 与 Google Gemini（LLM + Embedding）
- **引用提取**：自动提取并展示法律引用
- **置信度评分**：输出答案置信度
- **会话历史**：支持历史上下文管理

## 项目结构

```
qa_agent_legal_tax/
├── src/
│   ├── agents/                 # QA 与报告代理
│   ├── services/               # LLM、Embedding 与文档服务
│   ├── retrievers/             # 文档检索逻辑
│   ├── validators/             # 答案与报告校验
│   ├── utils/                  # 日志、文本处理工具
│   └── models/                 # 数据模型
├── data/
│   ├── acts_chunked/           # 分块后的法律文档
│   ├── acts_embedding/         # 向量嵌入数据
│   └── qa_pairs/               # 生成的问答数据
├── scripts/                    # 工具脚本
├── docs/                       # 项目文档
│   ├── eval_ground_truth_generation.md
│   ├── metrics_eval_report.md
├── notebooks/                  # Notebook Web UI
├── tests/                      # 单元测试
├── config.py                   # 配置管理
├── main.py                     # 启动入口
└── requirements.txt            # 依赖列表
```

## 文档说明

以下为 `docs/` 目录下维护的正式文档：

- `docs/eval_ground_truth_generation.md`：说明如何生成 `data/qa_pairs/eval_ground_truth.jsonl`，以及评估数据的输入/输出字段规范。
- `docs/metrics_eval_report.md`：说明检索与生成指标定义，汇总最近实验结果，并给出诊断与改进建议。

## 安装与快速开始

1. 克隆仓库：

```bash
git clone <repository-url>
cd qa_agent_legal_tax
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置环境变量：

```bash
cp .env.example .env
# 在 .env 中填写你的 API Key
```

4. 在 `.env` 中选择模型提供方：

```bash
# 方案 A：OpenAI
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# 方案 B：Gemini
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

5. 验证并运行：

```bash
python scripts/test_system.py
python scripts/demo_agent.py
python main.py
```

## 使用方式

### 交互式命令行

```bash
python main.py
```

示例：

```
You: What are the tax filing deadlines for Singapore companies?
Agent: <Answer with legal citations and sources>

You: How do I prepare a balance sheet according to SFRS?
Agent: <Financial reporting guidance>

You: history
Agent: <Show conversation history>

You: quit
```

### Web UI（Jupyter Notebook）

该 UI 与 `QAAgent` 主流程一致（基于 Gradio）：

```bash
jupyter notebook notebooks/web_ui.ipynb
```

然后：

1. 按顺序执行所有单元
2. 打开输出中的本地 Gradio 地址（通常是 `http://127.0.0.1:7860`）
3. 在浏览器里输入问题

UI 支持：

- 聊天式交互
- 置信度/耗时展示
- 来源与法律引用展示
- 清空历史会话

### 代码调用方式

```python
from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator

config = get_config().to_dict()
retriever = HybridRetriever(config)
llm = LLMService(config)
validator = LegalValidator(config)

agent = QAAgent(retriever, llm, validator, config)

response = agent.process_query("What is the GST rate in Singapore?")
print(response.answer)
print(f"Confidence: {response.confidence_score:.1%}")
```

## 配置说明

可在 `.env`（或 `config.py`）中设置：

- `LLM_PROVIDER`：`openai` 或 `gemini`（默认 `gemini`）
- `EMBEDDING_PROVIDER`：`openai` 或 `gemini`（默认 `gemini`）
- `OPENAI_API_KEY`：使用 OpenAI 时必填
- `GOOGLE_API_KEY`：使用 Gemini 时必填
- `LLM_MODEL`：Gemini 对话模型（默认 `emini-2.5-flash`）
- `GEMINI_LLM_MODEL`：Gemini 对话模型（默认 `gemini-2.5-flash`）
- `EMBEDDING_MODEL`：OpenAI embedding 模型（默认 `text-embedding-3-small`）
- `GEMINI_EMBEDDING_MODEL`：Gemini embedding 模型（默认 `models/text-embedding-004`）
- `RETRIEVAL_TOP_K`：检索文档数量（默认 5）
- `LOG_LEVEL`：日志等级（DEBUG、INFO、WARNING、ERROR）

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码质量

```bash
black src/
flake8 src/
```

### 日志

日志会写入 `logs/qa_agent.log`，并输出到控制台。

### 常见问题

- `No documents available for retrieval`
  - 请确认 `data/acts_chunked/` 中有分块 JSON。
- `OPENAI_API_KEY` 或 `GOOGLE_API_KEY` 报错
  - 请确认 provider 选择与对应 key 一致且配置正确。
- `Module not found`
  - 重新安装依赖：`pip install -r requirements.txt`。

## 数据流水线

### 1) 文档分块

新加坡法律文档分块后存储于 `data/acts_chunked/`。

### 2) 向量生成

按 `EMBEDDING_PROVIDER` 选择模型生成向量，并存储到 `data/acts_embedding/`。

### 3) 检索

混合检索由以下两部分组成：

- **BM25**：关键词相关性检索
- **Vector Search**：语义相似度检索

### 4) 答案生成

- 先检索并整理上下文
- 再由 `LLM_PROVIDER` 对应模型生成答案
- 最后提取并格式化引用

## 性能

- 平均响应时间：< 5 秒
- 置信度区间：60% - 95%
- 支持 50+ 新加坡法律与监管文档

## 限制

- 需提供所选 provider 的有效 API Key（`OPENAI_API_KEY` 或 `GOOGLE_API_KEY`）
- 回答质量依赖文档质量与问题表达
- 涉及关键法律决策时，建议咨询专业法律人士

## 后续规划

- [X] 多语言支持
- [X] Rerank
- [X] Evaluation Metrics
- [X] Web UI（Notebook + Gradio）
- [ ] API 接口
- [ ] 法律领域微调模型
- [ ] 财务报表生成能力
- [ ] 交互式文档上传

## 许可证

MIT License
