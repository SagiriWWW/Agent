# Aircraft Agent · 最小核心版

面向飞行遥测分析的本地 Agent 原型：接收规则库与 CSV/XLSX 数据，通过白名单工具完成画像、规则检查、异常分析、关联挖掘、结果校验与 HTML/Markdown 报告，并保存可核验的本地证据链。

此分支是 2026-11-11 从当前工作区提取的独立精简快照，保留核心源码及其依赖模块、配置、规则、核心回归测试和 40 行合成演示数据。原工作区、提交历史和远端 main 保留。未包含真实飞行数据、密钥、运行输出、研究实验与历史报告。

## 快速运行

Python 3.11+，在仓库根目录执行（使用源码可编辑安装，以便定位 configs、examples、dataset）：

```bash
git clone --branch codex/minimal-core --single-branch https://github.com/SagiriWWW/Agent.git
cd Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m aircraft_agent.interface.cli --demo
```

`--demo` 使用固定本地计划，无需模型 API。运行结果写入 `outputs/runs/`，CLI 输出报告位置。

```bash
# Web 页面：打开 http://127.0.0.1:8000
python -m uvicorn aircraft_agent.interface.app:app --host 127.0.0.1 --port 8000

# 核心回归
python -m pytest -q

# 校验已有运行；将路径替换为实际 run 目录
python -m aircraft_agent.interface.cli --verify-evidence-storage outputs/runs/<run_id> --json
```

## 核心结构

- `src/aircraft_agent/agent/`：计划执行、模型协议适配、工具调用循环与输入预算。
- `src/aircraft_agent/tools/`：工具契约、白名单注册与执行。
- `src/aircraft_agent/telemetry/`：解析、schema、规则应用、关联挖掘与报告。
- `src/aircraft_agent/evidence/`：哈希引用、追加日志、快照校验与显式恢复。
- `src/aircraft_agent/interface/`：CLI、FastAPI、历史报告及进度页面。
- `configs/`、`examples/`、`dataset/`：schema、规则和明确标注的合成样本。
- `tests/`：从完整项目保留的核心回归测试；`agent_mvp/` 保留旧导入兼容层。

关联挖掘默认使用安装依赖中的 mlxtend FP-growth；未携带可选的外部 Datamining 仓库。

## 模型调用与范围

可参考 `.env.example` 配置自有 OpenAI-compatible 服务；强制模型调用须设置 `AGENT_MVP_FORCE_MODEL_API=1`，并使用 `--rule examples/rules.yaml --dataset dataset/last_avg/synthetic_demo.csv`，不要同时使用会强制离线执行的 `--demo`。密钥只写入本地 `.env`。

离线流程跑通不证明模型能够自主完成任务。本快照未执行付费模型调用，也不提供真实故障准确率、独立飞行泛化或在线延迟结论。证据链校验只证明本地记录在当前哈希协议下的一致性，不提供签名或外部信任锚。

公开数据分析源码保留其内部依赖，但 `--public-data` 及对应 Web 入口需要完整项目的 RflyMAD 封存材料，本精简版不提供这些输入。批处理仅自带一份合成样本；更多数据需自行提供。
