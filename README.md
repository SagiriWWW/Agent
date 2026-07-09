# Agent 编排器 + 本地工具箱的数据处理与规则挖掘 MVP

这个 MVP 采用“Agent 为主结构，外围挂本地工具箱”的架构：Agent 负责任务计划、工具路由、结果校验和报告组织；本地工具箱负责数据画像、Schema/数据适配审查、缺失值检查、基础离群值提示、规则命中、关联规则挖掘和相关性分析；每个文件、Agent 步骤、工具调用和报告生成都会写入证据链。

没有配置模型 API 时，系统会使用本地 deterministic Agent demo 模式，仍然展示完整的“计划 -> 工具调用 -> 观察 -> 校验 -> 报告”闭环。

详细验收说明见 [docs/MVP验收说明.md](docs/MVP验收说明.md)。

## 目录结构

- `agent_mvp/`：Agent 编排器、工具箱、报告和 Web/API 入口。
- `configs/`：稳定配置文件，目前包含 `telemetry_schema.yaml` 字段展示配置和 `schema_semantics.yaml` 跨数据集语义角色配置。
- `dataset/`：当前 Demo 使用的数据集，保留 `last_avg` 和 `last_last` 两类数据。
- `examples/`：内置规则库样例。
- `tests/`：本地单元测试和离线 Agent 测试。
- `third_party/Datamining/`：轻量 FP-growth-only 关联规则后端。
- `docs/`：项目书、设计方案、验收说明和阶段记录的统一归档目录。

## 能力范围

- 支持数据集：CSV、XLSX
- 客户数据分析页支持规则库：MD、YAML/YML；其他规则来源需先转换为这两类格式
- 支持前端按任务选择模型策略：本地稳定模式、外部模型优先、强制外部模型；未配置 API Key 时使用本地 deterministic 演示流程
- 关联规则挖掘统一使用 FP-growth 生成树方法，并带最小命中数与小样本提示
- 内置 Demo 使用结构化 YAML 规则库，支持严重程度、建议动作、标签和复核要求
- 提供 `examples/rules.yaml` v1 基线规则库和 `examples/rules_v2.yaml` v2 优化草案，支持按批次生成规则版本对比报告
- 提供字段语义建议工具，把原始遥测字段映射为中文名、分组、角色和忽略字段建议
- 提供 `review_schema_mapping` Schema 语义映射审查，基于稳定语义角色、别名、单位和统计口径线索为未知客户字段生成候选映射
- 提供 `review_dataset_adaptation` 数据适配审查，在规则执行前区分可直接执行、需映射确认和不适用规则，避免只围绕当前 `last_avg` 样例数据调阈值
- 提供 `build_mapping_execution_plan` 映射执行计划，把已确认映射转成可审查字段重写预案，但不自动修改规则和阈值
- 提供 `validate_mapping_regression` 映射回归验证，检查已确认字段映射生成的重写预案是否能透明执行，但不证明阈值业务正确
- 提供 `review_generalization_readiness` 泛化验收门禁，检查系统是否把候选映射、不适用规则或当前数据特征误当作自动诊断结论
- 提供批量泛化门禁汇总，把多个 run 的 `generalization_readiness` 聚合为跨数据集反过拟合证据
- 提供显式 Agent 计划、Thought/Action/Observation 轨迹和 `validate_analysis` 结果校验
- 提供 `/demo-ui` 标准演示场景、`/runs/new` 客户数据分析页、`/batch-ui` 批量验收页、`/rule-versions` 规则优化页、`/review-promotions` 复核沉淀页、`/toolbox-ui` 分析能力中心，以及 `/demo`、`/batches`、`/api/toolbox` 等接口
- 提供 `/runs/<run_id>/progress` 运行进度页，通过 SSE 事件流展示 Agent 步骤、工具状态、即时输出和结果入口，轮询作为兜底
- 提供批量验收能力：遍历 `dataset/last_avg/`，为每个数据集生成独立 run，并生成批量汇总报告、批量数据适配审查和规则稳定性分析
- 提供规则库校验页、Schema 复核页和数据适配审查页，支持查看与下载复核产物
- 输出 Markdown/HTML 报告、可视化看板和 `evidence_chain.json`

第一版不做本地大模型部署、模型微调、生产级权限系统、流式遥测、涉密数据处理和独立异常检测项目接入。

## 当前内置数据选择

项目根目录的 `dataset/` 下有三类 CSV：

- `dataset/last_avg/`：窗口均值特征，字段后缀为 `_avg`
- `dataset/last_last/`：窗口末值特征，字段后缀为 `_last`
- `dataset/customer_variants/`：客户结构变体小样本，用于验证字段别名、统计口径变化和缺字段时的 Schema 语义映射/数据适配行为

`last_avg` 和 `last_last` 的文件数量、行数分布和字段语义基本一致，主要差异是统计口径不同。当前标准 Demo 仍以 `last_avg` 这一类均值遥测数据为主，因为均值特征更适合规则阈值、数据质量提示和关联模式探索，受瞬时噪声影响更小；`customer_variants` 用于验证泛化适配，不用于证明规则阈值业务正确。

`last_avg` 是当前 MVP 的内置验证样例，不是对未来客户数据结构的假设。后续客户数据可能在字段名、单位、采样口径、标签列和业务阶段上差异很大，因此新能力应优先提升字段语义映射、规则适用性检查和不兼容数据提示。

内置 Demo 使用：

- 规则库：`examples/rules.yaml`
- 数据集：`dataset/last_avg/all_features_10_db_2022_12_29_165551_last_avg.csv`

该数据集是 50 列飞行遥测特征，包含电池、电压、电流、姿态、振动、气压、温度、速度等字段。`label` 全部为 0，`diagnosis` 为空，因此当前报告不做有监督分类解释，而是做“规则命中 + 数据质量提示 + 关联挖掘”的本地工具箱分析。

## 字段语义配置

当前项目把“字段中文名/字段分组/忽略字段”独立为 Schema 配置和工具能力，而不是写死在报告模板里：

- 稳定配置：`configs/telemetry_schema.yaml`
- 语义角色配置：`configs/schema_semantics.yaml`
- 确认映射配置：`configs/schema_mapping_overrides.yaml`
- 工具注册名：`suggest_schema_metadata`
- 语义映射工具：`review_schema_mapping`
- 每次运行产物：`runtime/<run_id>/artifacts/schema_suggestions.yaml`、`schema_suggestions.json`、`schema_mapping.json` 和 `schema_mapping.md`

`suggest_schema_metadata` 会结合稳定配置、数据集字段和规则绑定字段，生成可人工复核的字段语义草案。`review_schema_mapping` 会进一步把数据字段映射到稳定语义角色，输出单位、统计口径、别名命中和规则字段候选映射。人工确认后的字段语义或规则字段映射可以写入 `schema_mapping_overrides.yaml`，下次运行会显示为“已确认映射”，而不是重新作为普通候选出现。报告和可视化会优先使用中文字段名展示，但底层数据计算仍使用原始字段名，避免因为中文展示影响规则执行、关联规则挖掘和证据链复核。`review_dataset_adaptation` 会读取语义映射结果，在规则执行前输出字段精确匹配、已确认映射、后缀口径不一致、语义候选和缺失字段；候选映射不会自动改写规则或阈值，必须先人工确认。已确认映射代表字段关系已经沉淀，下一步仍需生成映射执行计划和回归验证，不能直接等同于阈值、单位换算和统计窗口完全正确。

## 结构化规则库

内置 Demo 默认使用 `examples/rules.yaml`。每条规则建议包含：

- `rule_id`、`name`、`field`、`operator`、`value`
- `severity`：`critical`、`high`、`medium`、`low` 或 `info`
- `rationale`：规则解释
- `recommendation`：命中后的建议动作
- `tags`：规则分组标签

运行时会调用 `validate_rule_library` 生成 `rule_validation.json` 和 `rule_validation.md`，检查字段是否存在、条件能否执行、严重程度是否合法、建议动作是否齐全；随后调用 `review_dataset_adaptation` 生成 `dataset_adaptation.json` 和 `dataset_adaptation.md`，在规则执行前说明哪些规则可直接执行、哪些需要字段映射确认、哪些不适用于当前数据。`review_schema_mapping_candidates` 会生成 `schema_mapping_review.json` 和 `schema_mapping_review.md`，把候选映射分为“可考虑确认”“需专家复核”“不适用”和“已安全处理”，辅助人工复核但不写入决策或全局配置。`build_mapping_execution_plan` 会生成 `mapping_execution_plan.json` 和 `mapping_execution_plan.md`，把已确认映射转成可审查字段重写预案，并标出单位、统计口径和回归验证风险。`validate_mapping_regression` 会生成 `mapping_regression_validation.json` 和 `mapping_regression_validation.md`，验证重写后的规则条件是否能透明执行，并把需确认、需复核、失败和缺证据项显式暴露。`review_generalization_readiness` 会再生成 `generalization_readiness.json` 和 `generalization_readiness.md`，作为反过拟合门禁：只要仍有候选映射、已确认但未完成回归验证的映射、不适用规则或不可解析规则，就不会允许自动诊断解释。这样规则库可以逐步从样例配置过渡为可复核的专家知识库。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

配置模型 API。默认是 OpenAI；如果要使用百炼，把 provider 改成 `bailian`。

```bash
cp .env.example .env
# 编辑 .env；项目启动时会自动读取当前工作目录下的 .env。
# 也可以在当前 shell 中手动 export 对应变量，export 的值优先级更高。
```

OpenAI 示例：

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.5"
```

百炼/通义千问示例：

```bash
export LLM_PROVIDER=bailian
export DASHSCOPE_API_KEY="sk-..."
export BAILIAN_MODEL="qwen-plus"
export BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

如果你使用美国（弗吉尼亚）地域，可把 `BAILIAN_BASE_URL` 改成：

```bash
export BAILIAN_BASE_URL="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
```

如果你使用新加坡地域，需要按百炼文档填入自己的 WorkspaceId：

```bash
export BAILIAN_BASE_URL="https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
```

没有 key 时也能跑本地演示流程。

## 关联规则挖掘引擎

`mine_association_rules` 工具以 FP-growth 为唯一关联规则挖掘路径。项目当前已在 `third_party/Datamining` 放置轻量 FP-growth-only 后端，系统会自动扫描该路径，并优先复用其中的 FP-growth 调用与同传感器过滤逻辑。

如果你把同样结构的 Datamining FP-growth-only 后端放在其他位置，可以配置：

```bash
export DATAMINING_REPO_PATH="/path/to/Datamining"
```

未配置 `DATAMINING_REPO_PATH` 且项目内没有 `third_party/Datamining` 时，系统会使用本项目依赖中的 `mlxtend` FP-growth 实现；如果 Datamining FP-growth-only 后端和 mlxtend 都不可用，则会退回本地 demo fallback，输出格式仍保持一致。

如果 Datamining 的 FP-growth 核心文件位置不在默认 `scripts/core/` 扫描范围内，也可以直接指定：

```bash
export DATAMINING_FPGROWTH_CORE_PATH="/path/to/fpgrowth_core.py"
```

关联规则工具默认使用偏保守的 MVP 阈值：

- `min_support=0.2`
- `min_confidence=0.7`
- `min_lift=1.2`
- `min_count=2`
- `max_rules=15`

其中 `min_count` 表示一条规则至少要命中多少行才会输出。内置遥测 Demo 会进一步把关联挖掘限制在核心遥测字段和前 2000 行，并使用 `min_count=10`，避免时间戳、空标签或偶然组合影响报告。

可以用下面命令检查当前实际后端：

```bash
.venv/bin/python - <<'PY'
from agent_mvp.datamining_adapter import datamining_backend_status
print(datamining_backend_status())
PY
```

正常集成后应看到：

```text
engine: datamining_fpgrowth
summary: SagiriWWW/Datamining FP-growth 后端可用。
```

注意：如果直接使用系统 Python，而不是项目 `.venv`，可能因为缺少 `mlxtend` 导致 Datamining 核心脚本加载失败。建议先执行 `source .venv/bin/activate`，或直接使用 `.venv/bin/python` 运行命令。

## Web 运行

```bash
.venv/bin/uvicorn agent_mvp.app:app --reload --host 127.0.0.1 --port 8000
```

如果修改了 `.env`，请重启 uvicorn 进程。当前配置是在应用启动时读取的，不会在页面刷新时自动重新加载。

打开：

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/demo-ui
http://127.0.0.1:8000/runs/new
http://127.0.0.1:8000/batch-ui
http://127.0.0.1:8000/rule-versions
http://127.0.0.1:8000/review-promotions
http://127.0.0.1:8000/toolbox-ui
```

首页是客户演示工作台；`/demo-ui` 展示标准演示场景；`/runs/new` 用于客户数据分析；`/batch-ui` 用于多数据集批量验收；`/rule-versions` 用于规则库版本验证和 v1/v2 对比；`/review-promotions` 用于把多次人工复核意见聚合成全局配置候选；`/toolbox-ui` 展示分析能力中心。客户数据分析页只开放 MD/YAML/YML 规则库和 CSV/XLSX 数据集上传，后端也会做同样校验。点击“启动演示分析”即可使用结构化规则库 `examples/rules.yaml` 和 `dataset/last_avg/all_features_10_db_2022_12_29_165551_last_avg.csv` 跑完整闭环。运行会先进入进度页，页面通过事件流刷新 Agent 步骤、工具开始/完成状态和即时输出；完成后会出现分析报告、可视化看板、规则审核和字段审核入口。标准演示场景和批量验收会强制使用本地计划模式，避免被外部模型 API 影响；客户数据分析页可以为本次任务选择本地稳定模式、外部模型优先或强制外部模型，并填写模型服务商和模型名称，密钥与 Base URL 仍从 `.env` 读取。

命令行批量验收：

```bash
.venv/bin/python -m agent_mvp.cli --batch-last-avg --max-datasets 5
.venv/bin/python -m agent_mvp.cli --batch-last-avg --rule examples/rules_v2.yaml --max-datasets 5
.venv/bin/python -m agent_mvp.cli --batch-dir customer_variants --rule examples/rules_v2.yaml --max-datasets 3
```

不传 `--max-datasets` 时会处理所选目录下全部数据集。`--batch-last-avg` 等价于使用 `dataset/last_avg/`；`--batch-dir` 可以传 `last_avg`、`customer_variants` 或一个本地目录路径。批量产物写入 `runtime/batches/<batch_id>/`，每个数据集仍会生成独立 `runtime/<run_id>/`。批量任务完成后，可以打开 `/batches/<batch_id>/dataset-adaptation` 查看批量数据适配审查，打开 `/batches/<batch_id>/schema-mapping-review` 查看批量映射复核辅助，打开 `/batches/<batch_id>/review-decisions` 查看批量人工复核意见汇总，打开 `/batches/<batch_id>/mapping-regression` 查看批量映射回归验证，打开 `/batches/<batch_id>/generalization-readiness` 查看批量泛化门禁，也可以打开 `/batches/<batch_id>/rule-analysis` 查看规则稳定性分析；对应文件会写入 `batch_dataset_adaptation.*`、`batch_schema_mapping_review.*`、`batch_review_decisions.*`、`batch_mapping_regression.*`、`batch_generalization_readiness.*` 和 `batch_rule_analysis.*`。客户变体批次的预期通常是“需映射确认/需复核”，不是全部规则直接执行。

工具箱清单：

```text
http://127.0.0.1:8000/api/toolbox
```

运行完成后，报告页会出现复核入口：

```text
http://127.0.0.1:8000/runs/<run_id>/progress
http://127.0.0.1:8000/api/runs/<run_id>/events
http://127.0.0.1:8000/runs/<run_id>/rule-validation
http://127.0.0.1:8000/runs/<run_id>/schema-review
http://127.0.0.1:8000/runs/<run_id>/schema-mapping
http://127.0.0.1:8000/runs/<run_id>/schema-mapping-review
http://127.0.0.1:8000/runs/<run_id>/dataset-adaptation
http://127.0.0.1:8000/runs/<run_id>/mapping-execution-plan
http://127.0.0.1:8000/runs/<run_id>/mapping-regression
http://127.0.0.1:8000/runs/<run_id>/generalization-readiness
http://127.0.0.1:8000/batches/<batch_id>/schema-mapping-review
http://127.0.0.1:8000/batches/<batch_id>/review-decisions
http://127.0.0.1:8000/batches/<batch_id>/mapping-regression
http://127.0.0.1:8000/batches/<batch_id>/generalization-readiness
```

`/runs/<run_id>/schema-mapping` 和 `/runs/<run_id>/schema-mapping-review` 都支持对规则字段候选映射进行“确认/驳回”。人工意见会先保存到本次运行的 `review_decisions.json`，并刷新 Schema 语义映射、数据适配审查、映射执行计划、映射回归验证、泛化门禁和报告；它不会自动写入全局 `schema_mapping_overrides.yaml`，避免为了单个客户样本过拟合。

`/batches/<batch_id>/review-decisions` 会只在当前批次范围内汇总各 run 的 `review_decisions.json`，标记“可进入推广审核”“确认/驳回冲突”“缺少回归证据”“批内证据不足”和“仅驳回”。它不自动推广配置，只为后续 `/review-promotions` 的跨运行沉淀提供批内证据队列。

`/review-promotions` 会扫描各个 run 的 `review_decisions.json` 和 `mapping_regression_validation.json`，按照确认次数、不同数据指纹数量、映射回归通过证据和是否存在驳回冲突生成复核沉淀报告。默认只有至少 2 次确认、至少 2 个不同数据指纹、至少 2 次跨数据映射回归通过且无驳回冲突的映射，才会进入“可推广审核”候选；系统只生成 `runtime/review_promotions/review_decision_promotions.*` 和 YAML 预览，不自动写入全局配置。

每次运行目录还会生成 `runtime/<run_id>/run_events.jsonl`。这是当前 Agent 运行过程协议的落地文件，包含 `run_created`、`file_ingested`、`plan_created`、`step_started`、`tool_started`、`tool_finished`、`step_finished`、`run_completed` 和 `run_failed` 等事件。

对应产物可通过受限 artifact 路由下载，例如：

```text
http://127.0.0.1:8000/runs/<run_id>/artifacts/rule_validation.json
http://127.0.0.1:8000/runs/<run_id>/artifacts/schema_review.json
http://127.0.0.1:8000/runs/<run_id>/artifacts/dataset_adaptation.json
http://127.0.0.1:8000/runs/<run_id>/artifacts/mapping_execution_plan.json
http://127.0.0.1:8000/runs/<run_id>/artifacts/generalization_readiness.json
```

## CLI 运行

```bash
.venv/bin/python -m agent_mvp.cli \
  --rule examples/rules.yaml \
  --dataset dataset/last_avg/all_features_10_db_2022_12_29_165551_last_avg.csv \
  --goal "根据遥测规则库检查 last_avg 数据集，完成数据质量检查、规则命中和关联规则挖掘。"
```

一键 demo：

```bash
.venv/bin/python -m agent_mvp.cli --demo
```

运行结果默认写入 `runtime/<run_id>/artifacts/report.html`、`runtime/<run_id>/artifacts/visualizations.html` 和 `runtime/<run_id>/evidence_chain.json`。

## 测试

```bash
.venv/bin/python -m pytest
```

测试覆盖文件解析、工具输出、规则命中、关联规则和证据链哈希记录。

## 证据链字段

每个证据节点包含：

- `node_id`
- `parent_ids`
- `type`
- `tool_name`
- `input_hash`
- `output_hash`
- `arguments`
- `summary`
- `timestamp`
- `model_response_id`
