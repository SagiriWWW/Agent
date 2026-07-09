# Agent 工具箱 MVP 验收说明

本文档用于说明当前项目在“Agent 为主结构，外围工具箱负责本地执行”思路下，MVP 原型应该如何验收、如何运行，以及后续应优先补齐什么。

## 1. 项目目标

当前 MVP 的目标不是直接完成一个生产级智能决策系统，而是先证明下面这条闭环可运行、可解释、可复核：

- 用户输入目标、规则库和数据集。
- Agent 生成执行计划，并按计划调用本地工具箱。
- 工具箱完成数据画像、数据质量提示、规则命中、FP-growth 关联规则挖掘和相关性分析。
- 字段语义工具生成中文名、字段分组、角色和忽略字段建议，供报告与可视化展示使用。
- Agent 汇总工具输出，执行结果校验。
- 系统输出中文 Markdown/HTML 报告、可视化看板和证据链文件。

换句话说，本阶段重点是验证“Agent 调度工具”的工程骨架，而不是把所有业务算法一次性做完。

## 2. 当前验收范围

本轮应验收的内容：

- Agent 是否能稳定生成执行计划。
- 工具箱是否能按顺序完成本地工具调用。
- 规则库是否能解析为可执行 RuleCard，并完成结构化规则校验。
- 字段语义工具是否能生成可复核的 Schema 草案。
- 规则命中结果是否能输出命中行数、命中率和样例索引。
- 关联规则挖掘是否只使用 FP-growth 路径。
- 报告是否是中文，并包含验收摘要、可视化看板、数据质量摘要、规则命中摘要、关联规则结果、相关性摘要、局限说明和证据链说明。
- Web 是否能在 Agent 运行时打开进度页，并显示当前步骤、即时输出、证据节点数和完成后的结果入口。
- Web 是否能打开规则库校验页和 Schema 复核页，并提供对应 JSON/Markdown 产物下载。
- 每次运行是否生成独立 `runtime/<run_id>/` 目录，保留报告和 `evidence_chain.json`。

本轮暂不验收的内容：

- 独立异常检测项目接入。
- 多数据类型自动适配。
- 跨客户数据集的完整泛化适配。
- 生产级权限系统。
- 本地大模型部署和模型微调。
- 面向真实业务的故障诊断准确率评估。

当前代码中保留的离群值检查只作为数据质量提示，不等同于后续要接入的异常检测模块。

## 3. 数据与规则

内置 Demo 当前使用：

- 数据集：`dataset/last_avg/all_features_10_db_2022_12_29_165551_last_avg.csv`
- 规则库：`examples/rules.yaml`

`dataset/` 下存在 `last_avg` 和 `last_last` 两类数据。当前 MVP 只处理 `last_avg` 这一类均值遥测数据，原因是均值特征更适合规则阈值和关联模式探索，受瞬时噪声影响较小。

需要明确：`last_avg` 只是当前 MVP 验证样例，不应被视为后续客户数据的固定结构。客户后续数据可能在字段名、单位、采样口径、时间列、标签列、状态字段和业务阶段上与当前样例差异很大。因此后续优先级应从继续微调当前规则阈值，转向“跨数据集 Schema 识别、字段语义映射、规则适用性检查和不兼容数据提示”。

当前样例数据的 `label` 字段全部为 0，`diagnosis` 为空。因此报告不会输出监督分类、故障类型识别或模型准确率结论。

字段语义配置当前使用：

```text
configs/telemetry_schema.yaml
configs/schema_semantics.yaml
configs/schema_mapping_overrides.yaml
```

运行时会通过 `suggest_schema_metadata` 生成 `schema_suggestions.yaml` 和 `schema_suggestions.json`，用于记录本次数据字段对应的中文名、字段分组、角色和忽略字段建议；通过 `review_schema_mapping` 生成 `schema_mapping.json` 和 `schema_mapping.md`，用于记录字段语义角色、单位线索、统计口径、规则字段候选映射和已确认映射。Schema 语义映射页和映射候选复核辅助页都支持对候选规则字段映射进行“确认/驳回”，意见先保存到本次运行的 `review_decisions.json`，并刷新数据适配审查、映射执行计划、映射回归验证、泛化门禁和报告。`/review-promotions` 会跨运行聚合这些复核意见，只有确认次数、不同数据指纹数量、映射回归通过证据和无驳回冲突同时满足阈值时，才列入全局配置审核候选。`schema_mapping_overrides.yaml` 用于沉淀经过多数据集验证后的客户字段语义或规则字段映射，服务于后续新数据泛化，不用于针对单一当前数据集调参。底层计算仍使用原始字段名，中文名、语义候选和已确认映射只用于展示、复核与后续映射执行计划，不自动改写规则或阈值。

## 4. 架构闭环

当前主流程如下：

```text
用户目标/规则库/数据集
  -> Agent 计划
  -> ToolExecutor
  -> 本地工具箱
  -> 字段语义草案
  -> Schema 语义映射
  -> 数据适配审查
  -> 工具结果
  -> validate_analysis 校验
  -> 中文报告/可视化看板/证据链
```

这个结构是合理的：Agent 不直接承担重算法实现，而是负责计划、调度、解释和校验；算法能力以工具形式注册。后续的数据关联挖掘、领域规则执行、仿真接口、模型推理接口都可以按工具方式挂载。

## 5. 当前工具箱

当前核心工具包括：

- `profile_dataset`：读取数据并生成数据画像。
- `parse_rule_library`：解析规则库并生成 RuleCard。
- `validate_rule_library`：校验规则字段绑定、条件可执行性、严重程度和建议动作。
- `suggest_schema_metadata`：生成字段中文名、字段分组、角色和忽略字段建议。
- `review_schema_metadata`：生成 Schema 人工复核清单。
- `review_schema_mapping`：识别字段语义角色、单位和统计口径，为未知客户字段生成规则字段候选映射。
- `review_dataset_adaptation`：在规则执行前生成数据适配审查，区分可直接执行、已确认映射、需映射确认和不适用规则。
- `build_mapping_execution_plan`：把已确认映射生成可审查的规则字段重写预案，标出单位、统计口径和回归验证风险。
- `validate_mapping_regression`：验证映射重写预案是否能透明执行，不自动推广字段映射或确认阈值业务正确。
- `review_schema_mapping_candidates`：把候选规则字段映射分为可考虑确认、需专家复核、不适用和已安全处理，辅助人工复核但不自动写入决策或全局配置。
- `review_generalization_readiness`：执行泛化验收门禁，判断是否存在把当前数据集特征、候选映射或不适用规则误当作自动诊断结论的风险。
- `check_missing_values`：检查缺失值。
- `detect_outliers`：输出基础离群值提示。
- `apply_rules`：执行规则命中。
- `check_association_backend`：检查 FP-growth 后端。
- `mine_association_rules`：执行 FP-growth 关联规则挖掘。
- `compute_correlations`：计算数值字段相关性。
- `build_visualizations`：生成报告可嵌入的静态可视化看板。
- `summarize_findings`：汇总工具输出。
- `validate_analysis`：校验关键工具是否完成。
- `write_report`：生成报告文件。

## 6. FP-growth 后端

项目当前已去除 Apriori 相关内容，只保留 FP-growth 关联规则挖掘路径。

当前优先使用项目内置的轻量后端：

```text
third_party/Datamining/scripts/core/fpgrowth_core.py
```

如果配置了 `DATAMINING_REPO_PATH` 或 `DATAMINING_FPGROWTH_CORE_PATH`，系统会按配置寻找 FP-growth 后端。没有可用 Datamining 后端时，会回退到 `mlxtend` 的 FP-growth 实现；再不可用时才使用本项目的 demo fallback。

正常情况下，内置 Demo 输出中应看到：

```text
关联规则后端: datamining_fpgrowth
```

## 7. 运行方式

建议始终使用项目虚拟环境：

```bash
.venv/bin/python -m agent_mvp.cli --demo
```

指定规则库和数据集运行：

```bash
.venv/bin/python -m agent_mvp.cli \
  --rule examples/rules.yaml \
  --dataset dataset/last_avg/all_features_10_db_2022_12_29_165551_last_avg.csv \
  --goal "根据遥测规则库检查 last_avg 数据集，完成数据质量检查、规则命中和关联规则挖掘。"
```

启动 Web 服务：

```bash
.venv/bin/uvicorn agent_mvp.app:app --reload --host 127.0.0.1 --port 8000
```

运行完成后查看复核页面：

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/demo-ui
http://127.0.0.1:8000/runs/new
http://127.0.0.1:8000/batch-ui
http://127.0.0.1:8000/rule-versions
http://127.0.0.1:8000/review-promotions
http://127.0.0.1:8000/toolbox-ui
http://127.0.0.1:8000/runs/<run_id>/progress
http://127.0.0.1:8000/api/runs/<run_id>/events
http://127.0.0.1:8000/runs/<run_id>/rule-validation
http://127.0.0.1:8000/runs/<run_id>/schema-review
```

客户数据分析页 `/runs/new` 当前只开放 MD/YAML/YML 规则库和 CSV/XLSX 数据集上传，前端选择器和后端校验保持一致。该页面还支持按本次任务选择模型策略：本地稳定模式、外部模型优先、强制外部模型；模型密钥和 Base URL 仍由 `.env` 或环境变量提供。

批量验收页 `/batch-ui` 可以遍历 `dataset/last_avg/` 中的同口径数据集，也可以选择 `dataset/customer_variants/` 客户结构变体小样本。系统会为每个数据集生成独立 run，并在 `runtime/batches/<batch_id>/` 下生成批量汇总报告、批量数据适配审查、批量映射复核辅助、批量人工复核意见汇总、批量映射回归验证、批量泛化门禁、规则稳定性分析和关键命中样例。数据适配页面 `/batches/<batch_id>/dataset-adaptation` 会按数据集和规则聚合字段适配情况，区分“适配稳定”“需映射确认”“存在不适用”和“缺少适配证据”。映射复核辅助页面 `/batches/<batch_id>/schema-mapping-review` 会汇总每个 run 的 `schema_mapping_review.json`，把候选映射聚合为“可考虑确认”“需专家复核”“不适用”和“已安全处理”；它不自动写入人工决策或全局配置。人工复核意见页面 `/batches/<batch_id>/review-decisions` 会汇总每个 run 的 `review_decisions.json` 和映射回归证据，标记“可进入推广审核”“确认/驳回冲突”“缺少回归证据”“批内证据不足”和“仅驳回”；它仍然不自动写入全局配置。映射回归页面 `/batches/<batch_id>/mapping-regression` 会汇总每个 run 的 `mapping_regression_validation.json`，统计映射重写通过、需确认、需复核、失败和缺证据项；通过只代表技术执行验证，不代表阈值业务正确。泛化门禁页面 `/batches/<batch_id>/generalization-readiness` 会汇总每个 run 的 `generalization_readiness.json`，统计门禁证据覆盖、自动诊断允许数、未解决规则和检查项横向分布；缺少门禁产物会被标记为“证据不完整”，不会补造通过结论。规则稳定性页面 `/batches/<batch_id>/rule-analysis` 会按规则聚合多个数据集的命中率，标记“疑似过宽”“疑似过严”“局部高命中”“稀疏事件”等复核线索，用于指导下一轮规则库优化。命中样例页面 `/batches/<batch_id>/hit-samples` 会导出代表性命中行和关键上下文字段，方便专家复核阈值。命令行也可以运行：

```bash
.venv/bin/python -m agent_mvp.cli --batch-last-avg --max-datasets 5
.venv/bin/python -m agent_mvp.cli --batch-dir customer_variants --rule examples/rules_v2.yaml --max-datasets 3
```

`customer_variants` 批次用于验证泛化适配能力，预期结果可以是“需映射确认”或“存在不适用”，不应为了通过率而自动改写字段或强行执行规则。单次运行会生成 `mapping_execution_plan.json` 和 `mapping_execution_plan.md`，用于把已确认映射转成可审查执行预案；生成 `mapping_regression_validation.json` 和 `mapping_regression_validation.md`，用于验证重写预案是否可执行；还会生成 `generalization_readiness.json` 和 `generalization_readiness.md` 作为反过拟合门禁。只要仍有候选映射、已确认但未完成回归验证的映射、不适用规则或不可解析规则，就不应允许自动诊断解释。

复核沉淀页 `/review-promotions` 会扫描 `runtime/<run_id>/review_decisions.json` 和 `mapping_regression_validation.json`，生成 `runtime/review_promotions/review_decision_promotions.json`、`.md` 和 `.html`。默认阈值要求同一映射至少 2 次确认、至少 2 个不同数据指纹、至少 2 次跨数据映射回归通过且无驳回冲突；满足条件也只是进入“可推广审核”队列，不会自动写入 `configs/schema_mapping_overrides.yaml`。

规则优化页 `/rule-versions` 用于启动 v1 基线规则库和 v2 优化草案的批量验证，并生成规则版本对比报告。v2 当前保存在 `examples/rules_v2.yaml`，主要根据批量稳定性分析收窄 R001、R002、R007 的阈值。当前已经完成 v1/v2 在 30 个 `last_avg` 数据集上的全量对比，v2 批次 `07b79fd7-b0dc-4534-8cb6-6fa17c8ce012` 全部通过，并形成专家复核包 `docs/rule_reviews/rules_v2_expert_review_2026-07-02.md`。命令行可以直接验证 v2：

```bash
.venv/bin/python -m agent_mvp.cli --batch-last-avg --rule examples/rules_v2.yaml --max-datasets 5
```

检查关联规则后端：

```bash
.venv/bin/python - <<'PY'
from agent_mvp.datamining_adapter import datamining_backend_status
print(datamining_backend_status())
PY
```

运行测试：

```bash
.venv/bin/python -m pytest
```

## 8. 验收标准

一次运行可以视为 MVP 通过，至少应满足：

- CLI 或 Web Demo 能完成运行，不抛异常。
- 输出目录中存在 `report.md`、`report.html`、`visualizations.html` 和 `evidence_chain.json`。
- 输出目录中存在 `rule_validation.json`、`schema_suggestions.yaml` 或 `schema_suggestions.json`。
- 输出目录中存在 `schema_review.json` 或 `schema_review.md`。
- 输出目录中存在 `schema_mapping.json` 或 `schema_mapping.md`。
- 输出目录中存在 `dataset_adaptation.json` 或 `dataset_adaptation.md`。
- 输出目录中存在 `mapping_execution_plan.json` 或 `mapping_execution_plan.md`。
- 输出目录中存在 `mapping_regression_validation.json` 或 `mapping_regression_validation.md`。
- 输出目录中存在 `generalization_readiness.json` 或 `generalization_readiness.md`。
- 报告中包含 `MVP 验收摘要`、`可视化看板`、`规则命中摘要`、`关联规则挖掘结果` 和 `MVP 验收检查项`。
- 报告和可视化中的关键遥测字段能显示中文名，并保留必要的原始字段名复核线索。
- 运行提交后能进入 `/runs/<run_id>/progress`，并通过 `/api/runs/<run_id>/events` 事件流刷新 Agent 步骤、工具状态和最新观察结果。
- 运行记录 `manifest.json` 中包含本次任务的 `model_config`，进度页能展示模型策略、服务商和模型名称。
- 输出目录中存在 `run_events.jsonl`，能追踪运行创建、文件接入、计划生成、步骤执行、工具开始/完成和最终完成/失败事件。
- 报告导航或最近运行入口能打开规则库校验页、Schema 复核页、Schema 语义映射页、数据适配审查页、映射执行计划页、映射回归验证页、泛化验收门禁页、可视化页和证据链。
- 批量验收能生成 `batch_manifest.json`、`batch_report.md`、`batch_report.html`、`batch_dataset_adaptation.json`、`batch_dataset_adaptation.md`、`batch_dataset_adaptation.html`、`batch_schema_mapping_review.json`、`batch_schema_mapping_review.md`、`batch_schema_mapping_review.html`、`batch_review_decisions.json`、`batch_review_decisions.md`、`batch_review_decisions.html`、`batch_rule_analysis.json`、`batch_rule_analysis.md`、`batch_rule_analysis.html`、`key_hit_samples.json`、`key_hit_samples.md` 和 `key_hit_samples.html`，并能跳转到每个数据集对应的单次报告。
- 规则优化页能启动 v1/v2 批量验证，并能基于两个批次生成规则版本对比报告。
- `validate_analysis` 状态为通过，或报告明确列出告警原因。
- 关联规则后端不是 Apriori。
- 测试命令 `.venv/bin/python -m pytest` 通过。

## 9. 当前限制

- 规则解析仍以简单比较表达式为主，复杂自然语言规则需要继续结构化。
- 字段语义配置、Schema 语义映射、数据适配审查、映射执行计划和泛化验收门禁已能处理字段级精确匹配、后缀口径不一致、别名/语义角色候选、已确认映射和缺失字段；跨数据集自动执行仍需要继续补充单位换算、适用阶段和批量回归门禁。
- `rules_v2.yaml` 是用于验证规则迭代闭环的候选版本，不应被当作面向所有客户数据的最终规则库。
- 关联规则挖掘只能说明字段状态共现，不能说明因果关系。
- 报告是本地工具结果的解释与整理，不等同于专家最终判定。
- Web 前端目前是 MVP 演示界面，还不是完整产品交互。

## 10. 后续优先级

建议下一阶段按下面顺序推进：

1. 固化 MVP 验收：保持当前 demo、测试、报告和证据链稳定。
2. 扩展跨数据集 Schema 语义映射：在 `configs/schema_semantics.yaml` 中持续补充客户字段别名、单位、业务阶段和确认记录。
3. 增强规则适用性检查：在执行规则前判断客户数据是否具备必要字段、单位是否一致、规则是否需要人工确认。
4. 固化数据适配报告：对未知数据先说明可直接执行、需映射确认、不可执行的规则和字段，不强行套用当前 `last_avg` 规则。
5. 在 Schema/适配层稳定后，再基于专家反馈生成新的规则候选版本。
