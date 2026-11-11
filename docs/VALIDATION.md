# 精简快照验证 · 2026-11-11

验证针对本分支的独立目录进行，使用原项目已安装的 Python 3.14 环境；未进行全新环境的在线依赖重装。

- `PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider`：416 passed，1 个 Starlette/httpx 弃用警告。
- `PYTHONPATH=src:. python -m aircraft_agent.interface.cli --demo`：offline 模式，21 步，mlxtend_fpgrowth 后端，生成 HTML 报告。
- `--verify-evidence-storage <demo-run> --json`：快照/日志均 45 节点，consistent，write_state=ready，退出码 0。
- `python3 -m pip wheel --no-deps .`：成功构建 aircraft_agent-0.1.0-py3-none-any.whl（247,733 bytes）；仅隔离安装构建依赖，未重装应用依赖。原环境的 pip 启动停滞已终止，改用系统 Python 构建。
- FastAPI TestClient：`/` 与 `/docs/mvp` 均返回 HTTP 200。
- 上传文件按路径白名单整理，凭证格式扫描无匹配，不包含 `.env`、真实数据和运行产物。

以上只验证精简核心的软件行为。未调用真实模型，未验证需要完整归档材料的公开数据入口，也不代表完整项目的全量研究验收。
