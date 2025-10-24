# Repository Guidelines

## 项目结构与模块组织
仓库核心脚本为 `deepseek.py`，使用 DeepSeek API 生成信号，并借助 `ccxt` 下单至 Binance USDT 永续合约。另有 `deepseek_ok版本.py` 定位于 OKX，`deepseek_ok_带指标plus版本.py` 则扩展指标用于策略验证。根目录中的 `.env` 存放交易所与模型密钥，`requirements.txt` 记录依赖；`README.md` 提供基础部署说明。保持当前扁平脚本结构即可，若新增工具、回测脚本或数据文件，请创建命名清晰的子目录（例如 `strategies/`、`data/`）并在 README 补充入口说明。日志或样本数据应集中在 `logs/`、`samples/` 等目录，便于版本控制与排错；若需持久化模型权重或训练数据，请建立 `models/`、`datasets/` 并在 `.gitignore` 中列出不适合提交的文件。

## 构建、测试与开发命令
遵循现有 Conda 工作流初始化环境：
```bash
conda create -n ds python=3.10
conda activate ds
pip install -r requirements.txt
```
通过 `python deepseek.py` 启动默认策略，替换脚本名称即可运行其他版本。研发阶段可在命令后增加 `--test`（若自定义）控制模拟模式；生产部署推荐 `pm2 start python -- deepseek.py` 保持进程常驻，并用 `pm2 logs` 快速检查异常。若需热更新，可使用 `pm2 restart deepseek` 或 `pm2 reload all`；在服务器维护前运行 `pm2 save` 以保留进程配置。开发者可在 Conda 环境内执行 `python -m pip install --upgrade pip` 和 `pip check` 维持依赖健康，更新依赖后执行 `pip list --outdated` 核对版本漂移，必要时写入 `requirements.txt` 并同步到团队环境。若偏好 IDE，可在 VS Code `tasks.json` 或 Makefile 中封装上述命令，便于一键执行。

## 代码风格与命名约定
统一使用 PEP 8：四空格缩进、函数与变量采用 `snake_case`，常量集中在文件开头如 `TRADE_CONFIG`。敏感配置一律通过 `dotenv` 加载，避免硬编码。涉及交易逻辑时保持函数职责单一，例如拆分行情拉取、信号判断、执行模块。建议函数名描述策略意图（如 `generate_signal_from_history`），参数名体现单位或币种。撰写 Docstring 时说明输入输出、币种、风控假设，便于后续代理复用。当前未引入自动格式化工具，如需提交大规模格式化，请先讨论并在 PR 中说明范围，可附 `black` 或 `ruff` 命令作为参考而不要直接强制执行。

## 测试指南
当前缺少自动化测试，扩展策略时建议引入 `pytest` 并在 `tests/` 目录下按模块组织：如 `tests/test_deepseek.py`。使用 `pytest -q` 快速回归，必要时模拟 `ccxt` 交易所响应及 DeepSeek API 结果，覆盖风控、仓位管理和信号边界分支。新增策略前先补充至少一条断言验证下单条件，变更风控参数时附加对应的失败路径测试。若涉及实时数据，可通过 `pytest --maxfail=1 --disable-warnings` 快速定位，必要时补充 notebook 或 markdown 记录人工回测过程。提交前应运行全部测试，尤其是跨市场配置或新风控函数，确保没有回归；必要时增加 smoke test 脚本以验证与真实交易所连接的可达性。

## 提交与拉取请求规范
历史提交消息以简短命令式陈述为主，偶有中文描述，请保持 60 字以内的主题行，详细说明可写入正文。提交前执行 `git status` 与 `git diff`，确认未遗漏 `.env` 等本地配置；完成开发后运行 `pytest` 或列出手工验证步骤，写入 commit body 便于追溯。Always attach a short testing note, for example `Tests: pytest -q` 或者人工回测摘要，帮助审阅者快速理解验证范围。拉取请求需阐明策略变动目标、测试凭证（`pytest` 输出或人工回测说明）、新增环境变量与风险提示；若关联 Issue 或需要截图演示 CLI 行为，应在描述中附上链接或终端截屏。多人协作时在 PR 中列出风险评估、回滚方案与依赖任务，便于审核。

## 安全与配置提示
API 密钥与密码仅存放在 `.env`，部署前确认权限最小化（禁止提币、限制交易对）。真实下单前可利用 `TRADE_CONFIG['test_mode']` 或交易所沙箱，避免误触发资金操作。每次重大迭代后执行密钥轮换，并在服务器上限制访问权限与日志可见范围；本地日志若含敏感内容，应定期转储至受控存储并做脱敏处理。部署脚本中若需读取密钥，优先使用环境变量注入而非写入磁盘，定期检查 `pm2`、`screen` 等守护进程的日志滚动策略，避免泄露。建议定期审查依赖版本以防供应链风险，并记录合规检查日期；上线前运行一份 security checklist，覆盖密钥、风控与监控配置。

## Agent 协作提示
仓库主要面向自动化交易任务，编写或接入新 Agent 时，应复用既有配置加载与信号缓存模式，确保上下文一致。建议在代码开头声明角色、依赖与输出格式，并记录在 PR 描述，便于其他 Agent 或维护者理解接入点。多人并行改动时，提前在 Issue 中锁定脚本名称与变量前缀，减少策略冲突。协作前同步目标市场、交易周期及关键参数，保持 Agent 间信息一致性，可通过共享的 `docs/agents/` 目录记录行为契约。需要跨 Agent 调用时，请在文档附上示例输入输出，避免重复实现；涉及数据共享时给出最小化字段集合和脱敏策略。
