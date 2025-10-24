# Docker Compose 部署指南

本文档说明如何使用 `docker-compose` 快速部署交易策略与监控面板。默认情况下，每个服务都基于仓库根目录的 `Dockerfile` 构建镜像。

## 前置依赖
- Docker 24+ 与 Compose 插件（`docker compose` 命令）
- 已填好的 `.env` 文件（包含交易所与 DeepSeek API 密钥）
- 可选：为 `logs/` 目录配置持久化或日志收集方案

## 目录结构回顾
```
.
├── Dockerfile
├── docker-compose.yml
├── deepseek.py
├── deepseek_ok版本.py
├── deepseek_ok_带指标plus版本.py
├── monitoring/
└── logs/
```

## 构建基础镜像
```bash
docker compose build
```

这一步会安装 `requirements.txt` 中的依赖。若依赖更新，重新执行 `docker compose build` 即可。

## 启动监控面板
```bash
docker compose up monitoring
```

- 默认监听在 `0.0.0.0:8000`，浏览器访问 `http://localhost:8000`。
- `logs/monitor_status.json` 会实时记录各策略状态，可根据需要接入日志系统。
- 自定义刷新间隔或端口：在 `.env` 中设置 `MONITOR_REFRESH_SECONDS`、`MONITOR_PORT`，或在命令行追加 `-e` 覆盖。

## 启动交易机器人
选择需要的策略服务即可：

```bash
# 仅运行 Binance USDT 永续机器人
docker compose up binance_bot

# 运行 OKX 基础策略
docker compose up okx_bot

# 运行 OKX 指标增强策略
docker compose up okx_plus_bot
```

多个服务可以一起启动，例如：

```bash
docker compose up monitoring binance_bot okx_bot
```

### 后台运行
```bash
docker compose up -d monitoring binance_bot
```

使用 `docker compose logs -f <service>` 查看实时输出，`docker compose down` 停止全部服务。

## 环境变量与配置
- 所有服务默认加载根目录 `.env`。
- 如果需要针对某个策略修改行为，可在 `docker-compose.yml` 中为对应服务追加 `environment` 覆盖。
- 日志目录绑定到宿主机 `./logs`，便于调试及备份。

## 升级流程
1. 更新仓库代码与依赖：
   ```bash
   git pull
   docker compose build
   ```
2. 重启相关服务：
   ```bash
   docker compose up -d --force-recreate monitoring
   docker compose up -d --force-recreate binance_bot
   ```

## 故障排查
- `docker compose ps` 检查容器状态。
- `docker compose logs -f monitoring` 查看监控面板是否正常刷新。
- 如果看到 `ModuleNotFoundError`，确认 `requirements.txt` 是否同步更新并重新 `build`。
- 网络或 API 错误通常来自 `.env` 配置不全或密钥权限不足，必要时使用交易所测试网。

## 清理
```bash
docker compose down --volumes
```

该命令会移除容器与匿名卷，但会保留 `./logs` 中的持久化文件。
