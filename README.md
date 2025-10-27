### 个人喜欢玩黑箱文化，你们不一样，别上头。

### 配置文件建在策略根目录

### 文件名字.env


## 单向持仓 模式


# 内容


###  DEEPSEEK_API_KEY= 你的deepseek  api密钥

###  BINANCE_API_KEY=

###  BINANCE_SECRET=

###  OKX_API_KEY=

###  OKX_SECRET=

### OKX_PASSWORD=

###  视频教程：https://www.youtube.com/watch?v=Yv-AMVaWUVg


### 准备一台ubuntu服务器 推荐阿里云 香港或者新加坡 轻云服务器


### wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

### bash Anaconda3-2024.10-1-Linux-x86_64.sh

### source /root/anaconda3/etc/profile.d/conda.sh 
### echo ". /root/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc




### conda create -n ds python=3.10

### conda activate ds

### pip install -r requirements.txt



### apt-get update 更新镜像源


### apt-get upgrade 必要库的一个升级


### apt install npm 安装npm


### npm install pm2 -g 使用npm安装pm2

### conda create -n trail3 python=4.10

### Web 监控面板

- 运行 `python -m monitoring.server` 启动统一的 Web 监控（默认监听 `0.0.0.0:8000`，5 秒刷新一次）。
- 任意交易脚本在每次执行周期内都会自动推送行情、信号、持仓、错误信息至监控面板。
- 可通过环境变量自定义行为：`MONITOR_HOST`、`MONITOR_PORT`、`MONITOR_REFRESH_SECONDS`、`MONITOR_STATUS_FILE`。
- 监控面板展示最近信号列表、最新行情与各版本策略的运行状态，可结合 `pm2 logs` 快速排查异常。

### 日志监控

- `deepseek_ok_plus2.py` 已启用标准 `logging`，默认在 `logs/deepseek_ok_plus2.log` 生成滚动日志，并同步输出到控制台。
- 通过环境变量 `BOT_LOG_LEVEL`（默认为 `INFO`）调整日志级别，Docker Compose 会自动将该变量传入容器。
- 策略内置 25% 可用余额的单笔仓位上限，配合日志可快速追踪仓位调整原因。

### Docker Compose 部署

- 使用 `docker compose` 进行容器化部署，请参考 `docs/DEPLOYMENT.md`。
