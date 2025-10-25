"""Web server that renders monitoring data for all trading scripts."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template_string

try:  # Allow running as module or script
    from .state import STATUS_FILE, load_all_states
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    from monitoring.state import STATUS_FILE, load_all_states

DEFAULT_REFRESH = int(os.getenv("MONITOR_REFRESH_SECONDS", "5"))


def _format_size(num_bytes: int) -> str:
    if not num_bytes:
        return "0 KB"
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"

DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>交易策略监控面板</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            color-scheme: dark;
            --bg: #070b16;
            --surface: #0f172a;
            --card: rgba(15, 23, 42, 0.85);
            --border: rgba(99, 102, 241, 0.35);
            --text: #f8fafc;
            --muted: #94a3b8;
            --danger: #f87171;
            --success: #34d399;
            --warning: #facc15;
            --accent: #6366f1;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 35%, #05070f 100%);
            font-family: "Inter", "Segoe UI", "PingFang SC", sans-serif;
            color: var(--text);
        }
        header {
            padding: clamp(1.5rem, 4vw, 2.75rem);
        }
        .top-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-top: 0.8rem;
            font-size: 0.95rem;
            color: var(--muted);
        }
        .meta-pill {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 999px;
            padding: 0.35rem 0.95rem;
        }
        h1 {
            margin: 0;
            font-size: clamp(1.8rem, 5vw, 2.8rem);
            font-weight: 600;
        }
        main {
            padding: 0 clamp(1.25rem, 4vw, 2.75rem) 3rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.8rem;
        }
        .card {
            background: var(--card);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 1rem;
            padding: 1.4rem;
            backdrop-filter: blur(14px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.45);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .card-header .bot-name {
            font-size: 1.25rem;
            font-weight: 600;
        }
        .tag {
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid transparent;
        }
        .tag.buy { color: var(--success); border-color: rgba(52, 211, 153, 0.4); }
        .tag.sell { color: var(--danger); border-color: rgba(248, 113, 113, 0.4); }
        .tag.hold { color: var(--warning); border-color: rgba(250, 204, 21, 0.4); }
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.2rem;
        }
        .kpi {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 0.8rem;
            padding: 0.9rem;
        }
        .label {
            font-size: 0.8rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .value {
            margin-top: 0.35rem;
            font-size: 1.35rem;
            font-weight: 600;
        }
        canvas {
            width: 100% !important;
            height: 210px !important;
        }
        .section-title {
            margin: 1.2rem 0 0.4rem;
            font-size: 0.95rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th, td {
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }
        th { color: var(--muted); text-align: left; font-weight: 500; }
        .table-wrapper {
            max-height: 180px;
            overflow-y: auto;
            padding-right: 0.2rem;
        }
        .logs {
            max-height: 150px;
            overflow-y: auto;
            font-size: 0.85rem;
            line-height: 1.4;
            background: rgba(8, 12, 20, 0.75);
            border-radius: 0.8rem;
            border: 1px solid rgba(148, 163, 184, 0.15);
            padding: 0.9rem;
        }
        .log-entry + .log-entry {
            margin-top: 0.5rem;
            border-top: 1px solid rgba(148, 163, 184, 0.15);
            padding-top: 0.45rem;
        }
        .log-entry .level {
            font-size: 0.7rem;
            margin-right: 0.4rem;
            letter-spacing: 0.08em;
        }
        .log-entry .level.INFO,
        .log-entry .level.SUCCESS { color: var(--success); }
        .log-entry .level.ERROR { color: var(--danger); }
        .log-entry .level.WARNING { color: var(--warning); }
        .log-entry .level.DEBUG { color: var(--muted); }
        .ds-entry + .ds-entry {
            margin-top: 0.5rem;
            border-top: 1px dashed rgba(99, 102, 241, 0.3);
            padding-top: 0.5rem;
        }
        .empty {
            grid-column: 1 / -1;
            text-align: center;
            padding: 3rem 1rem;
            border-radius: 1rem;
            border: 1px dashed rgba(148, 163, 184, 0.3);
            color: var(--muted);
            background: rgba(15, 23, 42, 0.35);
        }
        @media (max-width: 640px) {
            header, main { padding: 1.25rem; }
            .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.5/dist/chart.umd.min.js" crossorigin="anonymous"></script>
</head>
<body>
    <header>
        <h1>策略监控面板</h1>
        <div class="top-meta">
            <span class="meta-pill">状态文件: {{ status_file }}</span>
            <span class="meta-pill">大小: <span id="status-size">{{ initial_size }}</span></span>
            <span class="meta-pill">自动刷新: {{ refresh // 1000 }} 秒</span>
            <span class="meta-pill">最近同步: <span id="last-sync">-</span></span>
        </div>
    </header>
    <main>
        <div id="grid" class="grid">
            <div class="empty">等待策略更新监控数据...</div>
        </div>
    </main>
    <template id="bot-card">
        <article class="card">
            <div class="card-header">
                <span class="bot-name"></span>
                <span class="tag latest-signal">-</span>
            </div>
            <div class="kpi-row">
                <div class="kpi">
                    <div class="label">当前价格</div>
                    <div class="value current-price">-</div>
                </div>
                <div class="kpi">
                    <div class="label">价格变化</div>
                    <div class="value price-change">-</div>
                </div>
                <div class="kpi">
                    <div class="label">浮动盈亏</div>
                    <div class="value position-pnl">-</div>
                </div>
            </div>
            <div class="kpi-row">
                <div class="kpi">
                    <div class="label">持仓</div>
                    <div class="value position-side">-</div>
                </div>
                <div class="kpi">
                    <div class="label">账户权益</div>
                    <div class="value account-equity">-</div>
                </div>
                <div class="kpi">
                    <div class="label">可用余额</div>
                    <div class="value account-free">-</div>
                </div>
            </div>
            <section>
                <div class="section-title">最新信号</div>
                <div class="label">理由</div>
                <div class="value signal-reason">-</div>
                <div class="label" style="margin-top:0.6rem;">时间</div>
                <div class="value signal-timestamp">-</div>
            </section>
            <section>
                <div class="section-title">订单列表</div>
                <div class="table-wrapper">
                    <table class="orders-table">
                        <thead>
                            <tr>
                                <th>时间</th>
                                <th>动作</th>
                                <th>方向</th>
                                <th>状态</th>
                                <th>数量</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>
            <section>
                <div class="section-title">运行日志</div>
                <div class="logs runtime-logs"></div>
            </section>
            <section>
                <div class="section-title">DeepSeek 通讯</div>
                <div class="logs deepseek-logs" style="max-height: 220px;"></div>
            </section>
            <section>
                <div class="section-title">最近信号</div>
                <div class="table-wrapper">
                    <table class="history-table">
                        <thead>
                        <tr><th>时间</th><th>信号</th><th>信心</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>
            <div class="section-title">最近更新</div>
            <div class="value last-update">-</div>
        </article>
    </template>
    <script>
        const template = document.getElementById('bot-card');
        const grid = document.getElementById('grid');
        const refreshInterval = {{ refresh }};

        function formatSize(bytes) {
            if (!bytes) return '0 KB';
            const kb = bytes / 1024;
            if (kb < 1024) return `${kb.toFixed(1)} KB`;
            const mb = kb / 1024;
            return `${mb.toFixed(2)} MB`;
        }

        function signalTagClass(signal) {
            if (!signal) return 'tag hold';
            const key = signal.toLowerCase();
            if (key === 'buy') return 'tag buy';
            if (key === 'sell') return 'tag sell';
            return 'tag hold';
        }

        function formatNumber(value, digits = 2) {
            if (value === null || value === undefined || Number.isNaN(value)) {
                return '-';
            }
            return Number(value).toFixed(digits);
        }

        function populateCard(name, data) {
            const node = template.content.firstElementChild.cloneNode(true);
            node.querySelector('.bot-name').textContent = name;

            const signal = data.latest_signal;
            const tag = node.querySelector('.latest-signal');
            if (signal && signal.signal) {
                tag.textContent = signal.signal;
                tag.className = signalTagClass(signal.signal);
            } else {
                tag.textContent = '未同步';
                tag.className = 'tag hold';
            }

            const snapshot = data.price_snapshot || {};
            const price = snapshot.price;
            node.querySelector('.current-price').textContent =
                price ? `$${formatNumber(price)}` : '-';

            const change = snapshot.price_change;
            node.querySelector('.price-change').textContent =
                change !== undefined ? `${formatNumber(change)}%` : '-';

            const position = data.position || {};
            const side = position.side || '无持仓';
            const size = position.size ? formatNumber(position.size, 4) : '';
            node.querySelector('.position-side').textContent = size ? `${side} ${size}` : side;
            const pnl = position.unrealized_pnl;
            node.querySelector('.position-pnl').textContent =
                pnl !== undefined ? `${formatNumber(pnl)} USDT` : '-';
            const account = data.account || {};
            const equity = account.equity ?? account.total ?? account.available;
            node.querySelector('.account-equity').textContent =
                equity !== undefined ? `${formatNumber(equity)} USDT` : '-';
            const freeBalance = account.available ?? account.free;
            node.querySelector('.account-free').textContent =
                freeBalance !== undefined ? `${formatNumber(freeBalance)} USDT` : '-';

            const reason = signal && signal.reason ? signal.reason : '-';
            node.querySelector('.signal-reason').textContent = reason;
            node.querySelector('.signal-timestamp').textContent =
                signal && signal.timestamp ? signal.timestamp : '-';

            node.querySelector('.last-update').textContent = data.last_update || '-';

            const historyBody = node.querySelector('.history-table tbody');
            const history = (data.signal_history || []).slice(-8).reverse();
            historyBody.innerHTML = '';
            history.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.timestamp || '-'}</td>
                    <td>${item.signal || '-'}</td>
                    <td>${item.confidence || '-'}</td>
                `;
                historyBody.appendChild(row);
            });

            const ordersBody = node.querySelector('.orders-table tbody');
            ordersBody.innerHTML = '';
            (data.orders || []).slice(-5).reverse().forEach(order => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${order.timestamp ? order.timestamp.replace('T', ' ').replace('Z', '') : '-'}</td>
                    <td>${order.action || '-'}</td>
                    <td>${order.side || '-'}</td>
                    <td>${order.status || '-'}</td>
                    <td>${order.amount !== undefined ? formatNumber(order.amount, 4) : '-'}</td>
                `;
                ordersBody.appendChild(row);
            });

            const logsContainer = node.querySelector('.runtime-logs');
            logsContainer.innerHTML = '';
            (data.logs || []).slice(-6).reverse().forEach(entry => {
                const level = entry.level || 'INFO';
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `
                    <div><span class="level ${level}">${level}</span>${entry.timestamp || '-'}</div>
                    <div>${entry.message || ''}</div>
                `;
                logsContainer.appendChild(div);
            });

            const dsContainer = node.querySelector('.deepseek-logs');
            dsContainer.innerHTML = '';
            (data.deepseek_messages || []).slice(-3).reverse().forEach(entry => {
                const wrapper = document.createElement('div');
                wrapper.className = 'ds-entry';
                wrapper.innerHTML = `
                    <div><span class="level ${entry.status || 'INFO'}">${entry.status || 'INFO'}</span>${entry.timestamp || '-'}</div>
                    <div class="ds-response">${entry.response || '-'}</div>
                    <div class="ds-prompt">Prompt: ${entry.prompt || '-'}</div>
                `;
                dsContainer.appendChild(wrapper);
            });

            return node;
        }

        async function refresh() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const bots = data.bots || {};
                const names = Object.keys(bots);
                grid.innerHTML = '';
                if (!names.length) {
                    const empty = document.createElement('div');
                    empty.className = 'empty';
                    empty.textContent = '等待策略更新监控数据...';
                    grid.appendChild(empty);
                } else {
                    names.sort().forEach(name => {
                        const card = populateCard(name, bots[name]);
                        grid.appendChild(card);
                    });
                }
                document.getElementById('last-sync').textContent = data.updated_at || '-';
                document.getElementById('status-size').textContent = formatSize(data.status_size || 0);
            } catch (error) {
                console.error('Failed to refresh monitor state', error);
            }
        }

        refresh();
        setInterval(refresh, refreshInterval);
    </script>
</body>
</html>
"""


def create_app(refresh_ms: int) -> Flask:
    app = Flask(__name__)

    @app.route("/api/status")
    def api_status():
        state = load_all_states()
        size = STATUS_FILE.stat().st_size if STATUS_FILE.exists() else 0
        payload: Dict[str, object] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "bots": state,
            "status_size": size,
        }
        return jsonify(payload)

    @app.route("/")
    def dashboard():
        return render_template_string(
            DASHBOARD_TEMPLATE,
            refresh=refresh_ms,
            status_file=str(STATUS_FILE),
            initial_size=_format_size(STATUS_FILE.stat().st_size) if STATUS_FILE.exists() else "0 KB",
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web dashboard for trading bots.")
    parser.add_argument("--host", default=os.getenv("MONITOR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MONITOR_PORT", "8000")))
    parser.add_argument(
        "--refresh",
        type=int,
        default=int(os.getenv("MONITOR_REFRESH_SECONDS", str(DEFAULT_REFRESH))),
        help="Dashboard auto-refresh interval in seconds.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable Flask debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refresh_ms = max(args.refresh, 1) * 1000
    app = create_app(refresh_ms=refresh_ms)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
