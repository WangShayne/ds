"""Web server that renders monitoring data for all trading scripts."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Dict

from flask import Flask, jsonify, render_template_string

from .state import STATUS_FILE, load_all_states

DEFAULT_REFRESH = int(os.getenv("MONITOR_REFRESH_SECONDS", "5"))

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
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --danger: #f87171;
            --success: #34d399;
            --warning: #facc15;
        }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            margin: 0;
            padding: 2rem clamp(1rem, 4vw, 3rem);
        }
        h1 {
            margin: 0 0 0.5rem;
            font-size: clamp(1.5rem, 4vw, 2.4rem);
        }
        .meta {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        .grid {
            display: grid;
            gap: 1.5rem;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
        }
        .card h2 {
            margin: 0;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .label {
            font-size: 0.8rem;
            color: var(--muted);
        }
        .value {
            font-size: 1.35rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .section {
            margin-top: 1rem;
            border-top: 1px solid var(--border);
            padding-top: 0.9rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 0.4rem 0;
            font-size: 0.9rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }
        th {
            color: var(--muted);
            font-weight: 500;
        }
        .tag {
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .tag.buy { background: rgba(52, 211, 153, 0.15); color: var(--success); }
        .tag.sell { background: rgba(248, 113, 113, 0.18); color: var(--danger); }
        .tag.hold { background: rgba(250, 204, 21, 0.15); color: var(--warning); }
        .empty {
            grid-column: 1 / -1;
            text-align: center;
            padding: 2rem;
            border: 1px dashed var(--border);
            border-radius: 0.75rem;
            color: var(--muted);
        }
        @media (max-width: 600px) {
            body { padding: 1.5rem; }
        }
    </style>
</head>
<body>
    <h1>策略监控面板</h1>
    <div class="meta">
        <span>状态文件: {{ status_file }}</span>
        <span style="margin-left: 1.5rem;">自动刷新: {{ refresh // 1000 }} 秒</span>
        <span style="margin-left: 1.5rem;">最近同步: <span id="last-sync">-</span></span>
    </div>
    <div id="grid" class="grid">
        <div class="empty">等待策略更新监控数据...</div>
    </div>
    <template id="bot-card">
        <article class="card">
            <h2>
                <span class="bot-name"></span>
                <span class="tag latest-signal">-</span>
            </h2>
            <div class="section">
                <div class="label">当前价格</div>
                <div class="value current-price">-</div>
                <div class="label price-change-label">价格变化</div>
                <div class="value price-change">-</div>
            </div>
            <div class="section">
                <div class="label">持仓</div>
                <div class="value position-side">-</div>
                <div class="label">浮动盈亏</div>
                <div class="value position-pnl">-</div>
            </div>
            <div class="section">
                <div class="label">最新信号</div>
                <div class="value signal-reason">-</div>
                <div class="label">信号时间</div>
                <div class="value signal-timestamp">-</div>
            </div>
            <div class="section">
                <div class="label">最近更新</div>
                <div class="value last-update">-</div>
            </div>
            <div class="section">
                <div class="label">最近信号</div>
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>时间</th>
                            <th>信号</th>
                            <th>信心</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </article>
    </template>
    <script>
        const template = document.getElementById('bot-card');
        const grid = document.getElementById('grid');
        const refreshInterval = {{ refresh }};

        function signalTagClass(signal) {
            if (!signal) return '';
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
        payload: Dict[str, object] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "bots": state,
        }
        return jsonify(payload)

    @app.route("/")
    def dashboard():
        return render_template_string(
            DASHBOARD_TEMPLATE,
            refresh=refresh_ms,
            status_file=str(STATUS_FILE),
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
