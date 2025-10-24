import os
import time
import schedule
from flask import Flask, jsonify, render_template_string
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
import threading
from pathlib import Path
from dotenv import load_dotenv

from monitoring import update_bot_state

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
})

# 交易参数配置
TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'amount': 0.01,  # 交易数量 (BTC)
    'leverage': 10,  # 杠杆倍数
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
}

MONITOR_CONFIG = {
    'host': os.getenv('MONITOR_HOST', '0.0.0.0'),
    'port': int(os.getenv('MONITOR_PORT', 5000)),
    'refresh_interval': 5,
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

# 监控数据存储
monitor_state = {
    'price_snapshot': None,
    'latest_signal': None,
    'position': None,
    'last_update': None,
}
monitor_lock = threading.Lock()

app = Flask(__name__)

BOT_NAME = Path(__file__).stem


def update_monitor_state(error=None, **kwargs):
    """Thread-safe monitor state update and shared monitor sync."""
    with monitor_lock:
        monitor_state.update({k: v for k, v in kwargs.items() if v is not None})
        if error:
            monitor_state['error'] = str(error)
        else:
            monitor_state.pop('error', None)

        snapshot = {
            'price_snapshot': monitor_state.get('price_snapshot'),
            'latest_signal': monitor_state.get('latest_signal'),
            'signal_history': signal_history[-30:],
            'position': monitor_state.get('position'),
            'last_update': monitor_state.get('last_update'),
            'trade_config': TRADE_CONFIG,
            'metadata': {
                'exchange': 'okx',
                'script': BOT_NAME,
                'timeframe': TRADE_CONFIG['timeframe'],
                'test_mode': TRADE_CONFIG['test_mode'],
            },
        }
        if monitor_state.get('error'):
            snapshot['error'] = monitor_state['error']

    try:
        update_bot_state(BOT_NAME, **snapshot)
    except Exception as monitor_err:
        print(f"共享监控状态更新失败: {monitor_err}")


def serialize_price_snapshot(price_data):
    """Prepare price data for JSON responses without mutating globals."""
    if not price_data:
        return None

    snapshot = dict(price_data)
    klines = []
    for entry in snapshot.get('kline_data', []):
        formatted = dict(entry)
        timestamp = formatted.get('timestamp')
        if hasattr(timestamp, 'strftime'):
            formatted['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted['timestamp'] = str(timestamp)
        klines.append(formatted)
    snapshot['kline_data'] = klines
    return snapshot


@app.route('/api/status')
def api_status():
    with monitor_lock:
        data = {
            'price_snapshot': monitor_state['price_snapshot'],
            'latest_signal': monitor_state['latest_signal'],
            'signal_history': signal_history[-10:],
            'position': monitor_state['position'],
            'last_update': monitor_state['last_update'],
            'trade_config': TRADE_CONFIG,
            'error': monitor_state.get('error'),
        }
    return jsonify(data)


@app.route('/')
def monitor_dashboard():
    refresh = MONITOR_CONFIG['refresh_interval'] * 1000
    html = """
    <!doctype html>
    <html lang="zh">
    <head>
        <meta charset="utf-8">
        <title>OKX 策略监控</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
            h1 { font-size: 1.8rem; margin-bottom: 1rem; }
            .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
            .card { background: #1e293b; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.4); }
            .label { color: #94a3b8; font-size: 0.85rem; }
            .value { font-size: 1.2rem; margin-top: 0.4rem; font-weight: 600; }
            table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
            th, td { border-bottom: 1px solid #334155; padding: 0.5rem; text-align: left; }
            th { color: #cbd5f5; }
            tr:hover { background: rgba(148, 163, 184, 0.1); }
            .signal-buy { color: #34d399; }
            .signal-sell { color: #f87171; }
            .signal-hold { color: #facc15; }
            .timestamp { font-size: 0.8rem; color: #64748b; }
        </style>
    </head>
    <body>
        <h1>OKX 策略监控面板</h1>
        <p class="timestamp">最后更新: <span id="last-update">-</span></p>
        <div class="grid">
            <div class="card">
                <div class="label">当前价格</div>
                <div id="current-price" class="value">-</div>
                <div class="label">价格变化</div>
                <div id="price-change" class="value">-</div>
            </div>
            <div class="card">
                <div class="label">最新信号</div>
                <div id="latest-signal" class="value">-</div>
                <div class="label">信号描述</div>
                <div id="latest-reason"></div>
            </div>
            <div class="card">
                <div class="label">当前持仓</div>
                <div id="current-position" class="value">-</div>
                <div class="label">浮动盈亏</div>
                <div id="unrealized-pnl" class="value">-</div>
            </div>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h2>近期信号</h2>
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>信号</th>
                        <th>信心</th>
                        <th>止损</th>
                        <th>止盈</th>
                    </tr>
                </thead>
                <tbody id="signal-history"></tbody>
            </table>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h2>最近K线</h2>
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>开盘</th>
                        <th>收盘</th>
                        <th>最高</th>
                        <th>最低</th>
                        <th>成交量</th>
                    </tr>
                </thead>
                <tbody id="kline-data"></tbody>
            </table>
        </div>

        <script>
            const refreshInterval = {{ refresh }};

            function formatSignal(signal) {
                if (!signal) return '-';
                const map = {
                    'BUY': 'signal-buy',
                    'SELL': 'signal-sell',
                    'HOLD': 'signal-hold'
                };
                const cls = map[signal.toUpperCase()] || '';
                return `<span class="${cls}">${signal}</span>`;
            }

            async function fetchStatus() {
                try {
                    const res = await fetch('/api/status');
                    const data = await res.json();

                    document.getElementById('last-update').textContent = data.last_update || '-';

                    const price = data.price_snapshot || {};
                    document.getElementById('current-price').textContent = price.price ? `$${price.price.toFixed(2)}` : '-';
                    document.getElementById('price-change').textContent = price.price_change ? `${price.price_change.toFixed(2)}%` : '-';

                    const latestSignal = data.latest_signal;
                    document.getElementById('latest-signal').innerHTML = latestSignal ? formatSignal(latestSignal.signal) : '-';
                    document.getElementById('latest-reason').textContent = latestSignal ? latestSignal.reason : '-';

                    const position = data.position;
                    document.getElementById('current-position').textContent = position ? `${position.side} ${position.size}` : '无持仓';
                    document.getElementById('unrealized-pnl').textContent = position ? `${position.unrealized_pnl.toFixed(2)} USDT` : '-';

                    const historyBody = document.getElementById('signal-history');
                    historyBody.innerHTML = '';
                    (data.signal_history || []).slice().reverse().forEach(item => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${item.timestamp || '-'}</td>
                            <td>${formatSignal(item.signal || '-')}</td>
                            <td>${item.confidence || '-'}</td>
                            <td>${item.stop_loss ? `$${item.stop_loss.toFixed(2)}` : '-'}</td>
                            <td>${item.take_profit ? `$${item.take_profit.toFixed(2)}` : '-'}</td>
                        `;
                        historyBody.appendChild(tr);
                    });

                    const klineBody = document.getElementById('kline-data');
                    klineBody.innerHTML = '';
                    (price.kline_data || []).slice().reverse().forEach(item => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${item.timestamp || '-'}</td>
                            <td>${Number(item.open).toFixed(2)}</td>
                            <td>${Number(item.close).toFixed(2)}</td>
                            <td>${Number(item.high).toFixed(2)}</td>
                            <td>${Number(item.low).toFixed(2)}</td>
                            <td>${Number(item.volume).toFixed(4)}</td>
                        `;
                        klineBody.appendChild(tr);
                    });
                } catch (err) {
                    console.error('Failed to fetch status', err);
                }
            }

            fetchStatus();
            setInterval(fetchStatus, refreshInterval);
        </script>
    </body>
    </html>
    """
    return render_template_string(html, refresh=refresh)


def start_monitor_server():
    """Launch the Flask monitoring server in a background thread."""
    def run():
        app.run(host=MONITOR_CONFIG['host'], port=MONITOR_CONFIG['port'], debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"Web监控已启动: http://{MONITOR_CONFIG['host']}:{MONITOR_CONFIG['port']}")


def setup_exchange():
    """设置交易所参数"""
    try:
        # OKX设置杠杆
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 全仓模式，也可用'isolated'逐仓
        )
        print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"当前USDT余额: {usdt_balance:.2f}")

        # # 设置持仓模式 (双向持仓)
        # exchange.set_position_mode(False, TRADE_CONFIG['symbol'])
        # print("设置单向持仓")

        return True
    except Exception as e:
        print(f"交易所设置失败: {e}")
        return False


def get_btc_ohlcv():
    """获取BTC/USDT的K线数据"""
    try:
        # 获取最近10根K线
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'], limit=10)

        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_dict('records')
        }
    except Exception as e:
        print(f"获取K线数据失败: {e}")
        return None


def get_current_position():
    """获取当前持仓情况"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号"""

    # 添加当前价格到历史记录
    price_history.append(price_data)
    if len(price_history) > 20:
        price_history.pop(0)

    # 构建K线数据文本
    kline_text = f"【最近5根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

    # 构建技术指标文本
    if len(price_history) >= 5:
        closes = [data['price'] for data in price_history[-5:]]
        sma_5 = sum(closes) / len(closes)
        price_vs_sma = ((price_data['price'] - sma_5) / sma_5) * 100

        indicator_text = f"【技术指标】\n5周期均价: {sma_5:.2f}\n当前价格相对于均线: {price_vs_sma:+.2f}%"
    else:
        indicator_text = "【技术指标】\n数据不足计算技术指标"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {indicator_text}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格,
        "take_profit": 具体价格,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，你现在需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            print(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return None


def execute_trade(signal_data, price_data):
    """执行交易"""
    global position

    current_position = get_current_position()

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    print(f"止损: ${signal_data['stop_loss']:,.2f}")
    print(f"止盈: ${signal_data['take_profit']:,.2f}")
    print(f"当前持仓: {current_position}")

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        return current_position

    try:
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                # 平空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                )
                time.sleep(1)
                # 开多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif not current_position:
                print("开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            else:
                print("已持有多仓，无需操作")

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                # 平多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                )
                time.sleep(1)
                # 开空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif not current_position:
                print("开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            else:
                print("已持有空仓，无需操作")

        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            return current_position

        print("订单执行成功")
        # 更新持仓信息
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")
        return position

    except Exception as e:
        print(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()
        update_monitor_state(error=e)
        return current_position

    return get_current_position()


def trading_bot():
    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取K线数据
    price_data = get_btc_ohlcv()
    if not price_data:
        update_monitor_state(
            signal_history=signal_history[-30:],
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            error="获取K线数据失败",
        )
        return

    print(f"BTC当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    serialized_price = serialize_price_snapshot(price_data)

    # 2. 使用DeepSeek分析
    signal_data = analyze_with_deepseek(price_data)
    if not signal_data:
        update_monitor_state(
            price_snapshot=serialized_price,
            signal_history=signal_history[-30:],
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            error="DeepSeek分析失败",
        )
        return

    # 3. 执行交易
    position_snapshot = execute_trade(signal_data, price_data)
    update_monitor_state(
        price_snapshot=serialized_price,
        latest_signal=signal_data,
        signal_history=signal_history[-30:],
        position=position_snapshot,
        last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        error=None,
    )


def main():
    """主函数"""
    print("BTC/USDT OKX自动交易机器人启动成功！")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用K线数据分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        update_monitor_state(error="交易所初始化失败")
        return

    update_monitor_state(
        position=get_current_position(),
        signal_history=signal_history[-30:],
        last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        error=None,
    )

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        schedule.every(15).minutes.do(trading_bot)
        print("执行频率: 每15分钟一次")
    else:
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
