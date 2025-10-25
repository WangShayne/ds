import os
import time
from collections import deque
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime, timezone
import json
import re
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

# 配置解析工具
def get_env_float(key, default):
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        log_event(f"环境变量 {key} 解析失败，使用默认值 {default}", level="WARNING")
        return default


def get_env_int(key, default):
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        log_event(f"环境变量 {key} 解析失败，使用默认值 {default}", level="WARNING")
        return default


def get_env_bool(key, default):
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    log_event(f"环境变量 {key} 解析失败，使用默认值 {default}", level="WARNING")
    return default


# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'symbol': os.getenv('TRADE_SYMBOL', 'BTC/USDT:USDT'),  # OKX的合约符号格式
    'amount': get_env_float('TRADE_AMOUNT', 0.01),  # 交易数量 (BTC)
    'leverage': get_env_int('TRADE_LEVERAGE', 10),  # 杠杆倍数
    'timeframe': os.getenv('TRADE_TIMEFRAME', '15m'),  # 使用15分钟K线
    'test_mode': get_env_bool('TRADE_TEST_MODE', False),  # 测试模式
    'data_points': get_env_int('TRADE_DATA_POINTS', 96),  # 24小时数据（96根15分钟K线）
    'analysis_periods': {
        'short_term': get_env_int('TRADE_SHORT_TERM', 20),  # 短期均线
        'medium_term': get_env_int('TRADE_MEDIUM_TERM', 50),  # 中期均线
        'long_term': get_env_int('TRADE_LONG_TERM', 96)  # 长期趋势
    }
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

order_history = deque(maxlen=30)
runtime_log = deque(maxlen=100)
deepseek_log = deque(maxlen=20)
account_snapshot = {}

BOT_NAME = Path(__file__).stem


def log_event(message, level="INFO", also_print=True):
    """记录运行日志并可选输出到控制台。"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = {
        'timestamp': timestamp,
        'level': level,
        'message': str(message),
    }
    runtime_log.append(entry)
    if also_print:
        print(f"[{timestamp}] [{level}] {message}")


def record_order(action, side, amount, params=None, response=None, note=None):
    """保存订单执行结果到历史记录。"""
    params = params or {}
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': action,
        'side': side,
        'amount': float(amount) if amount is not None else None,
        'symbol': response.get('symbol') if isinstance(response, dict) else TRADE_CONFIG['symbol'],
        'posSide': params.get('posSide'),
        'reduceOnly': params.get('reduceOnly'),
        'tdMode': params.get('tdMode'),
        'note': note or action,
    }

    if isinstance(response, dict):
        entry['id'] = response.get('id') or response.get('orderId')
        status = response.get('status')
        info = response.get('info') or {}
        if not status and isinstance(info, dict):
            status = info.get('state') or info.get('status')
        entry['status'] = status

        price = response.get('price')
        filled = response.get('filled')
        cost = response.get('cost')
        entry['price'] = float(price) if price not in (None, '') else None
        entry['filled'] = float(filled) if filled not in (None, '') else None
        entry['cost'] = float(cost) if cost not in (None, '') else None

        if isinstance(info, dict):
            entry['exchangeMessage'] = info.get('sMsg') or info.get('msg')
            entry['code'] = info.get('sCode') or info.get('code')

        fee = response.get('fee')
        if isinstance(fee, dict):
            entry['fee'] = {
                'cost': fee.get('cost'),
                'currency': fee.get('currency'),
            }

    order_history.append(entry)


def record_deepseek(prompt_text, response_text, status="success"):
    """记录与DeepSeek的通讯信息（截断以便展示）。"""
    status = (status or "info").upper()
    def _trim(text, limit=600):
        if text is None:
            return None
        text = str(text)
        return text if len(text) <= limit else text[:limit] + "..."

    deepseek_log.append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'prompt': _trim(prompt_text, 700),
        'response': _trim(response_text, 700),
    })


def update_account_snapshot(balance=None, usdt_balance=None, position_snapshot=None):
    """更新账户视图数据，供监控展示收益与持仓。"""
    snapshot = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'currency': 'USDT',
    }

    if usdt_balance is not None:
        try:
            snapshot['available'] = float(usdt_balance)
        except (TypeError, ValueError):
            snapshot['available'] = usdt_balance

    if isinstance(balance, dict):
        free_map = balance.get('free') or {}
        total_map = balance.get('total') or {}
        used_map = balance.get('used') or {}
        for label, value in (('free', free_map.get('USDT')), ('total', total_map.get('USDT')), ('used', used_map.get('USDT'))):
            if value is not None:
                try:
                    snapshot[label] = float(value)
                except (TypeError, ValueError):
                    snapshot[label] = value

        info = balance.get('info')
        if isinstance(info, dict):
            equity = info.get('equity') or info.get('totalEq')
            if equity is not None:
                try:
                    snapshot['equity'] = float(equity)
                except (TypeError, ValueError):
                    snapshot['equity'] = equity

    if position_snapshot:
        pnl = position_snapshot.get('unrealized_pnl')
        if pnl is not None:
            try:
                snapshot['unrealized_pnl'] = float(pnl)
            except (TypeError, ValueError):
                snapshot['unrealized_pnl'] = pnl
        snapshot['position_side'] = position_snapshot.get('side')
        snapshot['position_size'] = position_snapshot.get('size')
        snapshot['entry_price'] = position_snapshot.get('entry_price')

    account_snapshot.update({k: v for k, v in snapshot.items() if v is not None})


def sync_monitor(price_data=None, signal_data=None, position_snapshot=None, error=None, extra_metrics=None):
    """Persist monitoring data for the enhanced OKX strategy."""
    try:
        price_snapshot = None
        if price_data is not None:
            if isinstance(price_data, dict):
                price_snapshot = dict(price_data)
                price_snapshot.pop('full_data', None)
            else:
                price_snapshot = price_data

        payload = {
            'price_snapshot': price_snapshot,
            'latest_signal': signal_data,
            'signal_history': signal_history[-30:],
            'position': position_snapshot,
            'trade_config': TRADE_CONFIG,
            'metadata': {
                'exchange': 'okx',
                'script': BOT_NAME,
                'timeframe': TRADE_CONFIG['timeframe'],
                'test_mode': TRADE_CONFIG['test_mode'],
                'order_history_size': len(order_history),
                'log_entries': len(runtime_log),
                'deepseek_logs': len(deepseek_log),
            },
            'error': str(error) if error else None,
            'orders': list(order_history),
            'logs': list(runtime_log),
            'deepseek_messages': list(deepseek_log),
            'account': dict(account_snapshot) if account_snapshot else None,
        }
        if extra_metrics:
            payload['metadata'].update(extra_metrics)
        update_bot_state(BOT_NAME, **payload)
    except Exception as monitor_err:
        log_event(f"监控状态更新失败: {monitor_err}", level="ERROR", also_print=True)


def setup_exchange():
    """设置交易所参数"""
    try:
        # OKX设置杠杆
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 全仓模式
        )
        log_event(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        balance = exchange.fetch_balance()

        usdt_balance = None
        # OKX在不同账户模式下字段可能不同，逐项尝试
        if isinstance(balance, dict):
            usdt_entry = balance.get('USDT') or balance.get('USDT:USDT')
            if isinstance(usdt_entry, dict):
                usdt_balance = usdt_entry.get('free') or usdt_entry.get('total')

            if usdt_balance is None:
                free_map = balance.get('free') or {}
                total_map = balance.get('total') or {}
                usdt_balance = free_map.get('USDT') or total_map.get('USDT')

        if usdt_balance is None:
            raise KeyError(f"未找到USDT余额字段，可用字段: {list(balance.keys())}")

        log_event(f"当前USDT余额: {float(usdt_balance):.2f}")
        update_account_snapshot(balance=balance, usdt_balance=usdt_balance, position_snapshot=get_current_position())

        return True
    except Exception as e:
        log_event(f"交易所设置失败: {e}", level="ERROR")
        return False


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        log_event(f"技术指标计算失败: {e}", level="ERROR")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        log_event(f"支撑阻力计算失败: {e}", level="ERROR")
        return {}


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        log_event(f"趋势分析失败: {e}", level="ERROR")
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert exchange timestamps to timezone-aware UTC to satisfy monitoring serialization
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        kline_records = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).copy()
        kline_records['timestamp'] = kline_records['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': kline_records.to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        log_event(f"获取增强K线数据失败: {e}", level="ERROR")
        return None


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value and pd.notna(value) else default

    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
    - 20周期: {safe_float(tech['sma_20']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
    - 50周期: {safe_float(tech['sma_50']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
    """
    return analysis_text


def get_current_position():
    """获取当前持仓情况 - OKX版本"""
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
        log_event(f"获取持仓失败: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return None


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log_event(f"JSON解析失败，原始内容: {json_str}", level="ERROR", also_print=False)
            log_event(f"错误详情: {e}", level="ERROR", also_print=False)
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data['price'] * 0.98,  # -2%
        "take_profit": price_data['price'] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本
    kline_text = f"【最近5根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data'][-5:]):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

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

    {technical_analysis}

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

    【重要格式要求】
    - 必须返回纯JSON格式，不要有任何额外文本
    - 所有属性名必须使用双引号
    - 不要使用单引号
    - 不要添加注释
    - 确保JSON格式完全正确

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
                 "content": f"您是一位专业的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        log_event("DeepSeek请求已完成，正在解析结果", level="DEBUG")

        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)

            if signal_data is None:
                signal_data = create_fallback_signal(price_data)
        else:
            signal_data = create_fallback_signal(price_data)

        # 验证必需字段
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            signal_data = create_fallback_signal(price_data)

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)
        record_deepseek(prompt, result, status="success")

        # 信号统计
        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        log_event(f"信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)", level="DEBUG")

        # 信号连续性检查
        if len(signal_history) >= 3:
            last_three = [s['signal'] for s in signal_history[-3:]]
            if len(set(last_three)) == 1:
                log_event(f"⚠️ 注意：连续3次{signal_data['signal']}信号", level="WARNING")

        return signal_data

    except Exception as e:
        log_event(f"DeepSeek分析失败: {e}", level="ERROR")
        record_deepseek(prompt, str(e), status="error")
        return create_fallback_signal(price_data)


def execute_trade(signal_data, price_data):
    """执行交易 - OKX版本（修复保证金检查）"""
    global position

    current_position = get_current_position()

    log_event(f"交易信号: {signal_data['signal']}")
    log_event(f"信心程度: {signal_data['confidence']}", level="DEBUG")
    log_event(f"理由: {signal_data['reason']}", level="DEBUG")
    log_event(f"止损: ${signal_data['stop_loss']:,.2f}", level="DEBUG")
    log_event(f"止盈: ${signal_data['take_profit']:,.2f}", level="DEBUG")
    log_event(f"当前持仓: {current_position}", level="DEBUG")

    # 风险管理：低信心信号不执行
    if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
        log_event("⚠️ 低信心信号，跳过执行", level="WARNING")
        return current_position

    if TRADE_CONFIG['test_mode']:
        log_event("测试模式 - 仅模拟交易", level="INFO")
        return current_position

    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        update_account_snapshot(balance=balance, usdt_balance=usdt_balance, position_snapshot=current_position)

        # 智能保证金检查
        required_margin = 0

        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                # 平空仓 + 开多仓：需要额外保证金
                required_margin = price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']
                operation_type = "平空开多"
            elif not current_position:
                # 开多仓：需要保证金
                required_margin = price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']
                operation_type = "开多仓"
            else:
                # 已持有多仓：不需要额外保证金
                required_margin = 0
                operation_type = "保持多仓"

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                # 平多仓 + 开空仓：需要额外保证金
                required_margin = price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']
                operation_type = "平多开空"
            elif not current_position:
                # 开空仓：需要保证金
                required_margin = price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']
                operation_type = "开空仓"
            else:
                # 已持有空仓：不需要额外保证金
                required_margin = 0
                operation_type = "保持空仓"

        elif signal_data['signal'] == 'HOLD':
            log_event("建议观望，不执行交易", level="INFO")
            return current_position

        log_event(f"操作类型: {operation_type}, 需要保证金: {required_margin:.2f} USDT", level="DEBUG")

        # 只有在需要额外保证金时才检查
        if required_margin > 0:
            if required_margin > usdt_balance * 0.8:
                log_event(f"⚠️ 保证金不足，跳过交易。需要: {required_margin:.2f} USDT, 可用: {usdt_balance:.2f} USDT", level="WARNING")
                return current_position
        else:
            log_event("✅ 无需额外保证金，继续执行", level="INFO")

        # 执行交易逻辑   tag 是我的经纪商api（不拿白不拿），不会影响大家返佣，介意可以删除
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                log_event("平空仓并开多仓...")
                # 平空仓
                close_params = {
                    'reduceOnly': True,
                    'posSide': 'short',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_close = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params=close_params
                )
                record_order("close_short", 'buy', current_position['size'], params=close_params, response=response_close, note="平空仓")
                time.sleep(1)
                # 开多仓
                open_params = {
                    'posSide': 'long',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_open = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params=open_params
                )
                record_order("open_long", 'buy', TRADE_CONFIG['amount'], params=open_params, response=response_open, note="开多仓")
            elif current_position and current_position['side'] == 'long':
                log_event("已有多头持仓，保持现状", level="INFO")
            else:
                # 无持仓时开多仓
                log_event("开多仓...")
                open_params = {
                    'posSide': 'long',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_open = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params=open_params
                )
                record_order("open_long", 'buy', TRADE_CONFIG['amount'], params=open_params, response=response_open, note="开多仓")

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                log_event("平多仓并开空仓...")
                # 平多仓
                close_params = {
                    'reduceOnly': True,
                    'posSide': 'long',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_close = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params=close_params
                )
                record_order("close_long", 'sell', current_position['size'], params=close_params, response=response_close, note="平多仓")
                time.sleep(1)
                # 开空仓
                open_params = {
                    'posSide': 'short',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_open = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params=open_params
                )
                record_order("open_short", 'sell', TRADE_CONFIG['amount'], params=open_params, response=response_open, note="开空仓")
            elif current_position and current_position['side'] == 'short':
                log_event("已有空头持仓，保持现状", level="INFO")
            else:
                # 无持仓时开空仓
                log_event("开空仓...")
                open_params = {
                    'posSide': 'short',
                    'tdMode': 'cross',
                    'tag': 'f1ee03b510d5SUDE'
                }
                response_open = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params=open_params
                )
                record_order("open_short", 'sell', TRADE_CONFIG['amount'], params=open_params, response=response_open, note="开空仓")

        log_event("订单执行成功")
        time.sleep(2)
        position = get_current_position()
        log_event(f"更新后持仓: {position}")
        update_account_snapshot(position_snapshot=position)
        return position

    except Exception as e:
        log_event(f"订单执行失败: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        sync_monitor(price_data=price_data, signal_data=signal_data, position_snapshot=current_position, error=e)
        return current_position

    return get_current_position()


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            log_event(f"第{attempt + 1}次尝试失败，进行重试...", level="WARNING")
            time.sleep(1)

        except Exception as e:
            log_event(f"第{attempt + 1}次尝试异常: {e}", level="ERROR")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def trading_bot():
    """主交易机器人函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_event("=" * 60, level="DEBUG")
    log_event(f"执行时间: {timestamp}")
    log_event("=" * 60, level="DEBUG")

    # 1. 获取增强版K线数据
    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        sync_monitor(error="获取增强K线数据失败")
        return

    log_event(f"BTC当前价格: ${price_data['price']:,.2f}")
    log_event(f"数据周期: {TRADE_CONFIG['timeframe']}")
    log_event(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        log_event("⚠️ 使用备用交易信号", level="WARNING")

    # 3. 执行交易
    position_snapshot = execute_trade(signal_data, price_data)
    extra_metrics = {
        'overall_trend': price_data.get('trend_analysis', {}).get('overall'),
        'short_term_trend': price_data.get('trend_analysis', {}).get('short_term'),
        'rsi': price_data.get('technical_data', {}).get('rsi'),
        'is_fallback_signal': signal_data.get('is_fallback', False),
    }
    sync_monitor(
        price_data=price_data,
        signal_data=signal_data,
        position_snapshot=position_snapshot,
        error=None,
        extra_metrics=extra_metrics,
    )


def main():
    """主函数"""
    log_event("BTC/USDT OKX自动交易机器人启动成功！")
    log_event("融合技术指标策略 + OKX实盘接口")

    if TRADE_CONFIG['test_mode']:
        log_event("当前为模拟模式，不会真实下单", level="WARNING")
    else:
        log_event("实盘交易模式，请谨慎操作！", level="WARNING")

    log_event(f"交易周期: {TRADE_CONFIG['timeframe']}")
    log_event("已启用完整技术指标分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        log_event("交易所初始化失败，程序退出", level="ERROR")
        sync_monitor(error="交易所初始化失败")
        return

    sync_monitor(
        position_snapshot=get_current_position(),
        error=None,
        extra_metrics={'initialised': True},
    )

    # 根据时间周期设置执行频率
    timeframe = TRADE_CONFIG['timeframe']
    if timeframe == '1m':
        schedule.every().minute.do(trading_bot)
        log_event("执行频率: 每1分钟一次")
    elif timeframe == '5m':
        schedule.every(5).minutes.do(trading_bot)
        log_event("执行频率: 每5分钟一次")
    elif timeframe == '15m':
        schedule.every(15).minutes.do(trading_bot)
        log_event("执行频率: 每15分钟一次")
    elif timeframe == '30m':
        schedule.every(30).minutes.do(trading_bot)
        log_event("执行频率: 每30分钟一次")
    elif timeframe == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        log_event("执行频率: 每1小时一次（:01）")
    elif timeframe == '4h':
        schedule.every(4).hours.at(":01").do(trading_bot)
        log_event("执行频率: 每4小时一次（:01）")
    else:
        schedule.every().hour.at(":01").do(trading_bot)
        log_event(f"未识别周期 {timeframe}，默认每小时一次", level="WARNING")

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
