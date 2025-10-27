import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

from monitoring.state import update_bot_state

load_dotenv()

BOT_NAME = Path(__file__).stem


def configure_logging() -> logging.Logger:
    """Configure stdout and rotating file logging for the bot."""

    level_name = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(os.getenv("BOT_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(BOT_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(
            log_dir / f"{BOT_NAME}.log",
            maxBytes=int(os.getenv("BOT_LOG_MAX_BYTES", 5_242_880)),
            backupCount=int(os.getenv("BOT_LOG_BACKUP_COUNT", 5)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = configure_logging()

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

# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'leverage': 10,  # 杠杆倍数,只影响保证金不影响下单价值
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
    'data_points': 96,  # 24小时数据（96根15分钟K线）
    'analysis_periods': {
        'short_term': 20,  # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96  # 长期趋势
    },
    # 新增智能仓位参数
    'position_management': {
        'base_usdt_amount': 100,  # USDT投入下单基数
        'high_confidence_multiplier': 1.5,
        'medium_confidence_multiplier': 1.0,
        'low_confidence_multiplier': 0.5,
        'max_position_ratio': 0.25,  # 单次最大仓位比例（25%可用余额）
        'trend_strength_multiplier': 1.2
    }
}

ORDER_TAG = os.getenv("OKX_ORDER_TAG", "60bb4a8d3416BCDE")


def setup_exchange():
    """设置交易所参数 - 强制全仓模式"""
    try:
        logger.info("Loading OKX market metadata for %s", TRADE_CONFIG['symbol'])
        markets = exchange.load_markets()
        btc_market = markets[TRADE_CONFIG['symbol']]

        contract_size = float(btc_market['contractSize'])
        TRADE_CONFIG['contract_size'] = contract_size
        TRADE_CONFIG['min_amount'] = btc_market['limits']['amount']['min']
        logger.info("Contract spec: 1 contract = %.6f BTC", contract_size)
        logger.info("Minimum order size: %s contracts", TRADE_CONFIG['min_amount'])

        logger.info("Checking for existing isolated positions")
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] != TRADE_CONFIG['symbol']:
                continue

            contracts = float(pos.get('contracts', 0) or 0)
            mode = pos.get('mgnMode')
            if contracts > 0 and mode == 'isolated':
                logger.error(
                    "Detected isolated position %s with %s contracts at %s; aborting",
                    pos.get('side'),
                    contracts,
                    pos.get('entryPrice'),
                )
                return False

        logger.info("Ensuring single-side position mode")
        try:
            exchange.set_position_mode(False, TRADE_CONFIG['symbol'])
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.warning("Failed to set single-side mode (likely already set): %s", exc)

        logger.info("Setting cross margin leverage to %sx", TRADE_CONFIG['leverage'])
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}
        )

        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        logger.info("Available USDT balance: %.2f", usdt_balance)

        current_pos = get_current_position()
        if current_pos:
            logger.info(
                "Current position detected: %s %.2f contracts",
                current_pos['side'],
                current_pos['size'],
            )
        else:
            logger.info("No open positions detected")

        logger.info("Exchange configuration completed (cross margin + single side)")
        return True

    except Exception as exc:  # noqa: BLE001 - need full trace for exchange setup
        logger.exception("Exchange setup failed: %s", exc)
        return False


# 全局变量存储历史数据
price_history = []
signal_history = []


def calculate_intelligent_position(signal_data, price_data, current_position):
    """计算智能仓位大小 - 修复版"""
    config = TRADE_CONFIG['position_management']

    try:
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])

        base_usdt = float(config['base_usdt_amount'])

        confidence_multiplier = {
            'HIGH': float(config['high_confidence_multiplier']),
            'MEDIUM': float(config['medium_confidence_multiplier']),
            'LOW': float(config['low_confidence_multiplier'])
        }.get(signal_data.get('confidence'), 1.0)

        trend = price_data['trend_analysis'].get('overall', '震荡整理')
        trend_multiplier = (
            float(config['trend_strength_multiplier'])
            if trend in ['强势上涨', '强势下跌']
            else 1.0
        )

        rsi = float(price_data['technical_data'].get('rsi', 50))
        rsi_multiplier = 0.7 if rsi > 75 or rsi < 25 else 1.0

        suggested_usdt = base_usdt * confidence_multiplier * trend_multiplier * rsi_multiplier

        max_ratio = max(0.0, min(float(config.get('max_position_ratio', 0.25)), 1.0))
        max_usdt = usdt_balance * max_ratio
        final_usdt = max(0.0, min(suggested_usdt, max_usdt, usdt_balance))

        contract_denom = price_data['price'] * TRADE_CONFIG['contract_size']
        if contract_denom <= 0:
            raise ValueError("Invalid contract denominator for position sizing")

        contract_size = final_usdt / contract_denom
        contract_size = round(contract_size, 2)

        min_contracts = TRADE_CONFIG.get('min_amount', 0.01)
        if contract_size < min_contracts:
            contract_size = min_contracts
            logger.warning("Position size below minimum, adjusting to %.2f contracts", contract_size)

        logger.info(
            "Position sizing completed: balance %.2f USDT, final %.2f USDT, contracts %.2f",
            usdt_balance,
            final_usdt,
            contract_size,
        )
        logger.debug(
            "Sizing breakdown | base %.2f | confidence %.2f | trend %.2f | rsi %.2f | suggested %.2f | max %.2f",
            base_usdt,
            confidence_multiplier,
            trend_multiplier,
            rsi_multiplier,
            suggested_usdt,
            max_usdt,
        )

        return contract_size

    except Exception as e:
        logger.exception("Failed to calculate intelligent position size: %s", e)
        base_usdt = float(config['base_usdt_amount'])
        contract_size = (base_usdt * TRADE_CONFIG['leverage']) / (
            price_data['price'] * TRADE_CONFIG.get('contract_size', 0.01)
        )
        fallback_size = round(max(contract_size, TRADE_CONFIG.get('min_amount', 0.01)), 2)
        logger.info("Using fallback contract size %.2f", fallback_size)
        return fallback_size


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
        logger.exception("Failed to compute technical indicators: %s", e)
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
        logger.exception("Failed to calculate support/resistance levels: %s", e)
        return {}


def get_sentiment_indicators():
    """获取情绪指标 - 简洁版本"""
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "2b144650-4a16-4eb5-bbcd-70824577687b"

        # 获取最近4小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # 只保留核心指标
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": ["BTC"]
        }

        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = requests.post(API_URL, json=request_body, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                # 查找第一个有有效数据的时间段
                for period in time_periods:
                    period_data = period.get("data", [])

                    sentiment = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = item.get("value", "").strip()

                        if value:  # 只处理非空值
                            try:
                                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                    sentiment[endpoint] = float(value)
                                    valid_data_found = True
                            except (ValueError, TypeError):
                                continue

                    # 如果找到有效数据
                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment['CO-A-02-01']
                        negative = sentiment['CO-A-02-02']
                        net_sentiment = positive - negative

                        # 正确的时间延迟计算
                        data_delay = int((datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                        logger.info(
                            "Using sentiment data at %s (delay %s minutes)",
                            period['startTime'],
                            data_delay,
                        )

                        return {
                            'positive_ratio': positive,
                            'negative_ratio': negative,
                            'net_sentiment': net_sentiment,
                            'data_time': period['startTime'],
                            'data_delay_minutes': data_delay
                        }

                logger.warning("Sentiment API returned empty data for all periods")
                return None

        return None
    except Exception as e:
        logger.exception("Failed to retrieve sentiment indicators: %s", e)
        return None


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
        logger.exception("Failed to evaluate market trend: %s", e)
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
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
        logger.exception("Failed to fetch enhanced OHLCV data: %s", e)
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
        logger.exception("Failed to fetch positions: %s", e)
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
            logger.warning("Failed to parse JSON response: %s", e, exc_info=False)
            logger.debug("Original JSON payload: %s", json_str)
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

    # 获取情绪数据
    sentiment_data = get_sentiment_indicators()
    # 简化情绪文本 多了没用
    if sentiment_data:
        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        sentiment_text = f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
    else:
        sentiment_text = "【市场情绪】数据暂不可用"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
    pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']:.2f} USDT" if current_pos else ""

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {technical_analysis}

    {signal_text}

    {sentiment_text}  # 添加情绪分析

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}{pnl_text}

    【防频繁交易重要原则】
    1. **趋势持续性优先**: 不要因单根K线或短期波动改变整体趋势判断
    2. **持仓稳定性**: 除非趋势明确强烈反转，否则保持现有持仓方向
    3. **反转确认**: 需要至少2-3个技术指标同时确认趋势反转才改变信号
    4. **成本意识**: 减少不必要的仓位调整，每次交易都有成本

    【交易指导原则 - 必须遵守】
    1. **技术分析主导** (权重60%)：趋势、支撑阻力、K线形态是主要依据
    2. **市场情绪辅助** (权重30%)：情绪数据用于验证技术信号，不能单独作为交易理由  
    - 情绪与技术同向 → 增强信号信心
    - 情绪与技术背离 → 以技术分析为主，情绪仅作参考
    - 情绪数据延迟 → 降低权重，以实时技术指标为准
    3. **风险管理** (权重10%)：考虑持仓、盈亏状况和止损位置
    4. **趋势跟随**: 明确趋势出现时立即行动，不要过度等待
    5. 因为做的是btc，做多权重可以大一点点
    6. **信号明确性**:
    - 强势上涨趋势 → BUY信号
    - 强势下跌趋势 → SELL信号  
    - 仅在窄幅震荡、无明确方向时 → HOLD信号
    7. **技术指标权重**:
    - 趋势(均线排列) > RSI > MACD > 布林带
    - 价格突破关键支撑/阻力位是重要信号 


    【当前技术状况分析】
    - 整体趋势: {price_data['trend_analysis'].get('overall', 'N/A')}
    - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')} 
    - RSI状态: {price_data['technical_data'].get('rsi', 0):.1f} ({'超买' if price_data['technical_data'].get('rsi', 0) > 70 else '超卖' if price_data['technical_data'].get('rsi', 0) < 30 else '中性'})
    - MACD方向: {price_data['trend_analysis'].get('macd', 'N/A')}

    【智能仓位管理规则 - 必须遵守】

    1. **减少过度保守**：
       - 明确趋势中不要因轻微超买/超卖而过度HOLD
       - RSI在30-70区间属于健康范围，不应作为主要HOLD理由
       - 布林带位置在20%-80%属于正常波动区间

    2. **趋势跟随优先**：
       - 强势上涨趋势 + 任何RSI值 → 积极BUY信号
       - 强势下跌趋势 + 任何RSI值 → 积极SELL信号
       - 震荡整理 + 无明确方向 → HOLD信号

    3. **突破交易信号**：
       - 价格突破关键阻力 + 成交量放大 → 高信心BUY
       - 价格跌破关键支撑 + 成交量放大 → 高信心SELL

    4. **持仓优化逻辑**：
       - 已有持仓且趋势延续 → 保持或BUY/SELL信号
       - 趋势明确反转 → 及时反向信号
       - 不要因为已有持仓而过度HOLD

    【重要】请基于技术分析做出明确判断，避免因过度谨慎而错过趋势行情！

    【分析要求】
    基于以上分析，请给出明确的交易信号

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简要分析理由(包含趋势判断和技术依据)",
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
        logger.debug("DeepSeek raw response: %s", result)

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

        # 信号统计
        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        logger.debug(
            "Signal stats: %s occurred %s/%s times",
            signal_data['signal'],
            signal_count,
            total_signals,
        )

        # 信号连续性检查
        if len(signal_history) >= 3:
            last_three = [s['signal'] for s in signal_history[-3:]]
            if len(set(last_three)) == 1:
                logger.warning("Signal %s repeated three times consecutively", signal_data['signal'])

        return signal_data

    except Exception as e:
        logger.exception("DeepSeek analysis failed: %s", e)
        return create_fallback_signal(price_data)


def execute_intelligent_trade(signal_data, price_data):
    """执行智能交易 - OKX版本（支持同方向加仓减仓）"""
    current_position = get_current_position()

    logger.info(
        "Signal %s (confidence %s) | reason: %s",
        signal_data.get('signal'),
        signal_data.get('confidence'),
        signal_data.get('reason'),
    )
    logger.debug("Current position snapshot: %s", current_position)

    if signal_data.get('signal') == 'HOLD':
        logger.info("HOLD signal received; skip order placement")
        return current_position

    position_size = calculate_intelligent_position(signal_data, price_data, current_position)

    if signal_data.get('confidence') == 'LOW' and not TRADE_CONFIG['test_mode']:
        logger.info("Skipping low-confidence signal in live mode")
        return current_position

    if TRADE_CONFIG['test_mode']:
        logger.info(
            "Test mode active; simulated %s order of %.2f contracts",
            signal_data.get('signal'),
            position_size,
        )
        return current_position

    try:
        symbol = TRADE_CONFIG['symbol']
        target_side = signal_data.get('signal')

        if target_side == 'BUY':
            if current_position and current_position['side'] == 'short':
                if current_position['size'] > 0:
                    logger.info(
                        "Closing short %.2f contracts before opening long",
                        current_position['size'],
                    )
                    exchange.create_market_order(
                        symbol,
                        'buy',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': ORDER_TAG}
                    )
                    time.sleep(1)
                else:
                    logger.warning("Detected short position with zero size; skipping reduce step")

                logger.info("Opening long position of %.2f contracts", position_size)
                exchange.create_market_order(
                    symbol,
                    'buy',
                    position_size,
                    params={'tag': ORDER_TAG}
                )

            elif current_position and current_position['side'] == 'long':
                size_diff = round(position_size - current_position['size'], 2)
                if abs(size_diff) >= 0.01:
                    if size_diff > 0:
                        logger.info(
                            "Scaling in long position by %.2f contracts (current %.2f)",
                            size_diff,
                            current_position['size'],
                        )
                        exchange.create_market_order(
                            symbol,
                            'buy',
                            size_diff,
                            params={'tag': ORDER_TAG}
                        )
                    else:
                        reduce_size = abs(size_diff)
                        logger.info(
                            "Scaling out long position by %.2f contracts (current %.2f)",
                            reduce_size,
                            current_position['size'],
                        )
                        exchange.create_market_order(
                            symbol,
                            'sell',
                            reduce_size,
                            params={'reduceOnly': True, 'tag': ORDER_TAG}
                        )
                else:
                    logger.info(
                        "Existing long position aligned with target (current %.2f, target %.2f)",
                        current_position['size'],
                        position_size,
                    )
            else:
                logger.info("Opening new long position of %.2f contracts", position_size)
                exchange.create_market_order(
                    symbol,
                    'buy',
                    position_size,
                    params={'tag': ORDER_TAG}
                )

        elif target_side == 'SELL':
            if current_position and current_position['side'] == 'long':
                if current_position['size'] > 0:
                    logger.info(
                        "Closing long %.2f contracts before opening short",
                        current_position['size'],
                    )
                    exchange.create_market_order(
                        symbol,
                        'sell',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': ORDER_TAG}
                    )
                    time.sleep(1)
                else:
                    logger.warning("Detected long position with zero size; skipping reduce step")

                logger.info("Opening short position of %.2f contracts", position_size)
                exchange.create_market_order(
                    symbol,
                    'sell',
                    position_size,
                    params={'tag': ORDER_TAG}
                )

            elif current_position and current_position['side'] == 'short':
                size_diff = round(position_size - current_position['size'], 2)
                if abs(size_diff) >= 0.01:
                    if size_diff > 0:
                        logger.info(
                            "Scaling in short position by %.2f contracts (current %.2f)",
                            size_diff,
                            current_position['size'],
                        )
                        exchange.create_market_order(
                            symbol,
                            'sell',
                            size_diff,
                            params={'tag': ORDER_TAG}
                        )
                    else:
                        reduce_size = abs(size_diff)
                        logger.info(
                            "Scaling out short position by %.2f contracts (current %.2f)",
                            reduce_size,
                            current_position['size'],
                        )
                        exchange.create_market_order(
                            symbol,
                            'buy',
                            reduce_size,
                            params={'reduceOnly': True, 'tag': ORDER_TAG}
                        )
                else:
                    logger.info(
                        "Existing short position aligned with target (current %.2f, target %.2f)",
                        current_position['size'],
                        position_size,
                    )
            else:
                logger.info("Opening new short position of %.2f contracts", position_size)
                exchange.create_market_order(
                    symbol,
                    'sell',
                    position_size,
                    params={'tag': ORDER_TAG}
                )

        logger.info("Trade execution complete for signal %s", target_side)
        time.sleep(2)
        updated_position = get_current_position()
        logger.info("Updated position snapshot: %s", updated_position)
        return updated_position

    except Exception as exc:  # noqa: BLE001 - need stack for exchange failures
        logger.exception("Trade execution failed: %s", exc)

        if "don't have any positions" in str(exc).lower():
            logger.info("Attempting direct position open after position-not-found error")
            try:
                order_side = 'buy' if signal_data.get('signal') == 'BUY' else 'sell'
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    order_side,
                    position_size,
                    params={'tag': ORDER_TAG}
                )
                logger.info("Fallback order placement succeeded")
            except Exception as nested_exc:  # noqa: BLE001
                logger.exception("Fallback order placement failed: %s", nested_exc)

    return current_position


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            logger.warning("DeepSeek attempt %s returned fallback signal; retrying", attempt + 1)
            time.sleep(1)

        except Exception as e:
            logger.exception("DeepSeek attempt %s failed: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def publish_monitoring_snapshot(
    price_data: Optional[Dict[str, Any]],
    signal_data: Optional[Dict[str, Any]],
    current_position: Optional[Dict[str, Any]],
    error: Optional[str] = None,
) -> None:
    """Push the latest bot state to the shared monitoring store."""

    try:
        price_snapshot: Dict[str, Any] = {}
        if price_data:
            price_snapshot = {
                'price': price_data.get('price'),
                'timestamp': price_data.get('timestamp'),
                'high': price_data.get('high'),
                'low': price_data.get('low'),
                'volume': price_data.get('volume'),
                'timeframe': price_data.get('timeframe'),
                'price_change': price_data.get('price_change'),
                'trend_analysis': price_data.get('trend_analysis'),
                'levels_analysis': price_data.get('levels_analysis'),
            }

        payload: Dict[str, Any] = {
            'price_snapshot': price_snapshot or None,
            'latest_signal': signal_data,
            'signal_history': signal_history[-30:],
            'position': current_position,
            'trade_config': TRADE_CONFIG,
            'metadata': {
                'bot_name': BOT_NAME,
                'mode': 'test' if TRADE_CONFIG['test_mode'] else 'live',
            },
        }

        if error:
            payload['error'] = error

        update_bot_state(BOT_NAME, **payload)
    except Exception as exc:  # noqa: BLE001 - monitoring must not break trading
        logger.exception("Failed to publish monitoring snapshot: %s", exc)


def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        logger.info("Waiting %s minutes %s seconds for next interval", display_minutes, display_seconds)
    else:
        logger.info("Waiting %s seconds for next interval", display_seconds)

    return seconds_to_wait


def trading_bot():
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    logger.info("Starting trading cycle at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        logger.warning("Price data unavailable; skipping cycle")
        publish_monitoring_snapshot(None, None, get_current_position(), "price data unavailable")
        return

    logger.info(
        "Market snapshot | price %.2f | timeframe %s | change %.2f%%",
        price_data['price'],
        TRADE_CONFIG['timeframe'],
        price_data['price_change'],
    )

    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        logger.warning("Using fallback trading signal due to analysis issues")

    try:
        latest_position = execute_intelligent_trade(signal_data, price_data)
        publish_monitoring_snapshot(price_data, signal_data, latest_position)
    except Exception as exc:  # noqa: BLE001 - ensure monitoring is updated on failure
        logger.exception("Unexpected error during trade execution: %s", exc)
        publish_monitoring_snapshot(price_data, signal_data, get_current_position(), str(exc))


def main():
    """主函数"""
    logger.info("Starting OKX BTC/USDT trading bot")
    logger.info("Operating mode: %s", "test" if TRADE_CONFIG['test_mode'] else "live")
    logger.info("Timeframe: %s", TRADE_CONFIG['timeframe'])

    if not setup_exchange():
        logger.error("Exchange initialisation failed; aborting bot startup")
        publish_monitoring_snapshot(None, None, get_current_position(), "exchange setup failed")
        return

    logger.info("Execution cadence: every 15 minutes on the clock")

    while True:
        trading_bot()
        time.sleep(60)


if __name__ == "__main__":
    main()
