import importlib.util
import pathlib
import sys
import types


class FakeExchange:
    def __init__(self):
        self.markets = {"BTC/USDT:USDT": {}}
        self.orders = []

    def fetch_balance(self):
        return {"USDT": {"free": 1000}}

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.6f}"

    def create_market_order(self, symbol, side, amount, params=None):
        params = params or {}
        order = {
            "symbol": symbol,
            "side": side,
            "amount": float(amount),
            "params": dict(params),
        }
        self.orders.append(order)
        return {"symbol": symbol, "side": side, "amount": amount, "info": params}


def test_execute_trade_uses_dynamic_balance_fraction(monkeypatch):
    # Ensure environment variables are set before importing the module
    monkeypatch.setenv("TRADE_SYMBOL", "BTC/USDT:USDT")
    monkeypatch.setenv("TRADE_LEVERAGE", "10")
    monkeypatch.delenv("TRADE_AMOUNT", raising=False)
    monkeypatch.setenv("TRADE_BALANCE_FRACTION", "0.5")
    monkeypatch.setenv("TRADE_TEST_MODE", "false")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    script_path = repo_root / "deepseek_ok_带指标plus版本.py"
    # Provide stub modules so imports succeed during testing
    if "monitoring" not in sys.modules:
        monitoring_stub = types.ModuleType("monitoring")
        monitoring_stub.update_bot_state = lambda *args, **kwargs: None
        monitoring_stub.load_all_states = lambda: {}
        sys.modules["monitoring"] = monitoring_stub
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class _FakeOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        raise RuntimeError("DeepSeek client stub should not be used in tests")

        openai_stub.OpenAI = _FakeOpenAI
        sys.modules["openai"] = openai_stub
    spec = importlib.util.spec_from_file_location("deepseek_ok_plus", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["deepseek_ok_plus"] = module
    spec.loader.exec_module(module)

    fake_exchange = FakeExchange()
    monkeypatch.setattr(module, "exchange", fake_exchange)
    monkeypatch.setattr(module, "get_current_position", lambda: None)
    monkeypatch.setattr(module, "update_account_snapshot", lambda *args, **kwargs: None)
    recorded_orders = []

    def fake_record_order(action, side, amount, params=None, response=None, note=None):
        recorded_orders.append(
            {
                "action": action,
                "side": side,
                "amount": amount,
                "params": dict(params or {}),
                "note": note,
            }
        )

    monkeypatch.setattr(module, "record_order", fake_record_order)
    monkeypatch.setattr(module, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    signal_data = {
        "signal": "BUY",
        "confidence": "HIGH",
        "reason": "Test signal",
        "stop_loss": 19500,
        "take_profit": 22000,
    }
    price_data = {"price": 20000}

    module.execute_trade(signal_data, price_data)

    assert recorded_orders, "Expected record_order to capture at least one order"
    open_order = recorded_orders[-1]
    assert open_order["action"] == "open_long"
    assert abs(open_order["amount"] - 0.25) < 1e-6

    params = open_order["params"]
    assert params.get("posSide") == "long"
