"""
Shared state management for the web monitoring interface.

Trading scripts call :func:`update_bot_state` to persist their latest
runtime snapshot. The web server reads the JSON file via
:func:`load_all_states` and renders a dashboard.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, Dict

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    _np = None  # type: ignore

try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    _pd = None  # type: ignore

_DEFAULT_STATUS_FILE = (
    Path(__file__).resolve().parent.parent / "logs" / "monitor_status.json"
)
STATUS_FILE = Path(os.getenv("MONITOR_STATUS_FILE", str(_DEFAULT_STATUS_FILE))).resolve()
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

_write_lock = Lock()


def _json_ready(value: Any) -> Any:
    """Convert the provided value into a JSON-serialisable structure."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]

    if isinstance(value, (datetime,)):
        return value.astimezone(timezone.utc).isoformat()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    if isinstance(value, Decimal):
        return float(value)

    if _np is not None and isinstance(
        value, (_np.generic,)  # type: ignore[attr-defined]
    ):
        return value.item()

    if _pd is not None:
        if isinstance(value, _pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (_pd.Series, _pd.DataFrame)):
            return _json_ready(value.to_dict())

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


def _load_raw_state() -> Dict[str, Any]:
    if STATUS_FILE.exists():
        try:
            with STATUS_FILE.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            # Corrupted file – return empty state without breaking the bot
            return {}
    return {}


def load_all_states() -> Dict[str, Any]:
    """Return the full monitoring state indexed by bot identifier."""
    return _load_raw_state()


def update_bot_state(bot_name: str, **payload: Any) -> None:
    """
    Persist monitoring data for a trading bot.

    Parameters
    ----------
    bot_name:
        Identifier for the bot, typically the script file name sans extension.
    payload:
        Keyword arguments describing the snapshot. Recommended keys:
            - price_snapshot (dict)
            - latest_signal (dict)
            - signal_history (list)
            - position (dict)
            - trade_config (dict)
            - metadata (dict)
            - error (str)
    """

    if not bot_name:
        raise ValueError("bot_name must be provided")

    cleaned_payload = {k: _json_ready(v) for k, v in payload.items() if v is not None}

    state = _load_raw_state()
    bot_state: Dict[str, Any] = state.get(bot_name, {})

    if "signal_history" in cleaned_payload:
        history = cleaned_payload.pop("signal_history")
        if isinstance(history, list):
            # Keep the most recent 50 entries
            bot_state["signal_history"] = history[-50:]
    elif "latest_signal" in cleaned_payload and bot_state.get("signal_history"):
        bot_state["signal_history"] = (
            bot_state.get("signal_history", []) + [cleaned_payload["latest_signal"]]
        )[-50:]

    bot_state.update(cleaned_payload)
    bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
    state[bot_name] = bot_state

    with _write_lock:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(STATUS_FILE.parent), prefix="monitor_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_handle:
                json.dump(state, tmp_handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, STATUS_FILE)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
