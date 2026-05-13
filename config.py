import os
import sys

# --- Non-secret defaults ---
BASE_CURRENCY = "HKD"
USD_HKD_RATE = 7.8
INITIAL_CAPITAL_HKD = 1_600_000
BROKER_IBKR = "IBKR"
BROKER_CHIEF = "CHIEF"


def _load_toml_secrets() -> dict:
    """Read .streamlit/secrets.toml from next to the exe or script."""
    import tomllib

    # Candidate locations: next to .exe, next to script, cwd
    candidates = [
        os.path.join(os.path.dirname(sys.executable), ".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return tomllib.load(f)
    return {}


# Load once at import time
_toml: dict = {}
try:
    _toml = _load_toml_secrets()
except Exception:
    pass


def get_secret(key: str, default: str = "") -> str:
    # 1. Streamlit secrets (when running via streamlit)
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        pass

    # 2. secrets.toml read directly (exe / uvicorn context)
    if key in _toml:
        return str(_toml[key])

    # 3. Environment variable
    return os.environ.get(key, default)
