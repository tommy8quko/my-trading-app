import os

# Attempt to import streamlit; fall back to os.environ for tests / CLI use
try:
    import streamlit as st
    def get_secret(key: str, default: str = "") -> str:
        try:
            return st.secrets[key]
        except Exception:
            return os.environ.get(key, default)
except ImportError:
    def get_secret(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

# --- Hardcoded non-secret defaults ---
BASE_CURRENCY = "HKD"
USD_HKD_RATE = 7.8          # fallback; fetch live if needed
INITIAL_CAPITAL_HKD = 1_600_000   # total across both accounts

# --- Broker short codes (must match `brokers.short_code` in DB) ---
BROKER_IBKR = "IBKR"
BROKER_CHIEF = "CHIEF"
