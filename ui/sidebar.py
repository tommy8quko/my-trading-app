"""Shared sidebar rendered on every page."""
from __future__ import annotations
import streamlit as st


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Trading Journal")
        st.divider()
        page = st.radio(
            "Navigate",
            options=["Dashboard", "Journal", "Analysis", "Data"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Brokers: IBKR · Chief Securities HK")
    return page
