"""Streamlit entrypoint — run with: streamlit run app.py"""
import streamlit as st
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="Trading Journal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = render_sidebar()

if page == "Performance":
    from ui.pages.performance import render
elif page == "Positions":
    from ui.pages.positions import render
elif page == "Trade Replay":
    from ui.pages.trade_replay import render
elif page == "AI Coach":
    from ui.pages.ai_coach import render
elif page == "Review Queue":
    from ui.pages.review_queue import render
elif page == "Data Management":
    from ui.pages.data_mgmt import render
else:
    def render():
        st.write("Page not found")

render()
