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

if page == "Dashboard":
    from ui.pages.dashboard import render
elif page == "Journal":
    from ui.pages.journal import render
elif page == "Analysis":
    from ui.pages.analysis import render
elif page == "Data":
    from ui.pages.data import render
else:
    def render():
        st.write("Page not found")

render()
