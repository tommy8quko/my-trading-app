import streamlit as st
import pandas as pd
import os
import requests
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 核心配置 ---
FILE_NAME = "trade_ledger_v4.csv"
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.set_page_config(page_title="TradeMaster Pro", layout="wide")

def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp"
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    return pd.read_csv(FILE_NAME)

def save_transaction(data):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

# --- 2. 核心計算：風險與動量指標 ---
def get_advanced_stats(df):
    if df.empty: return None
    
    # 這裡計算已平倉交易的損益
    # 簡化計算：將買入賣出配對 (FIFO)
    closed_trades = []
    # (此處省略複雜的 FIFO 配對算法，直接沿用 v3 的平倉邏輯結果)
    # 假設我們已經有一個 history_df (已結算交易)
    return None

# --- 3. 繪製交易圖表 (進出場標註) ---
def plot_trade_execution(symbol, trade_date, entry_price, exit_price=None):
    try:
        start_dt = datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=5)
        end_dt = datetime.strptime(trade_date, '%Y-%m-%d') + timedelta(days=5)
        data = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
        
        if data.empty: return None

        fig = go.Figure()
        # 股價線
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='gray', width=1)))
        
        # 進場點
        fig.add_trace(go.Scatter(
            x=[trade_date], y=[entry_price],
            mode='markers+text', name='進場',
            text=['B'], textposition='bottom center',
            marker=dict(color='green', size=15, symbol='triangle-up')
        ))
        
        # 如果有出場點
        if exit_price:
            fig.add_trace(go.Scatter(
                x=[trade_date], y=[exit_price],
                mode='markers+text', name='出場',
                text=['S'], textposition='top center',
                marker=dict(color='red', size=15, symbol='triangle-down')
            ))
            
        fig.update_layout(title=f"{symbol} 交易執行回顧", template="plotly_white", height=400)
        return fig
    except:
        return None

# --- 4. UI 介面 ---
st.title("🛡️ TradeMaster Pro 決策系統")

df = load_data()

# --- 側邊欄：進階輸入 ---
with st.sidebar:
    st.header("⚡ 執行紀錄")
    with st.form("pro_trade_form", clear_on_submit=True):
        date_in = st.date_input("交易日期")
        s_raw = st.text_input("標的代號").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        
        act_in = st.radio("類型", ["買入 Buy", "賣出 Sell"], horizontal=True)
        col1, col2 = st.columns(2)
        qty_in = col1.number_input("數量", min_value=0.1)
        price_in = col2.number_input("價格", min_value=0.0)
        
        st.divider()
        # 動量與心理特有欄位
        emo_in = st.select_slider("心理狀態 (心理標記)", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        rr_in = st.number_input("預期盈虧比 (R:R)", min_value=0.0, value=2.0)
        strat_in = st.selectbox("策略類別", ["動量突破", "均線回歸", "新聞事件", "自訂"])
        
        note_in = st.text_area("決策過程 (市場條件重現)")
        
        if st.form_submit_button("寫入日誌"):
            save_transaction({
                "Date": date_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                "Strategy": strat_in, "Price": price_in, "Quantity": qty_in,
                "Fees": 0, "Emotion": emo_in, "Risk_Reward": rr_in, 
                "Notes": note_in, "Timestamp": int(time.time())
            })
            st.rerun()

# --- 主面板 ---
tab_dashboard, tab_replay, tab_psych = st.tabs(["📊 績效矩陣", "🔄 交易重播", "🧠 心理分析"])

with tab_dashboard:
    # 此處可放入 v3 的權益曲線與 Max Drawdown 計算
    st.subheader("📈 權益曲線與回撤 (Equity & Drawdown)")
    # 模擬數據或計算實體數據...
    st.info("這裡將顯示你的資金成長曲線與最大回撤幅度。")

with tab_replay:
    st.subheader("⏪ 決策重播 (Decision Replay)")
    if not df.empty:
        selected_trade = st.selectbox("選擇要回顧的交易", df.index, format_func=lambda x: f"{df.iloc[x]['Date']} - {df.iloc[x]['Symbol']} ({df.iloc[x]['Action']})")
        trade = df.iloc[selected_trade]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = plot_trade_execution(trade['Symbol'], trade['Date'], trade['Price'])
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("無法獲取該時段行情數據。")
            
        with c2:
            st.write(f"**策略：** {trade['Strategy']}")
            st.write(f"**當時情緒：** {trade['Emotion']}")
            st.write(f"**筆記：**")
            st.info(trade['Notes'])
    else:
        st.write("尚無交易紀錄可供重播。")

with tab_psych:
    st.subheader("🧠 心理對策略影響分析")
    if not df.empty:
        # 心理狀態分佈圖
        emo_counts = df['Emotion'].value_counts().reset_index()
        fig_emo = px.pie(emo_counts, values='count', names='Emotion', title="交易情緒占比")
        st.plotly_chart(fig_emo, use_container_width=True)
        
        # 簡單的相關性分析提示
        st.markdown("""
        **💡 職業觀察：**
        - 如果「衝動」標籤對應的是負損益，請在下週強制執行『下單前停頓 10 秒』。
        - 當你處於「平靜」狀態時，勝率是否明顯提高？
        """)
