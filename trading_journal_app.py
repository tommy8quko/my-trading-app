import streamlit as st
import pandas as pd
import os
import requests
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp"
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        return df
    except:
        return pd.DataFrame()

def save_transaction(data):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

# --- 2. 核心邏輯：計算持倉與損益曲線 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame()
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl = 0
    trade_history = [] 
    equity_curve = []
    running_pnl = 0

    for _, row in df.iterrows():
        sym = row['Symbol']
        action = row['Action']
        qty = float(row['Quantity'])
        price = float(row['Price'])
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0}
            
        curr = positions[sym]
        
        if "買入 Buy" in action:
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty != 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
        elif "賣出 Sell" in action:
            if curr['qty'] > 0:
                trade_pnl = (price - curr['avg_price']) * qty
                total_realized_pnl += trade_pnl
                curr['qty'] -= qty
                running_pnl += trade_pnl
                equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})
                trade_history.append({
                    "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                    "Action": action, "Price": price, "Cost": curr['avg_price'],
                    "Qty": qty, "PnL": trade_pnl, "Emotion": row.get('Emotion', '平靜')
                })

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0}
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

# --- 3. 繪製交易執行圖表 ---
def plot_trade_execution(symbol, trade_date, entry_price):
    try:
        t_date = pd.to_datetime(trade_date)
        start_dt = (t_date - timedelta(days=10)).strftime('%Y-%m-%d')
        end_dt = (t_date + timedelta(days=10)).strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
        
        if data.empty: return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='#636EFA', width=2)))
        fig.add_trace(go.Scatter(
            x=[t_date], y=[entry_price],
            mode='markers+text', name='執行點',
            text=['📍 EXEC'], textposition='top center',
            marker=dict(color='orange', size=15, symbol='star')
        ))
        fig.update_layout(title=f"{symbol} 執行當下行情重播", template="plotly_white", height=400, margin=dict(l=20,r=20,t=40,b=20))
        return fig
    except:
        return None

# --- 4. 即時報價與 AI ---
@st.cache_data(ttl=300)
def get_live_prices(symbols):
    if not symbols: return {}
    try:
        data = yf.download(list(symbols), period="1d", progress=False)['Close']
        return {s: float(data[s].iloc[-1]) if len(symbols)>1 else float(data.iloc[-1]) for s in symbols}
    except: return {}

def fetch_ai_insight(pnl_summary, open_summary):
    api_key = "" # 系統自動注入
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    prompt = f"你是交易教練。分析數據並給予建議：\n損益摘要:{pnl_summary}\n持倉:{open_summary}\n請提供表現評估、心理建設、及動量優化建議。"
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 分析暫時不可用。"

# --- 5. UI 介面 ---
df = load_data()
active_pos, realized_pnl, history_df, equity_df = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號", placeholder="700 或 TSLA").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        c1, c2 = st.columns(2)
        q_in = c1.number_input("股數", min_value=0.01)
        p_in = c2.number_input("價格", min_value=0.0)
        
        st.divider()
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        rr_in = st.number_input("R:R (風險回報比)", value=2.0)
        strat_in = st.selectbox("策略", ["動量突破", "均線拉回", "新聞驅動", "自訂..."])
        if strat_in == "自訂...": strat_in = st.text_input("輸入策略名稱")
        
        note_in = st.text_area("決策筆記 (市場條件紀錄)")
        
        if st.form_submit_button("執行"):
            if s_in and q_in > 0:
                save_transaction({"Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, "Strategy": strat_in, "Price": p_in, "Quantity": q_in, "Fees": 0, "Emotion": emo_in, "Risk_Reward": rr_in, "Notes": note_in, "Timestamp": int(time.time())})
                st.success(f"已儲存 {s_in}")
                st.rerun()

# 主畫面 Tab
t1, t2, t3, t4 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史"])

with t1:
    col1, col2, col3 = st.columns(3)
    col1.metric("已實現損益", f"${realized_pnl:,.2f}")
    win_r = (len(history_df[history_df['PnL']>0])/len(history_df)*100) if not history_df.empty else 0
    col2.metric("勝率", f"{win_r:.1f}%")
    col3.metric("平均 R:R", f"{df['Risk_Reward'].mean():.2f}" if not df.empty else "0")
    
    if not equity_df.empty:
        # 計算 Max Drawdown
        equity_df['Peak'] = equity_df['Cumulative PnL'].cummax()
        equity_df['Drawdown'] = equity_df['Cumulative PnL'] - equity_df['Peak']
        
        fig_equity = px.area(equity_df, x="Date", y="Cumulative PnL", title="帳戶權益成長曲線", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_equity, use_container_width=True)
        
        fig_dd = px.line(equity_df, x="Date", y="Drawdown", title="風險回撤圖 (Drawdown)", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_dd, use_container_width=True)

    if st.button("🤖 獲取 AI 專業分析"):
        with st.spinner("分析中..."):
            st.info(fetch_ai_insight(f"PnL:{realized_pnl}, 勝率:{win_r}%", str(list(active_pos.keys()))))

with t2:
    if active_pos:
        prices = get_live_prices(active_pos.keys())
        p_data = []
        for s, d in active_pos.items():
            now = prices.get(s)
            un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
            p_data.append({"代號": s, "股數": d['qty'], "成本": f"${d['avg_price']:.2f}", "現價": f"${now:.2f}" if now else "Loding...", "未實現": f"${un_pnl:,.2f}", "回報%": f"{(un_pnl/(d['qty']*d['avg_price'])*100):.1f}%" if now else "0%"})
        st.table(pd.DataFrame(p_data))
    else: st.info("目前無持倉")

with t3:
    st.subheader("⏪ 市場環境重播 (Market Replay)")
    if not df.empty:
        target = st.selectbox("選擇回顧交易", df.index, format_func=lambda x: f"{df.iloc[x]['Date']} | {df.iloc[x]['Symbol']} | {df.iloc[x]['Action']}")
        row = df.iloc[target]
        fig_replay = plot_trade_execution(row['Symbol'], row['Date'], row['Price'])
        if fig_replay:
            c1, c2 = st.columns([3, 1])
            c1.plotly_chart(fig_replay, use_container_width=True)
            c2.write("**當時筆記：**")
            c2.caption(row['Notes'])
            c2.write(f"**心理狀態：** {row['Emotion']}")
        else: st.warning("無法載入該時間段行情。")

with t4:
    c1, c2 = st.columns([1, 2])
    with c1:
        if not df.empty:
            emo_fig = px.pie(df, names="Emotion", title="交易情緒分佈")
            st.plotly_chart(emo_fig, use_container_width=True)
    with c2:
        st.subheader("📜 歷史流水帳")
        st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
