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
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp"
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        if not df.empty:
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
    equity_curve = [{"Date": df.iloc[0]['Date'], "Cumulative PnL": 0}] # 初始點
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
                sell_qty = min(qty, curr['qty'])
                trade_pnl = (price - curr['avg_price']) * sell_qty
                total_realized_pnl += trade_pnl
                curr['qty'] -= sell_qty
                running_pnl += trade_pnl
                equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})
                trade_history.append({
                    "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                    "Action": action, "Price": price, "Cost": curr['avg_price'],
                    "Qty": sell_qty, "PnL": trade_pnl, "Emotion": row.get('Emotion', '平靜')
                })

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0}
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

# --- 4. 即時報價與 AI ---
@st.cache_data(ttl=300)
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        data = yf.download(symbols_list, period="1d", progress=False)['Close']
        prices = {}
        for s in symbols_list:
            try:
                val = data[s].iloc[-1] if len(symbols_list) > 1 else data.iloc[-1]
                prices[s] = float(val)
            except:
                prices[s] = None
        return prices
    except:
        return {}

def fetch_ai_insight(pnl_summary, open_summary):
    api_key = "" 
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
        
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=1.0, value=None)
        p_in = col2.number_input("成交價格 (Price)", min_value=0.0, step=0.01, value=None)
        
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.01, value=None)
        
        st.divider()
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        rr_in = st.number_input("預期盈虧比 (R:R)", value=2.0, min_value=0.1)
        
        if p_in and sl_in and act_in == "買入 Buy":
            risk = p_in - sl_in
            if risk > 0:
                target = p_in + (risk * rr_in)
                st.caption(f"💡 風險: {risk:.2f} | 目標價: {target:.2f}")
            else:
                st.caption("⚠️ 停損價應低於成交價")

        # 策略選單簡化
        default_strategies = ["Pullback", "Breakout", "Buyable Gapup"]
        existing_custom = [s for s in df['Strategy'].unique().tolist() if s not in default_strategies] if not df.empty else []
        tags = default_strategies + existing_custom
        
        st_in = st.selectbox("策略 (Strategy)", tags + ["➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略名稱")
        
        note_in = st.text_area("決策筆記")
        
        if st.form_submit_button("儲存執行紀錄"):
            if not s_in:
                st.error("請輸入標的代號")
            elif q_in is None or q_in <= 0:
                st.error("請輸入有效的股數")
            elif p_in is None or p_in <= 0:
                st.error("請輸入有效的成交價格")
            else:
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), 
                    "Symbol": s_in, 
                    "Action": act_in, 
                    "Strategy": st_in, 
                    "Price": p_in, 
                    "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0,
                    "Fees": 0, 
                    "Emotion": emo_in, 
                    "Risk_Reward": rr_in, 
                    "Notes": note_in, 
                    "Timestamp": int(time.time())
                })
                st.success(f"✅ 已儲存 {s_in}")
                time.sleep(1)
                st.rerun()

# 主畫面 Tab
t1, t2, t3, t4 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史"])

with t1:
    max_dd = 0
    if not equity_df.empty:
        equity_df['Peak'] = equity_df['Cumulative PnL'].cummax()
        equity_df['Drawdown'] = equity_df['Cumulative PnL'] - equity_df['Peak']
        max_dd = equity_df['Drawdown'].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("已實現損益", f"${realized_pnl:,.2f}")
    win_r = (len(history_df[history_df['PnL']>0])/len(history_df)*100) if not history_df.empty else 0
    col2.metric("勝率", f"{win_r:.1f}%")
    col3.metric("平均 R:R", f"{df['Risk_Reward'].mean():.2f}" if not df.empty else "0")
    col4.metric("帳戶 MDD", f"${max_dd:,.2f}", delta_color="inverse")
    
    if not equity_df.empty:
        fig_equity = px.area(equity_df, x="Date", y="Cumulative PnL", title="帳戶權益成長曲線", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_equity, use_container_width=True)
        
        fig_dd = px.line(equity_df, x="Date", y="Drawdown", title="歷史回撤圖", color_discrete_sequence=['#EF553B'])
        fig_dd.add_hline(y=max_dd, line_dash="dash", line_color="red")
        st.plotly_chart(fig_dd, use_container_width=True)

with t2:
    if active_pos:
        prices = get_live_prices(list(active_pos.keys()))
        p_data = []
        for s, d in active_pos.items():
            now = prices.get(s)
            qty = d['qty']
            avg_p = d['avg_price']
            un_pnl = (now - avg_p) * qty if now else 0
            
            # 獲取該標的最後設定的停損價
            last_sl = df[df['Symbol'] == s]['Stop_Loss'].iloc[-1] if s in df['Symbol'].values else 0
            
            # 計算停損回撤金額 (SL Drawdown/Risk): (現價 - 停損價) * 股數
            sl_risk_amt = (now - last_sl) * qty if now and last_sl > 0 else 0

            p_data.append({
                "代號": s, 
                "股數": qty, 
                "成本": f"${avg_p:.2f}", 
                "停損價": f"${last_sl:.2f}", 
                "現價": f"${now:.2f}" if now else "讀取中...", 
                "未實現損益": f"${un_pnl:,.2f}", 
                "報酬%": f"{(un_pnl/(qty * avg_p)*100):.1f}%" if now and avg_p!=0 else "0%",
                "停損回撤 (SL Risk)": f"${sl_risk_amt:,.2f}" if now else "N/A"
            })
        st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)
        st.caption("💡 停損回撤 (SL Risk) = (現價 - 停損價) × 股數。代表若現在觸發停損，將從目前價值縮水的金額。")
        if st.button("🔄 刷新即時報價"): st.cache_data.clear(); st.rerun()
    else: st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 市場環境重播 (Market Replay)")
    if not df.empty:
        target = st.selectbox("選擇回顧交易", df.index, format_func=lambda x: f"{df.iloc[x]['Date']} | {df.iloc[x]['Symbol']} | {df.iloc[x]['Action']}")
        row = df.iloc[target]
        try:
            t_date = pd.to_datetime(row['Date'])
            start_dt = (t_date - timedelta(days=10)).strftime('%Y-%m-%d')
            end_dt = (t_date + timedelta(days=10)).strftime('%Y-%m-%d')
            data = yf.download(row['Symbol'], start=start_dt, end=end_dt, progress=False)
            if not data.empty:
                fig_replay = go.Figure()
                fig_replay.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價'))
                fig_replay.add_trace(go.Scatter(x=[t_date], y=[row['Price']], mode='markers+text', text=['📍 EXEC'], marker=dict(color='orange', size=15)))
                fig_replay.update_layout(title=f"{row['Symbol']} 執行當下行情", template="plotly_white")
                
                c1, c2 = st.columns([3, 1])
                c1.plotly_chart(fig_replay, use_container_width=True)
                c2.write(f"**策略：** {row['Strategy']}")
                c2.write(f"**執行價：** ${row['Price']}")
                c2.write(f"**停損價：** ${row['Stop_Loss']}")
            else: st.warning("無法載入數據")
        except: st.warning("重播載入出錯")

with t4:
    c1, c2 = st.columns([1, 2])
    with c1:
        if not df.empty:
            emo_fig = px.pie(df, names="Emotion", title="交易情緒分佈")
            st.plotly_chart(emo_fig, use_container_width=True)
    with c2:
        st.subheader("📜 完整歷史流水帳")
        st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)
