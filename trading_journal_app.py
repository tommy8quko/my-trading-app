import streamlit as st
import pandas as pd
import os
import requests
import time
import yfinance as yf
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v3.csv"
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.set_page_config(page_title="Momentum Pro Edge", layout="wide")

def init_csv():
    cols = ["Date", "Symbol", "Action", "Strategy", "Price", "Quantity", "Stop_Loss", "Fees", "Notes", "Img", "Timestamp", "Setup_Grade"]
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=cols)
        df.to_csv(FILE_NAME, index=False)
    else:
        df = pd.read_csv(FILE_NAME)
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    return pd.read_csv(FILE_NAME)

def save_all_data(df):
    df.to_csv(FILE_NAME, index=False)

# --- 2. 專業動能交易邏輯：金字塔加倉與 FIFO 結算 ---

def calculate_portfolio(df):
    """
    支援 Pyramiding (金字塔加倉) 的持倉計算
    使用 FIFO (先進先出) 邏輯處理賣出結算
    """
    positions = {} # {Symbol: [ {qty, price, sl, timestamp}, ... ]}
    df = df.sort_values(by="Timestamp")
    total_realized_pnl = 0
    trade_history = [] 
    equity_curve = []
    running_pnl = 0

    for idx, row in df.iterrows():
        sym = row['Symbol']
        action = row['Action']
        qty = float(row['Quantity']) if pd.notna(row['Quantity']) else 0
        price = float(row['Price']) if pd.notna(row['Price']) else 0
        sl = float(row['Stop_Loss']) if pd.notna(row['Stop_Loss']) else None
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = []
            
        if "買入 Buy" in action:
            # 加入新批次 (Lot)
            positions[sym].append({
                'qty': qty, 
                'price': price, 
                'sl': sl, 
                'timestamp': row['Timestamp']
            })
            
        elif "賣出 Sell" in action:
            remaining_to_sell = qty
            # FIFO 結算：從最早的批次開始賣
            while remaining_to_sell > 0 and positions[sym]:
                lot = positions[sym][0]
                sell_qty = min(remaining_to_sell, lot['qty'])
                
                # 計算該批次被賣出部分的 PnL
                pnl = (price - lot['price']) * sell_qty
                total_realized_pnl += pnl
                running_pnl += pnl
                
                # 計算 R/R Ratio (風險回報比)
                # Risk = 進場價 - 初始止損
                rr = "N/A"
                if lot['sl'] and lot['sl'] < lot['price']:
                    risk_per_share = lot['price'] - lot['sl']
                    reward_per_share = price - lot['price']
                    rr = round(reward_per_share / risk_per_share, 2) if risk_per_share != 0 else 0
                
                trade_history.append({
                    "Date": date, 
                    "Symbol": sym, 
                    "Strategy": row['Strategy'],
                    "PnL": round(pnl, 2), 
                    "R/R": rr, 
                    "Grade": row.get('Setup_Grade', 'C'),
                    "Notes": row['Notes']
                })
                
                lot['qty'] -= sell_qty
                remaining_to_sell -= sell_qty
                if lot['qty'] <= 0:
                    positions[sym].pop(0)
            
            equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})

    # 整理當前持倉摘要 (用於即時顯示)
    active_summary = {}
    for sym, lots in positions.items():
        total_q = sum(l['qty'] for l in lots)
        if total_q > 0:
            avg_p = sum(l['qty'] * l['price'] for l in lots) / total_q
            # 動能交易通常以「最後一次上移的止損」或「最新加倉的止損」為風險基準
            current_sl = lots[-1]['sl'] if lots[-1]['sl'] else None
            active_summary[sym] = {
                'qty': total_q, 
                'avg_price': avg_p, 
                'sl': current_sl,
                'lots_count': len(lots) # 加倉次數
            }

    return active_summary, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

@st.cache_data(ttl=300)
def get_momentum_data(symbols_list):
    """
    計算相對強度 (Relative Strength)
    """
    if not symbols_list: return {}, {}
    try:
        # 下載個股與 SPY 大盤數據
        data = yf.download(symbols_list + ["SPY"], period="3mo", progress=False)
        prices = {}
        rs_scores = {}
        
        spy_close = data['Close']['SPY']
        spy_perf = (spy_close.iloc[-1] / spy_close.iloc[0]) - 1
        
        for sym in symbols_list:
            s_close = data['Close'][sym] if len(symbols_list) > 1 else data['Close']
            prices[sym] = s_close.iloc[-1]
            s_perf = (s_close.iloc[-1] / s_close.iloc[0]) - 1
            # RS Score = 個股漲幅 - 大盤漲幅
            rs_scores[sym] = (s_perf - spy_perf) * 100
            
        return prices, rs_scores
    except:
        return {}, {}

# --- 3. UI 介面 ---
st.title("🏹 Momentum Pro Alpha (金字塔加倉版)")
st.markdown("""
<style>
    .stMetric { background: #1E1E1E; color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #00FFAA; }
    .status-card { background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

df_raw = load_data()
active_pos, realized_pnl, history_df, equity_df = calculate_portfolio(df_raw)

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚡ 交易錄入")
    with st.form("trade_form", clear_on_submit=True):
        col_d1, col_d2 = st.columns(2)
        d_in = col_d1.date_input("日期")
        grade = col_d2.selectbox("進場評級", ["A+", "A", "B", "C", "D"])
        
        s_raw = st.text_input("股票代號 (數字自動轉港股)").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        
        c1, c2, c3 = st.columns(3)
        q_in = c1.number_input("股數", min_value=0.0, step=1.0, value=None, format="%.0f")
        p_in = c2.number_input("價格", min_value=0.0, step=0.01, value=None, format="%.2f")
        sl_in = c3.number_input("止損價", min_value=0.0, step=0.01, value=None, format="%.2f")
        
        st_in = st.selectbox("動能策略", ["Breakout (突破)", "Pullback (回踩)", "VCP (收窄)", "High Tight Flag"])
        note_in = st.text_area("交易筆記 (形態、心理狀態)")
        
        if st.form_submit_button("儲存紀錄"):
            if s_in and q_in and p_in:
                new_row = {
                    "Date": d_in, "Symbol": s_in, "Action": act_in, "Strategy": st_in, 
                    "Price": p_in, "Quantity": q_in, "Stop_Loss": sl_in, "Setup_Grade": grade,
                    "Fees": 0, "Notes": note_in, "Timestamp": int(time.time())
                }
                add_df = pd.DataFrame([new_row])
                df_raw = pd.concat([df_raw, add_df], ignore_index=True)
                save_all_data(df_raw)
                st.success(f"成功紀錄 {s_in} {act_in}")
                st.rerun()

# --- 主畫面 ---
t1, t2, t3, t4 = st.tabs(["📊 績效矩陣", "🎯 即時加倉監控", "📖 交易日誌", "🛠️ 管理"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總已實現損益", f"${realized_pnl:,.0f}")
    win_rate = (len(history_df[history_df['PnL']>0]) / len(history_df) * 100) if not history_df.empty else 0
    c2.metric("交易勝率", f"{win_rate:.1f}%")
    
    avg_rr = 0
    if not history_df.empty:
        valid_rr = history_df[history_df["R/R"] != "N/A"]["R/R"]
        avg_rr = valid_rr.mean() if not valid_rr.empty else 0
    c3.metric("平均風險回報 (R)", f"{avg_rr:.2f}")
    
    # 計算全損回撤預估 (Portfolio Risk)
    portfolio_risk = 0
    if active_pos:
        cur_prices, _ = get_momentum_data(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = cur_prices.get(s, 0)
            if now and d['sl']:
                risk = (now - d['sl']) * d['qty']
                portfolio_risk += max(0, risk)
    c4.metric("總風險敞口 (Stop-out)", f"${portfolio_risk:,.0f}", delta_color="inverse")

    if not equity_df.empty:
        st.plotly_chart(px.line(equity_df, x="Date", y="Cumulative PnL", title="資金增長曲線"), use_container_width=True)

with t2:
    if active_pos:
        st.subheader("🔥 當前持倉動能追蹤 (Pyramiding Active)")
        prices, rs_scores = get_momentum_data(list(active_pos.keys()))
        p_list = []
        
        for s, d in active_pos.items():
            now = prices.get(s, 0)
            rs = rs_scores.get(s, 0)
            un_pnl = (now - d['avg_price']) * d['qty']
            
            # 單一標的風險
            risk_val = (now - d['sl']) * d['qty'] if (now and d['sl']) else 0
            
            p_list.append({
                "代號": s, 
                "RS 強度": f"{rs:+.1f}%",
                "加倉次數": d['lots_count'],
                "總股數": d['qty'], 
                "平均成本": round(d['avg_price'],2),
                "目前止損": d['sl'] if d['sl'] else "未設定",
                "現價": round(now,2), 
                "未實現損益": round(un_pnl,2),
                "止損預期虧損": f"-${risk_val:,.0f}" if risk_val > 0 else "無風險 (Free Trade)"
            })
        
        st.table(pd.DataFrame(p_list))
    else:
        st.info("目前無持倉，請從側邊欄錄入交易。")

with t3:
    if not history_df.empty:
        st.plotly_chart(px.bar(history_df, x="Date", y="PnL", color="Grade", title="各評級進場的盈虧分佈"), use_container_width=True)
    st.dataframe(df_raw.sort_values("Timestamp", ascending=False), use_container_width=True)

with t4:
    st.write("### 數據管理")
    if st.button("🚨 清空所有交易紀錄"):
        if os.path.exists(FILE_NAME): 
            os.remove(FILE_NAME)
            st.success("已清空數據")
            st.rerun()
