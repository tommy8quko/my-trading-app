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

# --- 2. 核心邏輯：金字塔加倉與 FIFO 結算 ---

def calculate_portfolio(df):
    positions = {} 
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
            positions[sym].append({
                'qty': qty, 
                'price': price, 
                'sl': sl, 
                'timestamp': row['Timestamp']
            })
            
        elif "賣出 Sell" in action:
            remaining_to_sell = qty
            while remaining_to_sell > 0 and positions[sym]:
                lot = positions[sym][0]
                sell_qty = min(remaining_to_sell, lot['qty'])
                pnl = (price - lot['price']) * sell_qty
                total_realized_pnl += pnl
                running_pnl += pnl
                
                rr = "N/A"
                if lot['sl'] and lot['sl'] < lot['price']:
                    risk_per_share = lot['price'] - lot['sl']
                    reward_per_share = price - lot['price']
                    rr = round(reward_per_share / risk_per_share, 2) if risk_per_share != 0 else 0
                
                trade_history.append({
                    "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                    "PnL": round(pnl, 2), "R/R": rr, "Grade": row.get('Setup_Grade', 'C')
                })
                
                lot['qty'] -= sell_qty
                remaining_to_sell -= sell_qty
                if lot['qty'] <= 0:
                    positions[sym].pop(0)
            
            equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})

    active_summary = {}
    for sym, lots in positions.items():
        total_q = sum(l['qty'] for l in lots)
        if total_q > 0:
            avg_p = sum(l['qty'] * l['price'] for l in lots) / total_q
            current_sl = lots[-1]['sl'] if lots[-1]['sl'] else None
            active_summary[sym] = {
                'qty': total_q, 'avg_price': avg_p, 'sl': current_sl, 'lots_count': len(lots)
            }
    return active_summary, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

@st.cache_data(ttl=60)
def get_momentum_data(symbols_list):
    if not symbols_list: return {}
    try:
        # 確保代號列表包含 SPY 用於大盤對照（雖然目前 UI 隱藏，但邏輯保留以防需要）
        search_list = list(set(symbols_list + ["SPY"]))
        data = yf.download(search_list, period="5d", progress=False)
        
        prices = {}
        
        # 處理 yfinance 可能回傳的 MultiIndex 結構
        if 'Close' in data:
            close_data = data['Close']
            for sym in symbols_list:
                try:
                    if isinstance(close_data, pd.DataFrame):
                        # 取得該標的最後一個非空價格
                        val = close_data[sym].dropna().iloc[-1]
                    else:
                        # 只有單一標的情況
                        val = close_data.dropna().iloc[-1]
                    prices[sym] = float(val)
                except Exception:
                    prices[sym] = 0.0
        return prices
    except Exception as e:
        st.error(f"抓取股價出錯: {e}")
        return {}

# --- 3. UI 介面 ---
st.title("🏹 Momentum Pro Alpha v3.9")
st.markdown("""
<style>
    .stMetric { background: #1E1E1E; color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #00FFAA; }
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
        
        s_raw = st.text_input("代號 (例如: TSLA, 0700)").upper().strip()
        if s_raw.isdigit():
            s_in = s_raw.zfill(4) + ".HK"
        else:
            s_in = s_raw
            
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        
        q_in = c1.number_input("股數", min_value=0.0, step=1.0, format="%.0f", value=None)
        p_in = c2.number_input("價格", min_value=0.0, step=0.01, format="%.2f", value=None)
        sl_in = c3.number_input("止損", min_value=0.0, step=0.01, format="%.2f", value=None)
        
        st_in = st.selectbox("策略", ["Breakout", "Pullback", "VCP", "High Tight Flag"])
        note_in = st.text_area("筆記")
        
        if st.form_submit_button("儲存紀錄"):
            if not s_in:
                st.error("請輸入標代號")
            elif q_in is None or q_in <= 0:
                st.error("請輸入正確的股數 (必須大於 0)")
            elif p_in is None or p_in <= 0:
                st.error("請輸入正確的價格 (必須大於 0)")
            else:
                try:
                    save_q = float(q_in)
                    save_p = float(p_in)
                    save_sl = float(sl_in) if sl_in is not None else 0.0
                    
                    new_row = {
                        "Date": d_in, "Symbol": s_in, "Action": act_in, "Strategy": st_in, 
                        "Price": save_p, "Quantity": save_q, "Stop_Loss": save_sl, 
                        "Setup_Grade": grade, "Fees": 0, "Notes": note_in, "Timestamp": int(time.time())
                    }
                    df_raw = pd.concat([df_raw, pd.DataFrame([new_row])], ignore_index=True)
                    save_all_data(df_raw)
                    st.success(f"成功紀錄 {s_in}")
                    st.rerun()
                except ValueError:
                    st.error("輸入格式錯誤")

# --- 主畫面 ---
t1, t2, t3, t4 = st.tabs(["📊 績效矩陣", "🎯 即時持倉監控", "📖 交易日誌", "🛠️ 管理"])

with t1:
    portfolio_risk = 0
    if active_pos:
        cur_prices = get_momentum_data(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = cur_prices.get(s, 0.0)
            if now > 0 and d['sl'] is not None:
                risk = (float(now) - float(d['sl'])) * d['qty']
                portfolio_risk += max(0, risk)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已實現損益", f"${realized_pnl:,.0f}")
    win_rate = (len(history_df[history_df['PnL']>0]) / len(history_df) * 100) if not history_df.empty else 0
    c2.metric("交易勝率", f"{win_rate:.1f}%")
    avg_rr = 0
    if not history_df.empty:
        valid_rr = history_df[history_df["R/R"] != "N/A"]["R/R"]
        avg_rr = valid_rr.mean() if not valid_rr.empty else 0
    c3.metric("平均 R/R", f"{avg_rr:.2f}")
    c4.metric("總風險敞口", f"${portfolio_risk:,.0f}", delta_color="inverse")

    if not equity_df.empty:
        st.plotly_chart(px.line(equity_df, x="Date", y="Cumulative PnL", title="資金增長曲線"), use_container_width=True)

with t2:
    if active_pos:
        prices = get_momentum_data(list(active_pos.keys()))
        p_list = []
        for s, d in active_pos.items():
            now = prices.get(s, 0.0)
            un_pnl = (now - d['avg_price']) * d['qty'] if now > 0 else 0.0
            risk_val = (now - d['sl']) * d['qty'] if (now > 0 and d['sl']) else 0.0
            
            p_list.append({
                "代號": s, 
                "加倉次數": d['lots_count'],
                "總股數": d['qty'], 
                "平均成本": round(d['avg_price'], 2),
                "止損價": round(d['sl'], 2) if d['sl'] else "未設定",
                "現價": round(now, 2) if now > 0 else "抓取中...", 
                "未實現損益": round(un_pnl, 2) if now > 0 else "--",
                "預期回撤風險": f"-${risk_val:,.0f}" if risk_val > 0 else "Free Trade"
            })
        st.table(pd.DataFrame(p_list))
    else: 
        st.info("目前無在場持倉。")

with t3:
    st.dataframe(df_raw.sort_values("Timestamp", ascending=False), use_container_width=True)

with t4:
    st.write("### 數據管理")
    if not df_raw.empty:
        st.write("選擇要刪除的交易紀錄：")
        df_for_del = df_raw.sort_values("Timestamp", ascending=False)
        to_del = st.multiselect("勾選時間戳記 (Timestamp)", df_for_del['Timestamp'].tolist())
        if st.button("確認刪除選中紀錄"):
            df_raw = df_raw[~df_raw['Timestamp'].isin(to_del)]
            save_all_data(df_raw)
            st.success("紀錄已更新")
            st.rerun()
            
    if st.button("🚨 清空所有數據"):
        if os.path.exists(FILE_NAME): 
            os.remove(FILE_NAME)
            st.rerun()
