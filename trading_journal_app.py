import streamlit as st
import pandas as pd
import os
import requests
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8 

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
            "Market_Condition", "Mistake_Tag" 
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def format_symbol(s_raw):
    if pd.isna(s_raw): return ""
    s_str = str(s_raw).upper().strip()
    if s_str.isdigit() and len(s_str) <= 5:
        return s_str.zfill(4) + ".HK"
    return s_str

def clean_strategy(s):
    s_str = str(s).strip()
    if "PULLBACK" in s_str.upper(): return "Pullback"
    if "BREAKOUT" in s_str.upper() or "BREAK OUT" in s_str.upper(): return "Breakout"
    return s_str

def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        if df.empty: return df
        if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)
        if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
        for col in ["Market_Condition", "Mistake_Tag"]:
            if col not in df.columns: df[col] = "N/A"
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Date'], errors='coerce').view('int64') // 10**9
            save_all_data(df)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Stop_Loss'] = pd.to_numeric(df['Stop_Loss'], errors='coerce').fillna(0)
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def save_all_data(df):
    df.to_csv(FILE_NAME, index=False)

def save_transaction(data):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    save_all_data(df)

def get_hkd_value(symbol, value):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return value
    return value * USD_HKD_RATE

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return "HK$"
    return "$"

# --- 2. 核心計算邏輯 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    cycle_tracker = {}
    completed_trades = [] 
    equity_curve = []

    for _, row in df.iterrows():
        sym = format_symbol(row['Symbol']) 
        action = str(row['Action']) if pd.notnull(row['Action']) else ""
        if not sym or not action: continue

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])
        date_str = row['Date']
        
        if sym not in positions: positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0}
        if sym not in cycle_tracker:
            cycle_tracker[sym] = {'cash_flow_raw': 0.0, 'start_date': date_str, 'is_active': False, 'initial_risk_raw': 0.0}
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        if not cycle_tracker[sym]['is_active'] and is_buy and qty > 0:
            cycle_tracker[sym]['is_active'] = True
            cycle_tracker[sym]['start_date'] = date_str
            cycle_tracker[sym]['cash_flow_raw'] = 0.0
            if sl > 0: cycle_tracker[sym]['initial_risk_raw'] = abs(price - sl) * qty
            else: cycle_tracker[sym]['initial_risk_raw'] = 0.0

        if is_buy:
            cycle_tracker[sym]['cash_flow_raw'] -= (qty * price)
            total_cost_base = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty > 0: curr['avg_price'] = total_cost_base / new_qty
            curr['qty'] = new_qty
        elif is_sell and curr['qty'] > 0:
            sell_qty = min(qty, curr['qty'])
            cycle_tracker[sym]['cash_flow_raw'] += (sell_qty * price)
            realized_pnl_hkd_item = get_hkd_value(sym, (price - curr['avg_price']) * sell_qty)
            total_realized_pnl_hkd += realized_pnl_hkd_item
            running_pnl_hkd += realized_pnl_hkd_item
            curr['qty'] -= sell_qty
            
            if curr['qty'] < 0.0001:
                d1, d2 = datetime.strptime(cycle_tracker[sym]['start_date'], '%Y-%m-%d'), datetime.strptime(date_str, '%Y-%m-%d')
                pnl_raw = cycle_tracker[sym]['cash_flow_raw']
                init_risk = cycle_tracker[sym]['initial_risk_raw']
                completed_trades.append({
                    "Exit_Date": date_str, "Entry_Date": cycle_tracker[sym]['start_date'], "Symbol": sym, 
                    "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((d2 - d1).days), "Trade_R": (pnl_raw / init_risk) if init_risk > 0 else 0.0
                })
                cycle_tracker[sym]['is_active'] = False
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    comp_df = pd.DataFrame(completed_trades)
    exp_hkd = 0
    exp_r = 0
    avg_dur = 0
    if not comp_df.empty:
        wins, losses = comp_df[comp_df['PnL_HKD'] > 0], comp_df[comp_df['PnL_HKD'] <= 0]
        wr = len(wins) / len(comp_df)
        exp_hkd = (wr * (wins['PnL_HKD'].mean() if not wins.empty else 0)) - ((1-wr) * (abs(losses['PnL_HKD'].mean()) if not losses.empty else 0))
        exp_r = comp_df['Trade_R'].mean()
        avg_dur = comp_df['Duration_Days'].mean()

    return {k: v for k, v in positions.items() if v['qty'] > 0.0001}, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur

@st.cache_data(ttl=60)
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        data = yf.download(symbols_list, period="1d", interval="1m", progress=False)
        prices = {}
        for s in symbols_list:
            try:
                val = data['Close'][s].dropna().iloc[-1] if len(symbols_list) > 1 else data['Close'].dropna().iloc[-1]
                prices[s] = float(val)
            except: prices[s] = None
        return prices
    except: return {}

# --- 3. UI 渲染 ---
df = load_data()
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_in = format_symbol(st.text_input("代號 (Ticker)").upper().strip())
        is_sell = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        q_in = st.number_input("股數 (Qty)", min_value=0.0)
        p_in = st.number_input("成交價格 (Price)", min_value=0.0)
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0)
        st.divider()
        mkt_cond = st.selectbox("市場環境", ["Trending Up", "Trending Down", "Range/Choppy", "N/A"])
        mistake_in = st.selectbox("錯誤標籤", ["None", "Fomo", "Revenge Trade", "Late Entry", "Moved Stop"])
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        rr_in = st.number_input("預期盈虧比 (R:R)", value=2.0)
        st_in = st.selectbox("策略 (Strategy)", ["Pullback", "Breakout", "➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略名稱")
        note_in = st.text_area("決策筆記")
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in > 0 and p_in > 0:
                save_transaction({"Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, "Stop_Loss": sl_in, "Fees": 0, "Emotion": emo_in, "Risk_Reward": rr_in, "Notes": note_in, "Timestamp": int(time.time()), "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in})
                st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    total_sl_risk_hkd = 0
    if active_pos:
        live_prices_for_risk = get_live_prices(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = live_prices_for_risk.get(s)
            if now and d['last_sl'] > 0:
                total_sl_risk_hkd += get_hkd_value(s, (now - d['last_sl']) * d['qty'])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${realized_pnl_total_hkd:,.2f}")
    m2.metric("期望值 (HKD / R)", f"${exp_val:,.0f} / {exp_r_val:.2f}R")
    m3.metric("總停損回撤 (Open Risk)", f"${total_sl_risk_hkd:,.2f}")
    m4.metric("平均持倉", f"{avg_dur_val:.1f} 天")
    m5.metric("勝率", f"{(len(completed_trades_df[completed_trades_df['PnL_HKD'] > 0]) / len(completed_trades_df) * 100) if not completed_trades_df.empty else 0:.1f}%")

    if not equity_df.empty: st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線 (HKD)"), use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
    current_symbols = list(active_pos.keys())
    live_prices = get_live_prices(current_symbols)
    processed_p_data = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
        processed_p_data.append({"代號": s, "持股數": f"{d['qty']:,.0f}", "平均成本": f"{d['avg_price']:,.2f}", "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{d['last_sl']:,.2f}", "未實現損益": f"{un_pnl:,.2f}"})
    if processed_p_data: st.dataframe(pd.DataFrame(processed_p_data), hide_index=True, use_container_width=True)
    else: st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 交易重播")
    if not df.empty:
        target = st.selectbox("選擇交易", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']}")
        row = df.iloc[target]
        # 下載前後 20 天的數據
        data = yf.download(row['Symbol'], start=(pd.to_datetime(row['Date']) - timedelta(days=20)).strftime('%Y-%m-%d'), progress=False)
        
        if not data.empty:
            # 處理 yfinance 可能返回的多層索引問題 (Multi-index)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 確保有 'Close' 欄位
            if 'Close' in data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='收盤價'))
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(row['Date'])], 
                    y=[row['Price']], 
                    mode='markers+text', 
                    marker=dict(size=12, color='orange', symbol='diamond'),
                    text=["執行點"],
                    textposition="top center",
                    name='執行點'
                ))
                fig.update_layout(title=f"{row['Symbol']} 執行回顧", xaxis_title="日期", yaxis_title="價格")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("無法獲取該時段的價格數據 (缺少 Close 欄位)")
        else:
            st.error("找不到該代號的歷史行情數據")

with t4:
    st.subheader("📜 歷史紀錄與心理分析")
    if not df.empty:
        history_display = df.sort_values("Timestamp", ascending=False).copy()
        history_display = history_display.rename(columns={"Stop_Loss": "執行時止損", "Price": "成交價", "Quantity": "股數"})
        cols = ["Date", "Symbol", "Action", "Strategy", "成交價", "股數", "執行時止損", "Emotion", "Market_Condition", "Notes"]
        st.dataframe(history_display[cols], use_container_width=True, hide_index=True)
        st.divider()
        mistake_counts = df['Mistake_Tag'].value_counts()
        if not mistake_counts.empty: st.plotly_chart(px.pie(names=mistake_counts.index, values=mistake_counts.values, title="錯誤標籤分布"), use_container_width=True)

with t5:
    st.subheader("🛠️ 數據管理")
    if not df.empty:
        selected_idx = st.selectbox("選擇紀錄進行編輯/刪除", df.index, format_func=lambda x: f"[{df.loc[x, 'Date']}] {df.loc[x, 'Symbol']}")
        if st.button("🗑️ 刪除此筆紀錄"):
            df = df.drop(selected_idx).reset_index(drop=True)
            save_all_data(df); st.rerun()
    if st.button("🚨 清空所有數據"):
        save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()
