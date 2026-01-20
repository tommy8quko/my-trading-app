import streamlit as st
import pandas as pd
import os
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 配置與初始化 ---
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
        # 確保必要欄位存在
        required_cols = {
            "Market_Condition": "N/A", "Mistake_Tag": "None", "Img": None, 
            "Fees": 0, "Risk_Reward": 0, "Timestamp": 0
        }
        for col, default in required_cols.items():
            if col not in df.columns: df[col] = default

        if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)
        if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
        
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
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

# --- 2. 核心計算邏輯 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 0, 0
    
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

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row.get('Stop_Loss', 0))
        date_str = row['Date']
        
        if sym not in positions: 
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'first_sl': 0.0}
        
        if sym not in cycle_tracker:
            cycle_tracker[sym] = {'cash_flow_raw': 0.0, 'start_date': date_str, 'is_active': False, 'initial_risk_raw': 0.0}
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        if not cycle_tracker[sym]['is_active'] and is_buy and qty > 0:
            cycle_tracker[sym].update({
                'is_active': True, 'start_date': date_str, 'cash_flow_raw': 0.0,
                'Strategy': row.get('Strategy', 'N/A'), 'Emotion': row.get('Emotion', '平靜')
            })
            if sl > 0:
                cycle_tracker[sym]['initial_risk_raw'] = abs(price - sl) * qty
                curr['first_sl'] = sl

        if is_buy:
            cycle_tracker[sym]['cash_flow_raw'] -= (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty > 0: curr['avg_price'] = ((curr['qty'] * curr['avg_price']) + (qty * price)) / new_qty
            curr['qty'] = new_qty
        elif is_sell and curr['qty'] > 0:
            sell_qty = min(qty, curr['qty'])
            cycle_tracker[sym]['cash_flow_raw'] += (sell_qty * price)
            pnl_hkd_item = get_hkd_value(sym, (price - curr['avg_price']) * sell_qty)
            total_realized_pnl_hkd += pnl_hkd_item
            running_pnl_hkd += pnl_hkd_item
            curr['qty'] -= sell_qty
            
            if curr['qty'] < 0.0001:
                d1, d2 = datetime.strptime(cycle_tracker[sym]['start_date'], '%Y-%m-%d'), datetime.strptime(date_str, '%Y-%m-%d')
                pnl_raw = cycle_tracker[sym]['cash_flow_raw']
                init_risk = cycle_tracker[sym]['initial_risk_raw']
                completed_trades.append({
                    "Exit_Date": date_str, "Symbol": sym, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Trade_R": (pnl_raw / init_risk) if init_risk > 0 else None,
                    "Strategy": cycle_tracker[sym].get('Strategy', 'N/A'),
                    "Duration": (d2 - d1).days
                })
                cycle_tracker[sym]['is_active'] = False
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    comp_df = pd.DataFrame(completed_trades)
    exp_hkd, exp_r, pl_ratio, mdd = 0, 0, 0, 0
    if not comp_df.empty:
        wins = comp_df[comp_df['PnL_HKD'] > 0]
        losses = comp_df[comp_df['PnL_HKD'] <= 0]
        pl_ratio = (wins['PnL_HKD'].mean() / abs(losses['PnL_HKD'].mean())) if not losses.empty and losses['PnL_HKD'].mean() != 0 else 0
        exp_r = comp_df['Trade_R'].mean() if 'Trade_R' in comp_df.columns else 0
        
        eq_series = pd.Series([e['Cumulative PnL'] for e in equity_curve])
        if not eq_series.empty:
            mdd = (eq_series - eq_series.cummax()).min()

    return {k: v for k, v in positions.items() if v['qty'] > 0.0001}, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, 0, pl_ratio, mdd

@st.cache_data(ttl=300)
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        data = yf.download(symbols_list, period="5d", interval="1d", progress=False, group_by='ticker')
        prices = {}
        for s in symbols_list:
            try:
                ticker_df = data[s] if len(symbols_list) > 1 else data
                prices[s] = float(ticker_df['Close'].dropna().iloc[-1])
            except: prices[s] = None
        return prices
    except: return {}

# --- 3. UI 渲染 ---
df = load_data()

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_in = format_symbol(st.text_input("代號 (Ticker)"))
        is_sell = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        q_in = st.number_input("股數", min_value=0.0, step=1.0)
        p_in = st.number_input("成交價格", min_value=0.0, step=0.01)
        sl_in = st.number_input("停損價格", min_value=0.0, step=0.01)
        st_in = st.selectbox("策略", ["Pullback", "Breakout", "➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略")
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        if st.form_submit_button("儲存紀錄"):
            if s_in and q_in > 0:
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in, "Emotion": emo_in, "Timestamp": int(time.time()),
                    "Market_Condition": "N/A", "Mistake_Tag": "None"
                })
                st.rerun()

active_pos, realized_total, comp_df, equity_df, _, exp_r, _, pl_ratio, mdd = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理分析", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("總實現損益", f"${realized_total:,.0f}", delta=f"{realized_total:,.0f}")
    m2.metric("期望值 (R)", f"{exp_r:.2f}R")
    m3.metric("盈虧比", f"{pl_ratio:.2f}")
    m4.metric("最大回撤", f"${mdd:,.0f}", delta_color="inverse")
    
    if not equity_df.empty:
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="資金曲線"), use_container_width=True)
    
    # 補回：交易排行榜
    if not comp_df.empty:
        st.divider()
        st.subheader("🏆 交易排行榜")
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**策略表現排行**")
            st.dataframe(comp_df.groupby("Strategy")["PnL_HKD"].sum().sort_values(ascending=False), use_container_width=True)
        with col_r:
            st.write("**標的獲利排行**")
            st.dataframe(comp_df.groupby("Symbol")["PnL_HKD"].sum().sort_values(ascending=False), use_container_width=True)

with t2:
    st.markdown("### 🟢 當前持倉")
    live_prices = get_live_prices(list(active_pos.keys()))
    pos_list = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
        pos_list.append({"代號": s, "持股": d['qty'], "成本價": d['avg_price'], "現價": now, "未實現損益": un_pnl})
    st.dataframe(pd.DataFrame(pos_list), column_config={"未實現損益": st.column_config.NumberColumn(format="$%.2f")}, use_container_width=True)

with t5:
    st.subheader("🛠️ 數據管理")
    
    # 補回：Bulk Upload
    st.markdown("### 📥 批次上傳 (Bulk Upload)")
    uploaded_file = st.file_uploader("選擇交易紀錄 CSV 檔案", type="csv")
    if uploaded_file:
        up_df = pd.read_csv(uploaded_file)
        if st.button("確認導入數據"):
            combined = pd.concat([df, up_df], ignore_index=True).drop_duplicates()
            save_all_data(combined)
            st.success("導入成功！")
            st.rerun()

    st.divider()
    # 補回：編輯與刪除功能
    st.markdown("### 📝 編輯 / 刪除交易紀錄")
    if not df.empty:
        edited_df = st.data_editor(df.sort_values("Timestamp", ascending=False), num_rows="dynamic", key="data_editor", use_container_width=True)
        if st.button("保存編輯內容"):
            save_all_data(edited_df)
            st.success("變更已保存")
            st.rerun()
        
        st.divider()
        st.download_button("📥 導出交易紀錄 CSV", df.to_csv(index=False), "trades.csv", "text/csv")
        if st.button("🚨 清空資料庫"):
            save_all_data(pd.DataFrame(columns=df.columns))
            st.rerun()
