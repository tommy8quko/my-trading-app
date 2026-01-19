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
USD_HKD_RATE = 7.8 # 固定匯率轉換

if not os.path.exists("images"):
    os.makedirs("images")

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
    if not str(symbol).endswith(".HK"):
        return value * USD_HKD_RATE
    return value

# --- 2. 核心邏輯 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame()
    
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    cycle_tracker = {}
    completed_trades = [] 
    equity_curve = []

    for _, row in df.iterrows():
        sym = row['Symbol']
        action = row['Action']
        qty = float(row['Quantity'])
        price = float(row['Price'])
        sl = float(row['Stop_Loss'])
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0}
            cycle_tracker[sym] = {'pnl_hkd': 0.0}
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        if "買入 Buy" in action:
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty > 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
        
        elif "賣出 Sell" in action:
            if curr['qty'] > 0:
                sell_qty = min(qty, curr['qty'])
                pnl_raw = (price - curr['avg_price']) * sell_qty
                pnl_hkd = get_hkd_value(sym, pnl_raw)
                total_realized_pnl_hkd += pnl_hkd
                running_pnl_hkd += pnl_hkd
                cycle_tracker[sym]['pnl_hkd'] += pnl_hkd
                curr['qty'] -= sell_qty
                
                if curr['qty'] < 0.0001:
                    completed_trades.append({
                        "Date": date, "Symbol": sym, "TotalPnL_HKD": cycle_tracker[sym]['pnl_hkd']
                    })
                    cycle_tracker[sym]['pnl_hkd'] = 0.0
                equity_curve.append({"Date": date, "Cumulative PnL": running_pnl_hkd})

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0.0001}
    return active_positions, total_realized_pnl_hkd, pd.DataFrame(completed_trades), pd.DataFrame(equity_curve)

@st.cache_data(ttl=60)
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        data = yf.download(symbols_list, period="1d", interval="1m", progress=False)
        prices = {}
        for s in symbols_list:
            try:
                if len(symbols_list) > 1:
                    val = data['Close'][s].dropna().iloc[-1]
                else:
                    val = data['Close'].dropna().iloc[-1]
                prices[s] = float(val)
            except:
                prices[s] = None
        return prices
    except:
        return {}

# --- 4. UI 介面 ---
df = load_data()
active_pos, _, _, _ = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號 (Ticker)", placeholder="例如: 700 或 TSLA").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        is_sell = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        toggle_color = "#EF553B" if is_sell else "#00CC96"
        st.markdown(f"<style>div[data-testid='stCheckboxToggle'] div[data-baseweb='checkbox'] div {{background-color: {toggle_color} !important;}}</style>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=1.0, value=None)
        p_in = col2.number_input("成交價格 (Price)", min_value=0.0, step=0.01, value=None)
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.01, value=None)
        st.divider()
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        rr_in = st.number_input("預期盈虧比 (R:R)", value=2.0, min_value=0.1)
        default_strategies = ["Pullback", "Breakout", "Buyable Gapup"]
        existing_custom = [s for s in df['Strategy'].unique().tolist() if s not in default_strategies] if not df.empty else []
        tags = default_strategies + existing_custom
        st_in = st.selectbox("策略 (Strategy)", tags + ["➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略名稱")
        note_in = st.text_area("決策筆記")
        if st.form_submit_button("儲存執行紀錄"):
            if not s_in or q_in is None or p_in is None or q_in <= 0 or p_in <= 0:
                st.error("請完整填寫代號、股數與價格")
            else:
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": st_in, "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": rr_in, "Notes": note_in, "Timestamp": int(time.time())
                })
                st.success(f"✅ 已儲存 {s_in}")
                time.sleep(0.5)
                st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_frame = st.selectbox("統計時間範圍", ["全部記錄", "今年", "本月", "最近 30 天"], index=0)
    filtered_df = df.copy()
    if not filtered_df.empty:
        filtered_df['Date_DT'] = pd.to_datetime(filtered_df['Date'])
        today = datetime.now()
        if time_frame == "今年": filtered_df = filtered_df[filtered_df['Date_DT'].dt.year == today.year]
        elif time_frame == "本月": filtered_df = filtered_df[(filtered_df['Date_DT'].dt.year == today.year) & (filtered_df['Date_DT'].dt.month == today.month)]
        elif time_frame == "最近 30 天": filtered_df = filtered_df[filtered_df['Date_DT'] >= (today - timedelta(days=30))]
            
    _, realized_pnl_hkd, completed_trades_df, equity_df = calculate_portfolio(filtered_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("已實現損益 (HKD)", f"${realized_pnl_hkd:,.2f}")
    win_r = (len(completed_trades_df[completed_trades_df['TotalPnL_HKD'] > 0]) / len(completed_trades_df) * 100) if not completed_trades_df.empty else 0
    col2.metric("勝率", f"{win_r:.1f}%")
    col3.metric("平均 R:R", f"{filtered_df['Risk_Reward'].mean():.2f}" if not filtered_df.empty else "0")
    col4.metric("策略數", f"{len(filtered_df['Strategy'].unique()) if not filtered_df.empty else 0}")

    if not equity_df.empty:
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", height=300), use_container_width=True)

# 獲取報價與計算即時風險
current_symbols = list(active_pos.keys())
live_prices = get_live_prices(current_symbols)
aggregate_sl_risk_hkd = 0
processed_p_data = []

if active_pos:
    for s, d in active_pos.items():
        now = live_prices.get(s)
        qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
        un_pnl_raw = (now - avg_p) * qty if now else 0
        sl_risk_amt_raw = (now - last_sl) * qty if (now and last_sl > 0) else 0
        aggregate_sl_risk_hkd += get_hkd_value(s, sl_risk_amt_raw)
        
        # 數據清理供表格使用 (Compact Format)
        processed_p_data.append({
            "Ticker": s,
            "Qty": qty,
            "Avg": avg_p,
            "Last": now if now else 0,
            "SL": last_sl,
            "PnL": un_pnl_raw,
            "Return%": (un_pnl_raw/(qty * avg_p)*100) if (now and avg_p!=0) else 0,
            "SL_Risk": sl_risk_amt_raw if now else 0
        })

with t2:
    st.markdown("### 🟢 持倉概覽 (Compact View)")
    if processed_p_data:
        p_df = pd.DataFrame(processed_p_data)
        
        # 使用 st.column_config 優化顯示，減少寬度並增加可視性
        st.dataframe(
            p_df,
            column_config={
                "Ticker": st.column_config.TextColumn("代號", width="small"),
                "Qty": st.column_config.NumberColumn("股數", format="%d", width="small"),
                "Avg": st.column_config.NumberColumn("成本", format="$%.2f", width="small"),
                "Last": st.column_config.NumberColumn("現價", format="$%.2f", width="small"),
                "SL": st.column_config.NumberColumn("停損", format="$%.2f", width="small"),
                "PnL": st.column_config.NumberColumn("損益", format="$%d", width="medium"),
                "Return%": st.column_config.ProgressColumn(
                    "報酬%",
                    help="未實現報酬率",
                    format="%.1f%%",
                    min_value=-20,
                    max_value=20,
                ),
                "SL_Risk": st.column_config.NumberColumn("SL風險", format="$%d", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            height=min(len(p_df) * 35 + 40, 800) # 動態高度，支撐 30 隻股票約 1000px 以內
        )
        
        col_risk1, col_risk2 = st.columns(2)
        col_risk1.metric("總持倉回撤風險 (SL Risk HKD)", f"${aggregate_sl_risk_hkd:,.2f}", delta_color="inverse")
        if st.button("🔄 刷新即時報價", use_container_width=True): st.cache_data.clear(); st.rerun()
    else:
        st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 市場環境重播")
    if not df.empty:
        target = st.selectbox("選擇交易", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']}")
        row = df.iloc[target]
        data = yf.download(row['Symbol'], start=(pd.to_datetime(row['Date']) - timedelta(days=15)).strftime('%Y-%m-%d'), end=(pd.to_datetime(row['Date']) + timedelta(days=15)).strftime('%Y-%m-%d'), progress=False)
        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=[pd.to_datetime(row['Date'])], y=[row['Price']], mode='markers+text', text=['📍 EXEC'], marker=dict(color='orange', size=12)))
            st.plotly_chart(fig, use_container_width=True)

with t4:
    st.subheader("📜 歷史紀錄")
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    with st.expander("📤 批量上傳"):
        uploaded_file = st.file_uploader("選擇 CSV/Excel", type=["xlsx", "csv"])
        if uploaded_file and st.button("🚀 確認上傳"):
            try:
                new_trades = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                new_trades['Action'] = new_trades['Action'].apply(lambda a: "買入 Buy" if str(a).upper()=="B" else ("賣出 Sell" if str(a).upper()=="S" else a))
                new_trades['Timestamp'] = int(time.time())
                df = pd.concat([df, new_trades], ignore_index=True)
                save_all_data(df); st.success("已完成！"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"錯誤: {e}")

    if not df.empty:
        st.markdown("### 📝 編輯紀錄")
        selected_idx = st.selectbox("選擇紀錄", df.index, format_func=lambda x: f"{df.loc[x, 'Symbol']} @ {df.loc[x, 'Date']}")
        t_edit = df.loc[selected_idx]
        col_e1, col_e2 = st.columns(2)
        n_p = col_e1.number_input("價格", value=float(t_edit['Price']))
        n_q = col_e2.number_input("股數", value=float(t_edit['Quantity']))
        if st.button("💾 更新"):
            df.loc[selected_idx, 'Price'] = n_p
            df.loc[selected_idx, 'Quantity'] = n_q
            save_all_data(df); st.rerun()
            
        st.divider()
        confirm = st.checkbox("確認刪除所有數據")
        if st.button("🔥 清空數據", disabled=not confirm):
            save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()
