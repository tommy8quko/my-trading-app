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

# --- 2. 核心邏輯：計算分批持倉與損益 ---
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
        sl = float(row['Stop_Loss'])
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0}
            
        curr = positions[sym]
        
        if sl > 0:
            curr['last_sl'] = sl
        
        if "買入 Buy" in action:
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty > 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
        
        elif "賣出 Sell" in action:
            if curr['qty'] > 0:
                sell_qty = min(qty, curr['qty'])
                pnl = (price - curr['avg_price']) * sell_qty
                total_realized_pnl += pnl
                running_pnl += pnl
                curr['qty'] -= sell_qty
                
                trade_history.append({
                    "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                    "Action": "賣出 Sell", "Price": price, "Cost": curr['avg_price'],
                    "Qty": sell_qty, "PnL": pnl, "Emotion": row.get('Emotion', '平靜')
                })
                equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0.0001}
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

# --- 3. 即時報價 ---
@st.cache_data(ttl=300)
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
active_pos, realized_pnl, history_df, equity_df = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號", placeholder="例如: 700 或 TSLA").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=1.0)
        p_in = col2.number_input("成交價格 (Price)", min_value=0.0, step=0.01)
        
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.01)
        
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
            if not s_in or q_in <= 0 or p_in <= 0:
                st.error("請完整填寫代號、股數與價格")
            else:
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), 
                    "Symbol": s_in, 
                    "Action": act_in, 
                    "Strategy": st_in, 
                    "Price": p_in, 
                    "Quantity": q_in, 
                    "Stop_Loss": sl_in,
                    "Fees": 0, 
                    "Emotion": emo_in, 
                    "Risk_Reward": rr_in, 
                    "Notes": note_in, 
                    "Timestamp": int(time.time())
                })
                st.success(f"✅ 已儲存 {s_in}")
                time.sleep(0.5)
                st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

# 先獲取當前報價以便計算總回撤風險
current_symbols = list(active_pos.keys())
live_prices = get_live_prices(current_symbols)

# 計算各標的 SL Risk 並加總
aggregate_sl_risk = 0
processed_p_data = []
if active_pos:
    for s, d in active_pos.items():
        now = live_prices.get(s)
        qty = d['qty']
        avg_p = d['avg_price']
        last_sl = d['last_sl']
        un_pnl = (now - avg_p) * qty if now else 0
        sl_risk_amt = (now - last_sl) * qty if (now and last_sl > 0) else 0
        aggregate_sl_risk += sl_risk_amt

        processed_p_data.append({
            "代號": s, "股數": f"{qty:,.0f}", "成本": f"${avg_p:.2f}", 
            "停損價": f"${last_sl:.2f}" if last_sl > 0 else "未設定", 
            "現價": f"${now:.2f}" if now else "讀取中...", 
            "未實現損益": f"${un_pnl:,.2f}", 
            "報酬%": f"{(un_pnl/(qty * avg_p)*100):.1f}%" if (now and avg_p!=0) else "0%",
            "停損回撤 (SL Risk)": f"${sl_risk_amt:,.2f}" if now else "N/A"
        })

with t1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("已實現損益", f"${realized_pnl:,.2f}")
    win_r = (len(history_df[history_df['PnL']>0])/len(history_df)*100) if not history_df.empty else 0
    col2.metric("勝率", f"{win_r:.1f}%")
    col3.metric("平均 R:R", f"{df['Risk_Reward'].mean():.2f}" if not df.empty else "0")
    # 將 MDD 定義為當前所有持倉的總停損風險金額
    col4.metric("總回撤風險 (SL Risk)", f"${aggregate_sl_risk:,.2f}", delta_color="inverse", help="當前持倉全部觸發停損時的預期資金回吐總額")
    
    if not equity_df.empty:
        fig_equity = px.area(equity_df, x="Date", y="Cumulative PnL", title="帳戶權益成長曲線", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_equity, use_container_width=True)

with t2:
    if active_pos:
        st.dataframe(pd.DataFrame(processed_p_data), use_container_width=True, hide_index=True)
        if st.button("🔄 刷新即時報價"): st.cache_data.clear(); st.rerun()
    else: st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 市場環境重播")
    if not df.empty:
        target = st.selectbox("選擇回顧交易", df.index, format_func=lambda x: f"{df.iloc[x]['Date']} | {df.iloc[x]['Symbol']} | {df.iloc[x]['Action']}")
        row = df.iloc[target]
        try:
            t_date = pd.to_datetime(row['Date'])
            data = yf.download(row['Symbol'], start=(t_date - timedelta(days=15)).strftime('%Y-%m-%d'), end=(t_date + timedelta(days=15)).strftime('%Y-%m-%d'), progress=False)
            if not data.empty:
                fig_replay = go.Figure()
                fig_replay.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價'))
                fig_replay.add_trace(go.Scatter(x=[t_date], y=[row['Price']], mode='markers+text', text=['📍 EXEC'], marker=dict(color='orange', size=12)))
                st.plotly_chart(fig_replay, use_container_width=True)
        except: st.warning("無法載入圖表數據")

with t4:
    st.subheader("📜 歷史紀錄")
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據編輯與管理")
    if not df.empty:
        edit_df = df.sort_values("Timestamp", ascending=False)
        selected_trade_idx = st.selectbox(
            "選擇要修改或刪除的交易", 
            edit_df.index, 
            format_func=lambda x: f"[{df.loc[x, 'Date']}] {df.loc[x, 'Symbol']} - {df.loc[x, 'Action']} (${df.loc[x, 'Price']})"
        )
        
        trade_to_edit = df.loc[selected_trade_idx].copy()
        
        st.markdown("---")
        col_e1, col_e2, col_e3 = st.columns(3)
        new_date = col_e1.date_input("修改日期", value=pd.to_datetime(trade_to_edit['Date']))
        new_price = col_e2.number_input("修改價格", value=float(trade_to_edit['Price']))
        new_qty = col_e3.number_input("修改股數", value=float(trade_to_edit['Quantity']))
        
        col_e4, col_e5, col_e6 = st.columns(3)
        new_sl = col_e4.number_input("修改停損價", value=float(trade_to_edit['Stop_Loss']))
        new_strategy = col_e5.text_input("修改策略", value=str(trade_to_edit['Strategy']))
        new_emotion = col_e6.selectbox("修改情緒", ["恐慌", "猶豫", "平靜", "自信", "衝動"], index=["恐慌", "猶豫", "平靜", "自信", "衝動"].index(trade_to_edit['Emotion']))
        
        new_notes = st.text_area("修改筆記", value=str(trade_to_edit['Notes']))
        
        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        
        if btn_col1.button("💾 更新此筆紀錄", use_container_width=True):
            df.loc[selected_trade_idx, 'Date'] = new_date.strftime('%Y-%m-%d')
            df.loc[selected_trade_idx, 'Price'] = new_price
            df.loc[selected_trade_idx, 'Quantity'] = new_qty
            df.loc[selected_trade_idx, 'Stop_Loss'] = new_sl
            df.loc[selected_trade_idx, 'Strategy'] = new_strategy
            df.loc[selected_trade_idx, 'Emotion'] = new_emotion
            df.loc[selected_trade_idx, 'Notes'] = new_notes
            save_all_data(df)
            st.success("更新成功！")
            time.sleep(0.5)
            st.rerun()
            
        if btn_col2.button("🗑️ 刪除此筆紀錄", use_container_width=True):
            df = df.drop(selected_trade_idx)
            save_all_data(df)
            st.warning("紀錄已刪除。")
            time.sleep(0.5)
            st.rerun()
    else:
        st.info("尚無紀錄可供編輯。")
