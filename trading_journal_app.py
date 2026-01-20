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
import json

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8 

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro - Full Edition", layout="wide")

def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
            "Market_Condition", "Mistake_Tag", "Trade_ID" 
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
        for col in ["Market_Condition", "Mistake_Tag", "Img", "Trade_ID"]:
            if col not in df.columns: df[col] = "N/A" if col not in ["Img", "Trade_ID"] else None
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

# --- 2. 核心計算邏輯 (Trade_ID & 損益計算) ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame()
    
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    cycle_tracker = {} 
    active_trade_map = {} 
    completed_trades = [] 
    equity_curve = []

    for _, row in df.iterrows():
        sym = format_symbol(row['Symbol']) 
        action = str(row['Action']) if pd.notnull(row['Action']) else ""
        if not sym or not action: continue

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])
        date_str = row['Date']
        ts = row['Timestamp']
        row_tid = row.get('Trade_ID')

        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        # ID 處理邏輯
        current_tid = None
        if is_buy:
            if sym in active_trade_map: current_tid = active_trade_map[sym]
            else:
                current_tid = row_tid if (pd.notnull(row_tid) and row_tid != "N/A") else f"gen_{sym}_{ts}"
                active_trade_map[sym] = current_tid
        elif is_sell:
            current_tid = active_trade_map.get(sym)
            if not current_tid: continue

        if current_tid not in cycle_tracker:
            cycle_tracker[current_tid] = {
                'Symbol': sym, 'cash_flow_raw': 0.0, 'start_date': date_str, 'is_active': True,
                'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'Entry_Price': price, 'Entry_SL': sl,
                'initial_risk_raw': abs(price - sl) * qty if sl > 0 else 0.0,
                'Strategy': row.get('Strategy', ''), 'Emotion': row.get('Emotion', ''),
                'Market_Condition': row.get('Market_Condition', ''), 'Mistake_Tag': row.get('Mistake_Tag', ''),
                'Notes': row.get('Notes', ''), 'Img': row.get('Img', None)
            }
            
        cycle = cycle_tracker[current_tid]
        if sl > 0: cycle['last_sl'] = sl

        if is_buy:
            cycle['cash_flow_raw'] -= (qty * price)
            total_cost_base = (cycle['qty'] * cycle['avg_price']) + (qty * price)
            new_qty = cycle['qty'] + qty
            if new_qty > 0: cycle['avg_price'] = total_cost_base / new_qty
            cycle['qty'] = new_qty
        elif is_sell:
            sell_qty = min(qty, cycle['qty'])
            cycle['cash_flow_raw'] += (sell_qty * price)
            pnl_item_hkd = get_hkd_value(sym, (price - cycle['avg_price']) * sell_qty)
            total_realized_pnl_hkd += pnl_item_hkd
            running_pnl_hkd += pnl_item_hkd
            cycle['qty'] -= sell_qty
            
            if cycle['qty'] < 0.0001:
                pnl_raw = cycle['cash_flow_raw']
                init_risk = cycle['initial_risk_raw']
                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                completed_trades.append({
                    "Exit_Date": date_str, "Entry_Date": cycle['start_date'], "Symbol": sym, 
                    "PnL_HKD": get_hkd_value(sym, pnl_raw), "Trade_R": trade_r,
                    "Strategy": cycle['Strategy'], "Emotion": cycle['Emotion'], "Notes": cycle['Notes'],
                    "Mistake_Tag": cycle['Mistake_Tag'], "Img": cycle['Img']
                })
                cycle['is_active'] = False
                if sym in active_trade_map: del active_trade_map[sym]
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    active_positions = {c['Symbol']: {
        'qty': c['qty'], 'avg_price': c['avg_price'], 'last_sl': c['last_sl'],
        'first_sl': c['Entry_SL'], 'first_price': c['Entry_Price'], 'Trade_ID': tid
    } for tid, c in cycle_tracker.items() if c['is_active'] and c['qty'] > 0.0001}

    return active_positions, total_realized_pnl_hkd, pd.DataFrame(completed_trades), pd.DataFrame(equity_curve)

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
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 交易錄入")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_in = format_symbol(st.text_input("代號 (Ticker)").upper().strip())
        is_sell = st.toggle("買入 Buy 🟢 / 賣出 Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數", min_value=0.0, step=1.0)
        p_in = col2.number_input("價格", min_value=0.0, step=0.01)
        sl_in = st.number_input("止損價", min_value=0.0, step=0.01)
        st_in = st.selectbox("策略", ["Pullback", "Breakout", "Mean Reversion"])
        emo_in = st.select_slider("情緒", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        mkt_cond = st.selectbox("市場", ["Trending Up", "Range", "Trending Down"])
        mistake_in = st.selectbox("標籤", ["None", "Fomo", "Early Exit", "Late Entry"])
        note_in = st.text_area("筆記")
        img_file = st.file_uploader("📸 圖表截圖", type=['png','jpg','jpeg'])
        
        if st.form_submit_button("儲存交易"):
            if s_in and q_in > 0:
                img_path = None
                if img_file:
                    img_path = os.path.join("images", f"{int(time.time())}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                
                tid = active_pos[s_in]['Trade_ID'] if s_in in active_pos else f"T_{int(time.time())}"
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": st_in, "Price": p_in, "Quantity": q_in, "Stop_Loss": sl_in,
                    "Emotion": emo_in, "Notes": note_in, "Timestamp": int(time.time()),
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in, "Img": img_path, "Trade_ID": tid
                })
                st.success("已儲存！"); time.sleep(0.5); st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效總覽", "🔥 實時持倉", "🔄 交易重播", "🧠 心理日誌", "🛠️ 管理"])

with t1:
    st.subheader("📊 績效指標")
    if not completed_trades_df.empty:
        best_trade = completed_trades_df.loc[completed_trades_df['PnL_HKD'].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("已實現損益", f"${realized_pnl_total_hkd:,.0f}")
        m2.metric("勝率", f"{(len(completed_trades_df[completed_trades_df['PnL_HKD']>0])/len(completed_trades_df)*100):.1f}%")
        m3.metric("平均 R 乘數", f"{completed_trades_df['Trade_R'].mean():.2f}R")
        m4.metric("單筆最強", f"${best_trade['PnL_HKD']:,.0f}", f"{best_trade['Symbol']}")
        
        st.plotly_chart(px.line(equity_df, x="Date", y="Cumulative PnL", title="資金曲線"), use_container_width=True)

with t2:
    st.subheader("🟢 當前持倉風險")
    if active_pos:
        prices = get_live_prices(list(active_pos.keys()))
        p_list = []
        for s, d in active_pos.items():
            now = prices.get(s, 0)
            un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
            risk = (now - d['last_sl']) * d['qty'] if (now and d['last_sl'] > 0) else 0
            p_list.append({"代號": s, "股數": d['qty'], "成本": d['avg_price'], "現價": now, "未實現": un_pnl, "當前風險": risk})
        st.table(pd.DataFrame(p_list))
        
        # 增加風險熱圖 (Risk Heatmap)
        fig = px.treemap(pd.DataFrame(p_list), path=['代號'], values='股數', color='未實現', color_continuous_scale='RdYlGn', title="持倉風險熱圖")
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.subheader("⏪ 交易重播")
    if not df.empty:
        idx = st.selectbox("選擇交易記錄", df.index, format_func=lambda x: f"{df.iloc[x]['Date']} {df.iloc[x]['Symbol']}")
        row = df.iloc[idx]
        if row['Img'] and os.path.exists(row['Img']):
            st.image(row['Img'], use_container_width=True)
        st.json(row.to_dict())

with t4:
    st.subheader("🧠 交易心理與錯誤複盤")
    if not completed_trades_df.empty:
        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(px.pie(completed_trades_df, names='Emotion', title="交易時情緒分佈"), use_container_width=True)
        with colB:
            st.plotly_chart(px.bar(completed_trades_df.groupby('Mistake_Tag')['PnL_HKD'].sum().reset_index(), x='Mistake_Tag', y='PnL_HKD', title="各類錯誤導致的損益"), use_container_width=True)
        
        # AI 導出功能
        st.markdown("### 🤖 AI 複盤導出")
        ai_json = completed_trades_df.to_json(orient='records')
        st.download_button("下載完整數據供 AI 分析", ai_json, "trade_history_for_ai.json", "application/json")

with t5:
    st.subheader("🛠️ 數據管理")
    st.dataframe(df)
    if st.button("🚨 清空數據庫"): 
        save_all_data(pd.DataFrame(columns=df.columns))
        st.rerun()
