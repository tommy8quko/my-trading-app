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
# 新增 Google Sheets 連線庫
from streamlit_gsheets import GSheetsConnection

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8 

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

# --- 資料讀取層 ---
def get_data_connection():
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except:
        return None

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
    conn = get_data_connection()
    df = pd.DataFrame()
    
    try:
        if conn:
            df = conn.read(worksheet="Log", ttl=0)
        else:
            raise Exception("No connection")
    except:
        init_csv()
        try:
            df = pd.read_csv(FILE_NAME)
        except:
            return pd.DataFrame()

    if df.empty: return df
    
    if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)
    if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
    for col in ["Market_Condition", "Mistake_Tag", "Img"]:
        if col not in df.columns: df[col] = "N/A" if col != "Img" else None
    
    if 'Trade_ID' not in df.columns:
        df['Trade_ID'] = pd.NA

    if 'Timestamp' not in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'], errors='coerce').view('int64') // 10**9
        save_all_data(df)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    # 修復點：確保數字欄位不會因為空值導致計算崩潰
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
    df['Stop_Loss'] = pd.to_numeric(df['Stop_Loss'], errors='coerce').fillna(0.0)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').fillna(0)
    return df

def save_all_data(df):
    conn = get_data_connection()
    try:
        if conn:
            conn.update(worksheet="Log", data=df)
        else:
            raise Exception("No connection")
    except:
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

def get_symbol_state(df, symbol):
    """判斷目前是否有持倉以及對應的 Trade_ID"""
    if df.empty: return False, None
    df = df.sort_values(by="Timestamp", kind='mergesort')
    sym_df = df[df['Symbol'] == symbol]
    if sym_df.empty: return False, None

    current_cycle_id = None
    current_qty = 0
    
    for _, row in sym_df.iterrows():
        action = str(row['Action']).upper()
        r_qty = float(row['Quantity'])
        r_tid = row.get('Trade_ID')
        
        is_buy = any(word in action for word in ["買入", "BUY", "B"])
        is_sell = any(word in action for word in ["賣出", "SELL", "S"])
        
        if is_buy:
            if current_qty < 0.0001:
                current_cycle_id = r_tid 
            current_qty += r_qty
        elif is_sell:
            current_qty -= r_qty
            if current_qty <= 0.0001:
                current_qty = 0
                current_cycle_id = None
    
    return (current_qty > 0.0001), current_cycle_id


# --- 2. 核心計算邏輯 (修正 P&L 穩定性) ---
@st.cache_data(ttl=60)
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    positions = {} 
    df = df.sort_values(by="Timestamp", kind='mergesort')
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    
    cycle_tracker = {} 
    completed_trades = [] 
    equity_curve = []
    
    active_trade_by_symbol = {}

    for _, row in df.iterrows():
        sym = format_symbol(row['Symbol']) 
        action = str(row['Action']).upper() if pd.notnull(row['Action']) else ""
        if not sym or not action: continue

        qty = float(row['Quantity'])
        price = float(row['Price'])
        sl = float(row['Stop_Loss'])
        date_str = row['Date']
        ts = row['Timestamp']
        
        trade_id = row.get('Trade_ID')
        
        is_buy = any(word in action for word in ["買入", "BUY", "B"])
        is_sell = any(word in action for word in ["賣出", "SELL", "S"])
        
        if sym not in positions: 
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'trade_id': None}
        curr_pos = positions[sym]
        
        if is_buy:
            if pd.isna(trade_id):
                trade_id = active_trade_by_symbol.get(sym, int(ts))

            if sym not in active_trade_by_symbol:
                active_trade_by_symbol[sym] = trade_id
                init_risk = abs(price - sl) * qty if sl > 0 else 0
                
                cycle_tracker[trade_id] = {
                    'Symbol': sym, 'start_date': date_str, 'cash_flow_raw': 0.0,
                    'initial_risk_raw': init_risk, 'Entry_Price': price, 'Entry_SL': sl,
                    'Strategy': row.get('Strategy', ''), 'Emotion': row.get('Emotion', ''),
                    'Market_Condition': row.get('Market_Condition', ''), 'Mistake_Tag': row.get('Mistake_Tag', '')
                }
                curr_pos['trade_id'] = trade_id
            
            target_tid = active_trade_by_symbol[sym]
            if target_tid in cycle_tracker:
                cycle_tracker[target_tid]['cash_flow_raw'] -= (qty * price)
            
            total_cost_base = (curr_pos['qty'] * curr_pos['avg_price']) + (qty * price)
            new_qty = curr_pos['qty'] + qty
            if new_qty > 0: curr_pos['avg_price'] = total_cost_base / new_qty
            curr_pos['qty'] = new_qty
            if sl > 0: curr_pos['last_sl'] = sl
            
        elif is_sell:
            active_tid = active_trade_by_symbol.get(sym)
            pos_tid = curr_pos.get('trade_id')
            target_tid = active_tid if active_tid else (pos_tid if pos_tid else trade_id)

            if target_tid and target_tid in cycle_tracker:
                sell_qty = min(qty, curr_pos['qty'])
                cycle_tracker[target_tid]['cash_flow_raw'] += (sell_qty * price)
                
                pnl_item_raw = (price - curr_pos['avg_price']) * sell_qty
                real_pnl_hkd = get_hkd_value(sym, pnl_item_raw)
                total_realized_pnl_hkd += real_pnl_hkd
                running_pnl_hkd += real_pnl_hkd
                
                curr_pos['qty'] -= sell_qty
                if sl > 0: curr_pos['last_sl'] = sl
                
                if curr_pos['qty'] < 0.0001:
                    c_data = cycle_tracker[target_tid]
                    d1 = datetime.strptime(c_data['start_date'], '%Y-%m-%d')
                    d2 = datetime.strptime(date_str, '%Y-%m-%d')
                    pnl_raw = c_data['cash_flow_raw']
                    init_risk = c_data['initial_risk_raw']
                    trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                    
                    completed_trades.append({
                        "Trade_ID": target_tid, "Exit_Date": date_str, "Entry_Date": c_data['start_date'], 
                        "Symbol": sym, "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                        "Duration_Days": float((d2 - d1).days), "Trade_R": trade_r,
                        "Strategy": c_data['Strategy'], "Emotion": c_data['Emotion'],
                        "Market_Condition": c_data['Market_Condition'], "Mistake_Tag": c_data['Mistake_Tag']
                    })
                    if sym in active_trade_by_symbol: del active_trade_by_symbol[sym]
                    curr_pos['qty'] = 0
                    
        equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    final_active_positions = {}
    for k, v in positions.items():
        if v['qty'] > 0.0001:
            tid = active_trade_by_symbol.get(k) or v.get('trade_id')
            if tid and tid in cycle_tracker:
                v['Entry_Price'] = cycle_tracker[tid]['Entry_Price']
                v['Entry_SL'] = cycle_tracker[tid]['Entry_SL']
            final_active_positions[k] = v

    comp_df = pd.DataFrame(completed_trades)
    exp_hkd, exp_r, avg_dur = 0, 0, 0
    if not comp_df.empty:
        wins = comp_df[comp_df['PnL_HKD'] > 0]
        losses = comp_df[comp_df['PnL_HKD'] <= 0]
        wr = len(wins) / len(comp_df)
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)
        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]
        exp_r = valid_r_trades['Trade_R'].mean() if not valid_r_trades.empty else 0
        avg_dur = comp_df['Duration_Days'].mean()

    return final_active_positions, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur

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

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_in = format_symbol(st.text_input("代號 (Ticker)").upper().strip())
        is_sell = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=1.0, value=None)
        p_in = col2.number_input("成交價格 (Price)", min_value=0.0, step=0.01, value=None)
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.01, value=None)
        st.divider()
        mkt_cond = st.selectbox("市場環境", ["Trending Up", "Trending Down", "Range/Choppy", "High Volatility", "N/A"])
        mistake_in = st.selectbox("錯誤標籤", ["None", "Fomo", "Revenge Trade", "Fat Finger", "Late Entry", "Moved Stop"])
        st_in = st.selectbox("策略 (Strategy)", ["Pullback", "Breakout", "➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略名稱")
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        note_in = st.text_area("決策筆記")
        img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'])
        
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in is not None and p_in is not None:
                img_path = None
                if img_file is not None:
                    ts_str = str(int(time.time()))
                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                
                is_active, active_tid = get_symbol_state(df, s_in)
                if not is_sell:
                    final_tid = active_tid if (is_active and active_tid) else int(time.time())
                else:
                    final_tid = active_tid if active_tid else int(time.time())

                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": 0, "Notes": note_in, "Timestamp": int(time.time()), 
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in, "Img": img_path, "Trade_ID": final_tid
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

active_pos, total_pnl_hkd, comp_trades_df, equity_df, exp_hkd, exp_r, avg_dur = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_frame = st.selectbox("統計時間範圍", ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"])
    
    f_comp = comp_trades_df.copy()
    if not f_comp.empty and time_frame != "全部記錄":
        f_comp['Entry_DT'] = pd.to_datetime(f_comp['Entry_Date'])
        f_comp['Exit_DT'] = pd.to_datetime(f_comp['Exit_Date'])
        today = datetime.now()
        start_date = None
        if "今年" in time_frame: start_date = datetime(today.year, 1, 1)
        elif "本月" in time_frame: start_date = datetime(today.year, today.month, 1)
        elif "本週" in time_frame: start_date = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif "3個月" in time_frame: start_date = today - timedelta(days=90)
            
        if start_date:
            f_comp = f_comp[(f_comp['Entry_DT'] >= start_date) & (f_comp['Exit_DT'] >= start_date)]

    f_pnl = f_comp['PnL_HKD'].sum() if not f_comp.empty else 0
    trade_count = len(f_comp)
    win_r = (len(f_comp[f_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")
    m2.metric("期望值 (R)", f"{exp_r:.2f}R")
    m3.metric("勝率", f"{win_r:.1f}%")
    m4.metric("交易次數", f"{trade_count}")
    m5.metric("平均持倉", f"{avg_dur:.1f} 天")

    if not equity_df.empty: st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線", height=300), use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
    current_symbols = list(active_pos.keys())
    live_prices = get_live_prices(current_symbols)
    processed_p_data = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
        entry_p = d.get('Entry_Price', avg_p)
        entry_sl = d.get('Entry_SL', 0)
        un_pnl = (now - avg_p) * qty if now else 0
        roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
        init_risk = abs(entry_p - entry_sl) * qty if entry_sl > 0 else 0
        curr_r = (un_pnl / init_risk) if (now and init_risk > 0) else 0
        
        processed_p_data.append({
            "代號": s, "持股數": f"{qty:,.0f}", "平均成本": f"{avg_p:,.2f}", 
            "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{last_sl:,.2f}", 
            "當前R": f"{curr_r:.2f}R", "未實現損益": f"{un_pnl:,.2f}", "報酬%": roi
        })
    if processed_p_data: 
        st.dataframe(pd.DataFrame(processed_p_data), column_config={"報酬%": st.column_config.ProgressColumn("報酬%", format="%.2f%%", min_value=-20, max_value=20)}, hide_index=True, use_container_width=True)
        if st.button("🔄 刷新即時報價", use_container_width=True): st.cache_data.clear(); st.rerun()

with t4:
    st.subheader("📜 歷史交易紀錄")
    if not df.empty:
        hist_df = df.sort_values("Timestamp", ascending=False).copy()
        st.dataframe(hist_df[["Date", "Symbol", "Action", "Price", "Quantity", "Strategy", "Trade_ID"]], use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    conn_status = get_data_connection()
    if conn_status: st.success("🟢 已連接至 Google Sheets")
    else: st.warning("🟠 目前使用本地 CSV 模式")

    if st.button("🚨 清空所有數據"):
        save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()
