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



# --- 改進部分：資料讀取層 (支援 Google Sheets 與 CSV 雙模式) ---

def get_data_connection():

    try:

        return st.connection("gsheets", type=GSheetsConnection)

    except:

        return None



def init_csv():

    if not os.path.exists(FILE_NAME):

        # Change 1: Added Trade_ID to schema

        df = pd.DataFrame(columns=[

            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 

            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",

            "Market_Condition", "Mistake_Tag", "Trade_ID"

        ])

        df.to_csv(FILE_NAME, index=False)



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

    

    # 數據類型轉換

    if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)

    if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)

    for col in ["Market_Condition", "Mistake_Tag", "Img", "Trade_ID"]:

        if col not in df.columns: df[col] = "N/A" if col != "Img" else None

    

    if 'Timestamp' not in df.columns:

        df['Timestamp'] = pd.to_datetime(df['Date'], errors='coerce').view('int64') // 10**9

        save_all_data(df)

    

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    df['Stop_Loss'] = pd.to_numeric(df['Stop_Loss'], errors='coerce').fillna(0)

    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')

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



# --- 2. 核心計算邏輯 (Change 2 & 3: Refactored Portfolio Calculation) ---

@st.cache_data(ttl=60)

def calculate_portfolio(df):

    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0

    

    positions = {} 

    df = df.sort_values(by="Timestamp")

    total_realized_pnl_hkd = 0

    running_pnl_hkd = 0

    

    # Change 2: Tracking with Trade_ID

    cycle_tracker = {} # Key: Trade_ID

    active_trade_by_symbol = {} # Key: Symbol, Value: Trade_ID

    completed_trades = [] 

    equity_curve = []



    for _, row in df.iterrows():

        sym = format_symbol(row['Symbol']) 

        action = str(row['Action']) if pd.notnull(row['Action']) else ""

        if not sym or not action: continue



        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])

        date_str = row['Date']

        

        # Handle Legacy Data: If Trade_ID is missing, create a temporary one for this session

        t_id = row.get('Trade_ID')

        if pd.isna(t_id) or t_id == "N/A":

            t_id = f"LEGACY_{sym}" 



        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])

        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])



        # Change 2: Logic to assign/find cycle

        current_trade_id = None

        if is_buy:

            if sym in active_trade_by_symbol:

                current_trade_id = active_trade_by_symbol[sym]

            else:

                current_trade_id = t_id

                active_trade_by_symbol[sym] = current_trade_id

                

            if current_trade_id not in cycle_tracker:

                # Change 3: Explicitly store Entry_Price and Entry_SL

                cycle_tracker[current_trade_id] = {

                    'symbol': sym,

                    'cash_flow_raw': 0.0, 

                    'start_date': date_str, 

                    'initial_risk_raw': 0.0,

                    'Entry_Price': price,

                    'Entry_SL': sl,

                    'qty_accumulated': 0.0,

                    'Strategy': row.get('Strategy', ''),

                    'Emotion': row.get('Emotion', ''),

                    'Market_Condition': row.get('Market_Condition', ''),

                    'Mistake_Tag': row.get('Mistake_Tag', '')

                }

                # Initial Risk calculation based on Entry Row

                if sl > 0:

                    cycle_tracker[current_trade_id]['initial_risk_raw'] = abs(price - sl) * qty

                

            # Update Position Data

            if sym not in positions:

                positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'trade_id': current_trade_id}

            

            curr = positions[sym]

            cycle_tracker[current_trade_id]['cash_flow_raw'] -= (qty * price)

            cycle_tracker[current_trade_id]['qty_accumulated'] += qty

            

            total_cost_base = (curr['qty'] * curr['avg_price']) + (qty * price)

            curr['qty'] += qty

            if curr['qty'] > 0: curr['avg_price'] = total_cost_base / curr['qty']

            if sl > 0: curr['last_sl'] = sl



        elif is_sell and sym in active_trade_by_symbol:

            current_trade_id = active_trade_by_symbol[sym]

            cycle_data = cycle_tracker[current_trade_id]

            curr = positions[sym]

            

            sell_qty = min(qty, curr['qty'])

            cycle_data['cash_flow_raw'] += (sell_qty * price)

            

            realized_pnl_hkd_item = get_hkd_value(sym, (price - curr['avg_price']) * sell_qty)

            total_realized_pnl_hkd += realized_pnl_hkd_item

            running_pnl_hkd += realized_pnl_hkd_item

            

            curr['qty'] -= sell_qty

            if sl > 0: curr['last_sl'] = sl



            if curr['qty'] < 0.0001:

                pnl_raw = cycle_data['cash_flow_raw']

                init_risk = cycle_data['initial_risk_raw']

                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None

                

                completed_trades.append({

                    "Trade_ID": current_trade_id,

                    "Exit_Date": date_str, 

                    "Entry_Date": cycle_data['start_date'], 

                    "Symbol": sym, 

                    "PnL_Raw": pnl_raw, 

                    "PnL_HKD": get_hkd_value(sym, pnl_raw),

                    "Duration_Days": float((datetime.strptime(date_str, '%Y-%m-%d') - datetime.strptime(cycle_data['start_date'], '%Y-%m-%d')).days), 

                    "Trade_R": trade_r,

                    "Strategy": cycle_data['Strategy'],

                    "Emotion": cycle_data['Emotion'],

                    "Market_Condition": cycle_data['Market_Condition'],

                    "Mistake_Tag": cycle_data['Mistake_Tag']

                })

                # Clean up trackers

                del active_trade_by_symbol[sym]

                if sym in positions: del positions[sym]

            

            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})



    # Prepare return values

    comp_df = pd.DataFrame(completed_trades)

    

    # Filter positions to only those still in active_trade_by_symbol

    active_output = {s: p for s, p in positions.items() if s in active_trade_by_symbol}

    # Attach cycle data for Tab 2

    for s, p in active_output.items():

        tid = active_trade_by_symbol[s]

        p['entry_price'] = cycle_tracker[tid]['Entry_Price']

        p['entry_sl'] = cycle_tracker[tid]['Entry_SL']



    exp_hkd, exp_r, avg_dur = 0, 0, 0

    if not comp_df.empty:

        wins, losses = comp_df[comp_df['PnL_HKD'] > 0], comp_df[comp_df['PnL_HKD'] <= 0]

        wr = len(wins) / len(comp_df)

        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0

        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0

        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)

        

        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]

        exp_r = valid_r_trades['Trade_R'].mean() if not valid_r_trades.empty else 0

        avg_dur = comp_df['Duration_Days'].mean()



    return active_output, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur



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



# Sidebar: Trade Form

with st.sidebar:

    st.header("⚡ 執行面板")

    # Change 1: Check for active cycles to determine Trade_ID

    active_pos_temp, _, _, _, _, _, _ = calculate_portfolio(df)

    

    with st.form("trade_form", clear_on_submit=True):

        d_in = st.date_input("日期")

        s_in = format_symbol(st.text_input("代號 (Ticker)").upper().strip())

        is_sell_toggle = st.toggle("Buy 🟢 / Sell 🔴", value=False)

        act_in = "賣出 Sell" if is_sell_toggle else "買入 Buy"

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

                assigned_tid = "N/A"

                if not is_sell_toggle: # Buy

                    if s_in in active_pos_temp:

                        assigned_tid = active_pos_temp[s_in]['trade_id']

                    else:

                        assigned_tid = int(time.time())

                else: # Sell

                    if s_in in active_pos_temp:

                        assigned_tid = active_pos_temp[s_in]['trade_id']

                    else:

                        st.error("找不到該標的的開倉紀錄，無法匹配 Trade_ID")



                img_path = None

                if img_file is not None:

                    if not os.path.exists("images"): os.makedirs("images")

                    ts_str = str(int(time.time()))

                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")

                    with open(img_path, "wb") as f:

                        f.write(img_file.getbuffer())

                

                save_transaction({

                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 

                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 

                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 

                    "Emotion": emo_in, "Risk_Reward": 0, 

                    "Notes": note_in, "Timestamp": int(time.time()), 

                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in,

                    "Img": img_path, "Trade_ID": assigned_tid

                })

                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()



# Pre-calculate main data

active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val = calculate_portfolio(df)



t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])



with t1:

    st.subheader("📊 績效概覽")

    time_options = ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"]

    time_frame = st.selectbox("統計時間範圍", time_options, index=0)

    

    filtered_comp = completed_trades_df.copy()

    if not filtered_comp.empty:

        filtered_comp['Entry_DT'] = pd.to_datetime(filtered_comp['Entry_Date'])

        filtered_comp['Exit_DT'] = pd.to_datetime(filtered_comp['Exit_Date'])

        today = datetime.now()

        

        if "今年" in time_frame:

            mask = (filtered_comp['Entry_DT'].dt.year == today.year) & (filtered_comp['Exit_DT'].dt.year == today.year)

        elif "本月" in time_frame:

            mask = (filtered_comp['Entry_DT'].dt.year == today.year) & (filtered_comp['Entry_DT'].dt.month == today.month) & \

                   (filtered_comp['Exit_DT'].dt.year == today.year) & (filtered_comp['Exit_DT'].dt.month == today.month)

        elif "本週" in time_frame: 

            start_week = today - timedelta(days=today.weekday())

            mask = (filtered_comp['Entry_DT'] >= start_week) & (filtered_comp['Exit_DT'] >= start_week)

        elif "3個月" in time_frame: 

            cutoff = today - timedelta(days=90)

            mask = (filtered_comp['Entry_DT'] >= cutoff) & (filtered_comp['Exit_DT'] >= cutoff)

        else: # 全部

            mask = [True] * len(filtered_comp)

        

        filtered_comp = filtered_comp[mask]



    f_pnl = filtered_comp['PnL_HKD'].sum() if not filtered_comp.empty else 0

    trade_count = len(filtered_comp)

    win_r = (len(filtered_comp[filtered_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0

    f_dur = filtered_comp['Duration_Days'].mean() if not filtered_comp.empty else 0

    

    if not filtered_comp.empty:

        wins = filtered_comp[filtered_comp['PnL_HKD'] > 0]

        losses = filtered_comp[filtered_comp['PnL_HKD'] <= 0]

        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0

        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0

        wr_dec = len(wins) / trade_count

        f_exp = (wr_dec * avg_win) - ((1-wr_dec) * avg_loss)

        f_exp_r = filtered_comp['Trade_R'].mean() if not filtered_comp.empty else 0

    else:

        f_exp, f_exp_r = 0, 0



    total_sl_risk_hkd = 0

    if active_pos:

        live_prices_for_risk = get_live_prices(list(active_pos.keys()))

        for s, d in active_pos.items():

            now = live_prices_for_risk.get(s)

            if now and d['last_sl'] > 0:

                total_sl_risk_hkd += get_hkd_value(s, (now - d['last_sl']) * d['qty'])



    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")

    m2.metric("期望值 (HKD / R)", f"${f_exp:,.0f} / {f_exp_r:.2f}R")

    m3.metric("總停損回撤 (Open Risk)", f"${total_sl_risk_hkd:,.2f}")

    m4.metric("平均持倉", f"{f_dur:.1f} 天")

    m5.metric("勝率 / 場數", f"{win_r:.1f}% ({trade_count})")



    if not equity_df.empty: st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線 (總體)", height=300), use_container_width=True)



    if not filtered_comp.empty:

        st.divider()

        st.subheader("🏆 週期成交排行榜")

        display_trades = filtered_comp.copy()

        display_trades['原始損益'] = display_trades.apply(lambda x: f"{get_currency_symbol(x['Symbol'])} {x['PnL_Raw']:,.2f}", axis=1)

        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: f"${x:,.2f}")

        display_trades['R 乘數'] =
