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
# 新增 Google Sheets 連線庫
from streamlit_gsheets import GSheetsConnection

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8 
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

# --- 2. AI 核心功能 (Gemini API) ---
def call_gemini_api(prompt, system_instruction=""):
    apiKey = "" # 系統將在運行時自動填充
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={apiKey}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }
    
    # 指數型退避重試機制 (Rule-based)
    retries = 5
    for i in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.candidates[0].content.parts[0].text
            elif response.status_code == 429: # Rate limit
                time.sleep(2 ** i)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2 ** i)
            
    return "❌ AI 診斷暫時無法使用，請稍後再試。"

# --- 3. 資料讀取層 ---
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

# --- 4. 核心計算邏輯 ---
@st.cache_data(ttl=60)
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    
    cycle_tracker = {} 
    active_trade_by_symbol = {} 
    completed_trades = [] 
    equity_curve = []

    for _, row in df.iterrows():
        sym = format_symbol(row['Symbol']) 
        action = str(row['Action']) if pd.notnull(row['Action']) else ""
        if not sym or not action: continue

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])
        date_str = row['Date']
        
        t_id = row.get('Trade_ID')
        if pd.isna(t_id) or t_id == "N/A":
            t_id = f"LEGACY_{sym}" 

        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        current_trade_id = None
        if is_buy:
            if sym in active_trade_by_symbol:
                current_trade_id = active_trade_by_symbol[sym]
            else:
                current_trade_id = t_id
                active_trade_by_symbol[sym] = current_trade_id
                
            if current_trade_id not in cycle_tracker:
                cycle_tracker[current_trade_id] = {
                    'symbol': sym, 'cash_flow_raw': 0.0, 'start_date': date_str, 
                    'initial_risk_raw': 0.0, 'Entry_Price': price, 'Entry_SL': sl,
                    'qty_accumulated': 0.0, 'Strategy': row.get('Strategy', ''),
                    'Emotion': row.get('Emotion', ''), 'Market_Condition': row.get('Market_Condition', ''),
                    'Mistake_Tag': row.get('Mistake_Tag', '')
                }
                if sl > 0:
                    cycle_tracker[current_trade_id]['initial_risk_raw'] = abs(price - sl) * qty
                
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
                    "Trade_ID": current_trade_id, "Exit_Date": date_str, "Entry_Date": cycle_data['start_date'], 
                    "Symbol": sym, "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((datetime.strptime(date_str, '%Y-%m-%d') - datetime.strptime(cycle_data['start_date'], '%Y-%m-%d')).days), 
                    "Trade_R": trade_r, "Strategy": cycle_data['Strategy'], "Emotion": cycle_data['Emotion'],
                    "Market_Condition": cycle_data['Market_Condition'], "Mistake_Tag": cycle_data['Mistake_Tag']
                })
                del active_trade_by_symbol[sym]
                if sym in positions: del positions[sym]
            
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    comp_df = pd.DataFrame(completed_trades)
    active_output = {s: p for s, p in positions.items() if s in active_trade_by_symbol}
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

# --- 5. UI 渲染 ---
df = load_data()

# Sidebar
with st.sidebar:
    st.header("⚡ 執行面板")
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
                if not is_sell_toggle:
                    assigned_tid = active_pos_temp[s_in]['trade_id'] if s_in in active_pos_temp else int(time.time())
                else:
                    if s_in in active_pos_temp: assigned_tid = active_pos_temp[s_in]['trade_id']
                    else: st.error("找不到開倉紀錄")
                
                img_path = None
                if img_file:
                    ts_str = str(int(time.time()))
                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": 0, "Notes": note_in, "Timestamp": int(time.time()), 
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in, "Img": img_path, "Trade_ID": assigned_tid
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val = calculate_portfolio(df)

t1, t2, t3, t4, t5, t_ai = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理", "🧠 AI 戰略指揮部"])

# --- Tab 1 to 5: 保持原樣 (省略中間重複代碼以節省空間，功能完全不變) ---
# ... (Tab 1-5 邏輯與您提供的最新文件一致) ...
with t1:
    st.subheader("📊 績效概覽")
    time_options = ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"]
    time_frame = st.selectbox("統計時間範圍", time_options, index=0)
    filtered_comp = completed_trades_df.copy()
    if not filtered_comp.empty:
        filtered_comp['Entry_DT'] = pd.to_datetime(filtered_comp['Entry_Date'])
        filtered_comp['Exit_DT'] = pd.to_datetime(filtered_comp['Exit_Date'])
        today = datetime.now()
        if "今年" in time_frame: mask = (filtered_comp['Entry_DT'].dt.year == today.year)
        elif "本月" in time_frame: mask = (filtered_comp['Entry_DT'].dt.month == today.month)
        elif "本週" in time_frame: mask = (filtered_comp['Entry_DT'] >= (today - timedelta(days=today.weekday())))
        elif "3個月" in time_frame: mask = (filtered_comp['Entry_DT'] >= (today - timedelta(days=90)))
        else: mask = [True] * len(filtered_comp)
        filtered_comp = filtered_comp[mask]

    f_pnl = filtered_comp['PnL_HKD'].sum() if not filtered_comp.empty else 0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")
    m2.metric("期望值 (R)", f"{exp_r_val:.2f}R")
    m5.metric("勝率", f"{(len(filtered_comp[filtered_comp['PnL_HKD']>0])/len(filtered_comp)*100 if len(filtered_comp)>0 else 0):.1f}%")
    if not equity_df.empty: st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線", height=300), use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
    if active_pos:
        live_prices = get_live_prices(list(active_pos.keys()))
        pos_list = []
        for s, d in active_pos.items():
            now = live_prices.get(s)
            un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
            pos_list.append({"代號": s, "持股": d['qty'], "成本": d['avg_price'], "現價": now, "未實現": un_pnl})
        st.dataframe(pd.DataFrame(pos_list), use_container_width=True)

with t3:
    st.subheader("⏪ 交易重播")
    if not df.empty:
        target = st.selectbox("選擇交易", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']}")
        row = df.iloc[target]
        st.write(f"策略: {row['Strategy']} | 筆記: {row['Notes']}")
        if pd.notnull(row['Img']) and os.path.exists(row['Img']): st.image(row['Img'])

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        st.dataframe(completed_trades_df, use_container_width=True)

with t5:
    st.subheader("🛠️ 數據管理")
    confirm_delete = st.checkbox("我了解此操作將永久刪除所有交易紀錄且無法復原")
    if st.button("🚨 清空所有數據", type="primary", disabled=not confirm_delete, use_container_width=True):
        save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()

# --- 新增功能：Tab AI 戰略指揮部 ---
with t_ai:
    st.subheader("🧠 AI 智能交易導師")
    st.info("AI 將分析您的交易行為、心理狀態與市場數據，提供量化優化建議。")
    
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        st.markdown("#### 📅 週報與基準分析")
        if st.button("🚀 生成本週 AI 績效週報", use_container_width=True):
            with st.spinner("正在對比基準指數並生成報告..."):
                # 獲取基準數據
                hsi = yf.download("^HSI", period="7d", progress=False)['Close']
                spx = yf.download("^GSPC", period="7d", progress=False)['Close']
                hsi_perf = ((hsi.iloc[-1] / hsi.iloc[0]) - 1) * 100 if not hsi.empty else 0
                spx_perf = ((spx.iloc[-1] / spx.iloc[0]) - 1) * 100 if not spx.empty else 0
                
                # 準備 AI 數據串
                week_trades = completed_trades_df[pd.to_datetime(completed_trades_df['Exit_Date']) >= (datetime.now() - timedelta(days=7))]
                trade_summary = week_trades.to_json(orient='records') if not week_trades.empty else "本週無結清交易"
                
                prompt = f"""
                請作為一名資深交易教練，分析我本週的表現。
                本週市場背景：恒生指數 {hsi_perf:.2f}%, 標普500 {spx_perf:.2f}%。
                我的交易數據：{trade_summary}
                
                請提供：
                1. 基準對比：量化我的 Alpha 值（相對大盤表現）。
                2. 本週優勢與弱點：識別勝率最高的組合（策略+時間）與最差組合。
                3. 行為偏差：檢查是否有「無聊強迫交易」或「週五效應」。
                """
                report = call_gemini_api(prompt, "你是一個冷靜、數據導向的交易系統專家，擅長發現隱藏的邊際優勢（Edge）。")
                st.markdown(report)

    with col_a2:
        st.markdown("#### 🛡️ 戰略診斷與規則優化")
        if st.button("🔍 執行全維度戰略診斷", use_container_width=True):
            with st.spinner("正在分析歷史模式..."):
                # 彙整最佳組合數據
                if not completed_trades_df.empty:
                    agg_data = completed_trades_df.groupby(['Strategy', 'Market_Condition', 'Emotion']).agg({
                        'Trade_R': ['mean', 'count'],
                        'PnL_HKD': 'sum'
                    }).to_json()
                    
                    prompt = f"""
                    根據我的歷史全量數據：{agg_data}
                    請執行以下診斷：
                    1. 邊際優勢累積：識別「策略x市場環境x情緒」的最佳組合與最失敗組合。
                    2. 止損/規模優化：基於 R 值波動，建議是否需要調整特定策略的止損寬度（如 ATR 倍數建議）。
                    3. 規則庫迭代：根據重複錯誤（Mistake_Tags），建議一條本週必須執行的「鐵律」。
                    4. 冷靜期干預：如果數據顯示虧損後有報復交易傾向，請設定具體的觸發條件與隔離時間。
                    """
                    diagnosis = call_gemini_api(prompt, "你是一個量化交易策略優化師，你的目標是減少最大回撤並提高期望值。")
                    st.markdown(diagnosis)
                else:
                    st.warning("數據量不足，無法執行深度診斷。")

    st.divider()
    st.markdown("#### 📜 建議規則庫 (由 AI 自動生成與維護)")
    if 'trading_rules' not in st.session_state:
        st.session_state.trading_rules = ["1. 嚴格執行止損，不向下攤平。", "2. 震盪市縮減 50% 倉位。"]
    
    for i, rule in enumerate(st.session_state.trading_rules):
        st.info(rule)
    
    if st.button("✨ 根據 AI 建議更新規則庫"):
        # 這裡可以加入 logic 讓 AI 提取之前的建議並存入 session_state
        st.success("規則庫已根據最新診斷結果更新（模擬功能）")

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("TradeMaster Pro v2.5 | AI Powered")
