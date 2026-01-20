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
        # Change 1: Added Trade_ID to schema
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

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return "HK$"
    return "$"

# --- 2. 核心計算邏輯 (Refactored for Change 2 & 3) ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    # Sort to ensure sequential processing
    df = df.sort_values(by="Timestamp")
    
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    
    # Change 2: Key is Trade_ID, not Symbol
    cycle_tracker = {} 
    # Helper to find active trade ID for a symbol {Symbol: Trade_ID}
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
        row_trade_id = row.get('Trade_ID')
        # Check for valid ID in row
        if pd.isna(row_trade_id) or row_trade_id == "": row_trade_id = None

        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        # Determine Trade ID
        current_trade_id = None
        
        if is_buy:
            if sym in active_trade_map:
                # Adding to existing active position
                current_trade_id = active_trade_map[sym]
            else:
                # New cycle start
                if row_trade_id:
                    current_trade_id = row_trade_id
                else:
                    # Generate unique ID for calculation if missing in legacy data
                    current_trade_id = f"gen_{sym}_{ts}"
                active_trade_map[sym] = current_trade_id
        
        elif is_sell:
            if sym in active_trade_map:
                current_trade_id = active_trade_map[sym]
            else:
                continue # Skip orphan sell

        # Initialize cycle in tracker
        if current_trade_id not in cycle_tracker:
            cycle_tracker[current_trade_id] = {
                'Symbol': sym,
                'cash_flow_raw': 0.0,
                'start_date': date_str,
                'is_active': True,
                'qty': 0.0,
                'avg_price': 0.0,
                'last_sl': 0.0,
                # Change 3: Explicit Entry stats
                'Entry_Price': price, 
                'Entry_SL': sl,
                'initial_risk_raw': 0.0,
                # Tags
                'Strategy': row.get('Strategy', ''),
                'Emotion': row.get('Emotion', ''),
                'Market_Condition': row.get('Market_Condition', ''),
                'Mistake_Tag': row.get('Mistake_Tag', '')
            }
            
            # Calculate Initial Risk based on FIRST Buy
            if sl > 0:
                cycle_tracker[current_trade_id]['initial_risk_raw'] = abs(price - sl) * qty
            
        cycle = cycle_tracker[current_trade_id]
        
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
            
            realized_pnl_raw = (price - cycle['avg_price']) * sell_qty
            realized_pnl_hkd_item = get_hkd_value(sym, realized_pnl_raw)
            total_realized_pnl_hkd += realized_pnl_hkd_item
            running_pnl_hkd += realized_pnl_hkd_item
            
            cycle['qty'] -= sell_qty
            
            # Check if cycle closed
            if cycle['qty'] < 0.0001:
                d1 = datetime.strptime(cycle['start_date'], '%Y-%m-%d')
                d2 = datetime.strptime(date_str, '%Y-%m-%d')
                
                pnl_raw = cycle['cash_flow_raw']
                init_risk = cycle['initial_risk_raw']
                
                # Trade_R based on Initial Risk
                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                
                completed_trades.append({
                    "Exit_Date": date_str, 
                    "Entry_Date": cycle['start_date'], 
                    "Symbol": sym, 
                    "PnL_Raw": pnl_raw, 
                    "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((d2 - d1).days), 
                    "Trade_R": trade_r,
                    "Strategy": cycle['Strategy'],
                    "Emotion": cycle['Emotion'],
                    "Market_Condition": cycle['Market_Condition'],
                    "Mistake_Tag": cycle['Mistake_Tag']
                })
                
                cycle['is_active'] = False
                if sym in active_trade_map:
                    del active_trade_map[sym]
            
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    # Prepare active positions for UI
    active_positions = {}
    for tid, c in cycle_tracker.items():
        if c['is_active'] and c['qty'] > 0.0001:
            active_positions[c['Symbol']] = {
                'qty': c['qty'],
                'avg_price': c['avg_price'],
                'last_sl': c['last_sl'],
                'first_sl': c['Entry_SL'],
                'first_price': c['Entry_Price'],
                'Trade_ID': tid
            }

    # Aggregate metrics based on completed trades (ALL TIME)
    # We return the full DF so Tab 1 can filter it later
    comp_df = pd.DataFrame(completed_trades)
    
    # These global metrics are just placeholders, actual display uses filtered data in Tab 1
    exp_hkd, exp_r, avg_dur = 0, 0, 0
    if not comp_df.empty:
        valid_r = comp_df[comp_df['Trade_R'].notna()]
        exp_r = valid_r['Trade_R'].mean() if not valid_r.empty else 0
        avg_dur = comp_df['Duration_Days'].mean()

    return active_positions, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur

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
# Run calculation on FULL dataset first
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, _, _, _ = calculate_portfolio(df)

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
        emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜")
        st_in = st.selectbox("策略 (Strategy)", ["Pullback", "Breakout", "➕ 新增..."])
        if st_in == "➕ 新增...": st_in = st.text_input("輸入新策略名稱")
        note_in = st.text_area("決策筆記")
        
        # 5. Screenshot Upload
        img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'])
        
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in is not None and p_in is not None:
                img_path = None
                if img_file is not None:
                    ts_str = str(int(time.time()))
                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                
                # Change 1: Assign Trade_ID logic
                trade_id_to_save = None
                if not is_sell:
                    # If Buy, check if existing active cycle
                    if s_in in active_pos:
                        trade_id_to_save = active_pos[s_in]['Trade_ID']
                    else:
                        trade_id_to_save = str(int(time.time())) # Generate new ID
                else:
                    # If Sell, attach to active cycle if exists
                    if s_in in active_pos:
                        trade_id_to_save = active_pos[s_in]['Trade_ID']
                
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": 0, 
                    "Notes": note_in, "Timestamp": int(time.time()), 
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in,
                    "Img": img_path, "Trade_ID": trade_id_to_save
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_options = ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"]
    time_frame = st.selectbox("統計時間範圍", time_options, index=0)
    
    # Change 4: Filter completed trades properly (Entry AND Exit within range)
    f_comp = completed_trades_df.copy()
    if not f_comp.empty and time_frame != "全部記錄":
        f_comp['Entry_DT'] = pd.to_datetime(f_comp['Entry_Date'])
        f_comp['Exit_DT'] = pd.to_datetime(f_comp['Exit_Date'])
        today = datetime.now()
        
        start_date = datetime(1900, 1, 1) # Default
        if "今年" in time_frame: start_date = datetime(today.year, 1, 1)
        elif "本月" in time_frame: start_date = datetime(today.year, today.month, 1)
        elif "本週" in time_frame: start_date = today - timedelta(days=today.weekday())
        elif "3個月" in time_frame: start_date = today - timedelta(days=90)
        
        # Strict filter: Trade must start AND end within period
        f_comp = f_comp[(f_comp['Entry_DT'] >= start_date) & (f_comp['Exit_DT'] >= start_date)]

    # Recalculate Metrics on Filtered Data
    f_pnl = f_comp['PnL_HKD'].sum() if not f_comp.empty else 0
    f_dur = f_comp['Duration_Days'].mean() if not f_comp.empty else 0
    
    f_exp_hkd = 0
    f_exp_r = 0
    if not f_comp.empty:
        wins, losses = f_comp[f_comp['PnL_HKD'] > 0], f_comp[f_comp['PnL_HKD'] <= 0]
        wr = len(wins) / len(f_comp)
        avg_w = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_l = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        f_exp_hkd = (wr * avg_w) - ((1-wr) * avg_l)
        
        valid_r = f_comp[f_comp['Trade_R'].notna()]
        f_exp_r = valid_r['Trade_R'].mean() if not valid_r.empty else 0

    # Risk (Live)
    total_sl_risk_hkd = 0
    if active_pos:
        live_prices_for_risk = get_live_prices(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = live_prices_for_risk.get(s)
            if now and d['last_sl'] > 0:
                total_sl_risk_hkd += get_hkd_value(s, (now - d['last_sl']) * d['qty'])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")
    m2.metric("期望值 (HKD / R)", f"${f_exp_hkd:,.0f} / {f_exp_r:.2f}R")
    m3.metric("總停損回撤 (Open Risk)", f"${total_sl_risk_hkd:,.2f}")
    m4.metric("平均持倉", f"{f_dur:.1f} 天")
    cnt = len(f_comp)
    wr_display = (len(f_comp[f_comp['PnL_HKD'] > 0]) / cnt * 100) if cnt > 0 else 0
    m5.metric("勝率 / 場數", f"{wr_display:.1f}% ({cnt})")

    # Equity Curve (Show All Time for context, or Filtered?) Usually All Time is better context
    if not equity_df.empty: st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線 (全歷史)", height=300), use_container_width=True)

    if not f_comp.empty:
        st.divider()
        st.subheader("🏆 交易排行榜 (篩選範圍內)")
        display_trades = f_comp.copy()
        display_trades['原始損益'] = display_trades.apply(lambda x: f"{get_currency_symbol(x['Symbol'])} {x['PnL_Raw']:,.2f}", axis=1)
        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: f"${x:,.2f}")
        display_trades['R 乘數'] = display_trades['Trade_R'].apply(lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A")
        display_trades = display_trades.rename(columns={"Exit_Date": "出場日期", "Symbol": "代號", "Duration_Days": "持有天數"})
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### 🟢 Top 獲利")
            st.dataframe(display_trades.sort_values(by="PnL_HKD", ascending=False).head(5)[['出場日期', '代號', '原始損益', 'HKD 損益', 'R 乘數']], hide_index=True, use_container_width=True)
        with r2:
            st.markdown("##### 🔴 Top 虧損")
            st.dataframe(display_trades.sort_values(by="PnL_HKD", ascending=True).head(5)[['出場日期', '代號', '原始損益', 'HKD 損益', 'R 乘數']], hide_index=True, use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
    current_symbols = list(active_pos.keys())
    live_prices = get_live_prices(current_symbols)
    processed_p_data = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        qty, avg_p, last_sl, first_sl = d['qty'], d['avg_price'], d['last_sl'], d.get('first_sl', 0)
        first_price = d.get('first_price', avg_p)
        
        un_pnl = (now - avg_p) * qty if now else 0
        roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
        sl_risk_raw = (now - last_sl) * qty if (now and last_sl > 0) else 0
        
        # Change 4: Position-level risk metrics
        init_risk = abs(first_price - first_sl) * qty if first_sl > 0 else 0
        curr_r = (un_pnl / init_risk) if (now and init_risk > 0) else 0
        
        processed_p_data.append({
            "代號": s, "持股數": f"{qty:,.0f}", "平均成本": f"{avg_p:,.2f}", 
            "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{last_sl:,.2f}", 
            "初始風險": f"{init_risk:,.2f}",
            "當前風險": f"{sl_risk_raw:,.2f}",
            "當前R": f"{curr_r:.2f}R",
            "未實現損益": f"{un_pnl:,.2f}", "報酬%": roi
        })
    if processed_p_data: 
        st.dataframe(pd.DataFrame(processed_p_data), column_config={"報酬%": st.column_config.ProgressColumn("報酬%", format="%.2f%%", min_value=-20, max_value=20, color="green" if 0>=0 else "red")}, hide_index=True, use_container_width=True)
        if st.button("🔄 刷新即時報價", use_container_width=True): st.cache_data.clear(); st.rerun()
    else: st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 交易重播")
    if not df.empty:
        target = st.selectbox("選擇交易", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']}")
        row = df.iloc[target]
        data = yf.download(row['Symbol'], start=(pd.to_datetime(row['Date']) - timedelta(days=20)).strftime('%Y-%m-%d'), progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='價格')])
            fig.add_trace(go.Scatter(x=[pd.to_datetime(row['Date'])], y=[row['Price']], mode='markers+text', marker=dict(size=15, color='orange', symbol='star'), text=["執行"], textposition="top center"))
            fig.update_layout(title=f"{row['Symbol']} K線圖回顧", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            if pd.notnull(row['Img']) and os.path.exists(row['Img']): st.image(row['Img'], caption="交易當下截圖")

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        c1, c2 = st.columns(2)
        valid_r = completed_trades_df[completed_trades_df['Trade_R'].notna()]
        with c1:
            mistake_r = valid_r[valid_r['Mistake_Tag'] != "None"].groupby('Mistake_Tag')['Trade_R'].mean().reset_index()
            if not mistake_r.empty: st.plotly_chart(px.bar(mistake_r, x='Mistake_Tag', y='Trade_R', title="平均 R 乘數 (按錯誤)", color='Trade_R'), use_container_width=True)
        with c2:
            emo_r = valid_r.groupby('Emotion')['Trade_R'].mean().reset_index()
            if not emo_r.empty: st.plotly_chart(px.bar(emo_r, x='Emotion', y='Trade_R', title="平均 R 乘數 (按情緒)", color='Trade_R'), use_container_width=True)

        st.markdown("### 🔍 多維度績效分析")
        with st.expander("查看詳細分類統計", expanded=False):
            group_by = st.selectbox("分組依據", ["Strategy", "Market_Condition", "Mistake_Tag", "Emotion"])
            if group_by:
                agg_df = completed_trades_df.groupby(group_by).agg(
                    Count=('Symbol', 'count'), Win_Rate=('PnL_HKD', lambda x: (x > 0).mean() * 100),
                    Avg_R=('Trade_R', 'mean'), Avg_HKD=('PnL_HKD', 'mean'),
                    Gross_Win=('PnL_HKD', lambda x: x[x > 0].sum()), Gross_Loss=('PnL_HKD', lambda x: abs(x[x <= 0].sum()))
                ).reset_index()
                agg_df['Profit Factor'] = agg_df['Gross_Win'] / agg_df['Gross_Loss'].replace(0, 1)
                st.dataframe(agg_df, hide_index=True, use_container_width=True)

    if not df.empty:
        st.divider()
        hist_df = df.sort_values("Timestamp", ascending=False).copy()
        hist_df['截圖'] = hist_df['Img'].apply(lambda x: "🖼️" if pd.notnull(x) and os.path.exists(x) else "")
        cols = ["Date", "Symbol", "Action", "Strategy", "Price", "Quantity", "Stop_Loss", "Emotion", "Mistake_Tag", "截圖"]
        st.dataframe(hist_df[cols], use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_file = st.file_uploader("📤 批量上傳 CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file and st.button("🚀 開始匯入"):
            try:
                new_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Symbol' in new_data.columns: new_data['Symbol'] = new_data['Symbol'].apply(format_symbol)
                if 'Timestamp' not in new_data.columns: new_data['Timestamp'] = int(time.time())
                df = pd.concat([df, new_data], ignore_index=True); save_all_data(df)
                st.success("匯入成功！"); st.rerun()
            except Exception as e: st.error(f"匯入失敗: {e}")
    
    if not df.empty:
        st.divider()
        selected_idx = st.selectbox("選擇紀錄進行編輯", df.index, format_func=lambda x: f"[{df.loc[x, 'Date']}] {df.loc[x, 'Symbol']} ({df.loc[x, 'Action']})")
        t_edit = df.loc[selected_idx]
        e1, e2, e3 = st.columns(3)
        n_p = e1.number_input("編輯價格", value=float(t_edit['Price']), key=f"ep_{selected_idx}")
        n_q = e2.number_input("編輯股數", value=float(t_edit['Quantity']), key=f"eq_{selected_idx}")
        n_sl = e3.number_input("編輯止損價", value=float(t_edit['Stop_Loss']), key=f"esl_{selected_idx}")
        b1, b2 = st.columns(2)
        if b1.button("💾 儲存修改", use_container_width=True):
            df.loc[selected_idx, ['Price', 'Quantity', 'Stop_Loss']] = [n_p, n_q, n_sl]
            save_all_data(df); st.success("已更新"); st.rerun()
        if b2.button("🗑️ 刪除此筆紀錄", use_container_width=True):
            df = df.drop(selected_idx).reset_index(drop=True)
            save_all_data(df); st.rerun()

    if st.button("🚨 清空所有數據"):
        save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()
