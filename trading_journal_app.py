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

# --- 改進部分：資料讀取層 (支援 Google Sheets 與 CSV 雙模式) ---
def get_data_connection():
    # 嘗試建立 Google Sheets 連線，如果失敗則返回 None
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except:
        return None

def init_csv():
    # 如果使用 CSV 模式，確保檔案存在
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
            "Market_Condition", "Mistake_Tag" 
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
    
    # 優先嘗試從 Google Sheets 讀取
    try:
        if conn:
            df = conn.read(worksheet="Log", ttl=0) # ttl=0 確保不快取舊數據
        else:
            raise Exception("No connection")
    except:
        # 降級使用本地 CSV
        init_csv()
        try:
            df = pd.read_csv(FILE_NAME)
        except:
            return pd.DataFrame()

    if df.empty: return df
    
    # 數據類型轉換 (保持原邏輯)
    if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)
    if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
    for col in ["Market_Condition", "Mistake_Tag", "Img"]:
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
        # 優先嘗試儲存到 Google Sheets
        if conn:
            conn.update(worksheet="Log", data=df)
        else:
            raise Exception("No connection")
    except:
        # 降級儲存到本地 CSV
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

# --- 2. 核心計算邏輯 (改進：添加 Cache 以提升效能) ---
@st.cache_data(ttl=60) # 加入 Cache，當 df 沒變時不會重複運算
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    positions = {} 
    # 確保按時間排序
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
        
        # 讀取標籤屬性
        strategy = row.get('Strategy', '')
        emotion = row.get('Emotion', '')
        mkt_cond = row.get('Market_Condition', '')
        mistake = row.get('Mistake_Tag', '')
        
        if sym not in positions: 
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'first_sl': 0.0}
        
        if sym not in cycle_tracker:
            cycle_tracker[sym] = {
                'cash_flow_raw': 0.0, 
                'start_date': date_str, 
                'is_active': False, 
                'initial_risk_raw': 0.0,
                # Store tags at start of cycle
                'Strategy': strategy,
                'Emotion': emotion,
                'Market_Condition': mkt_cond,
                'Mistake_Tag': mistake
            }
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        # Cycle Start Logic
        if not cycle_tracker[sym]['is_active'] and is_buy and qty > 0:
            cycle_tracker[sym]['is_active'] = True
            cycle_tracker[sym]['start_date'] = date_str
            cycle_tracker[sym]['cash_flow_raw'] = 0.0
            
            # Store initial tags
            cycle_tracker[sym].update({
                'Strategy': strategy, 'Emotion': emotion, 
                'Market_Condition': mkt_cond, 'Mistake_Tag': mistake
            })

            # Initial Risk Calculation
            if sl > 0:
                init_risk = abs(price - sl) * qty
                cycle_tracker[sym]['initial_risk_raw'] = init_risk
                curr['first_sl'] = sl # Track first SL for active position metrics
            else:
                cycle_tracker[sym]['initial_risk_raw'] = 0.0
                curr['first_sl'] = 0.0

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
                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                
                completed_trades.append({
                    "Exit_Date": date_str, "Entry_Date": cycle_tracker[sym]['start_date'], "Symbol": sym, 
                    "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((d2 - d1).days), "Trade_R": trade_r,
                    "Strategy": cycle_tracker[sym]['Strategy'],
                    "Emotion": cycle_tracker[sym]['Emotion'],
                    "Market_Condition": cycle_tracker[sym]['Market_Condition'],
                    "Mistake_Tag": cycle_tracker[sym]['Mistake_Tag']
                })
                cycle_tracker[sym]['is_active'] = False
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    comp_df = pd.DataFrame(completed_trades)
    exp_hkd, exp_r, avg_dur = 0, 0, 0
    if not comp_df.empty:
        wins, losses = comp_df[comp_df['PnL_HKD'] > 0], comp_df[comp_df['PnL_HKD'] <= 0]
        wr = len(wins) / len(comp_df)
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        # Expectancy formula: (Win% * Avg_Win) - (Loss% * Avg_Loss)
        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)
        
        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]
        exp_r = valid_r_trades['Trade_R'].mean() if not valid_r_trades.empty else 0
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

# Sidebar: Trade Form
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
        
        # 5. Screenshot Upload
        img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'])
        
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in is not None and p_in is not None:
                img_path = None
                if img_file is not None:
                    # Save image locally (Note: Will be lost on Cloud restart)
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
                    "Img": img_path
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

# Pre-calculate main data
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    # 3. Timeline Filter
    time_options = ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"]
    time_frame = st.selectbox("統計時間範圍", time_options, index=0)
    
    # Filter Logic
    filtered_df = df.copy()
    if not filtered_df.empty:
        filtered_df['Date_DT'] = pd.to_datetime(filtered_df['Date'])
        today = datetime.now()
        if "今年" in time_frame: filtered_df = filtered_df[filtered_df['Date_DT'].dt.year == today.year]
        elif "本月" in time_frame: filtered_df = filtered_df[(filtered_df['Date_DT'].dt.year == today.year) & (filtered_df['Date_DT'].dt.month == today.month)]
        elif "本週" in time_frame: 
            start_week = today - timedelta(days=today.weekday())
            filtered_df = filtered_df[filtered_df['Date_DT'] >= start_week]
        elif "3個月" in time_frame: 
            filtered_df = filtered_df[filtered_df['Date_DT'] >= (today - timedelta(days=90))]
    
    # Recompute metrics for filtered period
    _, f_pnl, f_comp, f_eq, f_exp, f_exp_r, f_dur = calculate_portfolio(filtered_df)
    
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
    trade_count = len(f_comp)
    win_r = (len(f_comp[f_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0
    m5.metric("勝率 / 場數", f"{win_r:.1f}% ({trade_count})")

    if not f_eq.empty: st.plotly_chart(px.area(f_eq, x="Date", y="Cumulative PnL", title=f"累計損益曲線 ({time_frame})", height=300), use_container_width=True)

    if not f_comp.empty:
        st.divider()
        st.subheader("🏆 交易排行榜")
        display_trades = f_comp.copy()
        display_trades['原始損益'] = display_trades.apply(lambda x: f"{get_currency_symbol(x['Symbol'])} {x['PnL_Raw']:,.2f}", axis=1)
        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: f"${x:,.2f}")
        display_trades['R 乘數'] = display_trades['Trade_R'].apply(lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A")
        display_trades = display_trades.rename(columns={"Exit_Date": "出場日期", "Symbol": "代號"})
        
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
        
        un_pnl = (now - avg_p) * qty if now else 0
        roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
        sl_risk_raw = (now - last_sl) * qty if (now and last_sl > 0) else 0
        
        # 4. New Metrics: Initial Risk, Current Risk, Current R
        init_risk = abs(avg_p - first_sl) * qty if first_sl > 0 else 0
        curr_risk = sl_risk_raw
        curr_r = (un_pnl / init_risk) if (now and init_risk > 0) else 0
        
        processed_p_data.append({
            "代號": s, "持股數": f"{qty:,.0f}", "平均成本": f"{avg_p:,.2f}", 
            "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{last_sl:,.2f}", 
            "初始風險": f"{init_risk:,.2f}",
            "當前風險": f"{curr_risk:,.2f}",
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
            
            # 5. Show Screenshot
            if pd.notnull(row['Img']) and os.path.exists(row['Img']):
                st.image(row['Img'], caption="交易當下截圖")

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        # 1. Bar Charts for Psychology
        c1, c2 = st.columns(2)
        valid_r = completed_trades_df[completed_trades_df['Trade_R'].notna()]
        
        with c1:
            mistake_r = valid_r[valid_r['Mistake_Tag'] != "None"].groupby('Mistake_Tag')['Trade_R'].mean().reset_index()
            if not mistake_r.empty:
                fig_m = px.bar(mistake_r, x='Mistake_Tag', y='Trade_R', title="平均 R 乘數 (按錯誤標籤)", color='Trade_R', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_m, use_container_width=True)
        
        with c2:
            emo_r = valid_r.groupby('Emotion')['Trade_R'].mean().reset_index()
            if not emo_r.empty:
                fig_e = px.bar(emo_r, x='Emotion', y='Trade_R', title="平均 R 乘數 (按情緒)", color='Trade_R', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_e, use_container_width=True)

        # 2. Breakdowns Table
        st.markdown("### 🔍 多維度績效分析")
        with st.expander("查看詳細分類統計", expanded=False):
            group_by = st.selectbox("分組依據", ["Strategy", "Market_Condition", "Mistake_Tag", "Emotion"])
            if group_by:
                agg_df = completed_trades_df.groupby(group_by).agg(
                    Count=('Symbol', 'count'),
                    Win_Rate=('PnL_HKD', lambda x: (x > 0).mean() * 100),
                    Avg_R=('Trade_R', 'mean'),
                    Avg_HKD=('PnL_HKD', 'mean'),
                    Gross_Win=('PnL_HKD', lambda x: x[x > 0].sum()),
                    Gross_Loss=('PnL_HKD', lambda x: abs(x[x <= 0].sum()))
                ).reset_index()
                agg_df['Profit Factor'] = agg_df['Gross_Win'] / agg_df['Gross_Loss'].replace(0, 1)
                
                # Formatting
                agg_df['Win_Rate'] = agg_df['Win_Rate'].map('{:.1f}%'.format)
                agg_df['Avg_R'] = agg_df['Avg_R'].map('{:.2f}R'.format)
                agg_df['Avg_HKD'] = agg_df['Avg_HKD'].map('${:,.0f}'.format)
                agg_df['Profit Factor'] = agg_df['Profit Factor'].map('{:.2f}'.format)
                
                st.dataframe(agg_df[[group_by, 'Count', 'Win_Rate', 'Avg_R', 'Avg_HKD', 'Profit Factor']], hide_index=True, use_container_width=True)

    if not df.empty:
        st.divider()
        hist_df = df.sort_values("Timestamp", ascending=False).copy()
        hist_df = hist_df.rename(columns={"Stop_Loss": "執行止損", "Price": "成交價", "Quantity": "股數"})
        
        # 5. Add Image Link Column
        hist_df['截圖'] = hist_df['Img'].apply(lambda x: "🖼️" if pd.notnull(x) and os.path.exists(x) else "")
        
        cols = ["Date", "Symbol", "Action", "Strategy", "成交價", "股數", "執行止損", "Emotion", "Mistake_Tag", "截圖"]
        st.dataframe(hist_df[cols], use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    
    # 提示目前的數據來源狀態
    conn_status = get_data_connection()
    if conn_status:
        st.success("🟢 已連接至 Google Sheets (雲端同步中)")
    else:
        st.warning("🟠 目前使用本地 CSV 模式 (雲端部署時數據將無法永久保存，請配置 secrets)")

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
