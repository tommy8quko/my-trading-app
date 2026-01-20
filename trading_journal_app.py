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

# --- 2. 核心計算邏輯 (Enhanced Metrics) ---
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

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])
        date_str = row['Date']
        
        strategy = row.get('Strategy', '')
        emotion = row.get('Emotion', '')
        mkt_cond = row.get('Market_Condition', '')
        mistake = row.get('Mistake_Tag', '')
        
        if sym not in positions: 
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'first_sl': 0.0}
        
        if sym not in cycle_tracker:
            cycle_tracker[sym] = {
                'cash_flow_raw': 0.0, 'start_date': date_str, 'is_active': False, 
                'initial_risk_raw': 0.0, 'Strategy': strategy, 'Emotion': emotion,
                'Market_Condition': mkt_cond, 'Mistake_Tag': mistake
            }
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        if not cycle_tracker[sym]['is_active'] and is_buy and qty > 0:
            cycle_tracker[sym]['is_active'] = True
            cycle_tracker[sym]['start_date'] = date_str
            cycle_tracker[sym]['cash_flow_raw'] = 0.0
            cycle_tracker[sym].update({'Strategy': strategy, 'Emotion': emotion, 'Market_Condition': mkt_cond, 'Mistake_Tag': mistake})
            if sl > 0:
                cycle_tracker[sym]['initial_risk_raw'] = abs(price - sl) * qty
                curr['first_sl'] = sl
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
                completed_trades.append({
                    "Exit_Date": date_str, "Entry_Date": cycle_tracker[sym]['start_date'], "Symbol": sym, 
                    "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((d2 - d1).days), "Trade_R": (pnl_raw / init_risk) if init_risk > 0 else None,
                    "Strategy": cycle_tracker[sym]['Strategy'], "Emotion": cycle_tracker[sym]['Emotion'],
                    "Market_Condition": cycle_tracker[sym]['Market_Condition'], "Mistake_Tag": cycle_tracker[sym]['Mistake_Tag']
                })
                cycle_tracker[sym]['is_active'] = False
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    comp_df = pd.DataFrame(completed_trades)
    exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown = 0, 0, 0, 0, 0
    
    if not comp_df.empty:
        wins, losses = comp_df[comp_df['PnL_HKD'] > 0], comp_df[comp_df['PnL_HKD'] <= 0]
        wr = len(wins) / len(comp_df)
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 1
        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]
        exp_r = valid_r_trades['Trade_R'].mean() if not valid_r_trades.empty else 0
        avg_dur = comp_df['Duration_Days'].mean()
        
        # Max Drawdown calculation
        eq_series = pd.Series([e['Cumulative PnL'] for e in equity_curve])
        if not eq_series.empty:
            rolling_max = eq_series.cummax()
            drawdown = eq_series - rolling_max
            max_drawdown = drawdown.min()

    return {k: v for k, v in positions.items() if v['qty'] > 0.0001}, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown

@st.cache_data(ttl=300) # 快取 5 分鐘
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        # 優化：Batch 調用 yfinance，減少請求次數
        # 使用 period='1d' 並獲取最近的價格
        data = yf.download(symbols_list, period="5d", interval="1d", progress=False, group_by='ticker')
        prices = {}
        for s in symbols_list:
            try:
                if len(symbols_list) > 1:
                    ticker_data = data[s]['Close'].dropna()
                else:
                    ticker_data = data['Close'].dropna()
                prices[s] = float(ticker_data.iloc[-1])
            except Exception:
                prices[s] = None
        return prices
    except Exception as e:
        # 錯誤緩衝處理
        st.warning(f"無法獲取部分即時報價: {e}")
        return {}

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
        img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'])
        
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in is not None and p_in is not None:
                img_path = None
                if img_file is not None:
                    ts_str = str(int(time.time()))
                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": 0, "Notes": note_in, "Timestamp": int(time.time()), 
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in, "Img": img_path
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

# 計算主要數據
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val, pl_ratio, mdd = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_options = ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"]
    time_frame = st.selectbox("統計時間範圍", time_options, index=0)
    
    filtered_df = df.copy()
    if not filtered_df.empty:
        filtered_df['Date_DT'] = pd.to_datetime(filtered_df['Date'])
        today = datetime.now()
        if "今年" in time_frame: filtered_df = filtered_df[filtered_df['Date_DT'].dt.year == today.year]
        elif "本月" in time_frame: filtered_df = filtered_df[(filtered_df['Date_DT'].dt.year == today.year) & (filtered_df['Date_DT'].dt.month == today.month)]
        elif "本週" in time_frame: 
            start_week = today - timedelta(days=today.weekday()); filtered_df = filtered_df[filtered_df['Date_DT'] >= start_week]
        elif "3個月" in time_frame: filtered_df = filtered_df[filtered_df['Date_DT'] >= (today - timedelta(days=90))]
    
    _, f_pnl, f_comp, f_eq, f_exp, f_exp_r, f_dur, f_pl, f_mdd = calculate_portfolio(filtered_df)
    
    # 指標列：使用 st.metric 的 delta 功能
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.0f}", delta=f"{f_pnl:,.0f}" if f_pnl != 0 else None)
    m2.metric("期望值 (R)", f"{f_exp_r:.2f}R", delta=f"{f_exp:,.0f} HKD")
    m3.metric("盈虧比 (P/L Ratio)", f"{f_pl:.2f}")
    m4.metric("最大回撤 (MDD)", f"${f_mdd:,.0f}", delta_color="inverse")
    trade_count = len(f_comp)
    win_r = (len(f_comp[f_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0
    m5.metric("勝率", f"{win_r:.1f}%", delta=f"共 {trade_count} 場")

    if not f_eq.empty: 
        st.plotly_chart(px.area(f_eq, x="Date", y="Cumulative PnL", title=f"累計損益曲線 ({time_frame})", height=300, color_discrete_sequence=['#00CC96']), use_container_width=True)

    # 星期表現分析 (New Metric)
    if not f_comp.empty:
        st.divider()
        st.subheader("🗓️ 時段分析 (星期表現)")
        f_comp['Weekday'] = pd.to_datetime(f_comp['Exit_Date']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekday_pnl = f_comp.groupby('Weekday')['PnL_HKD'].sum().reindex(day_order).fillna(0).reset_index()
        fig_day = px.bar(weekday_pnl, x='Weekday', y='PnL_HKD', color='PnL_HKD', 
                         color_continuous_scale='RdYlGn', title="各星期總損益")
        st.plotly_chart(fig_day, use_container_width=True)

with t2:
    st.markdown("### 🟢 當前持倉 & 即時監控")
    current_symbols = list(active_pos.keys())
    live_prices = get_live_prices(current_symbols)
    processed_p_data = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        qty, avg_p, last_sl, first_sl = d['qty'], d['avg_price'], d['last_sl'], d.get('first_sl', 0)
        un_pnl = (now - avg_p) * qty if now else 0
        roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
        init_risk = abs(avg_p - first_sl) * qty if first_sl > 0 else 0
        curr_r = (un_pnl / init_risk) if (now and init_risk > 0) else 0
        
        processed_p_data.append({
            "代號": s, "持股": qty, "成本": avg_p, "現價": now, "止損": last_sl, 
            "當前R": curr_r, "未實現損益": un_pnl, "報酬%": roi
        })
    
    if processed_p_data: 
        pos_df = pd.DataFrame(processed_p_data)
        # UI/UX: 條件格式化持倉列表
        st.dataframe(
            pos_df,
            column_config={
                "報酬%": st.column_config.NumberColumn("報酬%", format="%.2f%%"),
                "未實現損益": st.column_config.NumberColumn("損益", format="$%.2f"),
                "現價": st.column_config.NumberColumn("現價", format="%.2f"),
                "報酬%": st.column_config.ProgressColumn("表現", min_value=-15, max_value=15, format="%.2f%%")
            },
            hide_index=True, use_container_width=True
        )
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
            fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            if pd.notnull(row['Img']) and os.path.exists(row['Img']): st.image(row['Img'], caption="交易當下截圖")

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        c1, c2 = st.columns(2)
        valid_r = completed_trades_df[completed_trades_df['Trade_R'].notna()]
        with c1:
            mistake_r = valid_r[valid_r['Mistake_Tag'] != "None"].groupby('Mistake_Tag')['Trade_R'].mean().reset_index()
            if not mistake_r.empty:
                st.plotly_chart(px.bar(mistake_r, x='Mistake_Tag', y='Trade_R', title="平均 R (按錯誤)", color='Trade_R', color_continuous_scale='RdYlGn'), use_container_width=True)
        with c2:
            emo_r = valid_r.groupby('Emotion')['Trade_R'].mean().reset_index()
            if not emo_r.empty:
                st.plotly_chart(px.bar(emo_r, x='Emotion', y='Trade_R', title="平均 R (按情緒)", color='Trade_R', color_continuous_scale='RdYlGn'), use_container_width=True)

with t5:
    st.subheader("🛠️ 數據管理")
    if not df.empty:
        st.download_button("📥 導出交易日誌 CSV", df.to_csv(index=False), "my_trades.csv", "text/csv", use_container_width=True)
        if st.button("🚨 清空所有數據", use_container_width=True): save_all_data(pd.DataFrame(columns=df.columns)); st.rerun()
