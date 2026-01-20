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

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

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

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return "HK$"
    return "$"

# --- 2. 核心計算邏輯 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0
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
        row_trade_id = row.get('Trade_ID')
        if pd.isna(row_trade_id) or row_trade_id == "": row_trade_id = None
        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        current_trade_id = None
        if is_buy:
            if sym in active_trade_map: current_trade_id = active_trade_map[sym]
            else:
                current_trade_id = row_trade_id if row_trade_id else f"gen_{sym}_{ts}"
                active_trade_map[sym] = current_trade_id
        elif is_sell:
            if sym in active_trade_map: current_trade_id = active_trade_map[sym]
            else: continue

        if current_trade_id not in cycle_tracker:
            cycle_tracker[current_trade_id] = {
                'Symbol': sym, 'cash_flow_raw': 0.0, 'start_date': date_str, 'is_active': True,
                'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'Entry_Price': price, 'Entry_SL': sl,
                'initial_risk_raw': abs(price - sl) * qty if sl > 0 else 0.0,
                'Strategy': row.get('Strategy', ''), 'Emotion': row.get('Emotion', ''),
                'Market_Condition': row.get('Market_Condition', ''), 'Mistake_Tag': row.get('Mistake_Tag', '')
            }
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
            total_realized_pnl_hkd += get_hkd_value(sym, (price - cycle['avg_price']) * sell_qty)
            running_pnl_hkd += get_hkd_value(sym, (price - cycle['avg_price']) * sell_qty)
            cycle['qty'] -= sell_qty
            if cycle['qty'] < 0.0001:
                pnl_raw = cycle['cash_flow_raw']
                init_risk = cycle['initial_risk_raw']
                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                completed_trades.append({
                    "Exit_Date": date_str, "Entry_Date": cycle['start_date'], "Symbol": sym, 
                    "PnL_Raw": pnl_raw, "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": float((datetime.strptime(date_str, '%Y-%m-%d') - datetime.strptime(cycle['start_date'], '%Y-%m-%d')).days), 
                    "Trade_R": trade_r, "Strategy": cycle['Strategy'], "Emotion": cycle['Emotion'],
                    "Market_Condition": cycle['Market_Condition'], "Mistake_Tag": cycle['Mistake_Tag']
                })
                cycle['is_active'] = False
                if sym in active_trade_map: del active_trade_map[sym]
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})

    active_positions = {c['Symbol']: {'qty': c['qty'], 'avg_price': c['avg_price'], 'last_sl': c['last_sl'], 'first_sl': c['Entry_SL'], 'first_price': c['Entry_Price'], 'Trade_ID': tid} for tid, c in cycle_tracker.items() if c['is_active'] and c['qty'] > 0.0001}
    return active_positions, total_realized_pnl_hkd, pd.DataFrame(completed_trades), pd.DataFrame(equity_curve), 0, 0, 0

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
        img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'])
        if st.form_submit_button("儲存執行紀錄"):
            if s_in and q_in is not None and p_in is not None:
                img_path = None
                if img_file is not None:
                    img_path = os.path.join("images", f"{str(int(time.time()))}_{img_file.name}")
                    with open(img_path, "wb") as f: f.write(img_file.getbuffer())
                trade_id_to_save = active_pos[s_in]['Trade_ID'] if s_in in active_pos else (str(int(time.time())) if not is_sell else None)
                save_transaction({"Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, "Emotion": emo_in, "Risk_Reward": 0, "Notes": note_in, "Timestamp": int(time.time()), "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in, "Img": img_path, "Trade_ID": trade_id_to_save})
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史分析", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_frame = st.selectbox("統計時間範圍", ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"], index=0)
    f_comp = completed_trades_df.copy()
    if not f_comp.empty and time_frame != "全部記錄":
        today = datetime.now()
        start_date = datetime(1900, 1, 1)
        if "今年" in time_frame: start_date = datetime(today.year, 1, 1)
        elif "本月" in time_frame: start_date = datetime(today.year, today.month, 1)
        elif "本週" in time_frame: start_date = today - timedelta(days=today.weekday())
        elif "3個月" in time_frame: start_date = today - timedelta(days=90)
        f_comp = f_comp[(pd.to_datetime(f_comp['Entry_Date']) >= start_date) & (pd.to_datetime(f_comp['Exit_Date']) >= start_date)]
    
    f_pnl = f_comp['PnL_HKD'].sum() if not f_comp.empty else 0
    f_dur = f_comp['Duration_Days'].mean() if not f_comp.empty else 0
    f_exp_hkd, f_exp_r = 0, 0
    if not f_comp.empty:
        wins, losses = f_comp[f_comp['PnL_HKD'] > 0], f_comp[f_comp['PnL_HKD'] <= 0]
        wr = len(wins) / len(f_comp)
        f_exp_hkd = (wr * (wins['PnL_HKD'].mean() if not wins.empty else 0)) - ((1-wr) * (abs(losses['PnL_HKD'].mean()) if not losses.empty else 0))
        f_exp_r = f_comp['Trade_R'].mean() if not f_comp.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")
    m2.metric("期望值 (HKD / R)", f"${f_exp_hkd:,.0f} / {f_exp_r:.2f}R")
    m4.metric("平均持倉", f"{f_dur:.1f} 天")
    cnt = len(f_comp)
    m5.metric("勝率 / 場數", f"{(len(f_comp[f_comp['PnL_HKD'] > 0])/cnt*100 if cnt>0 else 0):.1f}% ({cnt})")
    
    # 補回：最優交易與損益熱圖
    if not f_comp.empty:
        best_trade = f_comp.loc[f_comp['PnL_HKD'].idxmax()]
        st.info(f"🏆 **最優交易回顧**: {best_trade['Symbol']} | 獲利: ${best_trade['PnL_HKD']:,.2f} | 盈虧比: {best_trade['Trade_R']:.2f}R")
        
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="資金曲線 (Equity Curve)", height=300), use_container_width=True)
        
        # 損益熱圖
        st.subheader("🌡️ 交易盈虧熱圖 (Trade PnL Heatmap)")
        fig_heat = px.bar(f_comp, x='Exit_Date', y='PnL_HKD', color='PnL_HKD', 
                          color_continuous_scale=['red', 'gray', 'green'], 
                          title="每日交易結果分布")
        st.plotly_chart(fig_heat, use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
    live_prices = get_live_prices(list(active_pos.keys()))
    processed_p_data = []
    for s, d in active_pos.items():
        now = live_prices.get(s)
        un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
        processed_p_data.append({"代號": s, "持股數": f"{d['qty']:,.0f}", "平均成本": f"{d['avg_price']:,.2f}", "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{d['last_sl']:,.2f}", "未實現損益": f"{un_pnl:,.2f}", "報酬%": (un_pnl/(d['qty']*d['avg_price'])*100 if now and d['avg_price']!=0 else 0)})
    if processed_p_data: st.dataframe(pd.DataFrame(processed_p_data), hide_index=True, use_container_width=True)

with t3:
    st.subheader("🔄 交易重播")
    if not df.empty:
        target = st.selectbox("選擇交易記錄", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']}")
        row = df.iloc[target]
        if pd.notnull(row['Img']) and os.path.exists(row['Img']): st.image(row['Img'], caption="交易截圖")
        st.write(row.to_dict())

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
                agg_df = completed_trades_df.groupby(group_by).agg(Count=('Symbol', 'count'), Win_Rate=('PnL_HKD', lambda x: (x > 0).mean() * 100), Avg_R=('Trade_R', 'mean'), Avg_HKD=('PnL_HKD', 'mean'), Gross_Win=('PnL_HKD', lambda x: x[x > 0].sum()), Gross_Loss=('PnL_HKD', lambda x: abs(x[x <= 0].sum()))).reset_index()
                agg_df['Profit Factor'] = agg_df['Gross_Win'] / agg_df['Gross_Loss'].replace(0, 1)
                st.dataframe(agg_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🤖 Free AI Review Export")
    review_mode = st.radio("Export for review:", ["Single Trade", "Period Summary", "Full Journal"])

    export_data = {}
    if review_mode == "Single Trade":
        if not df.empty:
            trade_idx = st.selectbox("Select trade:", df.index, format_func=lambda x: f"[{df.iloc[x]['Date']}] {df.iloc[x]['Symbol']} ({df.iloc[x]['Action']})")
            selected_trade = df.iloc[trade_idx]
            export_data = selected_trade.to_dict()
    elif review_mode == "Period Summary":
        if not completed_trades_df.empty:
            summary_stats = completed_trades_df.agg({'Trade_R': 'mean', 'PnL_HKD': 'mean', 'Duration_Days': 'mean'}).to_dict()
            export_data = {"period": "Current filtered period", "trades": len(completed_trades_df), **summary_stats}
    else:  # Full Journal
        if not completed_trades_df.empty:
            export_data = {"total_trades": len(completed_trades_df), "avg_R": completed_trades_df['Trade_R'].mean()}

    if export_data:
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button("📥 Download JSON for AI", json_str, f"ai-review.json", "application/json")

with t5:
    st.subheader("🛠️ 數據管理")
    
    # 補回：CSV 導入與備份功能
    col_mgmt1, col_mgmt2 = st.columns(2)
    with col_mgmt1:
        st.markdown("#### 📥 導入數據 (CSV)")
        uploaded_csv = st.file_uploader("選擇您的歷史 CSV 檔案", type="csv")
        if uploaded_csv:
            new_df = pd.read_csv(uploaded_csv)
            if st.button("確認合併數據"):
                merged_df = pd.concat([df, new_df]).drop_duplicates(subset=['Timestamp', 'Symbol', 'Price'], keep='last')
                save_all_data(merged_df)
                st.success("數據合併成功！")
                st.rerun()
                
    with col_mgmt2:
        st.markdown("#### 📤 備份數據")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("下載當前日誌備份", data=csv_data, file_name=f"trade_backup_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.divider()
    st.markdown("#### 📋 原始數據查看")
    st.dataframe(df, use_container_width=True)
    
    if st.button("🚨 警告：清空所有數據"):
        save_all_data(pd.DataFrame(columns=df.columns))
        st.rerun()
