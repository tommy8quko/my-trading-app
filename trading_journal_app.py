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
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
            "Market_Condition", "Mistake_Tag" # 新增欄位
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def format_symbol(s_raw):
    """ 統一處理代號格式 """
    if pd.isna(s_raw): return ""
    s_str = str(s_raw).upper().strip()
    if s_str.isdigit() and len(s_str) <= 5:
        return s_str.zfill(4) + ".HK"
    return s_str

def clean_strategy(s):
    """ 統一策略名稱 """
    s_str = str(s).strip()
    if "PULLBACK" in s_str.upper(): return "Pullback"
    if "BREAKOUT" in s_str.upper() or "BREAK OUT" in s_str.upper(): return "Breakout"
    return s_str

def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        if df.empty:
            return df
        
        # 數據清理與格式統一
        if 'Symbol' in df.columns:
            df['Symbol'] = df['Symbol'].apply(format_symbol)
        if 'Strategy' in df.columns:
            df['Strategy'] = df['Strategy'].apply(clean_strategy)
            
        # 確保新增欄位存在
        for col in ["Market_Condition", "Mistake_Tag"]:
            if col not in df.columns:
                df[col] = "N/A"

        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Date'], errors='coerce').view('int64') // 10**9
            df['Timestamp'] = df['Timestamp'].replace(-9223372036, int(time.time()))
            save_all_data(df)

        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Stop_Loss'] = pd.to_numeric(df['Stop_Loss'], errors='coerce').fillna(0)
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        return df
    except Exception as e:
        return pd.DataFrame()

def save_all_data(df):
    df.to_csv(FILE_NAME, index=False)

def save_transaction(data):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    save_all_data(df)

def get_hkd_value(symbol, value):
    if isinstance(symbol, str) and ".HK" in symbol.upper():
        return value
    return value * USD_HKD_RATE

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper():
        return "HK$"
    return "$"

# --- 2. 核心邏輯 ---
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0
    
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

        qty = float(row['Quantity']) if pd.notnull(row['Quantity']) else 0.0
        price = float(row['Price']) if pd.notnull(row['Price']) else 0.0
        sl = float(row['Stop_Loss']) if pd.notnull(row['Stop_Loss']) else 0.0
        date = row['Date']
        ts = row['Timestamp']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0}
        
        if sym not in cycle_tracker:
            cycle_tracker[sym] = {'cash_flow_raw': 0.0, 'start_date': date, 'start_ts': ts, 'is_active': False}
            
        curr = positions[sym]
        if sl > 0: curr['last_sl'] = sl
        
        if not cycle_tracker[sym]['is_active'] and qty > 0:
            cycle_tracker[sym]['is_active'] = True
            cycle_tracker[sym]['start_date'] = date
            cycle_tracker[sym]['start_ts'] = ts
            cycle_tracker[sym]['cash_flow_raw'] = 0.0

        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        if is_buy:
            cycle_tracker[sym]['cash_flow_raw'] -= (qty * price)
            total_cost_base = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty > 0:
                curr['avg_price'] = total_cost_base / new_qty
            curr['qty'] = new_qty
        
        elif is_sell:
            if curr['qty'] > 0:
                sell_qty = min(qty, curr['qty'])
                cycle_tracker[sym]['cash_flow_raw'] += (sell_qty * price)
                
                realized_pnl_raw = (price - curr['avg_price']) * sell_qty
                realized_pnl_hkd_item = get_hkd_value(sym, realized_pnl_raw)
                
                total_realized_pnl_hkd += realized_pnl_hkd_item
                running_pnl_hkd += realized_pnl_hkd_item
                curr['qty'] -= sell_qty
                
                if curr['qty'] < 0.0001:
                    duration_days = (ts - cycle_tracker[sym]['start_ts']) / 86400
                    completed_trades.append({
                        "Exit_Date": date,
                        "Entry_Date": cycle_tracker[sym]['start_date'],
                        "Symbol": sym, 
                        "PnL_Raw": cycle_tracker[sym]['cash_flow_raw'],
                        "PnL_HKD": get_hkd_value(sym, cycle_tracker[sym]['cash_flow_raw']),
                        "Duration_Days": duration_days
                    })
                    cycle_tracker[sym]['is_active'] = False
                
                equity_curve.append({"Date": date, "Cumulative PnL": running_pnl_hkd})

    completed_df = pd.DataFrame(completed_trades)
    
    # 計算 Expectancy 與 平均持倉時間
    expectancy = 0
    avg_duration = 0
    if not completed_df.empty:
        wins = completed_df[completed_df['PnL_HKD'] > 0]
        losses = completed_df[completed_df['PnL_HKD'] <= 0]
        win_rate = len(wins) / len(completed_df)
        loss_rate = 1 - win_rate
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        avg_duration = completed_df['Duration_Days'].mean()

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0.0001}
    return active_positions, total_realized_pnl_hkd, completed_df, pd.DataFrame(equity_curve), expectancy, avg_duration

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
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, expectancy_val, avg_dur_val = calculate_portfolio(df)

with st.sidebar:
    st.header("⚡ 執行面板")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號 (Ticker)", placeholder="例如: 700 或 TSLA").upper().strip()
        s_in = format_symbol(s_raw) 
        is_sell = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=1.0, value=None)
        p_in = col2.number_input("成交價格 (Price)", min_value=0.0, step=0.01, value=None)
        sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.01, value=None)
        st.divider()
        
        # 新增標籤屬性
        mkt_cond = st.selectbox("市場環境", ["Trending Up", "Trending Down", "Range/Choppy", "High Volatility", "N/A"])
        mistake_in = st.selectbox("錯誤標籤", ["None", "Fomo", "Revenge Trade", "Fat Finger", "Late Entry", "Moved Stop"])
        
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
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": rr_in, "Notes": note_in, 
                    "Timestamp": int(time.time()),
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in
                })
                st.success(f"✅ 已儲存 {s_in}")
                time.sleep(0.5)
                st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    
    total_sl_risk_hkd = 0
    if active_pos:
        live_prices_for_risk = get_live_prices(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = live_prices_for_risk.get(s)
            if now and d['last_sl'] > 0:
                risk_raw = (now - d['last_sl']) * d['qty']
                total_sl_risk_hkd += get_hkd_value(s, risk_raw)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("已實現損益 (HKD)", f"${realized_pnl_total_hkd:,.2f}")
    m2.metric("期望值 (Expectancy)", f"${expectancy_val:,.2f}")
    m3.metric("平均持倉", f"{avg_dur_val:.1f} 天")
    trade_count = len(completed_trades_df)
    m4.metric("勝率", f"{(len(completed_trades_df[completed_trades_df['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0:.1f}%")
    m5.metric("總交易場數", f"{trade_count}")

    if not equity_df.empty:
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線 (HKD 匯總)", height=300), use_container_width=True)

    if not completed_trades_df.empty:
        st.divider()
        st.subheader("🏆 交易排行榜 (按 HKD 價值排序)")
        
        display_trades = completed_trades_df.copy()
        # 準備顯示用的欄位
        display_trades['原始損益'] = display_trades.apply(lambda x: f"{get_currency_symbol(x['Symbol'])} {x['PnL_Raw']:,.2f}", axis=1)
        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: f"${x:,.2f}")
        
        # 改成中文欄位名
        display_trades = display_trades.rename(columns={
            "Exit_Date": "出場日期",
            "Entry_Date": "進場日期",
            "Symbol": "代號",
            "Duration_Days": "持有天數"
        })
        
        rank_col1, rank_col2 = st.columns(2)
        with rank_col1:
            st.markdown("##### 🟢 Top 獲利")
            # 依照 PnL_HKD 排序以確保美股港股比較基準一致
            top_profit = display_trades.sort_values(by="PnL_HKD", ascending=False).head(5)
            st.dataframe(top_profit[['出場日期', '代號', '原始損益', 'HKD 損益']], hide_index=True, use_container_width=True)
            
        with rank_col2:
            st.markdown("##### 🔴 Top 虧損")
            top_loss = display_trades.sort_values(by="PnL_HKD", ascending=True).head(5)
            st.dataframe(top_loss[['出場日期', '代號', '原始損益', 'HKD 損益']], hide_index=True, use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽 (原始幣種計)")
    current_symbols = list(active_pos.keys())
    live_prices = get_live_prices(current_symbols)
    processed_p_data = []
    if active_pos:
        for s, d in active_pos.items():
            now = live_prices.get(s)
            qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
            
            un_pnl_raw = (now - avg_p) * qty if now else 0
            sl_risk_raw = (now - last_sl) * qty if (now and last_sl > 0) else 0
            pos_size_raw = now * qty if now else 0
            roi = (un_pnl_raw/(qty * avg_p)*100) if (now and avg_p!=0) else 0

            processed_p_data.append({
                "代號": s, 
                "持股數": f"{qty:,.2f}", 
                "平均成本": f"{avg_p:,.2f}", 
                "現價": f"{now:,.2f}" if now else "0.00",
                "停損價": f"{last_sl:,.2f}", 
                "部位價值": f"{pos_size_raw:,.2f}",
                "停損回撤": f"{sl_risk_raw:,.2f}",
                "未實現損益": f"{un_pnl_raw:,.2f}", 
                "報酬%": roi
            })
        p_df = pd.DataFrame(processed_p_data)
        
        st.dataframe(
            p_df, 
            column_config={
                "報酬%": st.column_config.ProgressColumn(
                    "報酬%", 
                    format="%.2f%%", 
                    min_value=-20, 
                    max_value=20,
                    color="green" if p_df["報酬%"].mean() >= 0 else "red" 
                )
            }, 
            hide_index=True, 
            use_container_width=True
        )
        
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
    st.subheader("📜 歷史紀錄與心理分析")
    if not df.empty:
        # 顯示包含新標籤的歷史紀錄
        st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("⚠️ 錯誤標籤分析")
        mistake_counts = df['Mistake_Tag'].value_counts()
        if not mistake_counts.empty:
            st.plotly_chart(px.pie(names=mistake_counts.index, values=mistake_counts.values, title="交易錯誤分布"), use_container_width=True)

with t5:
    st.subheader("🛠️ 數據管理")
    # ... (原有管理功能保留) ...
    with st.expander("📤 批量上傳交易紀錄"):
        uploaded_file = st.file_uploader("選擇 CSV 或 Excel 檔案", type=["csv", "xlsx"])
        if uploaded_file and st.button("🚀 開始匯入"):
            try:
                new_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Symbol' in new_data.columns:
                    new_data['Symbol'] = new_data['Symbol'].apply(format_symbol)
                if 'Timestamp' not in new_data.columns:
                    new_data['Timestamp'] = int(time.time())
                df = pd.concat([df, new_data], ignore_index=True)
                save_all_data(df)
                st.success("✅ 數據匯入成功！")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"匯入失敗: {e}")

    if not df.empty:
        st.markdown("### 📝 編輯或刪除紀錄")
        selected_idx = st.selectbox("選擇紀錄進行操作", df.index, format_func=lambda x: f"[{df.loc[x, 'Date']}] {df.loc[x, 'Symbol']} - {df.loc[x, 'Action']} ({df.loc[x, 'Quantity']} 股)")
        t_edit = df.loc[selected_idx]
        col_e1, col_e2, col_e3 = st.columns(3)
        n_p = col_e1.number_input("編輯價格", value=float(t_edit['Price']), key=f"edit_price_{selected_idx}")
        n_q = col_e2.number_input("編輯股數", value=float(t_edit['Quantity']), key=f"edit_qty_{selected_idx}")
        n_sl = col_e3.number_input("編輯停損價格", value=float(t_edit['Stop_Loss']), key=f"edit_sl_{selected_idx}")
        
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("💾 更新此筆紀錄", use_container_width=True):
            df.loc[selected_idx, 'Price'] = n_p
            df.loc[selected_idx, 'Quantity'] = n_q
            df.loc[selected_idx, 'Stop_Loss'] = n_sl
            save_all_data(df)
            st.success(f"✅ {df.loc[selected_idx, 'Symbol']} 紀錄已更新！")
            time.sleep(0.5)
            st.rerun()
            
        if btn_col2.button("🗑️ 刪除此筆紀錄", use_container_width=True, type="secondary"):
            target_sym = df.loc[selected_idx, 'Symbol']
            df = df.drop(selected_idx).reset_index(drop=True)
            save_all_data(df)
            st.warning(f"🗑️ {target_sym} 紀錄已刪除。")
            time.sleep(0.5)
            st.rerun()

        st.divider()
        st.markdown("### ⚙️ 數據重設")
        confirm_reset = st.checkbox("我確定要清空所有交易歷史紀錄")
        if confirm_reset:
            if st.button("執行清空所有數據", use_container_width=True):
                empty_df = pd.DataFrame(columns=[
                    "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
                    "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
                    "Market_Condition", "Mistake_Tag"
                ])
                save_all_data(empty_df)
                st.error("💥 所有交易紀錄已清空！")
                time.sleep(0.5)
                st.rerun()
