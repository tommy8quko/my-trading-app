import streamlit as st
import pandas as pd
import os
import requests
import time
import yfinance as yf
import plotly.express as px
from datetime import datetime

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v3.csv"
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.set_page_config(page_title="Pro Trader Edge", layout="wide")

# 初始化 CSV (增加 Stop_Loss 欄位)
def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", 
            "Price", "Quantity", "Stop_Loss", "Fees", "Notes", "Img", "Timestamp"
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    df = pd.read_csv(FILE_NAME)
    # 確保舊資料也能相容新欄位
    if "Stop_Loss" not in df.columns:
        df["Stop_Loss"] = None
    return df

def save_transaction(data):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

# --- 2. 核心邏輯：計算持倉與損益曲線 ---
def calculate_portfolio(df):
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
        sl = float(row['Stop_Loss']) if pd.notna(row['Stop_Loss']) else None
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'initial_sl': None}
            
        curr = positions[sym]
        
        if "買入 Buy" in action:
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty != 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
            # 紀錄該標的的止損價（以最後一次買入為準）
            if sl is not None: curr['initial_sl'] = sl
            
        elif "賣出 Sell" in action:
            trade_pnl = (price - curr['avg_price']) * qty
            
            # 計算 Risk/Reward Ratio
            rr_ratio = "N/A"
            if curr['initial_sl'] and curr['initial_sl'] < curr['avg_price']:
                risk = curr['avg_price'] - curr['initial_sl']
                reward = price - curr['avg_price']
                rr_ratio = round(reward / risk, 2) if risk != 0 else 0
            
            total_realized_pnl += trade_pnl
            curr['qty'] -= qty
            running_pnl += trade_pnl
            equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})
            
            trade_history.append({
                "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                "Sell_Price": price, "Entry_Cost": round(curr['avg_price'], 2),
                "PnL": round(trade_pnl, 2), "R/R Ratio": rr_ratio, "Notes": row['Notes']
            })

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0}
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

# --- 3. 即時報價功能 ---
@st.cache_data(ttl=300)
def get_live_prices(symbols_list):
    if not symbols_list: return {}
    try:
        data = yf.download(symbols_list, period="1d", progress=False, multi_level=False)
        close_data = data['Close'] if 'Close' in data.columns else data
        prices = {}
        for sym in symbols_list:
            try:
                val = close_data.iloc[-1] if len(symbols_list) == 1 else close_data[sym].iloc[-1]
                prices[sym] = float(val) if pd.notna(val) else None
            except: prices[sym] = None
        return prices
    except: return {}

# --- 4. AI 分析 ---
def fetch_ai_insight(pnl_summary, open_summary, risk_summary):
    api_key = "" 
    if not api_key: return "⚠️ 未配置 API Key。"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    prompt = f"你是專業交易教練。分析數據並給予繁體中文建議：\n已實現損益:{pnl_summary}\n當前持倉:{open_summary}\n全損回撤預估:{risk_summary}\n請提供：1.風險集中度評估 2.如果發生全損的心理建設 3.操作建議。"
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 無法連線。"

# --- 5. UI 介面 ---
st.markdown("<style>div[data-testid='metric-container'] { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }</style>", unsafe_allow_html=True)
st.title("🚀 Pro Trader Edge (專業版 v3.2)")

df = load_data()
active_pos, realized_pnl, history_df, equity_df = calculate_portfolio(df)

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚡ 交易指令")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號 (如 700 或 TSLA)").upper().strip()
        s_in = s_raw.zfill(4) + ".HK" if s_raw.isdigit() else s_raw
        
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數", min_value=0.0, step=1.0, value=None, placeholder="輸入股數")
        p_in = col2.number_input("價格", min_value=0.0, step=0.01, value=None, placeholder="輸入價格")
        
        sl_in = st.number_input("預設止損價 (Stop Loss)", min_value=0.0, step=0.01, value=None, placeholder="買入時填寫")
        
        st_select = st.radio("策略標籤", ["Breakout", "Pullback", "Custom 自訂"], horizontal=True)
        st_in = st.text_input("請輸入自訂策略") if st_select == "Custom 自訂" else st_select
            
        note_in = st.text_area("交易心得")
        img_in = st.file_uploader("上傳截圖", type=['jpg', 'png'])
        
        if st.form_submit_button("儲存紀錄"):
            if s_in and q_in is not None and p_in is not None:
                i_path = ""
                if img_in:
                    i_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}.png")
                    with open(i_path, "wb") as f: f.write(img_in.getbuffer())
                save_transaction({
                    "Date": d_in, "Symbol": s_in, "Action": act_in, "Strategy": st_in, 
                    "Price": p_in, "Quantity": q_in, "Stop_Loss": sl_in, 
                    "Fees": 0, "Notes": note_in, "Img": i_path, "Timestamp": int(time.time())
                })
                st.success(f"已紀錄 {s_in}")
                st.rerun()

# --- 主畫面 ---
t1, t2, t3 = st.tabs(["📈 帳戶績效", "🔥 即時持倉", "📜 歷史紀錄"])

with t1:
    # --- 計算全損回撤 (Projected Drawdown) ---
    total_projected_drawdown = 0
    missing_sl = []
    
    if active_pos:
        prices = get_live_prices(list(active_pos.keys()))
        for s, d in active_pos.items():
            now = prices.get(s)
            if now and d['initial_sl']:
                # 風險 = (現價 - 止損價) * 股數
                risk_amount = (now - d['initial_sl']) * d['qty']
                total_projected_drawdown += risk_amount
            elif not d['initial_sl']:
                missing_sl.append(s)

    # 顯示指標
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已實現損益", f"${realized_pnl:,.0f}")
    win_r = (len(history_df[history_df['PnL']>0])/len(history_df)*100) if not history_df.empty else 0
    c2.metric("交易勝率", f"{win_r:.1f}%")
    
    # R/R 指標
    avg_rr = 0
    if not history_df.empty and "R/R Ratio" in history_df.columns:
        valid_rr = history_df[history_df["R/R Ratio"] != "N/A"]["R/R Ratio"]
        avg_rr = round(valid_rr.mean(), 2) if not valid_rr.empty else 0
    c3.metric("平均 R/R 比", f"{avg_rr}")
    
    # 新增：全損回撤指標
    c4.metric("全損回撤預估", f"-${total_projected_drawdown:,.0f}", delta_color="inverse")
    if missing_sl:
        st.caption(f"⚠️ 提醒：{', '.join(missing_sl)} 未設定止損價，未計入回撤。")

    if not equity_df.empty:
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益增長曲線"), use_container_width=True)

    if st.button("🤖 執行風險與績效 AI 診斷", use_container_width=True):
        with st.spinner("分析風險中..."):
            rep = fetch_ai_insight(
                f"${realized_pnl:,.0f}", 
                str(list(active_pos.keys())),
                f"-${total_projected_drawdown:,.0f}"
            )
            st.info(rep)

with t2:
    if active_pos:
        prices = get_live_prices(list(active_pos.keys()))
        p_data = []
        for s, d in active_pos.items():
            now = prices.get(s)
            un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
            # 顯示該倉位的風險
            risk_per_pos = (now - d['initial_sl']) * d['qty'] if (now and d['initial_sl']) else 0
            p_data.append({
                "代號": s, "股數": d['qty'], "成本": f"${d['avg_price']:.2f}", 
                "止損": f"${d['initial_sl']:.2f}" if d['initial_sl'] else "未設定",
                "現價": f"${now:.2f}" if now else "...", 
                "未實現損益": round(un_pnl, 2),
                "若止損虧損": f"-${risk_per_pos:,.2f}" if risk_per_pos > 0 else "$0"
            })
        st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)
    else: st.info("目前無持倉")

with t3:
    st.markdown("### 歷史結算紀錄")
    st.dataframe(history_df.sort_values("Date", ascending=False), use_container_width=True)
