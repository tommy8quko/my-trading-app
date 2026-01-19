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

# 初始化 CSV
def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", 
            "Price", "Quantity", "Fees", "Notes", "Img", "Timestamp"
        ])
        df.to_csv(FILE_NAME, index=False)

init_csv()

def load_data():
    try:
        return pd.read_csv(FILE_NAME)
    except:
        init_csv()
        return pd.read_csv(FILE_NAME)

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
        date = row['Date']
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0}
            
        curr = positions[sym]
        
        if "買入 Buy" in action:
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty != 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
        elif "賣出 Sell" in action:
            trade_pnl = (price - curr['avg_price']) * qty
            total_realized_pnl += trade_pnl
            curr['qty'] -= qty
            running_pnl += trade_pnl
            equity_curve.append({"Date": date, "Cumulative PnL": running_pnl})
            trade_history.append({
                "Date": date, "Symbol": sym, "Strategy": row['Strategy'],
                "Sell_Price": price, "Entry_Cost": curr['avg_price'],
                "Qty": qty, "PnL": trade_pnl, "Notes": row['Notes']
            })

    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0}
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history), pd.DataFrame(equity_curve)

# --- 3. 即時報價功能 (修正快取錯誤) ---
@st.cache_data(ttl=300)
def get_live_prices(symbols_list):
    """
    接收一個清單 (List) 而非 dict_keys
    """
    if not symbols_list: return {}
    try:
        # 下載數據
        data = yf.download(symbols_list, period="1d", progress=False, multi_level=False)
        
        # 取得最後一行的收盤價 (Close)
        if 'Close' in data.columns:
            close_data = data['Close']
        else:
            close_data = data # 有些版本的 yfinance 直接回傳 Series

        prices = {}
        for sym in symbols_list:
            try:
                # 處理單一標的與多標的不同格式
                if len(symbols_list) == 1:
                    val = close_data.iloc[-1]
                else:
                    val = close_data[sym].iloc[-1]
                
                if pd.notna(val):
                    prices[sym] = float(val)
                else:
                    prices[sym] = None
            except:
                prices[sym] = None
        return prices
    except Exception as e:
        st.sidebar.error(f"報價抓取失敗: {e}")
        return {}

# --- 4. AI 分析 ---
def fetch_ai_insight(pnl_summary, open_summary):
    api_key = "" # 系統會自動注入
    if not api_key: return "⚠️ 請於設定中配置 API Key。"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    prompt = f"你是專業交易教練。請分析數據並給予繁體中文建議：\n已實現:{pnl_summary}\n持倉:{open_summary}\n請提供：1.表現評估 2.風險警告 3.下週建議。"
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except: return "AI 目前無法連線。"

# --- 5. UI 介面 ---
st.markdown("<style>div[data-testid='metric-container'] { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }</style>", unsafe_allow_html=True)
st.title("🚀 Pro Trader Edge (專業版)")

df = load_data()
active_pos, realized_pnl, history_df, equity_df = calculate_portfolio(df)

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚡ 交易指令")
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_raw = st.text_input("代號 (如 700 或 TSLA)").upper().strip()
        
        # 港股自動補完邏輯
        if s_raw.isdigit():
            s_in = s_raw.zfill(4) + ".HK"
        else:
            s_in = s_raw
            
        act_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        col1, col2 = st.columns(2)
        q_in = col1.number_input("股數", min_value=0.01, step=1.0)
        p_in = col2.number_input("價格", min_value=0.0)
        
        tags = list(set(["趨勢", "突破", "反轉"] + (df['Strategy'].unique().tolist() if not df.empty else [])))
        st_in = st.selectbox("策略標籤", tags + ["➕ 新增..."])
        if st_in == "➕ 新增...":
            st_in = st.text_input("輸入新標籤")
            
        note_in = st.text_area("交易心得")
        img_in = st.file_uploader("上傳截圖", type=['jpg', 'png'])
        
        if st.form_submit_button("儲存紀錄"):
            if s_in and q_in > 0 and p_in > 0:
                i_path = ""
                if img_in:
                    i_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}.png")
                    with open(i_path, "wb") as f: f.write(img_in.getbuffer())
                save_transaction({"Date": d_in, "Symbol": s_in, "Action": act_in, "Strategy": st_in, "Price": p_in, "Quantity": q_in, "Fees": 0, "Notes": note_in, "Img": i_path, "Timestamp": int(time.time())})
                st.success(f"已紀錄 {s_in}")
                st.rerun()

# --- 主畫面 ---
t1, t2, t3 = st.tabs(["📈 帳戶績效", "🔥 即時持倉", "📜 歷史流水帳"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已實現損益", f"${realized_pnl:,.0f}")
    win_r = (len(history_df[history_df['PnL']>0])/len(history_df)*100) if not history_df.empty else 0
    c2.metric("勝率", f"{win_r:.1f}%")
    c3.metric("持倉檔數", len(active_pos))
    
    if not equity_df.empty:
        fig = px.area(equity_df, x="Date", y="Cumulative PnL", title="資金成長曲線")
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🤖 執行 AI 診斷", use_container_width=True):
        with st.spinner("AI 分析中..."):
            rep = fetch_ai_insight(f"PnL:{realized_pnl}, WinRate:{win_r}%", str(list(active_pos.keys())))
            st.info(rep)

with t2:
    if active_pos:
        # 修正這裡：將 .keys() 轉換為 list
        prices = get_live_prices(list(active_pos.keys()))
        p_data = []
        un_total = 0
        for s, d in active_pos.items():
            now = prices.get(s)
            un_pnl = (now - d['avg_price']) * d['qty'] if now else 0
            un_total += un_pnl
            p_data.append({
                "代號": s, 
                "股數": d['qty'], 
                "成本": f"${d['avg_price']:.2f}", 
                "現價": f"${now:.2f}" if now else "載入中", 
                "未實現損益": un_pnl, 
                "報酬率": f"{(un_pnl/(d['qty']*d['avg_price'])*100):.2f}%" if d['avg_price']!=0 else "0%"
            })
        
        st.metric("總未實現損益 (浮動)", f"${un_total:,.2f}", delta=f"{un_total:,.2f}")
        st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)
        if st.button("🔄 刷新報價"): st.cache_data.clear(); st.rerun()
    else: st.info("目前無持倉")

with t3:
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
    if st.checkbox("顯示最近截圖"):
        last_img = df[df['Img']!=""].tail(1)
        if not last_img.empty: st.image(last_img['Img'].values[0])
