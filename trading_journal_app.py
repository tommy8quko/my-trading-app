import streamlit as st
import pandas as pd
import os
import requests
import time
from datetime import datetime

# --- 1. 核心配置 ---
FILE_NAME = "trade_ledger.csv"  # 改名以區分舊版格式
UPLOAD_FOLDER = "images"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 初始化流水帳 (Ledger)
# 這裡紀錄每一筆「動作」，而不是每一筆「完整交易」
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
    # 轉換新資料為 DataFrame 並合併
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

# --- 2. 核心邏輯：計算持倉與損益 ---
# 這是一個會計引擎，它會重跑所有歷史紀錄來算出當前狀態
def calculate_portfolio(df):
    positions = {} # 格式: { 'AAPL': {'qty': 1000, 'avg_price': 150.0, 'realized_pnl': 5000} }
    
    # 確保數據按照時間排序
    df = df.sort_values(by="Timestamp")
    
    total_realized_pnl = 0
    trade_history = [] # 用來存儲每一筆結算的賣出紀錄

    for index, row in df.iterrows():
        sym = row['Symbol']
        action = row['Action']
        qty = float(row['Quantity'])
        price = float(row['Price'])
        fees = float(row['Fees']) if 'Fees' in row and pd.notna(row['Fees']) else 0
        
        if sym not in positions:
            positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'realized_pnl': 0.0}
            
        curr = positions[sym]
        
        # 簡單的做多邏輯 (Long Only Logic for simplicity)
        # 如果需要做空，邏輯會更複雜，這裡假設主要為做多
        if action == "買入 Buy":
            # 計算新的平均成本 (加權平均)
            total_cost = (curr['qty'] * curr['avg_price']) + (qty * price)
            new_qty = curr['qty'] + qty
            if new_qty != 0:
                curr['avg_price'] = total_cost / new_qty
            curr['qty'] = new_qty
            
        elif action == "賣出 Sell":
            # 計算已實現損益
            # 損益 = (賣出價 - 平均成本) * 賣出股數 - 手續費
            trade_pnl = ((price - curr['avg_price']) * qty) - fees
            curr['realized_pnl'] += trade_pnl
            total_realized_pnl += trade_pnl
            curr['qty'] -= qty
            
            # 紀錄這筆賣出的績效
            trade_history.append({
                "Date": row['Date'],
                "Symbol": sym,
                "Strategy": row['Strategy'],
                "Sell_Price": price,
                "Avg_Entry": curr['avg_price'],
                "Qty": qty,
                "PnL": trade_pnl,
                "Notes": row['Notes']
            })

    # 過濾掉股數為 0 的持倉，只回傳現有持倉
    active_positions = {k: v for k, v in positions.items() if v['qty'] > 0}
    
    return active_positions, total_realized_pnl, pd.DataFrame(trade_history)

# --- 3. AI 分析功能 ---
def fetch_ai_insight(pnl_text, open_pos_text):
    api_key = "" # 部署時請在 Streamlit Cloud Secrets 設定，或直接填入(不建議公開)
    if not api_key:
        return "⚠️ 請先配置 Gemini API Key 才能使用 AI 分析。"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    prompt = f"""
    你是專業的交易績效分析師。請分析以下數據 (繁體中文回覆)：
    
    [已實現損益紀錄]
    {pnl_text}
    
    [目前持倉風險]
    {open_pos_text}
    
    請簡短給出：
    1. 表現最好的策略與標的。
    2. 針對目前持倉的風險提示 (例如某檔股票佔比過重)。
    3. 下一步的操作建議。
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass
    return "❌ AI 連線逾時，請稍後再試。"

# --- 4. App 介面 ---
st.set_page_config(page_title="Pro Trader Journal", layout="centered")

# 手機版優化 CSS
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("💰 智能分批交易日誌")

# 讀取數據
df = load_data()
active_pos, total_pnl, history_df = calculate_portfolio(df)

# --- 側邊欄：輸入區 ---
with st.sidebar:
    st.header("📝 新增交易動作")
    with st.form("trade_form", clear_on_submit=True):
        date_in = st.date_input("日期")
        
        # 標的輸入 (自動大寫)
        symbol_in = st.text_input("股票代號 (Symbol)", placeholder="e.g. TSLA").upper()
        
        # 動作選擇
        action_in = st.radio("動作", ["買入 Buy", "賣出 Sell"], horizontal=True)
        
        # 股數與價格
        col1, col2 = st.columns(2)
        qty_in = col1.number_input("股數/口數", min_value=0.01, step=1.0)
        price_in = col2.number_input("成交價格", min_value=0.0, step=0.1)
        
        # 策略標籤 (Custom Tag)
        # 取得現有的策略列表
        existing_strategies = df['Strategy'].unique().tolist() if not df.empty else []
        default_opts = ["趨勢跟隨", "突破", "抄底", "當沖"]
        all_opts = list(set(default_opts + existing_strategies))
        
        # 讓使用者選擇或輸入新標籤
        strategy_select = st.selectbox("策略標籤", ["選取現有..."] + all_opts + ["➕ 新增自訂..."])
        
        final_strategy = ""
        if strategy_select == "➕ 新增自訂..." or strategy_select == "選取現有...":
            final_strategy = st.text_input("輸入新策略名稱")
        else:
            final_strategy = strategy_select

        notes_in = st.text_area("筆記")
        img_file = st.file_uploader("上傳截圖", type=['png', 'jpg'])
        
        submitted = st.form_submit_button("確認送出")
        
        if submitted:
            if qty_in > 0 and price_in > 0 and symbol_in:
                # 處理圖片
                img_path = ""
                if img_file:
                    img_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_file.getbuffer())
                
                # 儲存
                save_transaction({
                    "Date": date_in,
                    "Symbol": symbol_in,
                    "Action": action_in,
                    "Strategy": final_strategy if final_strategy else "未分類",
                    "Price": price_in,
                    "Quantity": qty_in,
                    "Fees": 0, # 未來可擴充手續費欄位
                    "Notes": notes_in,
                    "Img": img_path,
                    "Timestamp": int(time.time())
                })
                st.success("紀錄已更新！")
                st.rerun()
            else:
                st.error("請輸入完整的價格與股數")

# --- 主畫面：儀表板 ---

# 1. 帳戶摘要
st.markdown("### 📊 帳戶概況")
c1, c2, c3 = st.columns(3)
c1.metric("已實現損益", f"${total_pnl:,.0f}")
c2.metric("持倉檔數", len(active_pos))
# 估算持倉市值
total_market_value = sum([v['qty'] * v['avg_price'] for k, v in active_pos.items()])
c3.metric("持倉總成本", f"${total_market_value:,.0f}")

st.divider()

# 2. 目前持倉 (Open Positions)
st.subheader("🔥 目前持倉 (未平倉)")
if active_pos:
    pos_data = []
    for sym, data in active_pos.items():
        pos_data.append({
            "代號": sym,
            "持有股數": f"{data['qty']:,.0f}",
            "平均成本": f"${data['avg_price']:.2f}",
            "預估市值": f"${data['qty'] * data['avg_price']:.2f}"
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
else:
    st.info("目前空手，無持倉部位。")

# 3. AI 分析
st.divider()
if st.button("🤖 AI 投資組合診斷", use_container_width=True):
    with st.spinner("AI 正在分析您的分批進出場邏輯..."):
        # 準備資料給 AI
        pnl_summary = history_df.groupby('Strategy')['PnL'].sum().to_string() if not history_df.empty else "無已實現損益"
        pos_summary = str(active_pos)
        
        insight = fetch_ai_insight(pnl_summary, pos_summary)
        st.markdown(f"""
        <div style="background-color:#e8f4f9; padding:15px; border-radius:10px; border-left: 5px solid #2b8cbe;">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# 4. 近期已實現交易 (History)
st.subheader("📜 已平倉/部分獲利紀錄")
if not history_df.empty:
    # 格式化顯示
    show_df = history_df[['Date', 'Symbol', 'Strategy', 'Qty', 'Sell_Price', 'PnL']].copy()
    show_df['PnL'] = show_df['PnL'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(show_df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
else:
    st.write("尚無賣出紀錄。")

# 5. 完整流水帳 (Debug用)
with st.expander("查看完整交易流水帳 (Raw Data)"):
    st.dataframe(df.sort_values(by="Timestamp", ascending=False))
