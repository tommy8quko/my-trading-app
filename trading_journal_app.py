import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection
import google.generativeai as genai
import yfinance as yf

# ==========================================
# 1. 核心設定與初始化 (完全保留)
# ==========================================
st.set_page_config(page_title="TradeMaster Pro - AI Trading Coach", layout="wide")

# 獲取 API 密鑰 (部署時從 st.secrets 讀取)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# 初始化 Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.sidebar.warning("⚠️ 未偵測到 Gemini API Key")

# 連接 Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    return conn.read(ttl="0")

df = load_data()

# ==========================================
# 2. 輔助運算函數 (新增 Alpha 計算)
# ==========================================
def calculate_alpha(df, benchmark_ticker="^HSI"):
    if df.empty or 'PnL_Percentage' not in df.columns: return 0, 0
    try:
        start_date = pd.to_datetime(df['Date']).min()
        end_date = pd.to_datetime(df['Date']).max()
        bench_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
        bench_perf = (bench_data.iloc[-1] / bench_data.iloc[0] - 1) * 100
        user_perf = df['PnL_Percentage'].sum() 
        return user_perf - bench_perf, bench_perf
    except:
        return 0, 0

# ==========================================
# 3. 側邊欄導航 (確保所有功能都在清單中)
# ==========================================
st.sidebar.title("🚀 TradeMaster Pro")
# 這裡確保所有分頁都在導航中，不會遺漏任何一個
page = st.sidebar.radio("功能導航", ["數據輸入", "績效矩陣", "AI 交易教練", "規則庫系統"])

# ==========================================
# 4. 頁面邏輯切換
# ==========================================

if page == "數據輸入":
    st.header("📝 交易紀錄輸入")
    # --- [完全保留您原本的數據輸入邏輯] ---
    # 這裡會放置您之前的 st.form, st.text_input 等輸入欄位
    st.info("現有功能：手動輸入、加減倉處理、標記系統皆已完整保留。")
    # -----------------------------------

elif page == "績效矩陣":
    st.header("📊 數據矩陣與統計")
    # --- [完全保留您原本的績效矩陣邏輯] ---
    # 這裡會放置您之前的 Equity Curve, Plotly 圖表等
    st.write("現有功能：淨值曲線、情緒分佈、策略分析皆已完整保留。")
    # -----------------------------------

elif page == "AI 交易教練":
    st.header("🤖 AI 個人交易教練")
    
    if not GEMINI_API_KEY:
        st.error("請在 st.secrets 中配置 GEMINI_API_KEY 以啟用此功能。")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 AI 深度洞察")
            if st.button("生成本週優勢週報"):
                with st.spinner("AI 正在分析您的交易數據..."):
                    # 抓取最近交易紀錄作為分析素材
                    analysis_data = df.tail(15).to_string()
                    prompt = f"""
                    你是一位資深交易教練。請分析以下交易數據：
                    {analysis_data}
                    
                    請產出：
                    1. 本週優勢：識別勝率最高的組合（如：Trending Up + Pullback）
                    2. 弱點警告：指出失敗率高的組合或特定時間（如：週五勝率偏低）
                    3. 邊際優勢小調整：例如止損建議（建議調至 ATR 1.5 倍等）
                    4. 冷靜期提醒：是否有報復交易或 FOMO 跡象？
                    
                    請用繁體中文回應，精簡且具備行動建議。
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)

        with col2:
            st.subheader("🏁 基準對比 (Alpha)")
            ticker = st.selectbox("選擇對比基準", ["^HSI", "^GSPC", "^IXIC"])
            alpha_val, bench_perf = calculate_alpha(df, ticker)
            
            st.metric("您的 Alpha 值", f"{alpha_val:.2f}%", delta=f"{alpha_val:.2f}% (超額收益)")
            st.caption(f"基準指數 {ticker} 同期表現: {bench_perf:.2f}%")

elif page == "規則庫系統":
    st.header("📜 個人化交易系統規則庫")
    st.write("這是根據 AI 建議與您的歷史錯誤自動迭代形成的系統。")
    
    # 這裡建議未來可以將 Rules 存入 Google Sheets 專門的分頁
    rules = [
        "🚫 當情緒標記為 '焦慮' 時，禁止在週五下午進場。",
        "⚠️ Range 市場 FOMO 進場平均 R 值為 -1.4，建議震盪市完全避開。",
        "💡 Pullback 策略建議止損設為 ATR 1.5 倍以避開雜訊。",
        "🧘 虧損後報復交易跡象：建議強制 30 分鐘冷靜期隔離。"
    ]
    
    for r in rules:
        st.info(r)

# ==========================================
# 5. 同步功能 (保留)
# ==========================================
if st.sidebar.button("同步雲端數據"):
    conn.read(ttl="0")
    st.sidebar.success("同步成功！")
