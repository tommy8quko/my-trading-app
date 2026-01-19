import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
import time
# --- 配置與環境設定 ---
# 在雲端部署時，使用相對路徑確保相容性
FILE_NAME = "trade_data_v2.csv"
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)
# 初始化資料結構
def init_csv():
   if not os.path.exists(FILE_NAME):
       df = pd.DataFrame(columns=[
           "Date", "Symbol", "Setup", "Direction",
           "Entry", "Exit", "SL", "PnL", "RR", "Notes", "Img", "Status"
       ])
       df.to_csv(FILE_NAME, index=False)
init_csv()
def load_data():
   try:
       return pd.read_csv(FILE_NAME)
   except:
       init_csv()
       return pd.read_csv(FILE_NAME)
def save_trade(data):
   df = load_data()
   df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
   df.to_csv(FILE_NAME, index=False)
# --- AI 分析核心 (含重試邏輯) ---
def fetch_ai_insight(summary_text):
   api_key = "" # 系統運行時自動注入
   if not api_key:
       return "⚠️ 請檢查 API 配置。"
   url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
   prompt = f"""
   你是專業交易導師。請根據以下數據進行繁體中文短評：
   {summary_text}
   1. 找出最賺錢的模式。
   2. 給出一個針對風險控管的警告。
   3. 建議下週的一個改進動作。
   """
   payload = {"contents": [{"parts": [{"text": prompt}]}]}
   # 指數退避重試
   for i in [1, 2, 4, 8]:
       try:
           res = requests.post(url, json=payload, timeout=15)
           if res.status_code == 200:
               return res.json()['candidates'][0]['content']['parts'][0]['text']
       except:
           time.sleep(i)
   return "❌ AI 目前忙碌中，請稍後再試。"
# --- App 介面 ---
st.set_page_config(page_title="Trading Journal", layout="centered") # 手機版建議 centered
# 自定義 CSS 讓手機端更好看
st.markdown("""
<style>
   .main { background-color: #f8f9fa; }
   .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_stdio=True)
st.title("📱 AI 交易隨身筆記")
# --- 側邊欄紀錄 (iOS 側邊欄可收合) ---
with st.sidebar:
   st.header("新增紀錄")
   with st.form("add_trade", clear_on_submit=True):
       d = st.date_input("日期")
       s = st.text_input("標的 (Symbol)").upper()
       stp = st.selectbox("策略", ["趨勢", "突破", "反轉", "震盪"])
       dr = st.radio("方向", ["多 Long", "空 Short"], horizontal=True)
       stat = st.selectbox("狀態", ["持倉中", "已平倉"])
       c1, c2 = st.columns(2)
       en = c1.number_input("進場價", format="%.2f")
       sl = c2.number_input("止損", format="%.2f")
       ex = st.number_input("出場價 (未平倉填0)", format="%.2f")
       pic = st.file_uploader("上傳圖表", type=["jpg", "png"])
       note = st.text_area("筆記")
       if st.form_submit_button("確認儲存"):
           # 計算邏輯
           pnl = (ex - en) if dr == "多 Long" else (en - ex)
           risk = abs(en - sl)
           rr = round(abs(pnl/risk), 2) if risk != 0 else 0
           img_path = ""
           if pic:
               img_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}.png")
               with open(img_path, "wb") as f:
                   f.write(pic.getbuffer())
           save_trade({
               "Date": d, "Symbol": s, "Setup": stp, "Direction": dr,
               "Entry": en, "Exit": ex, "SL": sl, "PnL": pnl if stat=="已平倉" else 0,
               "RR": rr if stat=="已平倉" else 0, "Notes": note, "Img": img_path, "Status": stat
           })
           st.success("已同步至雲端")
           st.rerun()
# --- 主畫面顯示 ---
df = load_data()
if not df.empty:
   # 1. 頂部快報
   closed = df[df['Status'] == '已平倉']
   open_pos = df[df['Status'] == '持倉中']
   col1, col2 = st.columns(2)
   with col1:
       win_rate = (len(closed[closed['PnL'] > 0]) / len(closed) * 100) if not closed.empty else 0
       st.metric("總體勝率", f"{win_rate:.1f}%")
   with col2:
       st.metric("持倉中部位", len(open_pos))
   # 2. AI 按鈕 (針對手機優化為大按鈕)
   st.write("---")
   if st.button("✨ 執行 AI 績效診斷", use_container_width=True):
       summary = df.groupby('Setup')['PnL'].sum().to_string()
       with st.spinner("AI 正在閱讀您的帳單..."):
           insight = fetch_ai_insight(summary)
           st.info(insight)
   # 3. 歷史紀錄回顧
   st.subheader("📋 交易流水帳")
   # 只顯示重要資訊，節省空間
   display_df = df[['Date', 'Symbol', 'Setup', 'PnL', 'Status']].sort_values(by='Date', ascending=False)
   st.dataframe(display_df, use_container_width=True, hide_index=True)
   # 4. 圖片查看器
   if st.checkbox("查看最近截圖"):
       has_img = df[df['Img'] != ""].tail(1)
       if not has_img.empty:
           st.image(has_img['Img'].values[0])
else:
   st.write("目前還沒有交易紀錄，請點開左側選單開始記錄！")