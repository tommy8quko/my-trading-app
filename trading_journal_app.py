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
# 新增 Google Gemini AI 庫
import google.generativeai as genai

# --- 1. 核心配置與初始化 ---
FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

# --- AI 配置 ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_response(prompt):
    """呼叫 Gemini API 獲取分析結果"""
    if not GEMINI_API_KEY:
        return "⚠️ 請先在 Secrets 設定 GEMINI_API_KEY 才能使用 AI 功能。"
    try:
        with st.spinner("🤖 AI 交易教練正在分析數據中..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ AI 分析失敗: {str(e)}"

# --- 資料讀取層 ---
def get_data_connection():
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except:
        return None

def init_csv():
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "Date", "Symbol", "Action", "Strategy", "Price", "Quantity", 
            "Stop_Loss", "Fees", "Emotion", "Risk_Reward", "Notes", "Img", "Timestamp",
            "Market_Condition", "Mistake_Tag", "Trade_ID"
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
    
    try:
        if conn:
            df = conn.read(worksheet="Log", ttl=0) 
        else:
            raise Exception("No connection")
    except:
        init_csv()
        try:
            df = pd.read_csv(FILE_NAME)
        except:
            return pd.DataFrame()

    if df.empty: return df
    
    # 數據類型轉換
    if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].apply(format_symbol)
    if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
    for col in ["Market_Condition", "Mistake_Tag", "Img", "Trade_ID"]:
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
        if conn:
            conn.update(worksheet="Log", data=df)
        else:
            raise Exception("No connection")
    except:
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
@st.cache_data(ttl=60)
def calculate_portfolio(df):
    if df.empty: return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 0, 0
    
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    
    cycle_tracker = {} # Key: Trade_ID
    active_trade_by_symbol = {} # Key: Symbol, Value: Trade_ID
    completed_trades = [] 
    equity_curve = []

    for _, row in df.iterrows():
        sym = format_symbol(row['Symbol']) 
        action = str(row['Action']) if pd.notnull(row['Action']) else ""
        if not sym or not action: continue

        qty, price, sl = float(row['Quantity']), float(row['Price']), float(row['Stop_Loss'])
        date_str = row['Date']
        
        t_id = row.get('Trade_ID')
        if pd.isna(t_id) or t_id == "N/A":
            t_id = f"LEGACY_{sym}" 

        is_buy = any(word in action.upper() for word in ["買入", "BUY", "B"])
        is_sell = any(word in action.upper() for word in ["賣出", "SELL", "S"])

        current_trade_id = None
        if is_buy:
            if sym in active_trade_by_symbol:
                current_trade_id = active_trade_by_symbol[sym]
            else:
                current_trade_id = t_id
                active_trade_by_symbol[sym] = current_trade_id
                
            if current_trade_id not in cycle_tracker:
                cycle_tracker[current_trade_id] = {
                    'symbol': sym,
                    'cash_flow_raw': 0.0, 
                    'start_date': date_str, 
                    'initial_risk_raw': 0.0,
                    'Entry_Price': price,
                    'Entry_SL': sl,
                    'qty_accumulated': 0.0,
                    'Strategy': row.get('Strategy', ''),
                    'Emotion': row.get('Emotion', ''),
                    'Market_Condition': row.get('Market_Condition', ''),
                    'Mistake_Tag': row.get('Mistake_Tag', ''),
                    'Notes': row.get('Notes', '')
                }
                if sl > 0:
                    cycle_tracker[current_trade_id]['initial_risk_raw'] = abs(price - sl) * qty
                
            if sym not in positions:
                positions[sym] = {'qty': 0.0, 'avg_price': 0.0, 'last_sl': 0.0, 'trade_id': current_trade_id}
            
            curr = positions[sym]
            cycle_tracker[current_trade_id]['cash_flow_raw'] -= (qty * price)
            cycle_tracker[current_trade_id]['qty_accumulated'] += qty
            
            total_cost_base = (curr['qty'] * curr['avg_price']) + (qty * price)
            curr['qty'] += qty
            if curr['qty'] > 0: curr['avg_price'] = total_cost_base / curr['qty']
            if sl > 0: curr['last_sl'] = sl

        elif is_sell and sym in active_trade_by_symbol:
            current_trade_id = active_trade_by_symbol[sym]
            cycle_data = cycle_tracker[current_trade_id
