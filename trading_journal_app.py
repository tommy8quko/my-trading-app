import streamlit as st
import pandas as pd
import os
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

from supabase import create_client

@st.cache_resource
def get_supabase():
    """Supabase client"""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# Google Gemini AI 庫
import google.generativeai as genai

# 第三方備援庫 (OpenAI 兼容接口)
try:
    from openai import OpenAI
except ImportError:
    pass 


# --- 1. 核心配置與初始化 ---

USD_HKD_RATE = 7.8
INITIAL_CAPITAL = 1600000  # 初始本金 1.6M HKD

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

# --- AI 配置 (優化版：節省配額 + 雙引擎架構) ---

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
BACKUP_API_KEY = st.secrets.get("BACKUP_API_KEY", "") 
BACKUP_BASE_URL = st.secrets.get("BACKUP_BASE_URL", "https://api.deepseek.com") 

@st.cache_resource(ttl=3600, show_spinner=False)
def get_ai_model():
    """ 
    優化版初始化模型：
    1. 使用 @st.cache_resource 防止每次頁面刷新都消耗 API Quota。
    2. 優先使用免費額度高的模型。
    """
    if not GEMINI_API_KEY:
        return None, "未設定 API Key"
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    candidate_models = ['gemini-2.0-flash-lite', 
                        'gemini-1.5-flash', 
                        'gemini-1.5-pro',
    ]
    
    last_error = ""
    
    for model_name in candidate_models:
        try:
            m = genai.GenerativeModel(model_name)
            m.generate_content("ping", generation_config={"max_output_tokens": 1})
            return m, None
        except Exception as e:
            last_error = str(e)
            continue
            
    return None, last_error

model, init_error = get_ai_model()

def get_ai_response(prompt):
    """呼叫 Gemini API，如果失敗則嘗試備援"""
    if not GEMINI_API_KEY:
        return "⚠️ 請先在 Streamlit Secrets 設定 GEMINI_API_KEY。"
    
    if model:
        try:
            with st.spinner(f"🤖 AI 交易教練正在分析中 (Gemini)..."):
                response = model.generate_content(prompt)
                return response.text
        except Exception:
            pass
            
    if BACKUP_API_KEY:
        try:
            with st.spinner(f"⚠️ 切換至備援 AI 分析中..."):
                client = OpenAI(api_key=BACKUP_API_KEY, base_url=BACKUP_BASE_URL)
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                return f"🔄 [Backup AI] {response.choices[0].message.content}"
        except Exception as e:
            return f"❌ AI 分析失敗: {e}"
            
    return f"❌ 無法初始化 AI 模型或配額已滿。\nGemini 錯誤: {init_error}"

# --- ✅ 新增：用於 AI 匯出的持倉計算函數 ---
def get_hkd_value(symbol, value):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return value
    return value * USD_HKD_RATE

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return "HK$"
    return "$"
    
def calculate_position_percentage(active_pos, symbol, live_prices, current_equity):
    """
    計算該持倉佔整體帳戶百分比
    Returns: (市值HKD, 佔比%)
    """
    pos_data = active_pos[symbol]
    qty = pos_data['qty']
    current_price = live_prices.get(symbol)
    
    if not current_price:
        return 0, 0
    
    position_value_base = current_price * qty
    position_value_hkd = get_hkd_value(symbol, position_value_base)
    percentage = (position_value_hkd / current_equity) * 100 if current_equity > 0 else 0
    
    return position_value_hkd, percentage

# --- ✅ 修改：生成 AI 專用分析檔案（包含持倉詳情）---
def generate_llm_export_data(df, stats_summary, active_pos, live_prices, current_equity):
    """
    生成一個包含 Context + 統計 + 原始數據 + 持倉詳情 的文本，
    專門設計給外部 LLM (ChatGPT/Claude) 閱讀。
    """
    csv_data = df.to_csv(index=False)
    
    # ✅ 新增：生成持倉詳細列表
    active_positions_detail = "=== 📍 CURRENT ACTIVE POSITIONS ===\n"
    if active_pos:
        for s, d in active_pos.items():
            now = live_prices.get(s)
            qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
            un_pnl = (now - avg_p) * qty if now else 0
            un_pnl_hkd = get_hkd_value(s, un_pnl)
            
            pos_value_hkd, pos_pct = calculate_position_percentage(
                active_pos, s, live_prices, current_equity
            )
            
            active_positions_detail += f"""
symbol: {s}
  quantity: {qty:,.0f}
  Avg Entry: {avg_p:,.2f}
  Current price: {now:,.2f}
  Stop Loss: {last_sl:,.2f}
  Unrealized PnL (HKD): ${un_pnl_hkd:,.2f}
  Position Size %: {pos_pct:.2f}%
---"""
    else:
        active_positions_detail += "None\n"
    
    # 構建 Prompt 式的文本內容
    export_content = f"""
=== 🕵️‍♂️ AI TRADING JOURNAL REVIEW CONTEXT ===
You are an expert Trading Coach, Data Analyst, a panel of legendary stock traders Mark Minervini and David Ryan. The user has uploaded their trading journal data.
Your goal is to analyze this data to find patterns in their mistakes, evaluate their strategy performance, and suggest improvements. Be critical and direct.

=== 📊 CURRENT PERFORMANCE SUMMARY ===
- Total Realized PnL: {stats_summary.get('pnl_str', 'N/A')}
- Win Rate: {stats_summary.get('win_rate', 'N/A')}
- Profit Factor: {stats_summary.get('pf', 'N/A')}
- Expectancy (R): {stats_summary.get('exp_r', 'N/A')}
- Max Drawdown: {stats_summary.get('mdd', 'N/A')}
- Total Trades: {stats_summary.get('count', 'N/A')}
- Initial Capital: {INITIAL_CAPITAL} HKD
- Current Account Value: ${current_equity:,.0f} HKD

{active_positions_detail}

=== 📖 DATA DICTIONARY ===
- Trade_R: Risk multiple (Profit / Initial Risk). >1 is good, < -1 is bad risk management.
- Mistake_Tag: The specific error made (FOMO, Revenge Trade, etc.).
- Emotion: The psychological state at entry.
- Strategy: The setup used (Pullback, Breakout, etc.).

=== 📂 RAW TRADING LOG (CSV FORMAT) ===
{csv_data}

=== 📝 INSTRUCTIONS FOR AI ===
Please analyze the data above and provide:
1. A critique of the user's risk management based on 'Trade_R' and 'stop_loss'.
2. Correlation analysis: Which 'Emotion' or 'Mistake_Tag' leads to the biggest losses?
3. Strategy performance review: Which strategy is performing best?
4. Analysis of current open positions: Are they properly sized? Are the stop losses at risk?
5. Three actionable steps to improve profitability based on this specific data.
6. Answer in Traditional Chinese
"""
    return export_content

# --- 資料讀取層 ---

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
        client = get_supabase()
        result = client.table("trades").select("*").order("timestamp").execute()
        df = pd.DataFrame(result.data)
        
        if df.empty: return df
        
        # Your existing cleaning (unchanged)
        if 'symbol' in df.columns: df['symbol'] = df['symbol'].apply(format_symbol)
        if 'Strategy' in df.columns: df['Strategy'] = df['Strategy'].apply(clean_strategy)
        for col in ["Market_Condition", "Mistake_Tag", "Img", "Trade_ID"]:
            if col not in df.columns: df[col] = "N/A" if col != "Img" else None
        
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['stop_loss'] = pd.to_numeric(df['stop_loss'], errors='coerce').fillna(0)
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.to_datetime(df['date'], errors='coerce').view('int64') // 10**9

        
        return df
        
    except Exception as e:
        st.error(f"❌ Supabase load failed: {e}")
        return pd.DataFrame()

def save_all_data(df):
    try:
        client = get_supabase()
        client.table("trades").delete().neq("id", 0).execute()
        records = df.to_dict(orient="records")
        client.table("trades").insert(records).execute()
    except Exception as e:
        st.error(f"❌ Bulk save failed: {e}")

def save_transaction(data):
    try:
        client = get_supabase()
        client.table("trades").insert(data).execute()
        st.session_state['save_msg'] = {"type": "success", "msg": f"已儲存 {data.get('symbol', 'trade')}"}
    except Exception as e:
        st.session_state['save_msg'] = {"type": "error", "msg": f"❌ Save failed: {e}"}

def get_hkd_value(symbol, value):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return value
    return value * USD_HKD_RATE

def get_currency_symbol(symbol):
    if isinstance(symbol, str) and ".HK" in symbol.upper(): return "HK$"
    return "$"

# --- 2. 核心計算邏輯 ---

@st.cache_data(ttl=60)
def calculate_portfolio(df):
    if df.empty: 
        return {}, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 0, 0, 0, 0, 0
    
    positions = {} 
    df = df.sort_values(by="Timestamp")
    total_realized_pnl_hkd = 0
    running_pnl_hkd = 0
    
    cycle_tracker = {}
    active_trade_by_symbol = {}
    completed_trades = [] 
    equity_curve = []
    
    for _, row in df.iterrows():
        sym = format_symbol(row['symbol']) 
        action = str(row['action']) if pd.notnull(row['action']) else ""
        if not sym or not action: continue
        qty, price, sl = float(row['quantity']), float(row['price']), float(row['stop_loss'])
        date_str = row['date']
        
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
                    'Entry_price': price,
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
            cycle_data = cycle_tracker[current_trade_id]
            curr = positions[sym]
            
            sell_qty = min(qty, curr['qty'])
            cycle_data['cash_flow_raw'] += (sell_qty * price)
            
            realized_pnl_hkd_item = get_hkd_value(sym, (price - curr['avg_price']) * sell_qty)
            total_realized_pnl_hkd += realized_pnl_hkd_item
            running_pnl_hkd += realized_pnl_hkd_item
            
            curr['qty'] -= sell_qty
            if sl > 0: curr['last_sl'] = sl
            
            if curr['qty'] < 0.0001:
                pnl_raw = cycle_data['cash_flow_raw']
                init_risk = cycle_data['initial_risk_raw']
                trade_r = (pnl_raw / init_risk) if init_risk > 0 else None
                
                try:
                    duration = float((datetime.strptime(date_str, '%Y-%m-%d') - datetime.strptime(cycle_data['start_date'], '%Y-%m-%d')).days)
                except:
                    duration = 0
                
                completed_trades.append({
                    "Trade_ID": current_trade_id,
                    "Exit_date": date_str, 
                    "Entry_date": cycle_data['start_date'], 
                    "symbol": sym, 
                    "PnL_Raw": pnl_raw, 
                    "PnL_HKD": get_hkd_value(sym, pnl_raw),
                    "Duration_Days": duration, 
                    "Trade_R": trade_r,
                    "Strategy": cycle_data['Strategy'],
                    "Emotion": cycle_data['Emotion'],
                    "Market_Condition": cycle_data['Market_Condition'],
                    "Mistake_Tag": cycle_data['Mistake_Tag'],
                    "Notes": cycle_data.get('Notes', '')
                })
                del active_trade_by_symbol[sym]
                if sym in positions: del positions[sym]
            
            equity_curve.append({"date": date_str, "Cumulative PnL": running_pnl_hkd})
            
    comp_df = pd.DataFrame(completed_trades)
    active_output = {s: p for s, p in positions.items() if s in active_trade_by_symbol}
    
    for s, p in active_output.items():
        tid = active_trade_by_symbol[s]
        p['entry_price'] = cycle_tracker[tid]['Entry_price']
        p['entry_sl'] = cycle_tracker[tid]['Entry_SL']
    
    exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown = 0, 0, 0, 0, 0
    max_wins, max_losses = 0, 0
    avg_risk_per_trade = 0
    
    if not comp_df.empty:
        wins = comp_df[comp_df['PnL_HKD'] > 0]
        losses = comp_df[comp_df['PnL_HKD'] <= 0]
        
        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]
        if not valid_r_trades.empty:
            win_r_trades = valid_r_trades[valid_r_trades['Trade_R'] > 0]
            loss_r_trades = valid_r_trades[valid_r_trades['Trade_R'] <= 0]
            
            win_rate_r = len(win_r_trades) / len(valid_r_trades)
            avg_r_win = win_r_trades['Trade_R'].mean() if not win_r_trades.empty else 0
            avg_r_loss = abs(loss_r_trades['Trade_R'].mean()) if not loss_r_trades.empty else 0
            
            exp_r = (win_rate_r * avg_r_win) - ((1 - win_rate_r) * avg_r_loss)
        else:
            exp_r = 0
            
        wr = len(wins) / len(comp_df)
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)
        
        if avg_loss > 0:
            profit_loss_ratio = avg_win / avg_loss
        
        avg_dur = comp_df['Duration_Days'].mean()
        
        if equity_curve:
            eq_series = pd.DataFrame(equity_curve)['Cumulative PnL']
            rolling_max = eq_series.cummax()
            drawdown = eq_series - rolling_max
            max_drawdown = drawdown.min()
        
        if not comp_df.empty:
            comp_df_sorted = comp_df.sort_values('Exit_date').reset_index(drop=True)
            pnl_series = (comp_df_sorted['PnL_HKD'] > 0).astype(int)
            
            last_group = (pnl_series != pnl_series.shift()).cumsum().iloc[-1]
            current_streak_group = pnl_series.groupby((pnl_series != pnl_series.shift()).cumsum())
            current_streak = current_streak_group.last().iloc[-1]
            current_streak_length = len(current_streak_group.get_group(last_group))
            
            if current_streak == 1:
                max_wins = current_streak_length
                max_losses = 0
            else:
                max_losses = current_streak_length
                max_wins = 0
        else:
            max_wins, max_losses = 0, 0

        current_equity = INITIAL_CAPITAL + total_realized_pnl_hkd
        base_capital = current_equity if current_equity > 0 else INITIAL_CAPITAL
        
        comp_df['Risk_Per_Trade_Pct'] = (abs(comp_df['PnL_HKD']) / base_capital * 100)
        l_trades = comp_df[comp_df['PnL_HKD'] < 0]
        if not l_trades.empty:
            avg_risk_per_trade = l_trades['Risk_Per_Trade_Pct'].mean()
        
    return active_output, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown, max_wins, max_losses, avg_risk_per_trade

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
            except: prices[s] = None
        return prices
    except: return {}

# --- 3. UI 渲染 ---

df = load_data()

# Sidebar: Trade Form
with st.sidebar:
    st.header("⚡ 執行面板")

    active_pos_temp, realized_pnl_total_hkd_sb, _, _, _, _, _, _, _, _, _, _ = calculate_portfolio(df)
    current_equity_sb = INITIAL_CAPITAL + realized_pnl_total_hkd_sb
    if current_equity_sb <= 0: current_equity_sb = 1 

    if 'sb_qty' not in st.session_state: st.session_state.sb_qty = 0.0
    if 'sb_price' not in st.session_state: st.session_state.sb_price = 0.0
    if 'sb_sl' not in st.session_state: st.session_state.sb_sl = 0.0
    if 'sb_pos_pct' not in st.session_state: st.session_state.sb_pos_pct = 0.0
    if 'sb_risk_pct' not in st.session_state: st.session_state.sb_risk_pct = 0.0

    # --- update functions (unchanged) ---
    def update_pos_pct():
        """當 price 或 Qty 改變，更新 Pos% (考慮貨幣)"""
        try:
            symbol_val = st.session_state.sb_symbol.upper().strip()
            value_base = st.session_state.sb_price * st.session_state.sb_qty
            value_hkd = get_hkd_value(symbol_val, value_base)
            st.session_state.sb_pos_pct = (value_hkd / current_equity_sb) * 100
        except: pass

    def update_qty():
        """當 Pos% 改變，更新 Qty (考慮貨幣)"""
        try:
            symbol_val = st.session_state.sb_symbol.upper().strip()
            if st.session_state.sb_price > 0:
                val_hkd = current_equity_sb * (st.session_state.sb_pos_pct / 100)
                multiplier = 1.0 if ".HK" in symbol_val else USD_HKD_RATE
                val_base = val_hkd / multiplier
                st.session_state.sb_qty = val_base / st.session_state.sb_price
        except: pass

    def update_risk_pct():
        """當 price, Qty, 或 SL 改變，更新 Risk% (考慮貨幣)"""
        try:
            symbol_val = st.session_state.sb_symbol.upper().strip()
            risk_amt_base = abs(st.session_state.sb_price - st.session_state.sb_sl) * st.session_state.sb_qty
            risk_amt_hkd = get_hkd_value(symbol_val, risk_amt_base)
            st.session_state.sb_risk_pct = (risk_amt_hkd / current_equity_sb) * 100
        except: pass

    def update_sl():
        """當 Risk% 改變，更新 SL (考慮貨幣)"""
        try:
            symbol_val = st.session_state.sb_symbol.upper().strip()
            if st.session_state.sb_qty > 0:
                risk_amt_hkd = current_equity_sb * (st.session_state.sb_risk_pct / 100)
                multiplier = 1.0 if ".HK" in symbol_val else USD_HKD_RATE
                risk_amt_base = risk_amt_hkd / multiplier
                dist = risk_amt_base / st.session_state.sb_qty
                
                if st.session_state.sb_is_sell:
                    st.session_state.sb_sl = st.session_state.sb_price + dist
                else:
                    st.session_state.sb_sl = st.session_state.sb_price - dist
        except: pass

    def update_all_metrics():
        update_pos_pct()
        update_risk_pct()
    # ✅ FINAL FIXED: handle_save_transaction - 縮排完全正確 (4 spaces)
    def handle_save_transaction(active_pos_data):
        """儲存交易 + 正確處理 Supabase 錯誤"""
        s_in = format_symbol(st.session_state.sb_symbol.upper().strip())
        q_in = st.session_state.sb_qty
        p_in = st.session_state.sb_price
        sl_in = st.session_state.sb_sl
        is_sell = st.session_state.sb_is_sell
        act_in = "賣出 Sell" if is_sell else "買入 Buy"
        
        st_in = st.session_state.sb_strat
        if st_in == "➕ 新增...": 
            st_in = st.session_state.get('sb_strat_new', '')

        if not (s_in and q_in is not None and p_in is not None):
            st.session_state['save_msg'] = {"type": "error", "msg": "缺少必要欄位"}
            return

        # 決定 Trade_ID
        assigned_tid = "N/A"
        if not is_sell:  # Buy
            if s_in in active_pos_data:
                assigned_tid = active_pos_data[s_in]['trade_id']
            else:
                assigned_tid = int(time.time())
        else:  # Sell
            if s_in in active_pos_data:
                assigned_tid = active_pos_data[s_in]['trade_id']
            else:
                st.session_state['save_msg'] = {"type": "error", "msg": "找不到該標的的開倉紀錄，無法匹配 Trade_ID"}
                return

        img_path = None
        if st.session_state.sb_img is not None:
            img_path = f"chart_{int(time.time())}_{st.session_state.sb_img.name}"

        data = {
            "date": st.session_state.sb_date.strftime('%Y-%m-%d'), 
            "symbol": s_in, 
            "action": act_in, 
            "strategy": clean_strategy(st_in), 
            "price": p_in, 
            "quantity": q_in, 
            "stop_loss": sl_in if sl_in is not None else 0.0, 
            "fees": 0, 
            "emotion": st.session_state.sb_emo, 
            "risk_reward": 0, 
            "notes": st.session_state.sb_note, 
            "market_condition": st.session_state.sb_mkt, 
            "mistake_tag": st.session_state.sb_mistake,
            "img": img_path, 
            "trade_id": assigned_tid,
        }

        # 真正的 Supabase 呼叫
        save_transaction(data)

        # 清空表單
        st.session_state.sb_price = 0.0
        st.session_state.sb_qty = 0.0
        st.session_state.sb_sl = 0.0
        st.session_state.sb_pos_pct = 0.0
        st.session_state.sb_risk_pct = 0.0
        st.session_state.sb_note = ""
        if 'sb_img' in st.session_state:
            st.session_state.sb_img = None
def close_position_at_stop_loss(symbol, active_pos_data):
    """Close the entire position at the current Stop Loss price"""
    if symbol not in active_pos_data:
        st.error(f"找不到持倉: {symbol}")
        return
    
    pos = active_pos_data[symbol]
    sl_price = pos.get('last_sl', 0)
    
    if sl_price <= 0:
        st.error(f"{symbol} 沒有設定止損價，無法執行止損平倉")
        return
    
    # Prepare sell transaction data
    data = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "symbol": symbol,
        "action": "賣出 Sell (止損)",
        "strategy": pos.get('Strategy', 'Stop Loss'),
        "price": sl_price,
        "quantity": pos['qty'],
        "stop_loss": sl_price,
        "fees": 0,
        "emotion": "止損執行",
        "risk_reward": 0,
        "notes": f"自動止損平倉 @ {sl_price}",
        "market_condition": "N/A",
        "mistake_tag": "Stop Loss Hit",
        "img": None,
        "trade_id": pos.get('trade_id', f"SL_{int(time.time())}")
    }
    
    save_transaction(data)
    st.success(f"✅ 已執行止損平倉：{symbol} @ {sl_price:,.2f} (全數賣出)")
    st.rerun()
        # 決定 Trade_ID
    assigned_tid = "N/A"
    if not is_sell:
            if s_in in active_pos_data:
                assigned_tid = active_pos_data[s_in]['trade_id']
            else:
                assigned_tid = int(time.time())
        else:
            if s_in in active_pos_data:
                assigned_tid = active_pos_data[s_in]['trade_id']
        else:
            st.session_state['save_msg'] = {"type": "error", "msg": "找不到該標的的開倉紀錄，無法匹配 Trade_ID"}
            return

        img_path = None
        if st.session_state.sb_img is not None:
            img_path = f"chart_{int(time.time())}_{st.session_state.sb_img.name}"

        data = {
            "date": st.session_state.sb_date.strftime('%Y-%m-%d'), 
            "symbol": s_in, 
            "action": act_in, 
            "strategy": clean_strategy(st_in), 
            "price": p_in, 
            "quantity": q_in, 
            "stop_loss": sl_in if sl_in is not None else 0.0, 
            "fees": 0, 
            "emotion": st.session_state.sb_emo, 
            "risk_reward": 0, 
            "notes": st.session_state.sb_note, 
            "market_condition": st.session_state.sb_mkt, 
            "mistake_tag": st.session_state.sb_mistake,
            "img": img_path, 
            "trade_id": assigned_tid,
            # Remove the manual timestamp line completely
            # PostgreSQL will use DEFAULT now()
        }

        # 真正的 Supabase 呼叫
        save_transaction(data)

        # 清空表單
        st.session_state.sb_price = 0.0
        st.session_state.sb_qty = 0.0
        st.session_state.sb_sl = 0.0
        st.session_state.sb_pos_pct = 0.0
        st.session_state.sb_risk_pct = 0.0
        st.session_state.sb_note = ""

    # --- 以下為 Sidebar UI 表單（縮排必須與上面的 def 對齊）---
    d_in = st.date_input("日期", value=datetime.now(), key='sb_date')
    s_in = st.text_input("代號 (Ticker)", key='sb_symbol')
    is_sell_toggle = st.toggle("Buy 🟢 / Sell 🔴", value=False, key='sb_is_sell', on_change=update_sl)

    col1, col2 = st.columns(2)
    q_in = col1.number_input("股數 (Qty)", min_value=0.0, step=100.0, key='sb_qty', on_change=update_all_metrics)
    p_in = col2.number_input("成交價格 (price)", min_value=0.0, step=0.05, key='sb_price', on_change=update_all_metrics)

    sl_in = st.number_input("停損價格 (Stop Loss)", min_value=0.0, step=0.05, key='sb_sl', on_change=update_risk_pct)
        
    st.divider()
    pos_pct_in = st.number_input("該筆交易佔整體倉位的 %", min_value=0.0, max_value=100.0, step=1.0, key='sb_pos_pct', on_change=update_qty)  

    risk_pct_in = st.number_input("停損幅度佔整體倉位的 %", min_value=0.0, max_value=100.0, step=0.1, key='sb_risk_pct', on_change=update_sl)
    st.divider()

    mkt_cond = st.selectbox("市場環境", ["Trending Up", "Trending Down", "Range/Choppy", "High Volatility", "N/A"], key='sb_mkt')
    mistake_in = st.selectbox("錯誤標籤", ["None", "Fomo", "Revenge Trade", "Fat Finger", "Late Entry", "Moved Stop"], key='sb_mistake')
    st_in = st.selectbox("策略 (Strategy)", ["Pullback", "Breakout", "➕ 新增..."], key='sb_strat')
    if st_in == "➕ 新增...": 
        st.text_input("輸入新策略名稱", key='sb_strat_new')
    
    emo_in = st.select_slider("心理狀態", options=["恐慌", "猶豫", "平靜", "自信", "衝動"], value="平靜", key='sb_emo')
    note_in = st.text_area("決策筆記", key='sb_note')
    img_file = st.file_uploader("📸 上傳圖表截圖", type=['png','jpg','jpeg'], key='sb_img')

    st.button("儲存執行紀錄", type="primary", use_container_width=True, 
              on_click=handle_save_transaction, args=(active_pos_temp,))

    if 'save_msg' in st.session_state:
        msg = st.session_state.pop('save_msg')
        if msg['type'] == 'success':
            st.success(msg['msg'])
        else:
            st.error(msg['msg'])


active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val, pl_ratio_val, mdd_val, max_wins_val, max_losses_val, avg_risk_val = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    
    c_header, c_toggle = st.columns([5, 2])
    with c_header:
        st.subheader("📊 績效概覽")
        time_frame = st.selectbox("統計時間範圍", ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"], index=0)
    with c_toggle:
        st.write("")
        st.write("") 
        private_mode = st.toggle("🙈 隱私模式", value=False, help="隱藏敏感金額數據，適合公開展示")

    filtered_comp = completed_trades_df.copy()
    if not filtered_comp.empty:
        filtered_comp['Entry_DT'] = pd.to_datetime(filtered_comp['Entry_date'])
        filtered_comp['Exit_DT'] = pd.to_datetime(filtered_comp['Exit_date'])
        today = datetime.now()
        
        if "今年" in time_frame:
            mask = (filtered_comp['Exit_DT'].dt.year == today.year)
        elif "本月" in time_frame:
            mask = (filtered_comp['Exit_DT'].dt.year == today.year) & (filtered_comp['Exit_DT'].dt.month == today.month)
        elif "本週" in time_frame: 
            start_week = today - timedelta(days=today.weekday())
            mask = (filtered_comp['Exit_DT'] >= start_week)
        elif "3個月" in time_frame: 
            cutoff = today - timedelta(days=90)
            mask = (filtered_comp['Exit_DT'] >= cutoff)
        else: mask = [True] * len(filtered_comp)
        filtered_comp = filtered_comp[mask]
    
    f_pnl = filtered_comp['PnL_HKD'].sum() if not filtered_comp.empty else 0
    trade_count = len(filtered_comp)
    win_r = (len(filtered_comp[filtered_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0
    
    live_prices = get_live_prices(list(active_pos.keys()))
    potential_stop_loss_impact = 0
    for s, d in active_pos.items():
        curr_price = live_prices.get(s)
        if curr_price and d['last_sl'] > 0:
            impact = (curr_price - d['last_sl']) * d['qty']
            potential_stop_loss_impact += get_hkd_value(s, impact)
    
    mask_val = lambda v, fmt: "****" if private_mode else fmt.format(v)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("已實現損益 (HKD)", mask_val(f_pnl, "${:,.2f}"))
    m2.metric("期望值 (R)", f"{exp_r_val:.2f}R", help="修正公式：(勝率 x 平均贏R) - (敗率 x 平均輸R)")
    m3.metric("勝率", f"{win_r:.1f}%")
    m4.metric("盈虧比", f"{pl_ratio_val:.2f}")
    m5.metric("最大回撤", mask_val(mdd_val, "${:,.0f}"), delta_color="inverse")
    m6.metric("交易場數", f"{trade_count}")
    
    st.divider()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("若全體止損回撤", mask_val(potential_stop_loss_impact, "-${:,.0f}"), delta_color="inverse", help="若所有當前持倉立刻打到止損價，帳戶市值將減少的金額")
    if max_wins_val > 0:
        k2.metric("🔥 連勝狀態", f"🔥{max_wins_val} ")
    elif max_losses_val > 0:
        k2.metric("🧊 連敗狀態", f"🧊{max_losses_val}")
    else:
        k2.metric("交易狀態", "無連續紀錄")

    k3.metric("平均單筆風險 %", f"{avg_risk_val:.2f}%", help="平均每筆虧損單佔當時本金的百分比 (建議控制在 1-2%)")
    k4.metric("目前帳戶預估", mask_val(INITIAL_CAPITAL + realized_pnl_total_hkd, "${:,.0f}"))
    
    if not equity_df.empty:
        fig_equity = px.area(equity_df, x="date", y="Cumulative PnL", title="累計損益曲線")
        if private_mode:
            fig_equity.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_equity, use_container_width=True)
    
    st.divider()
    st.subheader("🤖 AI 交易教練洞察")
    if st.button("生成本期 AI 檢討報告"):
        if filtered_comp.empty:
            st.warning("目前無已平倉數據供 AI 分析。")
        else:
            stats = {
                "PnL": f_pnl, 
                "WinRate": f"{win_r:.1f}%",
                "ExpR": exp_r_val, 
                "Mistakes": filtered_comp['Mistake_Tag'].value_counts().to_dict(),
                "ConsecutiveLosses": max_losses_val
            }
            prompt = f"請根據以下交易統計給出深度專業建議：{stats}。請分析錯誤標籤，並給出三個下週改進動作。請用繁體中文，語氣要像專業交易導師。"
            st.markdown(get_ai_response(prompt))
            
    if not filtered_comp.empty:
        st.divider()
        st.subheader("🏆 週期成交排行榜")
        display_trades = filtered_comp.copy()
        display_trades['原始損益'] = display_trades.apply(lambda x: mask_val(x['PnL_Raw'], "{} {:,.2f}".format(get_currency_symbol(x['symbol']), x['PnL_Raw'])) if not private_mode else "****", axis=1)
        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: mask_val(x, "${:,.2f}"))
        display_trades['R 乘數'] = display_trades['Trade_R'].apply(lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A")
        display_trades = display_trades.rename(columns={"Exit_date": "出場日期", "symbol": "代號"})
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### 🟢 Top 獲利")
            st.dataframe(display_trades.sort_values(by="PnL_HKD", ascending=False).head(5)[['出場日期', '代號', '原始損益', 'HKD 損益', 'R 乘數']], hide_index=True, use_container_width=True)
        with r2:
            st.markdown("##### 🔴 Top 虧損")
            st.dataframe(display_trades.sort_values(by="PnL_HKD", ascending=True).head(5)[['出場日期', '代號', '原始損益', 'HKD 損益', 'R 乘數']], hide_index=True, use_container_width=True)

with t2:
    st.markdown("### 🟢 持倉概覽")
         
    if active_pos:
        live_prices = get_live_prices(list(active_pos.keys()))
        processed_p_data = []
        
        total_position_value_hkd = 0
        
        for s, d in active_pos.items():
            now = live_prices.get(s)
            qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
            entry_p, entry_sl = d.get('entry_price', avg_p), d.get('entry_sl', 0)
            
            un_pnl = (now - avg_p) * qty if now else 0
            un_pnl_hkd = get_hkd_value(s, un_pnl)
            roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
            
            init_risk = abs(entry_p - entry_sl) * qty if entry_sl > 0 else 0
            init_risk_hkd = get_hkd_value(s, init_risk)
            curr_risk = (now - last_sl) * qty if (now and last_sl > 0) else 0
            curr_risk_hkd = get_hkd_value(s, curr_risk)
            curr_r = (un_pnl_hkd / init_risk_hkd) if (now and init_risk_hkd > 0) else 0
            
            # Calculate position percentage
            pos_value_hkd, pos_pct = calculate_position_percentage(
                active_pos, s, live_prices, 
                INITIAL_CAPITAL + realized_pnl_total_hkd
            )
            total_position_value_hkd += pos_value_hkd
            
            processed_p_data.append({
                "代號": s, 
                "持股數": f"{qty:,.0f}", 
                "平均成本": f"{avg_p:,.2f}", 
                "現價": f"{now:,.2f}" if now else "N/A", 
                "當前止損": f"{last_sl:,.2f}", 
                "初始風險": f"{init_risk_hkd:,.2f}",
                "當前風險(Open)": f"{curr_risk_hkd:,.2f}",
                "當前R": f"{curr_r:.2f}R",
                "未實現損益(HKD)": f"{un_pnl_hkd:,.2f}", 
                "報酬%": roi,
                "佔整體帳戶%": f"{pos_pct:.2f}%",
                "操作": s   # We will use this for the button later
            })
        
        # Create DataFrame
        pos_df = pd.DataFrame(processed_p_data)
        
        # Display with custom column config
        st.dataframe(
            pos_df.drop(columns=["操作"]),  # Hide the helper column
            column_config={
                "報酬%": st.column_config.ProgressColumn(
                    "報酬%", format="%.2f%%", min_value=-20, max_value=20, color="green"
                ),
                "佔整體帳戶%": st.column_config.ProgressColumn(
                    "佔整體帳戶%", format="%.2f%%", min_value=0, max_value=100
                )
            }, 
            hide_index=True, 
            use_container_width=True
        )
        
        # === NEW: Add Stop Loss buttons for each position ===
        st.divider()
        st.markdown("### ⚠️ 快速止損平倉")
        st.caption("點擊「❌ 止損平倉」將以當前止損價全數賣出該持倉")
        
        cols = st.columns(len(active_pos)) if len(active_pos) <= 4 else st.columns(4)
        
        for idx, (symbol, pos_data) in enumerate(active_pos.items()):
            col = cols[idx % len(cols)]
            with col:
                sl_price = pos_data.get('last_sl', 0)
                current_price = live_prices.get(symbol)
                
                st.markdown(f"**{symbol}**")
                st.caption(f"止損價: {sl_price:,.2f} | 現價: {current_price:,.2f}" if current_price else f"止損價: {sl_price:,.2f}")
                
                if st.button(f"❌ 止損平倉", key=f"sl_btn_{symbol}", type="secondary"):
                    close_position_at_stop_loss(symbol, active_pos)
        
        # Summary section remains the same
        st.divider()
        current_account_value = INITIAL_CAPITAL + realized_pnl_total_hkd
        total_pos_pct = (total_position_value_hkd / current_account_value) * 100 if current_account_value > 0 else 0
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        col_summary1.metric("總持倉市值 (HKD)", f"${total_position_value_hkd:,.0f}")
        col_summary2.metric("總倉位佔比", f"{total_pos_pct:.2f}%")
        col_summary3.metric("帳戶現金", f"${current_account_value - total_position_value_hkd:,.0f}")
        
        if st.button("🔄 刷新即時報價", use_container_width=True): 
            st.cache_data.clear()
            st.rerun()
            
    else:
        st.info("目前無持倉部位")

with t3:
    st.subheader("⏪ 交易重播")
    if not df.empty:
        target = st.selectbox("選擇交易", df.index, format_func=lambda x: f"[{df.iloc[x]['date']}] {df.iloc[x]['symbol']}")
        row = df.iloc[target]
        data = yf.download(row['symbol'], start=(pd.to_datetime(row['date']) - timedelta(days=20)).strftime('%Y-%m-%d'), progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='價格')])
            fig.add_trace(go.Scatter(x=[pd.to_datetime(row['date'])], y=[row['price']], mode='markers+text', marker=dict(size=15, color='orange', symbol='star'), text=["執行"], textposition="top center"))
            fig.update_layout(title=f"{row['symbol']} K線圖回顧", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        if st.button("🤖 AI 單筆深度診斷"):
            prompt = f"請檢討這筆交易：代號 {row['symbol']}, 進場 {row['price']}, 策略 {row['Strategy']}, 情緒 {row['Emotion']}, 錯誤 {row['Mistake_Tag']}。請評估其進場合理性。"
            st.markdown(get_ai_response(prompt))

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        
        st.markdown("#### 🚨 錯誤代價分析 (Cost of Mistakes)")
        mistake_impact = completed_trades_df.groupby('Mistake_Tag').agg({
            'PnL_HKD': ['sum', 'count'],
            'Trade_R': 'mean'
        }).reset_index()
        mistake_impact.columns = ['錯誤類型', '總虧損(HKD)', '次數', '平均R']
        mistake_impact['總虧損(HKD)'] = mistake_impact['總虧損(HKD)'].round(0)
        mistake_impact['平均R'] = mistake_impact['平均R'].round(2)
        
        c_mis1, c_mis2 = st.columns([1, 2])
        with c_mis1:
            st.dataframe(mistake_impact.sort_values('總虧損(HKD)'), hide_index=True, use_container_width=True)
        with c_mis2:
             st.plotly_chart(px.bar(mistake_impact, x='錯誤類型', y='總虧損(HKD)', color='總虧損(HKD)', color_continuous_scale='RdYlGn', title="哪種錯誤最燒錢？"), use_container_width=True)

        st.divider()

        c_st1, c_st2 = st.columns(2)
        
        with c_st1:
            st.markdown("#### ⚔️ 策略勝率 (Strategy Breakdown)")
            strat_stats = completed_trades_df.groupby('Strategy').agg({
                'PnL_HKD': 'sum',
                'Trade_R': 'mean',
                'symbol': 'count'
            }).reset_index().rename(columns={'symbol': '次數', 'PnL_HKD': '總損益'})
            strat_stats['總損益'] = strat_stats['總損益'].round(0)
            strat_stats['Trade_R'] = strat_stats['Trade_R'].round(2)
            st.dataframe(strat_stats.sort_values('Trade_R', ascending=False), hide_index=True, use_container_width=True)

        with c_st2:
            st.markdown("#### 🌊 市場環境適應性 (Market Condition)")
            mkt_stats = completed_trades_df.groupby('Market_Condition').agg({
                'PnL_HKD': 'sum',
                'Trade_R': 'mean'
            }).reset_index()
            mkt_stats['PnL_HKD'] = mkt_stats['PnL_HKD'].round(0)
            mkt_stats['Trade_R'] = mkt_stats['Trade_R'].round(2)
            st.dataframe(mkt_stats.sort_values('Trade_R', ascending=False), hide_index=True, use_container_width=True)

        st.divider()

        st.markdown("#### ⏳ 持倉時間與獲利關係 (Time vs PnL)")
        dur_bins = [0, 1, 5, 20, 100, 999]
        dur_labels = ['當沖 (0-1天)', '短線 (2-5天)', '波段 (6-20天)', '長波段 (20-100天)', '長線 (>100天)']
        
        temp_df = completed_trades_df.copy()
        temp_df['Duration_Bin'] = pd.cut(temp_df['Duration_Days'], bins=dur_bins, labels=dur_labels, right=True)
        
        dur_stats = temp_df.groupby('Duration_Bin', observed=True)['PnL_HKD'].agg(['sum', 'mean', 'count']).reset_index()
        dur_stats.columns = ['持有週期', '總損益', '平均損益', '次數']
        dur_stats['總損益'] = dur_stats['總損益'].round(0)
        dur_stats['平均損益'] = dur_stats['平均損益'].round(0)
        
        st.plotly_chart(px.bar(dur_stats, x='持有週期', y='總損益', color='總損益', title="不同週期的獲利表現", color_continuous_scale='RdYlGn'), use_container_width=True)

    if not df.empty:
        st.divider()
        hist_df = df.sort_values("timestamp", ascending=False).copy()   # ← changed to lowercase
        
        # Safe column list - only show columns that actually exist
        desired_cols = ["date", "symbol", "action", "trade_id", "price", 
                       "quantity", "stop_loss", "emotion", "mistake_tag", "img"]
        
        # Use only columns that exist in the DataFrame
        available_cols = [col for col in desired_cols if col in hist_df.columns]
        
        # Rename for nicer display (optional but recommended)
        display_df = hist_df[available_cols].copy()
        display_df = display_df.rename(columns={
            "date": "日期",
            "symbol": "代號",
            "action": "買賣",
            "trade_id": "Trade_ID",
            "price": "價格",
            "quantity": "股數",
            "stop_loss": "停損",
            "emotion": "情緒",
            "mistake_tag": "錯誤標籤",
            "img": "截圖"
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    st.success("🟢 已連接至 Supabase (永久儲存)")
    
    st.divider()
    st.markdown("#### 🤖 匯出給 AI 分析 (Export for LLM)")
    st.info("下載此檔案後，直接上傳給 ChatGPT / Claude / DeepSeek，它們會自動為您進行全方位帳戶診斷。")
    
    if not df.empty:
        live_prices_export = get_live_prices(list(active_pos.keys())) if active_pos else {}
        current_equity_export = INITIAL_CAPITAL + realized_pnl_total_hkd
        
        export_stats = {
            "pnl_str": f"${realized_pnl_total_hkd:,.2f}",
            "win_rate": f"{(len(completed_trades_df[completed_trades_df['PnL_HKD'] > 0])/len(completed_trades_df)*100):.1f}%" if not completed_trades_df.empty else "N/A",
            "pf": f"{pl_ratio_val:.2f}",
            "exp_r": f"{exp_r_val:.2f}R",
            "mdd": f"${mdd_val:,.0f}",
            "count": len(completed_trades_df)
        }
        
        # ✅ 修改：新增參數以包含持倉詳情
        export_text = generate_llm_export_data(
            df, export_stats, active_pos, 
            live_prices_export, current_equity_export
        )
        
        st.download_button(
            label="📥 下載 AI 專用分析報告 (.txt)",
            data=export_text,
            file_name=f"TradeMaster_AI_Review_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    else:
        st.caption("尚無交易紀錄可供匯出。")
    
    st.divider()
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_file = st.file_uploader("📤 批量上傳 CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file and st.button("🚀 開始匯入"):
            try:
                new_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'symbol' in new_data.columns: new_data['symbol'] = new_data['symbol'].apply(format_symbol)
                if 'Timestamp' not in new_data.columns: new_data['Timestamp'] = int(time.time())
                df = pd.concat([df, new_data], ignore_index=True); save_all_data(df)
                st.success("匯入成功！"); st.rerun()
            except Exception as e: st.error(f"匯入失敗: {e}")
    
    if not df.empty:
        st.divider()
        selected_idx = st.selectbox("選擇紀錄進行編輯", df.index, format_func=lambda x: f"[{df.loc[x, 'date']}] {df.loc[x, 'symbol']} ({df.loc[x, 'action']})")
        t_edit = df.loc[selected_idx]
        e1, e2, e3 = st.columns(3)
        n_p = e1.number_input("編輯價格", value=float(t_edit['price']), key=f"ep_{selected_idx}")
        n_q = e2.number_input("編輯股數", value=float(t_edit['quantity']), key=f"eq_{selected_idx}")
        n_sl = e3.number_input("編輯止損價", value=float(t_edit['stop_loss']), key=f"esl_{selected_idx}")
        
        b1, b2 = st.columns(2)
        if b1.button("💾 儲存修改", use_container_width=True):
            df.loc[selected_idx, ['price', 'quantity', 'stop_loss']] = [n_p, n_q, n_sl]
            save_all_data(df); st.success("已更新"); st.rerun()
        if b2.button("🗑️ 刪除此筆紀錄", use_container_width=True):
            df = df.drop(selected_idx).reset_index(drop=True)
            save_all_data(df); st.rerun()
            
    st.divider()
    st.markdown("#### 🚨 危險區域")
    confirm_delete = st.checkbox("我了解此操作將永久刪除所有交易紀錄且無法復原")
    if st.button("🚨 清空所有數據", type="primary", disabled=not confirm_delete, use_container_width=True):
        save_all_data(pd.DataFrame(columns=df.columns))
        st.success("數據已清空")
        st.rerun()
