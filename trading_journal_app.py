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

# Google Sheets 連線庫
try:
    from streamlit_gsheets import GSheetsConnection
except ImportError:
    st.error("⚠️ 缺少 streamlit_gsheets 庫。請執行 `pip install st-gsheets-connection`")

# Google Gemini AI 庫
import google.generativeai as genai


# --- 1. 核心配置與初始化 ---

FILE_NAME = "trade_ledger_v_final.csv"
USD_HKD_RATE = 7.8

if not os.path.exists("images"):
    os.makedirs("images")

st.set_page_config(page_title="TradeMaster Pro UI", layout="wide")

# --- AI 配置 (優化版：節省配額 + 雙引擎架構) ---

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
# 如果你有備援的 Key (例如 DeepSeek 或 Groq)，可以設定在這裡，否則留空
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
    
    # 策略：包含 1.5-flash 以確保穩定
    candidate_models = ['gemini-3-flash-preview',
                        'gemini-2.5-flash',
                        'gemini-2.5-flash-lite',
                        'gemini-2.0-flash-lite',
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

# 初始化模型
model, init_error = get_ai_model()

def get_ai_response(prompt):
    """呼叫 Gemini API，如果失敗則嘗試備援"""
    if not GEMINI_API_KEY:
        return "⚠️ 請先在 Streamlit Secrets 設定 GEMINI_API_KEY。"
    
    # 1. 嘗試 Gemini
    if model:
        try:
            with st.spinner(f"🤖 AI 交易教練正在分析中 (Gemini)..."):
                response = model.generate_content(prompt)
                return response.text
        except Exception:
            pass # 失敗則進入備援
            
    # 2. 嘗試備援 (如果有的話)
    if BACKUP_API_KEY:
        try:
            with st.spinner(f"⚠️ 切換至備援 AI 分析中..."):
                client = OpenAI(api_key=BACKUP_API_KEY, base_url=BACKUP_BASE_URL)
                response = client.chat.completions.create(
                    model="deepseek-chat", # 或 gpt-4o-mini, llama-3 等
                    messages=[{"role": "user", "content": prompt}]
                )
                return f"🔄 [Backup AI] {response.choices[0].message.content}"
        except Exception as e:
            return f"❌ AI 分析失敗: {e}"
            
    return f"❌ 無法初始化 AI 模型或配額已滿。\nGemini 錯誤: {init_error}"

# --- 新增功能：生成 AI 專用分析檔案 ---
def generate_llm_export_data(df, stats_summary):
    """
    生成一個包含 Context + 統計 + 原始數據的文本，
    專門設計給外部 LLM (ChatGPT/Claude) 閱讀。
    """
    csv_data = df.to_csv(index=False)
    
    # 構建 Prompt 式的文本內容
    export_content = f"""
=== 🕵️‍♂️ AI TRADING JOURNAL REVIEW CONTEXT ===
You are an expert Trading Coach and Data Analyst. The user has uploaded their trading journal data.
Your goal is to analyze this data to find patterns in their mistakes, evaluate their strategy performance, and suggest improvements.

=== 📊 CURRENT PERFORMANCE SUMMARY ===
- Total Realized PnL: {stats_summary.get('pnl_str', 'N/A')}
- Win Rate: {stats_summary.get('win_rate', 'N/A')}
- Profit Factor: {stats_summary.get('pf', 'N/A')}
- Expectancy (R): {stats_summary.get('exp_r', 'N/A')}
- Max Drawdown: {stats_summary.get('mdd', 'N/A')}
- Total Trades: {stats_summary.get('count', 'N/A')}

=== 📖 DATA DICTIONARY ===
- Trade_R: Risk multiple (Profit / Initial Risk). >1 is good, < -1 is bad risk management.
- Mistake_Tag: The specific error made (FOMO, Revenge Trade, etc.).
- Emotion: The psychological state at entry.
- Strategy: The setup used (Pullback, Breakout, etc.).

=== 📂 RAW TRADING LOG (CSV FORMAT) ===
{csv_data}

=== 📝 INSTRUCTIONS FOR AI ===
Please analyze the data above and provide:
1. A critique of the user's risk management based on 'Trade_R' and 'Stop_Loss'.
2. Correlation analysis: Which 'Emotion' or 'Mistake_Tag' leads to the biggest losses?
3. Strategy performance review: Which strategy is performing best?
4. Three actionable steps to improve profitability based on this specific data.
"""
    return export_content

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
                    "Exit_Date": date_str, 
                    "Entry_Date": cycle_data['start_date'], 
                    "Symbol": sym, 
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
            
            equity_curve.append({"Date": date_str, "Cumulative PnL": running_pnl_hkd})
    comp_df = pd.DataFrame(completed_trades)
    active_output = {s: p for s, p in positions.items() if s in active_trade_by_symbol}
    for s, p in active_output.items():
        tid = active_trade_by_symbol[s]
        p['entry_price'] = cycle_tracker[tid]['Entry_Price']
        p['entry_sl'] = cycle_tracker[tid]['Entry_SL']
    exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown = 0, 0, 0, 0, 0
    if not comp_df.empty:
        wins, losses = comp_df[comp_df['PnL_HKD'] > 0], comp_df[comp_df['PnL_HKD'] <= 0]
        wr = len(wins) / len(comp_df)
        avg_win = wins['PnL_HKD'].mean() if not wins.empty else 0
        avg_loss = abs(losses['PnL_HKD'].mean()) if not losses.empty else 0
        exp_hkd = (wr * avg_win) - ((1-wr) * avg_loss)
        
        if avg_loss > 0:
            profit_loss_ratio = avg_win / avg_loss
        
        valid_r_trades = comp_df[comp_df['Trade_R'].notna()]
        exp_r = valid_r_trades['Trade_R'].mean() if not valid_r_trades.empty else 0
        avg_dur = comp_df['Duration_Days'].mean()
        
        if equity_curve:
            eq_series = pd.DataFrame(equity_curve)['Cumulative PnL']
            rolling_max = eq_series.cummax()
            drawdown = eq_series - rolling_max
            max_drawdown = drawdown.min()
    return active_output, total_realized_pnl_hkd, comp_df, pd.DataFrame(equity_curve), exp_hkd, exp_r, avg_dur, profit_loss_ratio, max_drawdown

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
    active_pos_temp, _, _, _, _, _, _, _, _ = calculate_portfolio(df)
    
    with st.form("trade_form", clear_on_submit=True):
        d_in = st.date_input("日期")
        s_in = format_symbol(st.text_input("代號 (Ticker)").upper().strip())
        is_sell_toggle = st.toggle("Buy 🟢 / Sell 🔴", value=False)
        act_in = "賣出 Sell" if is_sell_toggle else "買入 Buy"
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
                assigned_tid = "N/A"
                if not is_sell_toggle: # Buy
                    if s_in in active_pos_temp:
                        assigned_tid = active_pos_temp[s_in]['trade_id']
                    else:
                        assigned_tid = int(time.time())
                else: # Sell
                    if s_in in active_pos_temp:
                        assigned_tid = active_pos_temp[s_in]['trade_id']
                    else:
                        st.error("找不到該標的的開倉紀錄，無法匹配 Trade_ID")
                img_path = None
                if img_file is not None:
                    ts_str = str(int(time.time()))
                    img_path = os.path.join("images", f"{ts_str}_{img_file.name}")
                    with open(img_path, "wb") as f:
                        f.write(img_file.getbuffer())
                
                save_transaction({
                    "Date": d_in.strftime('%Y-%m-%d'), "Symbol": s_in, "Action": act_in, 
                    "Strategy": clean_strategy(st_in), "Price": p_in, "Quantity": q_in, 
                    "Stop_Loss": sl_in if sl_in is not None else 0.0, "Fees": 0, 
                    "Emotion": emo_in, "Risk_Reward": 0, 
                    "Notes": note_in, "Timestamp": int(time.time()), 
                    "Market_Condition": mkt_cond, "Mistake_Tag": mistake_in,
                    "Img": img_path, "Trade_ID": assigned_tid
                })
                st.success(f"已儲存 {s_in}"); time.sleep(0.5); st.rerun()

# 計算主要數據
active_pos, realized_pnl_total_hkd, completed_trades_df, equity_df, exp_val, exp_r_val, avg_dur_val, pl_ratio_val, mdd_val = calculate_portfolio(df)

t1, t2, t3, t4, t5 = st.tabs(["📈 績效矩陣", "🔥 持倉 & 報價", "🔄 交易重播", "🧠 心理 & 歷史", "🛠️ 數據管理"])

with t1:
    st.subheader("📊 績效概覽")
    time_frame = st.selectbox("統計時間範圍", ["全部記錄", "本週 (This Week)", "本月 (This Month)", "最近 3個月 (Last 3M)", "今年 (YTD)"], index=0)
    
    filtered_comp = completed_trades_df.copy()
    if not filtered_comp.empty:
        filtered_comp['Entry_DT'] = pd.to_datetime(filtered_comp['Entry_Date'])
        filtered_comp['Exit_DT'] = pd.to_datetime(filtered_comp['Exit_Date'])
        today = datetime.now()
        
        if "今年" in time_frame:
            mask = (filtered_comp['Entry_DT'].dt.year == today.year)
        elif "本月" in time_frame:
            mask = (filtered_comp['Entry_DT'].dt.year == today.year) & (filtered_comp['Entry_DT'].dt.month == today.month)
        elif "本週" in time_frame: 
            start_week = today - timedelta(days=today.weekday())
            mask = (filtered_comp['Entry_DT'] >= start_week)
        elif "3個月" in time_frame: 
            cutoff = today - timedelta(days=90)
            mask = (filtered_comp['Entry_DT'] >= cutoff)
        else: mask = [True] * len(filtered_comp)
        filtered_comp = filtered_comp[mask]
    f_pnl = filtered_comp['PnL_HKD'].sum() if not filtered_comp.empty else 0
    trade_count = len(filtered_comp)
    win_r = (len(filtered_comp[filtered_comp['PnL_HKD'] > 0]) / trade_count * 100) if trade_count > 0 else 0
    f_dur = filtered_comp['Duration_Days'].mean() if not filtered_comp.empty else 0
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("已實現損益 (HKD)", f"${f_pnl:,.2f}")
    m2.metric("期望值 (R)", f"{exp_r_val:.2f}R")
    m3.metric("勝率", f"{win_r:.1f}%")
    m4.metric("盈虧比", f"{pl_ratio_val:.2f}")
    m5.metric("最大回撤", f"${mdd_val:,.0f}", delta_color="inverse")
    m6.metric("交易場數", f"{trade_count}")
    if not equity_df.empty:
        st.plotly_chart(px.area(equity_df, x="Date", y="Cumulative PnL", title="累計損益曲線"), use_container_width=True)
    
    # --- AI 交易教練洞察 ---
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
                "Mistakes": filtered_comp['Mistake_Tag'].value_counts().to_dict()
            }
            prompt = f"請根據以下交易統計給出深度專業建議：{stats}。請分析錯誤標籤，並給出三個下週改進動作。請用繁體中文，語氣要像專業交易導師。"
            st.markdown(get_ai_response(prompt))
    # --- 還原交易排行榜格式 ---
    if not filtered_comp.empty:
        st.divider()
        st.subheader("🏆 週期成交排行榜")
        display_trades = filtered_comp.copy()
        display_trades['原始損益'] = display_trades.apply(lambda x: f"{get_currency_symbol(x['Symbol'])} {x['PnL_Raw']:,.2f}", axis=1)
        display_trades['HKD 損益'] = display_trades['PnL_HKD'].apply(lambda x: f"${x:,.2f}")
        display_trades['R 乘數'] = display_trades['Trade_R'].apply(lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A")
        display_trades = display_trades.rename(columns={"Exit_Date": "出場日期", "Symbol": "代號"})
        
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
        for s, d in active_pos.items():
            now = live_prices.get(s)
            qty, avg_p, last_sl = d['qty'], d['avg_price'], d['last_sl']
            entry_p, entry_sl = d.get('entry_price', avg_p), d.get('entry_sl', 0)
            
            un_pnl = (now - avg_p) * qty if now else 0
            roi = (un_pnl / (qty * avg_p) * 100) if (now and avg_p != 0) else 0
            
            init_risk = abs(entry_p - entry_sl) * qty if entry_sl > 0 else 0
            curr_risk = (now - last_sl) * qty if (now and last_sl > 0) else 0
            curr_r = (un_pnl / init_risk) if (now and init_risk > 0) else 0
            
            processed_p_data.append({
                "代號": s, "持股數": f"{qty:,.0f}", "平均成本": f"{avg_p:,.2f}", 
                "現價": f"{now:,.2f}" if now else "N/A", "當前止損": f"{last_sl:,.2f}", 
                "初始風險": f"{init_risk:,.2f}",
                "當前風險": f"{curr_risk:,.2f}",
                "當前R": f"{curr_r:.2f}R",
                "未實現損益": f"{un_pnl:,.2f}", "報酬%": roi
            })
        
        st.dataframe(
            pd.DataFrame(processed_p_data), 
            column_config={
                "報酬%": st.column_config.ProgressColumn(
                    "報酬%", format="%.2f%%", min_value=-20, max_value=20, color="green"
                )
            }, 
            hide_index=True, use_container_width=True
        )
        if st.button("🔄 刷新即時報價", use_container_width=True): st.cache_data.clear(); st.rerun()
    else:
        st.info("目前無持倉部位")

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
            fig.update_layout(title=f"{row['Symbol']} K線圖回顧", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            if pd.notnull(row['Img']) and os.path.exists(row['Img']):
                st.image(row['Img'], caption="交易當下截圖")
        
        st.divider()
        if st.button("🤖 AI 單筆深度診斷"):
            prompt = f"請檢討這筆交易：代號 {row['Symbol']}, 進場 {row['Price']}, 策略 {row['Strategy']}, 情緒 {row['Emotion']}, 錯誤 {row['Mistake_Tag']}。請評估其進場合理性。"
            st.markdown(get_ai_response(prompt))

with t4:
    st.subheader("📜 心理 & 歷史分析")
    if not completed_trades_df.empty:
        c1, c2 = st.columns(2)
        valid_r = completed_trades_df[completed_trades_df['Trade_R'].notna()]
        with c1:
            mistake_r = valid_r[valid_r['Mistake_Tag'] != "None"].groupby('Mistake_Tag')['Trade_R'].mean().reset_index()
            if not mistake_r.empty:
                st.plotly_chart(px.bar(mistake_r, x='Mistake_Tag', y='Trade_R', title="平均 R 乘數 (按錯誤標籤)", color='Trade_R', color_continuous_scale='RdYlGn'), use_container_width=True)
        with c2:
            emo_r = valid_r.groupby('Emotion')['Trade_R'].mean().reset_index()
            if not emo_r.empty:
                st.plotly_chart(px.bar(emo_r, x='Emotion', y='Trade_R', title="平均 R 乘數 (按情緒)", color='Trade_R', color_continuous_scale='RdYlGn'), use_container_width=True)
    if not df.empty:
        st.divider()
        hist_df = df.sort_values("Timestamp", ascending=False).copy()
        hist_df['截圖'] = hist_df['Img'].apply(lambda x: "🖼️" if pd.notnull(x) and os.path.exists(x) else "")
        cols = ["Date", "Symbol", "Action", "Trade_ID", "Price", "Quantity", "Stop_Loss", "Emotion", "Mistake_Tag", "截圖"]
        st.dataframe(hist_df[cols], use_container_width=True, hide_index=True)

with t5:
    st.subheader("🛠️ 數據管理")
    conn_status = get_data_connection()
    if conn_status:
        st.success("🟢 已連接至 Google Sheets (雲端同步中)")
    else:
        st.warning("🟠 目前使用本地 CSV 模式")
    
    # --- NEW: AI 匯出功能 ---
    st.divider()
    st.markdown("#### 🤖 匯出給 AI 分析 (Export for LLM)")
    st.info("下載此檔案後，直接上傳給 ChatGPT / Claude / DeepSeek，它們會自動為您進行全方位帳戶診斷。")
    
    # 準備匯出數據
    if not df.empty:
        # 計算匯出用的摘要數據
        export_stats = {
            "pnl_str": f"${realized_pnl_total_hkd:,.2f}",
            "win_rate": f"{(len(completed_trades_df[completed_trades_df['PnL_HKD'] > 0])/len(completed_trades_df)*100):.1f}%" if not completed_trades_df.empty else "N/A",
            "pf": f"{pl_ratio_val:.2f}",
            "exp_r": f"{exp_r_val:.2f}R",
            "mdd": f"${mdd_val:,.0f}",
            "count": len(completed_trades_df)
        }
        
        export_text = generate_llm_export_data(df, export_stats)
        
        st.download_button(
            label="📥 下載 AI 專用分析報告 (.txt)",
            data=export_text,
            file_name=f"TradeMaster_AI_Review_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    else:
        st.caption("尚無交易紀錄可供匯出。")
    
    st.divider()
    
    # 原有的匯入功能
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_file = st.file_uploader("📤 批量上傳 CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file and st.button("🚀 開始匯入"):
            try:
                new_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Symbol' in new_data.columns: new_data['Symbol'] = new_data['Symbol'].apply(format_symbol)
                if 'Timestamp' not in new_data.columns: new_data['Timestamp'] = int(time.time())
                df = pd.concat([df, new_data], ignore_index=True); save_all_data(df)
                st.success("匯入成功！"); st.rerun()
            except Exception as e: st.error(f"匯入失敗: {e}")
    
    if not df.empty:
        st.divider()
        selected_idx = st.selectbox("選擇紀錄進行編輯", df.index, format_func=lambda x: f"[{df.loc[x, 'Date']}] {df.loc[x, 'Symbol']} ({df.loc[x, 'Action']})")
        t_edit = df.loc[selected_idx]
        e1, e2, e3 = st.columns(3)
        n_p = e1.number_input("編輯價格", value=float(t_edit['Price']), key=f"ep_{selected_idx}")
        n_q = e2.number_input("編輯股數", value=float(t_edit['Quantity']), key=f"eq_{selected_idx}")
        n_sl = e3.number_input("編輯止損價", value=float(t_edit['Stop_Loss']), key=f"esl_{selected_idx}")
        
        b1, b2 = st.columns(2)
        if b1.button("💾 儲存修改", use_container_width=True):
            df.loc[selected_idx, ['Price', 'Quantity', 'Stop_Loss']] = [n_p, n_q, n_sl]
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


