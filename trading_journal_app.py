# ... (前面代碼保持不變，直接定位到 t4 區塊)

with t4:
    st.subheader("📜 歷史紀錄與心理分析")
    if not df.empty:
        # 優化歷史表格顯示，避免欄位名稱混淆
        history_display = df.sort_values("Timestamp", ascending=False).copy()
        
        # 重新命名欄位讓意義更明確
        history_display = history_display.rename(columns={
            "Stop_Loss": "執行時止損",
            "Price": "成交價",
            "Quantity": "股數",
            "Risk_Reward": "預期 R:R"
        })
        
        # 隱藏不需要在歷史表顯示的技術欄位
        cols_to_show = ["Date", "Symbol", "Action", "Strategy", "成交價", "股數", "執行時止損", "Emotion", "Market_Condition", "Mistake_Tag", "Notes"]
        st.dataframe(history_display[cols_to_show], use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("⚠️ 錯誤標籤分析")
        # ... (後續分析圖表保持不變)
