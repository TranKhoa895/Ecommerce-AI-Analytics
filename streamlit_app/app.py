import streamlit as st 
import requests
import pandas as pd 
import numpy as np 
import sqlite3
import plotly.express as px 
import plotly.graph_objects as go 
import lightgbm as lgb 
import shap 
import matplotlib.pyplot as plt 

# Cấu hình trang Streamlit 
st.set_page_config(
    page_title="E-Commerce AI Analytics & Recommendation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

FASTAPI_URL = "http://127.0.0.1:8000"
DB_PATH = "ecommerce.db"
FORECAST_PARQUET_PATH = "data/forecast.parquet"

# Sidebar: Health Check & Navigation 
st.sidebar.header("⚙️ System Status")
try:
    res = requests.get(f"{FASTAPI_URL}/health", timeout=2)
    if res.status_code == 200:
        st.sidebar.success("FastAPI Service: ONLINE 🟢")
    else:
        st.sidebar.success("FastAPI Service: ISSUES 🟡")
except Exception:
    st.sidebar.error("FastAPI Service: OFFLINE 🔴\n(Hãy chạy `uvicorn app.main:app`)")

# Tạo 2 Tab chính theo yêu cầu
tab1, tab2 = st.tabs(["🎯 Real-Time Recommendations & RFM", "📈 Demand Forecasting & SHAP Explainability"])

# TAB 1: RECOMMENDATIONS & CUSTOMER SEGMENTATION DEMO
with tab1:
    st.header("1. Product Recommendation Engine")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Search")
        stock_code_input = st.text_input("Nhập StockCode (ví dụ: 85123A, 84997C, hoặc ký tự rác 'A'):", value="85123A")
        top_k_input = st.slider("Số lượng gợi ý:", min_value=1, max_value=10, value=5)

        btn_recommend = st.button("🚀 Gợi ý Sản Phẩm", use_container_width=True)

    with col2:
        st.subheader("Gợi ý Real-time FastAPI")
        if btn_recommend:
            payload = {"stock_code": stock_code_input, "top_k": top_k_input}
            try:
                response = requests.post(f"{FASTAPI_URL}/api/v1/recommend", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    recs = data.get("recommendations", [])

                    if recs:
                        df_recs = pd.DataFrame(recs)
                        st.write(f"**Kết quả gợi ý cho mã:** `{data['input_stock_code']}`")

                        # Hiển thị Badge chiến lược Fallback được kích hoạt 
                        strategy_used = df_recs['strategy'].iloc[0]
                        if strategy_used == "Co-occurrence":
                            st.success(f"Chiến lược áp dụng: **{strategy_used}** (Tầng 1 - Mua kèm cùng giỏ hàng)")
                        elif strategy_used == "TF-IDF Content-Based":
                            st.info(f"Chiến lược áp dụng: **{strategy_used}** (Tầng 2 - Tương đồng về nội dung)")
                        else:
                            st.warning(f"Chiến lược áp dụng: **{strategy_used}** (Tầng 3 - Popular Fallback)")

                        st.dataframe(df_recs, use_container_width=True)
                    else:
                        st.info("Không tìm thấy sản phẩm gợi ý phù hợp.")
                else:
                    st.error(f"Lỗi API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến FastAPI Server: {e}")

    st.divider()
    st.header("2. RFM Customer Segmentation Simulator")

    col_rfm1, col_rfm2 = st.columns([1, 2])
    with col_rfm1:
        r_val = st.number_input("Recency (Số ngày chưa mua hàng):", min_value=0.0, value=15.0)
        f_val = st.number_input("Frequency (Tổng số đơn hàng):", min_value=1.0, value=5.0)
        m_val = st.number_input("Monetary (Tổng chi tiêu):", min_value=0.0, value=450.0)

        btn_rfm = st.button("🔍 Phân Nhóm Khách Hàng", use_container_width=True)

    with col_rfm2:
        if btn_rfm:
            rfm_payload = {"recency": r_val, "frequency": f_val, "monetary": m_val}
            try:
                res_rfm = requests.post(f"{FASTAPI_URL}/api/v1/segment-customer", json=rfm_payload)
                if res_rfm.status_code == 200:
                    rfm_data = res_rfm.json()
                    st.metric(label="Nhóm khách hàng (Cluster Label)", value=f"Cluster {rfm_data['cluster_label']}")
                    st.subheader(f"Phân loại: **{rfm_data['cluster_name']}**")
                else:
                    st.error(f"Lỗi API: {res_rfm.status_code}")
            except Exception as e:
                st.error(f"Lỗi kết nối FastAPI: {e}")


# TAB 2: DEMAND FORECASTING & SHAP EXPLAINABILITY
with tab2:
    st.header("📈 30-90 Days Revenue & Demand Forecast")

    # Khởi tạo biến mặc định tránh lỗi NameError
    selected_sku = "Tất cả (Tổng Doanh Thu Toàn Sàn)"

    # 1. Tải dữ liệu từ Parquet & SQLite
    try:
        df_forecast = pd.read_parquet(FORECAST_PARQUET_PATH)

        conn = sqlite3.connect(DB_PATH)
        df_hist = pd.read_sql_query("SELECT * FROM time_series_features", conn)
        conn.close()

        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        date_fc_col = 'Forecast_Date' if 'Forecast_Date' in df_forecast.columns else ('Date' if 'Date' in df_forecast.columns else df_forecast.columns[0])
        df_forecast[date_fc_col] = pd.to_datetime(df_forecast[date_fc_col])

        # Kiểm tra an toàn sự tồn tại của cột StockCode (Tránh KeyError)
        avail_skus = []
        if 'StockCode' in df_forecast.columns:
            avail_skus = sorted(df_forecast['StockCode'].dropna().astype(str).unique().tolist())
        elif 'StockCode' in df_hist.columns:
            avail_skus = sorted(df_hist['StockCode'].dropna().astype(str).unique().tolist())

        col_select, _ = st.columns([1.5, 1.5])
        with col_select:
            sku_options = ["Tất cả (Tổng Doanh Thu Toàn Sàn)"] + avail_skus
            selected_sku = st.selectbox("🎯 Chọn sản phẩm để xem dự báo (StockCode):", options=sku_options)

        # Lọc dữ liệu theo StockCode đã chọn
        if selected_sku != "Tất cả (Tổng Doanh Thu Toàn Sàn)" and ('StockCode' in df_hist.columns or 'StockCode' in df_forecast.columns):
            df_hist_sub = df_hist[df_hist['StockCode'].astype(str) == selected_sku].sort_values('Date') if 'StockCode' in df_hist.columns else df_hist
            df_fc_sub = df_forecast[df_forecast['StockCode'].astype(str) == selected_sku].sort_values(date_fc_col) if 'StockCode' in df_forecast.columns else df_forecast
            chart_title = f"Chuỗi Thời Gian Lịch Sử & Dự Báo SKU: {selected_sku}"
        else:
            df_hist_sub = df_hist.groupby('Date', as_index=False)['Daily_Revenue'].sum().sort_values('Date') if 'Daily_Revenue' in df_hist.columns else df_hist
            rev_fc_col = 'Forecasted_Revenue' if 'Forecasted_Revenue' in df_forecast.columns else ('Daily_Revenue' if 'Daily_Revenue' in df_forecast.columns else df_forecast.columns[1])
            df_fc_sub = df_forecast.groupby(date_fc_col, as_index=False)[rev_fc_col].sum().sort_values(date_fc_col) if rev_fc_col in df_forecast.columns else df_forecast
            chart_title = "Chuỗi Thời Gian Doanh Thu Lịch Sử & Dự Báo Toàn Sàn"

        # Biểu đồ Plotly
        fig = go.Figure()
        
        y_hist_col = 'Daily_Revenue' if 'Daily_Revenue' in df_hist_sub.columns else df_hist_sub.columns[1]
        y_fc_col = 'Forecasted_Revenue' if 'Forecasted_Revenue' in df_fc_sub.columns else ('Daily_Revenue' if 'Daily_Revenue' in df_fc_sub.columns else df_fc_sub.columns[1])

        fig.add_trace(go.Scatter(
            x=df_hist_sub['Date'].tail(60), 
            y=df_hist_sub[y_hist_col].tail(60), 
            name="Lịch sử (60 ngày)", 
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_fc_sub[date_fc_col].tail(60), 
            y=df_fc_sub[y_fc_col].tail(60), 
            name="Dự báo (30-90 ngày tới)", 
            line=dict(color='orange', dash='dash')
        ))

        fig.update_layout(
            title=chart_title, 
            xaxis_title="Ngày", 
            yaxis_title="Doanh thu ($)", 
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Không thể tải file dữ liệu dự báo `forecast.parquet`: {e}")

    st.divider()
    st.header("🧠 SHAP Model Explainability (LightGBM)")
    st.write("Giải thích tác động của các chỉ số (`Lag`, `Rolling Mean`) đến mô hình dự báo.")

    # 2. Biểu đồ SHAP
    try: 
        conn = sqlite3.connect(DB_PATH)
        df_ts = pd.read_sql_query("SELECT * FROM time_series_features", conn)
        conn.close()

        if selected_sku != "Tất cả (Tổng Doanh Thu Toàn Sàn)" and 'StockCode' in df_ts.columns:
            df_ts_clean = df_ts[df_ts['StockCode'].astype(str) == selected_sku].dropna().reset_index(drop=True)
            if df_ts_clean.empty:
                df_ts_clean = df_ts.groupby('Date', as_index=False).sum(numeric_only=True).dropna().reset_index(drop=True)
        else:
            if 'Date' in df_ts.columns:
                df_ts_clean = df_ts.groupby('Date', as_index=False).sum(numeric_only=True).dropna().reset_index(drop=True)
            else:
                df_ts_clean = df_ts.dropna().reset_index(drop=True)

        feature_cols = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
        avail_features = [col for col in feature_cols if col in df_ts_clean.columns]

        if avail_features and len(df_ts_clean) > 5:
            X_train = df_ts_clean[avail_features]
            y_col = 'Daily_Revenue' if 'Daily_Revenue' in df_ts_clean.columns else df_ts_clean.columns[1]
            y_train = np.log1p(df_ts_clean[y_col])

            model_lgb = lgb.LGBMRegressor(learning_rate=0.05, num_leaves=15, max_depth=4, random_state=42, verbose=-1)
            model_lgb.fit(X_train, y_train)

            explainer = shap.TreeExplainer(model_lgb)
            shap_values = explainer(X_train)

            col_shap1, col_shap2 = st.columns(2)
            with col_shap1:
                st.subheader("1. SHAP Feature Summary Plot")
                fig_summary, _ = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                st.pyplot(fig_summary)
                plt.close(fig_summary)

            with col_shap2:
                st.subheader("2. SHAP Waterfall Plot (Dự báo ngày gần nhất)")
                fig_waterfall = plt.figure(figsize=(6, 4))
                shap.plots.waterfall(shap_values[-1], show=False)
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

    except Exception as e:
        st.error(f"Lỗi tính toán SHAP Explainability: {e}")