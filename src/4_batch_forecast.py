import os 
import sqlite3
import pandas as pd 
import numpy as np 
import lightgbm as lgb

def batch_forecasting(db_path='ecommerce.db', output_dir='data', forecast_days=30):
    print("BẮT ĐẦU CHẠY BATCH FORECAST SERVICE")

    # 1. Đảm bảo thư mục lưu dữ liệu đầu ra tồn tại 
    os.makedirs(output_dir, exist_ok=True)

    # 2. Tải dữ liệu chuỗi thời gian từ Feature Store
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT Date, Daily_Revenue, lag_1, lag_7, lag_30, rolling_mean_7, rolling_mean_30
        FROM time_series_features
        ORDER BY Date
    """, conn)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().reset_index(drop=True)

    # 3. Chuẩn bị Feature Matrix và Target Variable 
    feature_cols = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
    X = df[feature_cols]
    y = np.log1p(df['Daily_Revenue'])

    # 4. Huấn luyện LightGBM Regressor trên toàn bộ dữ liệu 
    params = {
            'objective': 'regression', 
            'metric' : 'rmse',
            'learning_rate': 0.05, 
            'num_leaves': 15, 
            'max_depth': 4,
            'random_state': 42,
            'verbose': -1
        }
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    # 5. Khởi tạo danh sách chuỗi lịch sử cho Vòng lặp Dự báo Đệ quy (Recursive Loop)
    revenue_history = list(df['Daily_Revenue'].values)
    last_date = df['Date'].max()

    forecast_results = []

    print(f"Đang tính toán dự báo đệ quy cho {forecast_days} ngày tiếp theo...")

    for day_step in range(1, forecast_days + 1):
        # Tính toán các chỉ số lag và rolling mean dựa trên lịch sử mới nhất
        lag_1 = revenue_history[-1]
        lag_7 = revenue_history[-7]
        lag_30 = revenue_history[-30]
        rolling_mean_7 = np.mean(revenue_history[-7:])
        rolling_mean_30 = np.mean(revenue_history[-30:])

        # Tạo vector đặc trưng cho ngày tiếp theo 
        X_next = pd.DataFrame([{
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_30': lag_30,
            'rolling_mean_7': rolling_mean_7,
            'rolling_mean_30': rolling_mean_30
        }])

        # Dự báo doanh thu dạng log1p và chuyển đổi về doanh thu thực ($)
        pred_log = model.predict(X_next)[0]
        pred_revenue = float(np.expm1(pred_log))
        pred_revenue = max(0.0, pred_revenue) # Triệt tiêu giá trị âm 

        # Cập nhật ngày và chuỗi giá trị lịch sử 
        next_date = last_date + pd.Timedelta(days=day_step)
        revenue_history.append(pred_revenue)

        forecast_results.append({
            'Forecast_Date': next_date.strftime('%Y-%m-%d'),
            'Forecasted_Revenue': round(pred_revenue, 2), 
            'Lag_1': round(lag_1, 2),
            'Rolling_mean_7': round(rolling_mean_7, 2)
        })

    df_forecast = pd.DataFrame(forecast_results)

    # 6. Xuất kết quả và lưu vào DB 
    parquet_path = os.path.join(output_dir, 'forecast.parquet')
    df_forecast.to_parquet(parquet_path, index=False)
    df_forecast.to_sql('daily_batch_forecasts', conn, if_exists='replace', index=False)
    conn.close()

    print("HOÀN THÀNH BATCH FORECAST")
    print(f"File Parquet lưu tại: {parquet_path}")
    print(f"Bảng SQL lưu tại: 'daily_batch_forecasts' trong {db_path}")
    print("\nPreview 10 ngày dự báo đầu tiên:")
    print(df_forecast.head(10).to_string(index=False))

if __name__ == "__main__":
    batch_forecasting()    
        