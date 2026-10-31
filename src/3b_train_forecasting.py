import sqlite3
import pandas as pd 
import numpy as np 
import mlflow 
import mlflow.lightgbm
import lightgbm as lgb 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def demand_forecasting(db_path='ecommerce.db'):
    # 1. Cấu hình Tracking URI và Experiment trên MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("3.2_Demand_Forecasting_LightGBM")

    # 2. Tải dữ liệu chuỗi thời gian từ Feature Store
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT Date, Daily_Revenue, lag_1, lag_7, lag_30, rolling_mean_7, rolling_mean_30
        FROM time_series_features
        ORDER BY Date
    """, conn)
    conn.close()

    # 3. Làm sạch các dòng chứa giá trị NaN do tạo Lags/Rolling
    df = df.dropna().reset_index(drop=True)

    X = df[['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']]
    # Áp dụng log1p biến đổi doanh thu giảm ảnh hưởng của nhiễu bùng nổ (Outliers)
    y = np.log1p(df['Daily_Revenue'])

    # 4. Khởi tạo TimeSeriesSplit (3 Folds)
    tscv = TimeSeriesSplit(n_splits=3)

    params = {
        'objective': 'regression', 
        'metric' : 'rmse',
        'learning_rate': 0.05, 
        'num_leaves': 15, 
        'max_depth': 4,
        'random_stae': 42,
        'verbose': -1
    }

    rmse_list, mae_list, r2_list = [], [], []

    with mlflow.start_run(run_name="LightGBM_Log1p_TimeSeriesSplit"):
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            # Dự báo trên tập Validation 
            preds_log = model.predict(X_val)

            # Khôi phục về giá trị doanh thu thực ($)
            y_val_orig = np.expm1(y_val)
            preds_orig = np.expm1(preds_log)
            preds_orig = np.clip(preds_orig, a_min=0, a_max=None)

            # Tính toán các chỉ số 
            rmse = root_mean_squared_error(y_val_orig, preds_orig)
            mae = mean_absolute_error(y_val_orig, preds_orig)
            r2 = r2_score(y_val_orig, preds_orig)

            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)

        avg_rmse = float(np.mean(rmse_list))
        avg_mae = float(np.mean(mae_list))
        avg_r2 = float(np.mean(r2_list))

        # Log Parameters & Metrics lên MLflow 
        mlflow.log_params(params)

        mlflow.log_metric("avg_rmse", avg_rmse)
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("avg_r2", avg_r2)

        # 5. Huấn luyện mô hình trên toàn bộ dự liệu 
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X, y)
        mlflow.lightgbm.log_model(final_model, artifact_path="lightgbm_forecast_model")

        print(f"Số lượng ngày huấn luyện: {len(df):,} ngày")
        print(f"Average RMSE: ${avg_rmse:,.2f}")
        print(f"Average MAE: ${avg_mae:,.2f}")
        print(f"Average R2 Score: ${avg_r2:,.2f}")
        print("Đã ghi nhận thành công mô hình Dự báo Nhu cầu lên MLflow!")

if __name__ == "__main__":
    demand_forecasting()




