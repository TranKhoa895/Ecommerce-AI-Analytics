import sqlite3
import pandas as pd 
import numpy as np 
import mlflow 

def recommender_pipeline(db_path='ecommerce.db'):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # 1. Tạo Experiment trên MLflow
    mlflow.set_experiment("3.3_Hybrid_Recommender_Engine")

    # 2. Đọc dữ liệu RFM từ Feature Store
    conn = sqlite3.connect(db_path)
    df_co = pd.read_sql_query("SELECT * FROM recommender_co_occurrence", conn)
    df_tfidf = pd.read_sql_query("SELECT * FROM recommender_tfidf_similarity", conn)

    # Lấy danh sách sản phẩm trong Catalog 
    try:
        df_sales = pd.read_sql_query("SELECT DISTINCT StockCode FROM clean_sales", conn)
    except Exception:
        df_sales = pd.read_sql_query("SELECT DISTINCT StockCode FROM time_series_features", conn)
    conn.close()

    # 3. Tính toán độ phủ Catalog (Catalog Coverage)
    total_catalog = len(df_sales)
    co_products = set(df_co['StockCode_A'].unique()) if 'StockCode_A' in df_co.columns else set()
    tfidf_products = set(df_tfidf['StockCode'].unique()) if 'StockCode' in df_tfidf.columns else set()

    coveraged_products = co_products.union(tfidf_products)
    catalog_coverage = (len(coveraged_products) / total_catalog * 100) if total_catalog > 0 else 0.0

    # 4. Log Parameters và Metrics lên MLflow 
    with mlflow.start_run(run_name="Hybrid_CoOccurrence_TFIDF_Engine"):
        mlflow.log_param("total_catalog_items", total_catalog)
        mlflow.log_param("co_occurrence_pairs", len(df_co))
        mlflow.log_param("tfidf_similarity_pairs", len(df_tfidf))
        mlflow.log_param("fallback_strategy", "Co-occurrence -> TF-IDF")

        mlflow.log_metric("catalog_coverage_pct", round(catalog_coverage, 2))
        if 'Co_Occurrence_Count' in df_co.columns and len(df_co) > 0:
            mlflow.log_metric("max_co_occurrence_count", int(df_co['Co_Occurrence_Count'].max()))
        if 'Similarity_Score' in df_tfidf.columns and len(df_tfidf) > 0:
            mlflow.log_metric("avg_tfidf_similarity", float(df_tfidf['Similarity_Score'].mean()))

        print(f"Tổng sản phẩm trong Catalog: {total_catalog:,}")
        print(f"Số sản phẩm tạo được gợi ý: {len(coveraged_products):,}")
        print(f"Độ phủ Catalog (Catalog Coverage): {catalog_coverage:.2f}%")
        print(f"Cặp mua kèm (Co-occurrence): {len(df_co):,}")
        print(f"Cặp tương đồng mô tả (TF-IDF): {len(df_tfidf):,}")
        print("Đã ghi nhận thành công chỉ số Recommender Engine lên MLflow!")

if __name__ == "__main__":
    recommender_pipeline()

