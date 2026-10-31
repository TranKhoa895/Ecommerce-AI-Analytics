import sqlite3
import pandas as pd 
import numpy as np 
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def rfm_clustering(db_path='ecommerce.db'):
    # 1. Khởi tạo Tracking URI (SQLite Local DB) và Experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("3.1_RFM_Customer_Clustering")

    # 2. Đọc dữ liệu RFM từ Feature Store
    conn = sqlite3.connect(db_path)
    df_rfm = pd.read_sql_query("SELECT CustomerID, Recency, Frequency, Monetary FROM rfm_segmentation", conn)

    print(f"HUẤN LUYỆN K-MEANS CLUSTERING (TỔNG SỐ MẪU: {len(df_rfm):,})")

    # 3. Tiền xử lý dữ liệu 
    # Biến đổi log1p để giảm lệch (skewness)
    X = df_rfm[['Recency', 'Frequency', 'Monetary']].copy()
    X_log = np.log1p(X)

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    best_k = 3
    best_silhouette = -1.0
    best_model = None

    print(f"HUẤN LUYỆN K-MEANS CLUSTERING (TỔNG SỐ MẪU: {len(df_rfm):,})")

    # 4. Thử nghiệm các cụm K khác nhau và log vào MLflow 
    for k in [3, 4, 5]:
        with mlflow.start_run(run_name=f"KMeans_K_{k}"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, cluster_labels)
            inertia = kmeans.inertia_

            # Log Parameters & Metrics
            mlflow.log_param("k_clusters", k)
            mlflow.log_param("transformation", "log1p")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_param("random_state", 42)
            
            mlflow.log_param("silhouette_score", sil_score)
            mlflow.log_param("inertia", inertia)

            # Log Model Artifact
            mlflow.sklearn.log_model(
                sk_model=kmeans, 
                artifact_path=f"kmeans_k{k}_model")

            print(f"-> K = {k} | Silhouette Score: {sil_score:.4f} | Inetia: {inertia:,.2f}")

            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_k = k
                best_labels = cluster_labels

    # 5. Lưu kết quả phân cụm tối ưu vào DB
    df_rfm['Cluster_KMeans'] = best_labels
    df_rfm.to_sql('rfm_kmeans_clusters', conn, if_exists='replace', index=False)
    conn.close()

    print(f"\nKẾT QUẢ TỐI ƯU: K={best_k} đạt Silhouette Score cao nhất: {best_silhouette:.4f}")
    print("Đã lưu kết quả phân cụm tối ưu vào bảng 'rfm_kmeans_clusters' trong ecommerce.db")

if __name__ == "__main__":
    rfm_clustering()
    