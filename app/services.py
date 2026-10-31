import sqlite3
import pandas as pd 
import numpy as np 

DB_PATH = 'ecommerce.db'

class InferenceService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def recommend_items(self, stock_code: str, top_k: int = 5):
        conn = self._get_db_connection()
        recommendations = []

        # TẦNG 1: Co-occurrence Matrix (Mùa kèm cùng giỏ hàng)
        querry_co = """
                SELECT StockCode_B AS stock_code, Co_Occurrence_Count AS score
                FROM recommender_co_occurrence
                WHERE StockCode_A = ?
                ORDER BY Co_Occurrence_Count DESC
                LIMIT ?
        """
        df_co = pd.read_sql_query(querry_co, conn, params=(stock_code, top_k))

        if not df_co.empty:
            for _, row in df_co.iterrows():
                recommendations.append({
                    "stock_code": str(row['stock_code']), 
                    "score": float(row['score']),
                    "strategy": "Co-occurrence"
                })
            conn.close()
            return recommendations

        # TẦNG 2: Fallack TF-IDF Content-Based (Tương đồng tên/Mô tả sản phẩm)
        query_tfidf = """
                SELECT Similar_StockCode AS stock_code, Similarity_Score AS score
                FROM recommender_tfidf_similarity
                WHERE StockCode = ?
                ORDER BY Similarity_Score DESC
                LIMIT ?
        """
        df_tfidf = pd.read_sql_query(query_tfidf, conn, params=(stock_code, top_k))

        if not df_tfidf.empty:
            for _, row in df_tfidf.iterrows():
                recommendations.append({
                    "stock_code": str(row['stock_code']), 
                    "score": float(row['score']),
                    "strategy": "TF-IDF Content-Based"
                })
            conn.close()
            return recommendations

        # TẦNG 3: Fallback Popular Items (Sản phẩm bán chạy nhất toàn hệ thống)
        querry_pop = """
                SELECT StockCode_A AS stock_code, SUM(Co_Occurrence_Count) AS total_freq
                FROM recommender_co_occurrence
                GROUP BY StockCode_A
                ORDER BY total_freq DESC
                LIMIT ?
        """
        df_pop = pd.read_sql_query(querry_pop, conn, params=(top_k,))
        for _, row in df_pop.iterrows():
            recommendations.append({
                "stock_code": str(row['stock_code']), 
                "score": float(row['total_freq']),
                "strategy": "Popular_Fallback"
            })
        conn.close()
        return recommendations

    def predict_rfm_cluster(self, recency: float, frequency: float, monetary: float):
        # Chuẩn hóa biến đổi log1p
        r_log = np.log1p(recency)
        f_log = np.log1p(frequency)
        m_log = np.log1p(monetary)

        # Đọc dữ liệu cụm để tìm tâm cụm (Centroid) gần nhất
        conn = self._get_db_connection()
        df_clusters = pd.read_sql_query("SELECT Recency, Frequency, Monetary, Cluster_KMeans FROM rfm_kmeans_clusters", conn)
        conn.close()

        df_clusters['r_log'] = np.log1p(df_clusters['Recency'])
        df_clusters['f_log'] = np.log1p(df_clusters['Frequency'])
        df_clusters['m_log'] = np.log1p(df_clusters['Monetary'])

        centroids = df_clusters.groupby('Cluster_KMeans')[['r_log', 'f_log', 'm_log']].mean()

        # Tính khoảng cách Euclidean đến các tâm cụm 
        point = np.array([r_log, f_log, m_log])
        distances = {cluster: np.linalg.norm(point - centroids.loc[cluster].values) for cluster in centroids.index}

        best_cluster = int(min(distances, key=distances.get))

        cluster_names = {
            0: "Standard/Regular Customers", 
            1: "High-Value/VIP Customers", 
            2: "At-Risk/Churned Customers"
        }

        return {
            "cluster_label": best_cluster,
            "cluster_name": cluster_names.get(best_cluster, f"Cluster {best_cluster}")
        }