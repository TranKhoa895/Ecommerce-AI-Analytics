'''
STAGE 2: EDA & RFM ANALYSIS
'''
import sqlite3
import pandas as pd 
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalFeatureStore:
    def __init__(self, db_path: str = 'ecommerce.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.df = None

    def load_data(self):
        """Tải dữ liệu đã làm sạch từ SQLite."""
        logging.info("Đang đọc dữ liệu 'clean_sales' từ SQLite...")
        self.df = pd.read_sql_query("SELECT * FROM clean_sales", self.conn)
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])

        # Đảm bảo cột Revenue tồn tại 
        if 'Revenue' not in self.df.columns:
            self.df['Revenue'] = self.df['Quantity'] * self.df['UnitPrice']
        logging.info(f"Đã tải thành công {len(self.df):,} dòng dữ liệu.")

    def calculate_monthly_kpis(self) -> pd.DataFrame:
        """Phân tích doanh thu và số lượng đơn hàng theo tháng."""
        logging.info("Đang tính toán KPIs theo tháng...")
        self.df['YearMonth'] = self.df['InvoiceDate'].dt.to_period('M')

        monthly_df = self.df.groupby('YearMonth').agg(
            Revenue=('Revenue', 'sum'),
            Total_Orders=('InvoiceNo', 'nunique'),
            Unique_Customers=('CustomerID', 'nunique')
        ).reset_index()

        monthly_df['YearMonth'] = monthly_df['YearMonth'].astype(str)
        # Lưu vào SQLite
        monthly_df.to_sql('monthly_kpis', self.conn, if_exists='replace', index=False)
        return monthly_df

    # 1. USER FEATURES (RFM Scores & Segments)
    def user_features(self) -> pd.DataFrame:
        """Phân tích và gán nhãn nhóm khách hàng theo mô hình RFM."""
        logging.info("Đang khởi tạo User Features (RFM)...")

        # Lọc bỏ các giao dịch không có Customer ID (nếu có)
        rfm_df_raw = self.df.dropna(subset=['CustomerID']).copy()

        # Mốc thời gian tham chiếu (ngày giao dịch cuối cùng + 1 ngày)
        max_date = rfm_df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)

        # Tính toán R (ngày), F (số đơn), M (tổng chi tiêu)
        rfm = rfm_df_raw.groupby('CustomerID').agg({
            # Recency
            'InvoiceDate': lambda x: (max_date - x.max()).days,
            # Frequency
            'InvoiceNo': 'nunique',
            # Montary
            'Revenue': 'sum'
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Tính điểm RFM (thang điểm 1 = Kém/Thấp, 2 = Trung bình, 3 = Tốt/Mới)
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 3, labels=[3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 3, labels=[1, 2, 3])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 3, labels=[1, 2, 3])

        # Phân nhóm khách hàng (dựa trên R và F)
        def segment_customer(row):
            r = int(row['R_Score'])
            f = int(row['F_Score'])
            m = int(row['M_Score'])
            # 1. VIP/Champions: Mua gần đây + Tần suất hoặc Doanh số ở top cao nhất
            if r >= 2 and (f ==3 or m==3):
                return 'VIP/Champions'
            # 2. Inactive: Recency tệ nhất (lâu không quay lại)
            elif r == 1:
                return 'Inactive'
            # 3. Regulars: Duy trì mua ổn định / Khách hàng mới phát sinh đơn 
            else:
                return 'Regulars'

        rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1) 

        # Lưu vào SQLite DB
        rfm.to_sql('rfm_segmentation', self.conn, if_exists='replace', index=False)
        logging.info("Đã hoàn thành phân nhóm RFM và lưu vào SQLite 'rfm_segmentation'.")
        return rfm

    # 2. TIME SERIES FEATURES (Lags & Rolling)
    def time_series_features(self) -> pd.DataFrame:
        logging.info("Đang tạo Time Series Features (Lags & Rolling)...")

        # Gom nhóm doanh thu & số lượng bán theo từng ngày 
        daily_ts = self.df.groupby(['StockCode', self.df['InvoiceDate'].dt.date]).agg(
            Daily_Revenue=('Revenue', 'sum'),
            Daily_Quantity=('Quantity', 'sum'),
            Total_Orders=('InvoiceNo', 'nunique')
        ).reset_index()

        daily_ts.rename(columns={'InvoiceDate': 'Date'}, inplace=True)
        daily_ts['Date'] = pd.to_datetime(daily_ts['Date'])

        # Đảm bảo chuỗi ngày liên tục (fill ngày thiếu bằng 0)
        all_dates = pd.date_range(start=daily_ts['Date'].min(), end=daily_ts['Date'].max(), freq='D')
        all_skus = daily_ts['StockCode'].unique()

        multi_idx = pd.MultiIndex.from_product([all_skus, all_dates], names=['StockCode', 'Date'])
        daily_ts = daily_ts.set_index(['StockCode', 'Date']).reindex(multi_idx, fill_value=0).reset_index()
        daily_ts = daily_ts.sort_values(['StockCode', 'Date'])

        # Tạo Lags (1, 7, 30 ngày)
        grouped_rev = daily_ts.groupby('StockCode')['Daily_Revenue']

        daily_ts['lag_1'] = grouped_rev.shift(1)
        daily_ts['lag_7'] = grouped_rev.shift(7)
        daily_ts['lag_30'] = grouped_rev.shift(30)

        # Tạo Rolling Mean (7, 30 ngày) - Shift(1) tránh Data Leakage
        daily_ts['rolling_mean_7'] = grouped_rev.transform(lambda x: x.shift(1).rolling(window=7).mean())
        daily_ts['rolling_mean_30'] = grouped_rev.transform(lambda x: x.shift(1).rolling(window=30).mean())

        daily_ts['Date'] = daily_ts['Date'].astype(str)

        daily_ts.to_sql('time_series_features', self.conn, if_exists='replace', index=False)
        return daily_ts

    # 3. RECOMMENDER FEATURES (Co-occurrence & TF-IDF)
    def recommender_features(self) -> pd.DataFrame:
        logging.info("Đang xây dựng Recommender Features (Co-occurrence Matrix & TF-IDF)...")
        
        # Đảm bảo lấy đúng cột dữ liệu
        df_rec = self.df.dropna(subset=['StockCode', 'Description', 'InvoiceNo']).copy()
        
        # Chuyển kiểu dữ liệu rõ ràng
        df_rec['InvoiceNo'] = df_rec['InvoiceNo'].astype(str).str.strip()
        df_rec['StockCode'] = df_rec['StockCode'].astype(str).str.strip()
        df_rec['Description'] = df_rec['Description'].astype(str).str.strip()

        # Bỏ các mã dịch vụ/phí không phải sản phẩm thật (POST, D, M, BANK CHARGES,...)
        df_rec = df_rec[~df_rec['StockCode'].isin(['POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT'])]

        # A. Co-occurrance Matrix 
        basket = df_rec[['InvoiceNo', 'StockCode']].drop_duplicates()

        # Lọc top 500 sản phẩm bán chạy nhất 
        top_products = basket['StockCode'].value_counts().head(500).index
        filtered_basket = basket[basket['StockCode'].isin(top_products)]

        # Self-join để tìm các cặp sản phẩm mua cùng đơn hàng 
        merged = pd.merge(filtered_basket, filtered_basket, on='InvoiceNo', suffixes=('_A', '_B'))
        co_matrix = merged[merged['StockCode_A'] != merged['StockCode_B']]
        
        co_occurrence = co_matrix.groupby(['StockCode_A', 'StockCode_B']).size().reset_index(name='Co_Occurrence_Count')
        
        co_occurrence.to_sql('recommender_co_occurrence', self.conn, if_exists='replace', index=False)
        logging.info(f"-> Đã ghi {len(co_occurrence):,} cặp sản phẩm mua kèm vào bảng 'recommender_co_occurrence'.")

        # B. TF-IDF Vectors theo mô tả sản phẩm (Description)
        products = self.df[['StockCode', 'Description']].dropna().drop_duplicates(subset=['StockCode'])

        tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = tfidf.fit_transform(products['Description'])

        # Tính Cosine Similarity giữa các sản phẩm 
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuyển ma trận Cosine Similarity thành bảng Top-5 sản phẩm tương đồng
        sim_list = []
        stock_codes = products['StockCode'].values

        for idx, row in enumerate(cosine_sim):
            # Lấy top 5 sản phẩm tương đồng nhất (độ tương đồng < 0.99 để tránh trùng chính nó)
            top_indices = row.argsort()[-10:][::-1]
            for top_idx in top_indices:
                score = float(row[top_idx])
                if 0.2 < score < 0.99:
                    sim_list.append({
                        'StockCode': stock_codes[idx],
                        'Similar_StockCode': stock_codes[top_idx],
                        'Similarity_Score': round(row[top_idx], 4)
                    })

        tfidf_sim_df = pd.DataFrame(sim_list).drop_duplicates(subset=['StockCode', 'Similar_StockCode'])
        tfidf_sim_df.to_sql('recommender_tfidf_similarity', self.conn, if_exists='replace', index=False)
        logging.info(f"-> Đã ghi {len(tfidf_sim_df):,} cặp tương đồng TF-IDF vào bảng 'recommender_tfidf_similarity'.")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    fs = LocalFeatureStore()
    fs.load_data()

    # 1. KPIs theo tháng 
    monthly_summary  = fs.calculate_monthly_kpis()
    print("\n BÁO CÁO DOANH THU HÀNG THÁNG (TOP 10)")
    print(monthly_summary.head(10))

    # 2. User Features
    rfm_summary = fs.user_features()
    print("\n PHÂN BỔ 3 NHÓM KHÁCH HÀNG")
    print(rfm_summary['Customer_Segment'].value_counts())

    print("\n GIÁ TRỊ TRUNG BÌNH THEO NHÓM (R-F-M)")
    print(rfm_summary.groupby('Customer_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2))

    # 3. Time Series Features
    ts_df = fs.time_series_features()
    print("\n TOP 5 NGÀY MỚI NHẤT")
    print(ts_df.tail())

    # 4. Recommender Features
    fs.recommender_features()

    fs.close()
    print("\n HOÀN THÀNH PHẦN LOCAL FEATURE STORE")