'''
STAGE 1: DATA INGESTION, CLEANING & PIPELINE OPTIMIZATION
'''
import pandas as pd 
import numpy as np 
import sqlite3
import os 
import logging 

logging .basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    """Pipeline  làm sạch và chuẩn hóa dữ liệu giao dịch bán lẻ."""

    NON_PRODUCT_CODES = [
        'POST', 'D', 'M', 'BANK CHARGES', 'PADS', 
        'DOT', 'CRUK', 'AMAZONFEE', 'S', 'B', 'POSTAGE'
    ]

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df = None
        self.clean_df = None
        self.returns_df = None

    def load_data(self, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
        """Nạp dữ liệu thô từ file CSV."""
        logging.info(f"Đang nạp dữ liệu từ: {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path, encoding=encoding)
        logging.info(f"Kích thước ban đầu: {self.df.shape[0]:,} dòng, {self.df.shape[1]} cột")
        return self.df

    def clean(self) -> pd.DataFrame:
        """Thực hiện chuỗi quy tắc làm sạch dữ liệu."""
        if self.df is None:
            raise ValueError("Dữ liệu chưa được nạp. Hãy gọi load_data() trước.")

        df_work = self.df.copy()

        # 1. Chuẩn hóa InvoiceDate & Text fields 
        df_work['InvoiceDate'] = pd.to_datetime(df_work['InvoiceDate'])
        df_work['StockCode'] = df_work['StockCode'].astype(str).str.strip().str.upper()
        df_work['InvoiceNo'] = df_work['InvoiceNo'].astype(str).str.strip()
        df_work['Description'] = df_work['Description'].fillna('UNKNOWN ITEM').astype(str).str.strip()

        # 2. Nhận diện đơn hủy / trả hàng
        is_cancelled = df_work['InvoiceNo'].str.startswith('C') | (df_work['Quantity'] < 0)
        self.returns_df = df_work[is_cancelled].copy()
        logging.info(f"Phát hiện {len(self.returns_df):,} giao dịch hủy/trả hàng.")

        # 3. Lọc bỏ mã phi hàng hóa & đơn giá bất hợp lệ
        is_non_product = df_work['StockCode'].isin(self.NON_PRODUCT_CODES) | df_work['StockCode'].str.contains('POSTAGE', regex=False)
        is_invalid_price = df_work['UnitPrice'] <= 0 

        # 4. Giữ lại giao dịch bán hàng hợp lệ (Net Sales)
        valid_mask = ~is_cancelled & ~is_non_product & ~is_invalid_price
        self.clean_df = df_work[valid_mask].copy()

        # Tính toán giá trị đơn hàng (Revenue)
        self.clean_df['Revenue'] = self.clean_df['Quantity'] * self.clean_df['UnitPrice']

        logging.info(f"Hoàn thành làm sạch. Dữ liệu bán hàng hợp lệ: {len(self.clean_df):,} dòng")
        return self.clean_df

    def export(self, db_path: str = 'ecommerce.db', csv_gz_path: str = 'data_clean.csv.gz'):
        """Xuất hiện dữ liệu sạch ra SQLite Database và Compressed CSV."""
        if self.clean_df is None:
            raise ValueError("Chưa có dữ liệu sạch đề suất. Hãy gọi clean() trước.")

        # Xuất file CSV nén
        self.clean_df.to_csv(csv_gz_path, compression='gzip', index=False)
        logging.info(f"Đã lưu file nén: {csv_gz_path}")

        # Lưu vào SQLite DB
        conn = sqlite3.connect(db_path)
        self.clean_df.to_sql('clean_sales', conn, if_exists='replace', index=False)
        self.returns_df.to_sql('cancelled_sales', conn, if_exists='replace', index=False)
        conn.close()
        logging.info(f"Đã lưu bảng 'clean_sales' và 'cancelled_sales' vào SQLite: {db_path}")

if __name__ == '__main__':
    pipeline = DataPipeline(raw_data_path='data/data.csv')
    pipeline.load_data()
    pipeline.clean()
    pipeline.export()