# 📊 E-Commerce AI Analytics & Real-Time Recommendation System

Hệ thống Phân tích & Gợi ý Sản phẩm thời gian thực cho E-Commerce: Tích hợp thuật toán Machine Learning (LightGBM, K-Means, Co-occurrence/TF-IDF), API Backend (FastAPI), Giải thích mô hình (SHAP) và Giao diện Dashboard quản trị (Streamlit).

---

## 📌 1. Bối Cảnh Kinh Doanh (Business Context) & Mục Tiêu

### 🛍️ Bài toán 1: Tối ưu hóa Doanh thu & Cá nhân hóa (Cross-sell / Up-sell)

* **Thực trạng:**
  * Tỷ lệ giá trị đơn hàng trung bình (AOV) thấp do khách hàng thường chỉ mua các mặt hàng đơn lẻ.
  * **24.93%** số dòng giao dịch không có `CustomerID` (khách vãng lai), làm vô hiệu hóa các thuật toán User-based Collaborative Filtering truyền thống (Bài toán Cold-start).

* **Mục tiêu:**
  * Xây dựng cơ chế Gợi ý lai (Hybrid Recommendation) kết hợp *Item-based Co-occurrence Matrix* và *Content-based TF-IDF Similarity* dựa trên mô tả sản phẩm (`Description`).
  * Cho phép gợi ý thời gian thực cho cả người dùng cũ lẫn khách vãng lai, hướng tới tăng Tỷ lệ nhấp (CTR) và tăng AOV từ 8–10%.

---

### 📦 Bài toán 2: Tối ưu hóa Chi phí & Quản trị Tồn kho (Demand & Inventory Optimization)

* **Thực trạng:**
  * Đội ngũ Procurement thu mua hàng dựa trên cảm tính hoặc trung bình động đơn giản, gây ra tình trạng đứt hàng (Stockout) các SKU bán chạy trong mùa cao điểm (Tháng 10 - Tháng 12), đồng thời dư thừa vốn ở các mã hàng bán chậm.
  * Tập dữ liệu gốc chứa **10,624** giao dịch hủy/trả (`Quantity < 0`) gây nhiễu nghiêm trọng cho biểu đồ chuỗi thời gian.

* **Mục tiêu:**
  * Tiền xử lý lọc nhiễu đơn hủy.
  * Xây dựng mô hình chuỗi thời gian LightGBM Regressor dự báo nhu cầu bán ra theo cửa sổ 30–90 ngày cho 4,070 SKUs.
  * Kiểm soát sai số WAPE < 18% và giảm 15% tỷ lệ đứt hàng.
---

## 🏗️ 2. Kiến Trúc Dữ Liệu & Hệ Thống (Data Architecture)

Hệ thống được thiết kế theo luồng kiến trúc end-to-end rời rạc (Decoupled Architecture) giữa Backend Serving và Frontend Dashboard:

```text
[ Data Sources / Raw Logs ]
           │
           ▼
[ 1. Data Cleaning Pipeline (src/1_data_processing.py) ]
           │
           ▼
[ 2. Feature Store Builder (src/2_feature_store.py) ]
           │
           ├───────────────────────────────┐
           ▼                               ▼
[ SQLite Feature Store ]        [ Storage (data/forecast.parquet) ]
(recommender_co_occurrence,                │
 time_series_features, etc.)               │
           │                               │
           ▼                               │
[ FastAPI Backend (app/main.py) ]          │
- GET  /health                             │
- POST /api/v1/recommend                   │
- POST /api/v1/segment-customer            │
           │                               │
           └───────────────┬───────────────┘
                           ▼
          [ Streamlit Frontend (streamlit_app/app.py) ]
- Tab 1: Recommender Engine & RFM Simulator
- Tab 2: Demand Forecasting & SHAP Explainability

```

---

## 📁 3. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
Project_02/
├── app/                           # FastAPI Backend Service
│   ├── __init__.py
│   ├── main.py                    # Khởi tạo REST API Routes & App Lifecycle
│   ├── schemas.py                 # Pydantic Schemas (Input/Output Validation)
│   └── services.py                # Core Business Logic & Database Engine Queries
├── data/                          # Thư mục lưu trữ dữ liệu đầu ra
│   ├── data_clean.csv.gz          # Dữ liệu sạch đã nén
│   └── forecast.parquet           # Kết quả dự báo 30-90 ngày dạng Parquet
├── src/                           # Machine Learning Scripts & Pipeline
│   ├── __init__.py
│   ├── 1_data_processing.py       # Data Cleaning & Validation (Xử lý 10,624 đơn hủy)
│   ├── 2_feature_store.py         # Feature Store Engine (Lag, Rolling Mean per SKU)
│   ├── 3a_train_rfm_kmeans.py     # Train Scaler & Model K-Means RFM Segmentation
│   ├── 3b_train_forecasting.py    # Offline Training Model LightGBM Dự Báo Nhu Cầu
│   ├── 3c_train_recommender.py    # Offline Training Engine Recommender (Co-occurrence & TF-IDF)
│   └── 4_batch_forecast.py        # Batch Service: LightGBM SKU-Level Recursive Forecasting
├── streamlit_app/                 # Streamlit Frontend UI Dashboard
│   └── app.py                     # Dynamic Interactive Dashboard
├── ecommerce.db                   # Feature Store & Analytical SQLite Database
├── mlruns/                        # Thư mục quản lý thực nghiệm MLflow (nếu có)
├── Dockerfile                     # Docker Build Recipe
├── docker-compose.yml             # Cấu hình triển khai Docker Container
├── requirements.txt               # Danh sách phiên bản thư viện hệ thống
├── .gitignore                     # Cấu hình bỏ qua file rác / môi trường ảo
└── README.md                      # Tài liệu hướng dẫn vận hành dự án

```

---

## ⚙️ 4. Luồng Xử Lý & Nghiệp Vụ Logics (ETL & Business Rules)

### A. Recommender Engine (3-Tier Fallback)

1. **Tầng 1 (Co-occurrence Matrix):** Tìm các sản phẩm hay được mua cùng nhau nhất dựa trên lịch sử hóa đơn (`recommender_co_occurrence`).
2. **Tầng 2 (TF-IDF Similarity Fallback):** Nếu sản phẩm chưa có giao dịch mua kèm, hệ thống chuyển sang đo độ tương đồng về mặt mô tả/tên sản phẩm (`recommender_tfidf`).
3. **Tầng 3 (Popularity Fallback):** Trường hợp mã sản phẩm mới hoặc chưa có lịch sử, hệ thống gợi ý Top các sản phẩm bán chạy nhất toàn sàn (giải quyết bài toán Cold-start cho 24.93% lượt truy cập vô danh).

### B. RFM Segmentation Simulator

* **Input Vector:**

$$X = [\log(1 + \text{Recency}), \log(1 + \text{Frequency}), \log(1 + \text{Monetary})]$$


* **Pipeline:** $\text{StandardScaler} \rightarrow \text{K-Means Clustering} \rightarrow \text{Ép kiểu int JSON Serializable} \rightarrow \text{Mapping nhãn Business Rules}$:
* **VIP/Champions:** Khách hàng mua gần đây ($R \ge 2$) và có Tần suất hoặc Doanh số ở top cao nhất ($F = 3$ hoặc $M = 3$).
* **Inactive:** Khách hàng lâu không quay lại mua hàng ($R = 1$).
* **Regulars:** Khách hàng mới hoặc khách hàng duy trì lịch sử mua hàng ổn định.



### C. Time Series Forecasting & SHAP

* **Target Variable:**
* 
  $$\text{Target} = \log(1 + \text{Daily Revenue})$$


* **Feature Matrix:** `lag_1`, `lag_7`, `lag_30`, `rolling_mean_7`, `rolling_mean_30`.
* **Recursive Prediction:** Vòng lặp đệ quy 30 bước, mỗi bước tính toán lại biến Lag/Rolling từ kết quả dự báo của bước trước đó.
* **SHAP Explainability:** Giải thích mô hình bằng `TreeExplainer`, xuất đồ thị Summary Bar Plot và Waterfall Plot chi tiết tác động của từng đặc trưng.

---

## 🛠️ 5. Cài Đặt Môi Trường (Installation)

### Bước 1: Clone Repository & Chuẩn bị thư mục

```powershell
git clone [https://github.com/your-username/ecommerce-ai-analytics.git](https://github.com/your-username/ecommerce-ai-analytics.git)
cd ecommerce-ai-analytics

```

### Bước 2: Tạo và Kích hoạt Môi trường ảo (.venv)

* **Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate

```


* **macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate

```



### Bước 3: Cài đặt Dependencies

```powershell
pip install -r requirements.txt

```

---

## 🚀 6. Hướng Dẫn Vận Hành Hệ Thống (Usage)

Vận hành dự án theo 3 bước chuẩn:

1️⃣ Bước 1: Chạy Data Pipeline, Huấn Luyện Mô Hình & Batch Service
Khởi tạo Feature Store SQLite, huấn luyện mô hình và xuất file kết quả dự báo mới nhất:
python src/1_data_processing.py
python src/2_feature_store.py
python src/3a_train_rfm_kmeans.py
python src/3b_train_forecasting.py
python src/3c_train_recommender.py
python src/4_batch_forecast.py

2️⃣ Bước 2: Khởi chạy FastAPI Backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

* Truy cập Swagger UI kiểm thử API tại: http://127.0.0.1:8000/docs

3️⃣ Bước 3: Khởi chạy Streamlit Frontend (Terminal 2)
Mở một cửa sổ Terminal mới, kích hoạt .venv và chạy:
streamlit run streamlit_app/app.py

* Truy cập Dashboard quản trị tại: http://localhost:8501

---

## 📌 7. Giả Định & Hạn Chế (Assumptions & Limitations)

* **Feature Store:** Dự án hiện dùng SQLite local nhằm phục vụ mục đích thử nghiệm và demo local. Khi đưa lên Production quy mô lớn, nên chuyển sang PostgreSQL hoặc Snowflake/BigQuery.
* **Dự báo đệ quy (Recursive Loop):** Phương pháp dự báo đệ quy 30 ngày có thể bị tích lũy sai số (error accumulation) ở các ngày cuối chu kỳ.
* **Thời gian thực:** Engine recommender đọc từ bảng tính toán sẵn (Pre-computed Co-occurrence matrix). Việc cập nhật matrix cần các Job chạy định kỳ hàng đêm (Nightly Batch Jobs).

---

## 📈 8. Kết Quả Kỹ Thuật & Giá Trị Doanh Nghiệp (Business Impact)

| Thành phần hệ thống | Thuật toán / Kỹ thuật | Giải pháp cho bài toán | Chỉ số mục tiêu (Target KPI) |
| --- | --- | --- | --- |
| **Real-time Recommender** | Co-occurrence + TF-IDF Description | Giải quyết Cold-start (24.93% khách vãng lai) bằng Item-based Filtering | Tăng 8–10% AOV, Latency $< 50\text{ms}$ |
| **RFM Segmentation** | K-Means + Rule-based Mapping | Phân cụm 3 nhóm khách hàng chiến lược (VIP/Champions, Regulars, Inactive) | Tối ưu hóa ngân sách Marketing |
| **Demand Forecasting** | LightGBM + Lag/Rolling Features | Làm sạch 10,624 đơn hủy, dự báo đệ quy 30–90 ngày cho 4,070 SKUs | $\text{WAPE} < 18\%$, Giảm 15% Stockout |
| **Model Explainability** | SHAP TreeExplainer | Biểu diễn yếu tố ảnh hưởng đến doanh thu (Summary & Waterfall Plot) | Minh bạch hóa quyết định kinh doanh |

---

## 👤 9. Tác Giả & Bản Quyền (Contributors & Data Source)

* **Tác giả:** Tran Phuoc Khoa - Data Scientist / Machine Learning Engineer
* **Nguồn dữ liệu:** [E-Commerce Data (Kaggle)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)


