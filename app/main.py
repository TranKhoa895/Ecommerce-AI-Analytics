from fastapi import FastAPI, HTTPException
from app.schemas import RecommenderRequest, RecommenderResponse, RFMRequest, RFMResponse
from app.services import InferenceService

app = FastAPI(
    title="E-commerce Real-time Serving API",
    description="Hệ thống API theo thời gian thực phục vụ nhu cầu gợi ý sản phẩm (Recommender) và phân cụm nhóm khách hàng (RFM)",
    version="1.0.0"
)

inference_service = InferenceService()

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy", "service": "E-commerce Fast API Service"}

@app.post("/api/v1/recommend", response_model=RecommenderResponse, tags=["Recommender Engine"])
def recommend_products(payload: RecommenderRequest):
    try:
        results = inference_service.recommend_items(
            stock_code=payload.stock_code,
            top_k=payload.top_k
        )
        return {
            "input_stock_code": payload.stock_code, 
            "recommendations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Inference Recommender: {str(e)}")

@app.post("/api/v1/segment-customer", response_model=RFMResponse)
def segment_customer(payload: RFMRequest):
    r = payload.recency
    f = payload.frequency
    m = payload.monetary

    # 1. Tính điểm RFM Score (Thang điểm 1-3)
    # Recency: Số ngày càng nhỏ điểm càng cao
    r_score = 3 if r <= 15 else (2 if r <= 60 else 1)

    # Frequency: Tần suất đặt hàng
    f_score = 3 if f >= 10 else (2 if m >= 3 else 1)

    # Monetary: Tổng tiền chi tiêu
    m_score = 3 if m >= 1000 else (2 if m >= 200 else 1)

    # 2. Phân loại 3 nhóm khách hàng 
    # VIP/Champion: Mua gần đây (R >= 2) và (F=3 hoặc M=3)
    if r_score >= 2 and (f_score == 3 or m_score == 3):
        cluster_label = 1
        cluster_name = "VIP/Champion"
    # Inactive: Lâu không quay lại mua hàng (R=1)
    elif r_score == 1:
        cluster_label = 3
        cluster_name = "Inactive"
    # Regulars: Duy trì mua hàng ổn định
    else: 
        cluster_label = 2
        cluster_name = "Regulars"
    
    return {
        "recency": r,
        "frequency": f,
        "monetary": m,
        "cluster_label": cluster_label,
        "cluster_name": cluster_name
    }
    