from pydantic import BaseModel, Field
from typing import List 
# RECOMMENDER SCHEMAS 
class RecommenderRequest(BaseModel):
    stock_code: str = Field(..., example="85123A", description="Mã sản phẩm khách đang xem")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng gợi ý cần trả về")

class RecommenderItem(BaseModel):
    stock_code: str
    score: float
    strategy: str   # "Co-occurrence", "TF-IDF Content-Based", hoặc "Popular_Fallback"

class RecommenderResponse(BaseModel):
    input_stock_code: str
    recommendations: List[RecommenderItem]

# RFM CLUSTERING SCHEMAS
class RFMRequest(BaseModel):
    recency: float = Field(..., ge=0, example=15.0, description="Số ngày từ lần mua thứ nhất")
    frequency: float = Field(..., ge=1, example=5.0, description="Tổng số đơn hàng đã đặt")
    monetary: float = Field(..., ge=0, example=450.0, description='Tổng số tiền đã chi tiêu ($)')

class RFMResponse(BaseModel):
    recency: float
    frequency: float
    monetary: float
    cluster_label: int
    cluster_name: str