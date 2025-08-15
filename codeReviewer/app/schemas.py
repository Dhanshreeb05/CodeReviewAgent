from pydantic import BaseModel
from typing import Optional

class QualityRequest(BaseModel):
    code_diff: str
    request_id: Optional[str] = None

class QualityResponse(BaseModel):
    needs_review: bool
    confidence: float
    reasoning: Optional[str] = None
    request_id: str
    processing_time_ms: int

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: str
    timestamp: str