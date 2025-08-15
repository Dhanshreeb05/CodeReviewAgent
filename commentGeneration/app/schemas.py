from pydantic import BaseModel
from typing import Optional

class CommentRequest(BaseModel):
    code_diff: str
    request_id: Optional[str] = None

class CommentResponse(BaseModel):
    generated_comment: str
    confidence: float
    request_id: str
    processing_time_ms: int

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: str
    timestamp: str