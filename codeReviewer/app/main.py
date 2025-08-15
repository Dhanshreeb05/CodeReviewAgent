from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import logging
from datetime import datetime

from .schemas import QualityRequest, QualityResponse, ErrorResponse
from .model import model_wrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Code Review Quality Estimation API",
    description="Determines if code diff needs review",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_wrapper.load_model()
        logger.info("Quality service started successfully")
    except Exception as e:
        logger.error(f"Failed to start quality service: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper.model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict-quality", response_model=QualityResponse)
async def predict_quality(request: QualityRequest):
    """Predict if code diff needs review"""
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing quality prediction request {request_id}")
        
        # Validate input
        if not request.code_diff.strip():
            raise HTTPException(status_code=400, detail="Code diff cannot be empty")
        
        # Run prediction
        needs_review, confidence, reasoning = model_wrapper.predict(request.code_diff)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = QualityResponse(
            needs_review=needs_review,
            confidence=confidence,
            reasoning=reasoning,
            request_id=request_id,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Request {request_id} completed in {processing_time}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)