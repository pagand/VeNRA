from fastapi import FastAPI, HTTPException
from venra.logging_config import logger
from venra.db import init_db
from venra.models import VerificationRequest, VerificationResponse
from venra.sentinel import SentinelJudge

app = FastAPI(title="VeNRA: Verifiable Numerical Reasoning Agent")
judge = SentinelJudge()

@app.on_event("startup")
def on_startup():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized.")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "VeNRA Sentinel Service Online"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/verify", response_model=VerificationResponse)
async def verify_response(request: VerificationRequest):
    """
    Verifies the groundedness of an agent's response.
    Breakdown of analysis:
    1. Splits response into sentences.
    2. Checks each sentence against the provided context and trace.
    3. Calculates an overall groundedness score.
    """
    logger.info(f"Received verification request for query: {request.query[:50]}...")
    try:
        results = judge.verify_answer(request)
        logger.info(f"Verification complete. Score: {results.overall_groundedness_score}")
        return results
    except Exception as e:
        logger.error(f"Internal error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
