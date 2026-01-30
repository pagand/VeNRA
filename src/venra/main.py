from fastapi import FastAPI
from venra.logging_config import logger
from venra.db import init_db

app = FastAPI(title="VeNRA: Verifiable Numerical Reasoning Agent")

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
