import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory (assuming config.py is in src/venra)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # --- Models ---
    # We use Groq for fast inference
    SLM_MODEL_FAST: str = "llama-3.1-8b-instant"
    SLM_MODEL_PRECISION: str = "llama-3.3-70b-versatile"
    
    # --- Confidence Thresholds ---
    CONFIDENCE_TABLE: float = 0.95
    CONFIDENCE_TEXT_HIGH: float = 0.85
    CONFIDENCE_TEXT_LOW: float = 0.60

    # --- Prompt Paths ---
    PROMPTS_PATH: str = str(PROJECT_ROOT / "assets" / "PROMPTS.md")

    # --- Storage ---
    DATA_DIR: str = str(PROJECT_ROOT / "data")
    CHROMA_DB_PATH: str = str(PROJECT_ROOT / "data" / "chroma_db")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore" # Ignore unknown keys in .env
    )

settings = Settings()
