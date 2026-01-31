import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- API Keys ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
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
    PROMPTS_PATH: str = "assets/PROMPTS.md"

    # --- Storage ---
    CHROMA_DB_PATH: str = "data/chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore" # Ignore unknown keys in .env
    )

settings = Settings()
