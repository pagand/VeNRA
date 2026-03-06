import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory (assuming config.py is in src/venra)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    @property
    def GROQ_KEYS(self) -> List[str]:
        key_names = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_8"]
        keys = [os.getenv(k) for k in key_names if os.getenv(k)]
        if not keys and self.GROQ_API_KEY:
            keys = [self.GROQ_API_KEY]
        return keys

    @property
    def NVIDIA_KEYS(self) -> List[str]:
        key_names = ["NVIDIA_API_KEY", "NVIDIA_API_KEY_2"]
        keys = [os.getenv(k) for k in key_names if os.getenv(k)]
        if not keys and self.NVIDIA_API_KEY:
            keys = [self.NVIDIA_API_KEY]
        return keys

    @property
    def OPENROUTER_KEYS(self) -> List[str]:
        key_names = ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]
        keys = [os.getenv(k) for k in key_names if os.getenv(k)]
        if not keys and self.OPENROUTER_API_KEY:
            keys = [self.OPENROUTER_API_KEY]
        return keys

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

    # --- Agent Flags ---
    ENABLE_AGENT_SYNTHESIS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore" # Ignore unknown keys in .env
    )

settings = Settings()

