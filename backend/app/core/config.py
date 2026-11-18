import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# If local cached models exist, default to offline mode to avoid network retry delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "ClinicalNoteAI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Base LLM and LoRA Adapter
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    USE_LORA_ADAPTER: bool = False
    ADAPTER_PATH: str = str(REPO_ROOT / "outputs" / "qwen_clinical_lora_mts")

    # Speech Recognition (Whisper)
    WHISPER_MODEL: str = "openai/whisper-small.en"

    # Storage paths
    UPLOAD_DIR: str = str(REPO_ROOT / "uploads")
    NOTES_DIR: str = str(REPO_ROOT / "generated_notes")

    # Audio limits
    MAX_AUDIO_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB
    MAX_AUDIO_DURATION_SECONDS: int = 3600  # 1 hour

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.NOTES_DIR).mkdir(parents=True, exist_ok=True)
