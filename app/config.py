from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment or .env")
    origins = os.getenv("CORS_ORIGINS", "*").strip()
    return Settings(supabase_url=url, supabase_key=key, cors_origins=origins)
