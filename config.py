"""
Configuration for AI Quote Reel Bot (Google AI Stack)
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Google AI API
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_cloud_project: str = Field(default="", env="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", env="GOOGLE_CLOUD_LOCATION")
    
    # Google TTS
    google_tts_voice: str = Field(default="en-US-Studio-O", env="GOOGLE_TTS_VOICE")
    google_tts_language: str = Field(default="en-US", env="GOOGLE_TTS_LANGUAGE")
    
    # YouTube API
    youtube_client_id: str = Field(default="", env="YOUTUBE_CLIENT_ID")
    youtube_client_secret: str = Field(default="", env="YOUTUBE_CLIENT_SECRET")
    
    # Stock Video APIs (free)
    pexels_api_key: str = Field(default="", env="PEXELS_API_KEY")
    pixabay_api_key: str = Field(default="", env="PIXABAY_API_KEY")
    
    # Paths
    output_dir: Path = Field(default=Path("./output"))
    temp_dir: Path = Field(default=Path("./temp"))
    
    # Video Settings
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 30
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Quote categories
QUOTE_CATEGORIES = [
    "motivation", "success", "mindset", "discipline",
    "perseverance", "wisdom", "leadership", "growth"
]

# Story categories
STORY_CATEGORIES = [
    "moral", "funny", "anime", "horror", "inspirational"
]

# Content types for output organization
CONTENT_TYPES = ["quotes", "stories", "anime"]

# Visual themes for backgrounds
VISUAL_THEMES = [
    {"name": "sunset", "prompt": "golden sunset, warm cinematic lighting", "text_color": "#FFFFFF"},
    {"name": "ocean", "prompt": "calm ocean waves, serene blue hour", "text_color": "#FFFFFF"},
    {"name": "mountains", "prompt": "majestic mountains, dramatic sky", "text_color": "#FFFFFF"},
    {"name": "forest", "prompt": "misty forest, sunbeams through trees", "text_color": "#FFFFFF"},
    {"name": "space", "prompt": "cosmic nebula, stars, deep space", "text_color": "#FFFFFF"},
    {"name": "minimalist", "prompt": "soft gradient, clean minimal background", "text_color": "#333333"},
]


def get_settings() -> Settings:
    return Settings()
