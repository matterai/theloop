import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingSettings(BaseModel):
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    enable_file_logging: bool = Field(default=False)
    log_file_path: str | None = Field(default=None)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    backup_count: int = Field(default=3, ge=1, le=10)


class Settings(BaseModel):
    chunk_size: int = Field(default=1024 * 1024, ge=1024)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


class ConfigManager:
    def __init__(
        self, app_name: str = "theloop", config_filename: str = "settings.json"
    ) -> None:
        self.config_dir = Path.home() / f".{app_name}"
        self.config_file = self.config_dir / config_filename

    def ensure_config_dir(self) -> None:
        self.config_dir.mkdir(exist_ok=True, parents=True)

    def load_config(self) -> Settings:
        self.ensure_config_dir()

        if not self.config_file.exists():
            return Settings()

        with open(self.config_file, "r") as f:
            data = json.load(f)

        return Settings(**data)

    def save_config(self, settings: Settings) -> None:
        self.ensure_config_dir()

        with open(self.config_file, "w") as f:
            json.dump(settings.model_dump(), f, indent=2)

    def update_logging_level(self, level: LogLevel) -> None:
        settings = self.load_config()
        settings.logging.level = level
        self.save_config(settings)
