import logging
import logging.handlers
from pathlib import Path

from theloop.services.config_manager import Settings


class LoggingConfigurator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def setup_logging(self) -> None:
        logger = logging.getLogger()
        logger.setLevel(self.settings.logging.level.value)
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(self.settings.logging.format)
        )
        logger.addHandler(console_handler)

        if (
            self.settings.logging.enable_file_logging
            and self.settings.logging.log_file_path
        ):
            log_path = Path(self.settings.logging.log_file_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=self.settings.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.settings.logging.backup_count,
            )
            file_handler.setFormatter(
                logging.Formatter(self.settings.logging.format)
            )
            logger.addHandler(file_handler)
