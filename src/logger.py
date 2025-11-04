from pathlib import Path
import logging
from datetime import datetime

logs_dir = Path.cwd() / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = logs_dir / LOG_FILE

logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)