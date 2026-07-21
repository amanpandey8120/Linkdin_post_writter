import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(BASE_DIR / "linkedin_post_writer_lora"))

# Model configuration
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/qwen3-8b-unsloth-bnb-4bit")
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")

# Generation parameters
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TOP_K = int(os.getenv("TOP_K", "50"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
