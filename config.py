"""
Configuration file for the PDF processing pipeline.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # Will raise error when get_db_connection is called

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "manuals"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"

# File paths
IMAGE_METADATA_FILE = PROCESSED_DIR / "image_metadata.json"
TEXT_CHUNKS_FILE = PROCESSED_DIR / "text_chunks.json"
LEXICAL_COMPONENTS_FILE = PROCESSED_DIR / "lexical_components.json"

# Processing parameters
MIN_CHUNK_LENGTH = 10  # Minimum length for text chunks
MAX_CHUNK_LENGTH = 500  # Maximum length for text chunks
IMAGE_MIN_SIZE = (50, 50)  # Minimum image dimensions to keep

# Caption patterns
CAPTION_PATTERNS = [
    r"Fig\.?\s*\d+[:\s]+.*?(?=\n|$)",
    r"Figure\s*\d+[:\s]+.*?(?=\n|$)",
    r"Abb\.?\s*\d+[:\s]+.*?(?=\n|$)",
    r"Image\s*\d+[:\s]+.*?(?=\n|$)",
    r"Bild\s*\d+[:\s]+.*?(?=\n|$)",  # German
    r"Рис\.?\s*\d+[:\s]+.*?(?=\n|$)",  # Russian
]

# Instruction patterns
INSTRUCTION_PATTERNS = [
    r"^\d+\.",  # Numbered list
    r"^[•·▪▫]",  # Bullet points
    r"^[a-zA-Z]\.",  # Lettered list
    r"^(Step|Procedure|Instruction|Note|Warning|Caution)",
    r"^[A-Z][a-z]+:",  # Bold headers
]

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Output format
OUTPUT_FORMAT = "json"

# Database mode: 'local' or 'remote' (default: 'remote')
DB_MODE = os.getenv("DB_MODE", "remote")

# Remote database connection parameters (used when DB_MODE=remote)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# Local database connection parameters (used when DB_MODE=local)
LOCAL_DB_HOST = os.getenv("LOCAL_DB_HOST", "localhost")
LOCAL_DB_PORT = int(os.getenv("LOCAL_DB_PORT", "5433"))
LOCAL_DB_NAME = os.getenv("LOCAL_DB_NAME", "multimodal_align")
LOCAL_DB_USER = os.getenv("LOCAL_DB_USER", "postgres")
LOCAL_DB_PASSWORD = os.getenv("LOCAL_DB_PASSWORD", "postgres")

# CLIP model configuration
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
CLIP_DIM = int(os.getenv("CLIP_DIM", "512"))


def get_db_connection(**kwargs):
    """
    Get a PostgreSQL database connection (local or remote based on DB_MODE).

    Args:
        **kwargs: Additional connection parameters (e.g., connect_timeout)

    Returns:
        psycopg2.connection: Database connection object

    Raises:
        ValueError: If required database environment variables are not set
        ImportError: If psycopg2 is not installed
    """
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is not installed. Install with: pip install psycopg2-binary"
        )

    if DB_MODE == "local":
        # Auto-start local database if not running
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from utils.manage_local_db import start_local_db, is_container_running

            if not is_container_running():
                print("🔄 Auto-starting local database...")
                start_local_db()
        except ImportError:
            # utils may not be importable in all contexts, that's okay
            pass
        except Exception as e:
            # If auto-start fails, continue anyway - user might start manually
            print(f"⚠️  Could not auto-start local database: {e}")

        host = LOCAL_DB_HOST
        port = LOCAL_DB_PORT
        dbname = LOCAL_DB_NAME
        user = LOCAL_DB_USER
        password = LOCAL_DB_PASSWORD
    else:
        # Remote database (existing logic)
        host = DB_HOST
        port = DB_PORT
        dbname = DB_NAME
        user = DB_USER
        password = DB_PASSWORD

        if not host:
            raise ValueError("DB_HOST environment variable is not set")
        if not dbname:
            raise ValueError("DB_NAME environment variable is not set")
        if not user:
            raise ValueError("DB_USER environment variable is not set")
        if not password:
            raise ValueError("DB_PASSWORD environment variable is not set")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        **kwargs,
    )
