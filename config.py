"""
Configuration file for the PDF processing pipeline.
"""

import os
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

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# CLIP model configuration
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
CLIP_DIM = int(os.getenv("CLIP_DIM", "512"))


def get_db_connection(**kwargs):
    """
    Get a PostgreSQL database connection.

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

    if not DB_HOST:
        raise ValueError("DB_HOST environment variable is not set")
    if not DB_NAME:
        raise ValueError("DB_NAME environment variable is not set")
    if not DB_USER:
        raise ValueError("DB_USER environment variable is not set")
    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD environment variable is not set")

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        **kwargs,
    )
