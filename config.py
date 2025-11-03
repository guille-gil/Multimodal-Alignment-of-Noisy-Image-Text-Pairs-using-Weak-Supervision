"""
Configuration file for the PDF processing pipeline.
"""

from pathlib import Path

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
