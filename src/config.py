from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

WHO_DIR = RAW_PDF_DIR / "who"
CDC_DIR = RAW_PDF_DIR / "cdc"

CHUNK_OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"

# Chunking parameters
CHUNK_WORDS = 500
CHUNK_OVERLAP = 80
