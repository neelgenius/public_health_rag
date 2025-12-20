# Public Health RAG

A Retrieval-Augmented Generation (RAG) system for public health documents.

## Project Structure

- `data/`: Data directory containing raw PDFs and processed documents
- `src/`: Source code for the RAG pipeline
  - `ingest/`: Document ingestion and preprocessing
  - `indexing/`: Indexing modules (vector and BM25)
  - `retrieval/`: Document retrieval module
  - `config.py`: Configuration settings

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Place raw PDFs in `data/raw_pdfs/`
3. Run the pipeline to process and index documents

## Dependencies

See `requirements.txt` for required packages.
