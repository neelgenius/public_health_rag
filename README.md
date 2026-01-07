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


## Snapshots

Below are example outputs from the Retrieval-Augmented Generation (RAG) pipeline.

### Retrieved Context Chunks

<p align="center">
  <img width="1906" height="1080" alt="retrieved_chunks"
       src="https://github.com/user-attachments/assets/ebfaabbf-3dbb-42fb-bba8-e2077316875f" />
</p>

<br><br>

### Final Answer Generation

<p align="center">
  <img width="1920" height="1080" alt="final_answer"
       src="https://github.com/user-attachments/assets/26f8b2de-e20b-4936-a3ba-aac9844ea989" />
</p>





