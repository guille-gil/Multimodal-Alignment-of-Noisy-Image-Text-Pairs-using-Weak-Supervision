# Multimodal Alignment in Industrial Maintenance Manuals Using Weakly Supervision based on Positional Proximity

Industrial maintenance manuals combine text and images to describe procedures, yet their alignment is often weak or inconsistent. Most multimodal large language models assume perfectly paired data, limiting their applicability to such noisy domains. This work investigates weak supervision based on positional proximity of image-text to infer potential correspondences. By treating spatial and sequential closeness between textual segments and figures as a proxy for semantic relatedness, pseudo-pairs are generated without manual annotation. A lightweight adapter is then trained on these pairs to align embeddings from a pretrained CLIP model. Evaluation on document-style datasets demonstrates that proximity-based weak supervision improves retrieval precision and multimodal coherence compared to vanilla CLIP, providing a foundation for adapting MLLMs to low-alignment industrial corpora.

## Citation 

Gil de Avalle, G.; Maruster, M.; Emmanouilidis, C. (2025).

## Overview

This system implements a full pipeline for:
1. **Document Processing**: Extract images and text chunks from PDF/Word documents
2. **Vector Database Setup**: PostgreSQL with pgvector for efficient similarity search
3. **CLIP Embedding Computation**: OpenCLIP-based embeddings (fully local, open-source)
4. **Weak Supervision**: Local and global proximity alignment strategies
5. **Evaluation**: Comprehensive metrics and visualizations

## Features

- **Multi-format Support**: PDF and Word documents (.pdf, .docx, .doc)
- **Robust Image Extraction**: Native, vector, and fallback bounding box detection
- **Intelligent Text Chunking**: Instruction-level granularity with bounding boxes
- **Weak Supervision Strategies**: 
  - Local proximity (bounding box proximity on same page)
  - Global proximity (page distance across document)
  - Combined (both local and global)
- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Open-Source CLIP**: Fully local embedding computation (no data leakage)
- **Comprehensive Evaluation**: Metrics and visualizations

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Multimodal-Alignment-of-Noisy-Image-Text-Pairs-using-Weak-Supervision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install spaCy Language Model

For Dutch (default):
```bash
python3 -m spacy download nl_core_news_sm
```

For English:
```bash
python3 -m spacy download en_core_web_sm
```

### 4. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
# Database connection
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432

# CLIP model configuration
CLIP_MODEL=ViT-B-32  # Options: ViT-B-32, ViT-L-14, ViT-H-14
CLIP_DIM=512  # Must match model (512, 768, or 1024)
CLIP_PRETRAINED=openai  # or laion2b_s34b_b79k for fully open-source

# Processing settings
LANGUAGE=nl  # nl (Dutch) or en (English)
```

## Quick Start

### Run Complete Pipeline

```bash
python3 src/run_pipeline.py
```

This will:
1. Process PDF/Word documents in `data/raw/manuals/`
2. Set up database (if needed)
3. Compute and insert CLIP embeddings
4. Run evaluation and generate metrics

### Skip Steps (if already completed)

```bash
# Skip database setup (if already configured)
python3 src/run_pipeline.py --skip-db

# Skip embedding insertion (if already done)
python3 src/run_pipeline.py --skip-embeddings

# Force re-run everything
python3 src/run_pipeline.py --force
```

## Pipeline Steps

### Step 1: Document Processing

Extracts images and text chunks from PDF/Word documents.

```bash
python3 src/pdf_processor.py
```

**Output:**
- `data/processed/images/`: Extracted images
- `data/processed/image_metadata.json`: Image metadata with bounding boxes
- `data/processed/text_chunks.json`: Text chunks with bounding boxes

### Step 2: Database Setup

Creates PostgreSQL schemas and tables with pgvector support.

```bash
python3 src/setup_vector_db.py
```

Creates 4 schemas:
- `vanilla_clip`: Pure CLIP embeddings (no weak supervision)
- `clip_local`: CLIP + local proximity (bounding box proximity on same page)
- `clip_global`: CLIP + global proximity (page distance across document)
- `clip_combined`: CLIP + both local and global proximity

### Step 3: Embedding Insertion

Computes CLIP embeddings using OpenCLIP and inserts into database.

```bash
# Insert into all schemas
python3 src/insert_clip_embeddings.py

# Insert into specific schema
python3 src/insert_clip_embeddings.py vanilla_clip
```

**Note:** First run will download the CLIP model (cached locally afterward).

### Step 4: Evaluation

Computes metrics and generates visualizations.

```bash
python3 src/evaluate_alignments.py
```

**Output:**
- `evaluation_results/metrics.json`: Comprehensive metrics
- `evaluation_results/similarity_distributions.png`: Similarity score distributions
- `evaluation_results/top_k_comparison.png`: Top-K accuracy comparison
- `evaluation_results/weak_supervision_scores.png`: Weak supervision distributions

## Evaluation Metrics

The evaluation script computes:

- **Top-K Accuracy**: Percentage of true pairs found in top K results (K=1, 5, 10, 20)
- **Mean Reciprocal Rank (MRR)**: Average rank of true matches
- **Average Similarity**: Mean cosine similarity for true pairs
- **Weak Supervision Stats**: Distribution and quality of alignment scores

Visualizations include:
- Similarity score distributions across schemas
- Top-K accuracy comparison
- Weak supervision score distributions

## Directory Structure

```
.
├── data/
│   ├── raw/
│   │   └── manuals/          # Input PDF/Word documents
│   └── processed/
│       ├── images/            # Extracted images
│       ├── image_metadata.json
│       ├── text_chunks.json
├── src/                        # Main source code
│   ├── pdf_processor.py           # Document processing
│   ├── setup_vector_db.py         # Database setup
│   ├── insert_clip_embeddings.py  # Embedding computation & insertion
│   ├── evaluate_alignments.py     # Evaluation
│   └── run_pipeline.py            # Main orchestrator
├── utils/                       # Utility scripts
│   ├── bbox_image_check.py      # Bounding box checker
│   └── test_installation.py     # Installation test
├── docs/                        # Documentation
│   ├── DATABASE_SETUP.md
│   ├── QUICK_START.md
│   └── SECURITY.md
├── evaluation_results/          # Evaluation metrics and charts
├── config.py                    # Configuration
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- **Database**: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`
- **CLIP Model**: `CLIP_MODEL`, `CLIP_DIM`, `CLIP_PRETRAINED`
- **Processing**: `LANGUAGE`, `USE_OCR_FALLBACK`, `MAX_FILE_SIZE_MB`

### Model Options

**CLIP Models:**
- `ViT-B-32`: 512 dim, fastest (recommended)
- `ViT-L-14`: 768 dim, better quality
- `ViT-H-14`: 1024 dim, best quality, slowest

**Pretrained Weights:**
- `openai`: OpenAI weights (compatible, cached locally)
- `laion2b_s34b_b79k`: Fully open-source LAION weights

## Technical Details

### Image Extraction
- Uses PyMuPDF for raster image extraction
- Detects vector graphics as figures
- Multiple fallback strategies for bounding box detection
- Filters invalid bounding boxes

### Text Chunking
- Groups words into lines by vertical proximity
- Identifies instruction patterns (numbered lists, bullets, headers)
- Splits non-instruction text by sentences
- Computes bounding boxes for spatial alignment

### Weak Supervision

**Local Proximity:**
- Computes bounding box proximity on the same page
- Uses Intersection over Union (IoU) for overlapping bounding boxes
- Distance-based scoring for non-overlapping elements on the same page

**Global Proximity:**
- Computes page distance across the entire document
- Same page pairs get score of 1.0
- Exponential decay with page distance

**Combined Alignment:**
- Average of local and global proximity scores

### Vector Database

- PostgreSQL with pgvector extension
- HNSW indexes for fast similarity search
- Normalized embeddings for cosine similarity
- Supports queries like:
  ```sql
  SELECT image_id, 1 - (clip_embedding <=> %s::vector) AS similarity
  FROM vanilla_clip.images
  ORDER BY clip_embedding <=> %s::vector
  LIMIT 10;
  ```

## Error Handling

The pipeline includes comprehensive error handling:
- Graceful handling of corrupted documents
- Skip problematic images while continuing processing
- Fallback strategies for missing bounding boxes
- Clear error messages and logging

## Performance Considerations

- Processes documents sequentially to avoid memory issues
- Efficient vector indexing with HNSW/IVFFlat
- Batch insertion for embeddings
- Progress bars for long-running operations
- Cached CLIP model after first download

## Security

- Sensitive data (manuals, processed data) excluded from Git
- Environment variables for configuration
- Local-only CLIP processing (no data sent externally)
- See `SECURITY.md` for details

## Troubleshooting

### Database Connection Issues
- Verify credentials in `.env`
- Check PostgreSQL is running
- Ensure pgvector extension is installed

### CLIP Model Download Fails
- Check internet connection (first download only)
- Verify disk space
- Models are cached locally after download

### Missing Dependencies
```bash
pip install -r requirements.txt
python3 -m spacy download nl_core_news_sm  # or en_core_web_sm
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
