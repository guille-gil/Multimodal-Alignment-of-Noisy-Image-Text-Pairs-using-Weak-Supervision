# Multimodal Alignment of Noisy Image-Text Pairs using Weak Supervision

A complete pipeline for processing technical manuals (PDF/Word), extracting multimodal data, computing CLIP embeddings, and evaluating alignment quality with weak supervision strategies.

## Overview

This system implements a full pipeline for:
1. **Document Processing**: Extract images, text chunks, and lexical components from PDF/Word documents
2. **Lexical Component Filtering**: Operator-in-the-loop filtering of vocabulary terms
3. **Vector Database Setup**: PostgreSQL with pgvector for efficient similarity search
4. **CLIP Embedding Computation**: OpenCLIP-based embeddings (fully local, open-source)
5. **Weak Supervision**: Lexical and positional alignment strategies
6. **Evaluation**: Comprehensive metrics and visualizations

## Features

- **Multi-format Support**: PDF and Word documents (.pdf, .docx, .doc)
- **Robust Image Extraction**: Native, vector, and fallback bounding box detection
- **Intelligent Text Chunking**: Instruction-level granularity with bounding boxes
- **Lexical Component Analysis**: Noun extraction with frequency tracking
- **Weak Supervision Strategies**: 
  - Lexical component alignment
  - Positional (bounding box) alignment
  - Combined alignment
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
2. Prompt you to filter lexical components
3. Set up database (if needed)
4. Compute and insert CLIP embeddings
5. Run evaluation and generate metrics

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

Extracts images, text chunks, and lexical components from PDF/Word documents.

```bash
python3 src/pdf_processor.py
```

**Output:**
- `data/processed/images/`: Extracted images
- `data/processed/image_metadata.json`: Image metadata with bounding boxes
- `data/processed/text_chunks.json`: Text chunks with bounding boxes
- `data/processed/lexical_components.json`: Vocabulary with frequencies

### Step 2: Lexical Component Filtering

Operator-in-the-loop step to filter non-relevant terms.

1. Review extracted components:
```bash
python3 -c "import json; d=json.load(open('data/processed/lexical_components.json')); [print(f\"{i+1:3d}. {c['term']:30s} ({c['count']})\") for i,c in enumerate(d['components'][:30])]"
```

2. Edit `src/filter_lexical_components.py` and add terms to exclude:
```python
EXCLUDE_TERMS = {
    "proce",      # Example: truncation
    "visionplaa", # Example: truncation
    # Add your terms here
}
```

3. Run filtering:
```bash
python3 src/filter_lexical_components.py
```

### Step 3: Database Setup

Creates PostgreSQL schemas and tables with pgvector support.

```bash
python3 src/setup_vector_db.py
```

Creates 4 schemas:
- `vanilla_clip`: Standard CLIP embeddings
- `clip_lexical`: CLIP + lexical weak supervision
- `clip_positional`: CLIP + positional weak supervision
- `clip_combined`: CLIP + both alignment strategies

### Step 4: Embedding Insertion

Computes CLIP embeddings using OpenCLIP and inserts into database.

```bash
# Insert into all schemas
python3 src/insert_clip_embeddings.py

# Insert into specific schema
python3 src/insert_clip_embeddings.py vanilla_clip
```

**Note:** First run will download the CLIP model (cached locally afterward).

### Step 5: Evaluation

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
│       ├── lexical_components.json
│       └── filtered_lexical_components.json
├── src/                        # Main source code
│   ├── pdf_processor.py           # Document processing
│   ├── filter_lexical_components.py  # Lexical filtering
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

**Lexical Alignment:**
- Computes overlap between text chunks and lexical components
- Normalized score based on component frequency

**Positional Alignment:**
- Uses Intersection over Union (IoU) for overlapping bounding boxes
- Distance-based scoring for non-overlapping elements

**Combined Alignment:**
- Average of lexical and positional scores

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

## Future Enhancements

- Support for additional languages
- Improved spatial relationship analysis
- Better caption-to-image linking
- Parallel processing for large document sets
- Active learning for lexical component filtering

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the test installation script
2. Review error logs
3. Ensure all dependencies are properly installed
4. Create an issue with detailed error information
