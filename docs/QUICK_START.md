# Quick Start Guide

## Complete Pipeline Execution

### First Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python3 -m spacy download nl_core_news_sm  # or en_core_web_sm
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Place documents:**
   ```bash
   # Add PDF/Word files to data/raw/manuals/
   ```

4. **Run complete pipeline:**
   ```bash
   python3 src/run_pipeline.py
   ```

### Subsequent Runs

The pipeline automatically skips steps that are already completed:

```bash
# Run pipeline (skips completed steps)
python3 src/run_pipeline.py

# Force re-run everything
python3 src/run_pipeline.py --force

# Skip specific steps
python3 src/run_pipeline.py --skip-db --skip-embeddings
```

## Individual Steps

### 1. Process Documents
```bash
python3 src/pdf_processor.py
```

### 2. Setup Database
```bash
python3 src/setup_vector_db.py
```

### 3. Insert Embeddings
```bash
python3 src/insert_clip_embeddings.py
```

### 4. Evaluate
```bash
python3 src/evaluate_alignments.py
```

## Common Workflows

### Process New Documents
```bash
# Add new files to data/raw/manuals/
python3 src/run_pipeline.py --skip-db --skip-embeddings --skip-eval
```

### Re-compute Embeddings
```bash
python3 src/run_pipeline.py --skip-pdf --skip-db
```

## Output Locations

- **Processed Data**: `data/processed/`
- **Evaluation Results**: `evaluation_results/`
- **Logs**: Console output

## Troubleshooting

### "Schema already exists"
- Normal - pipeline skips if already set up
- Use `--force` to recreate

### "No files found"
- Add PDF/Word files to `data/raw/manuals/`

### "CLIP model download failed"
- Check internet connection (first run only)
- Verify disk space

### "Database connection failed"
- Check `.env` configuration
- Verify PostgreSQL is running
- Ensure pgvector extension is installed

