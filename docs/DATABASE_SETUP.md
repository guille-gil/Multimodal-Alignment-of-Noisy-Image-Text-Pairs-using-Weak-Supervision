# Database Setup Guide

This guide explains how to set up the PostgreSQL vector database with pgvector for storing CLIP embeddings.

## Prerequisites

1. PostgreSQL database with pgvector extension installed
2. Database credentials (host, name, user, password)
3. CLIP model (for computing embeddings)

## Setup Steps

### 1. Configure Environment Variables

Add your database credentials to `.env`:

```bash
DB_HOST=DB_HOST
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_PORT=5432

# CLIP model configuration
CLIP_MODEL=ViT-B/32  # Options: ViT-B/32 (512 dim), ViT-L/14 (768 dim)
CLIP_DIM=512
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `psycopg2-binary`: PostgreSQL adapter
- `torch`, `torchvision`: For CLIP model (if using OpenAI CLIP)
- `numpy`: For array operations

### 3. Set Up Database Schemas

Run the setup script to create all schemas and tables:

```bash
python3 setup_vector_db.py
```

This creates 4 schemas:
- **vanilla_clip**: Standard CLIP embeddings without weak supervision
- **clip_lexical**: CLIP + lexical component weak supervision
- **clip_positional**: CLIP + positional (bbox) weak supervision
- **clip_combined**: CLIP + both lexical and positional weak supervision

Each schema contains:
- `images` table: Image metadata and CLIP embeddings
- `text_chunks` table: Text chunks and CLIP embeddings
- `alignments` table: Weak supervision alignment scores

### 4. CLIP Encoding (Already Implemented!)

The code already uses **OpenCLIP** (fully open-source, runs locally). The implementation is complete in `insert_clip_embeddings.py`.

**Configuration Options:**

Set in `.env`:
```bash
# CLIP model (default: ViT-B-32 = 512 dimensions)
CLIP_MODEL=ViT-B-32  # Options: ViT-B-32, ViT-L-14, ViT-H-14
CLIP_DIM=512  # Must match model dimensions

# Pretrained weights (default: openai)
CLIP_PRETRAINED=openai  # Uses OpenAI weights (cached locally)
# Alternative: CLIP_PRETRAINED=laion2b_s34b_b79k  # Fully open-source LAION weights
```

**Model Options:**
- **ViT-B-32** (512 dim): Fastest, recommended for most use cases
- **ViT-L-14** (768 dim): Better quality, slower
- **ViT-H-14** (1024 dim): Best quality, slowest

**How it works:**
1. Models are downloaded once and cached locally (no data sent anywhere)
2. All processing runs on your machine (CPU or GPU if available)
3. Embeddings are normalized for cosine similarity search
4. Fully open-source implementation

**Note:** The model will be downloaded automatically on first run. After that, it's cached locally and all inference runs completely offline.

### 5. Insert Embeddings

After implementing CLIP encoding, insert embeddings into all schemas:

```bash
# Insert into all schemas at once
python3 insert_clip_embeddings.py

# Or insert into specific schema
python3 insert_clip_embeddings.py vanilla_clip
python3 insert_clip_embeddings.py clip_lexical
python3 insert_clip_embeddings.py clip_positional
python3 insert_clip_embeddings.py clip_combined
```

## Database Schema Details

### Images Table
- `image_id`: Unique identifier (from processed data)
- `manual_id`: Source manual/document
- `page`: Page number
- `bbox`: Bounding box coordinates [x0, y0, x1, y1]
- `bbox_source`: Source of bbox (native, dict_fallback, vector, unknown)
- `caption`: Figure caption (if available)
- `filename`: Image filename
- `image_type`: Type (raster_image, vector_figure)
- `clip_embedding`: CLIP embedding vector (512 or 768 dimensions)

### Text Chunks Table
- `chunk_id`: Unique identifier
- `manual_id`: Source manual/document
- `page`: Page number
- `bbox`: Bounding box coordinates [x0, y0, x1, y1]
- `text`: Text content
- `clip_embedding`: CLIP embedding vector

### Alignments Table (Weak Supervision)
- `image_id`: Reference to image
- `chunk_id`: Reference to text chunk
- `weak_score`: Confidence score (0-1)
- `alignment_type`: 'lexical', 'positional', or 'combined'

## Query Examples

### Find similar images to a text query
```sql
SELECT image_id, manual_id, filename, 
       1 - (clip_embedding <=> %s::vector) AS similarity
FROM vanilla_clip.images
ORDER BY clip_embedding <=> %s::vector
LIMIT 10;
```

### Find text chunks similar to an image
```sql
SELECT chunk_id, text, 
       1 - (clip_embedding <=> %s::vector) AS similarity
FROM vanilla_clip.text_chunks
ORDER BY clip_embedding <=> %s::vector
LIMIT 10;
```

### Get weak supervision alignments
```sql
SELECT a.image_id, a.chunk_id, a.weak_score, a.alignment_type,
       i.filename, t.text
FROM clip_combined.alignments a
JOIN clip_combined.images i ON a.image_id = i.image_id
JOIN clip_combined.text_chunks t ON a.chunk_id = t.chunk_id
WHERE a.weak_score > 0.3
ORDER BY a.weak_score DESC;
```

## Indexes

The setup script creates HNSW indexes for efficient vector similarity search. If HNSW is not available, it falls back to IVFFlat indexes.

## Notes

- Embeddings are normalized vectors suitable for cosine similarity
- The `<=>` operator in PostgreSQL computes cosine distance (lower = more similar)
- Similarity score = 1 - distance (higher = more similar)
- Weak supervision scores are computed heuristically and can be tuned based on your data

