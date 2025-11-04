"""
Insert CLIP embeddings into PostgreSQL vector database.

Supports 4 schemas:
- vanilla_clip: Pure CLIP embeddings (no weak supervision)
- clip_local: CLIP + local proximity (bounding box proximity on same page)
- clip_global: CLIP + global proximity (page distance across document)
- clip_combined: CLIP + both local and global proximity
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from psycopg2 import sql
from psycopg2.extras import execute_values

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import CLIP_DIM, CLIP_MODEL, get_db_connection  # noqa: E402

IMAGE_METADATA_FILE = BASE_DIR / "data/processed/image_metadata.json"
TEXT_CHUNKS_FILE = BASE_DIR / "data/processed/text_chunks.json"
IMAGES_DIR = BASE_DIR / "data/processed/images"


def load_clip_model():
    """
    Load OpenCLIP model - fully open-source, runs locally.

    Model options:
    - 'ViT-B-32': 512 dim (recommended for speed)
    - 'ViT-L-14': 768 dim (better quality, slower)
    - 'ViT-H-14': 1024 dim (best quality, slowest)

    Pretrained options:
    - 'openai': OpenAI weights (compatible, cached locally)
    - 'laion2b_s34b_b79k': LAION-2B trained weights (fully open-source)
    """
    try:
        import open_clip
        import torch
    except ImportError:
        raise ImportError(
            "open-clip-torch not installed. Install with: pip install open-clip-torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse model name from env (supports formats like "ViT-B-32" or "ViT-B/32")
    model_name = CLIP_MODEL.replace("/", "-")

    # Map common model names to OpenCLIP format
    model_map = {
        "ViT-B-32": "ViT-B-32",
        "ViT-B/32": "ViT-B-32",
        "ViT-L-14": "ViT-L-14",
        "ViT-L/14": "ViT-L-14",
        "ViT-H-14": "ViT-H-14",
        "ViT-H/14": "ViT-H-14",
    }

    model_name = model_map.get(model_name, model_name)

    # Use OpenAI weights for compatibility (cached locally after first download)
    # Alternative: Use 'laion2b_s34b_b79k' for fully open-source weights
    pretrained = os.getenv("CLIP_PRETRAINED", "openai")

    print(f"Loading OpenCLIP model: {model_name} with {pretrained} weights...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    print(f"OpenCLIP model loaded on {device}")
    return model, preprocess, tokenizer, device


def compute_image_embedding(image_path: Path, model, preprocess, device):
    """
    Compute CLIP embedding for an image using OpenCLIP.

    Args:
        image_path: Path to image file
        model: OpenCLIP model
        preprocess: Image preprocessing function
        device: torch device (cpu/cuda)

    Returns:
        Normalized embedding vector (np.float32)
    """
    # Conditional imports for optional dependencies
    try:
        from PIL import Image
        import torch
    except ImportError:
        raise ImportError("PIL and torch are required for image embeddings")

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor).cpu().numpy()

    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding[0].astype(np.float32)


def compute_text_embedding(text: str, model, tokenizer, device):
    """
    Compute CLIP embedding for text using OpenCLIP.

    Args:
        text: Text string to encode
        model: OpenCLIP model
        tokenizer: OpenCLIP tokenizer
        device: torch device (cpu/cuda)

    Returns:
        Normalized embedding vector (np.float32)
    """
    # Conditional import for optional dependency
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for text embeddings")

    # Truncate text if too long (CLIP has a max token limit)
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(text_tokens).cpu().numpy()

    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding[0].astype(np.float32)


def compute_local_proximity(image: Dict, chunk: Dict) -> float:
    """
    Compute weak supervision score based on local proximity (bounding box on same page).

    Uses IoU (Intersection over Union) if boxes overlap, otherwise distance-based score.
    """
    if not image.get("bbox") or not chunk.get("bbox"):
        return 0.0

    img_bbox = image["bbox"]
    chunk_bbox = chunk["bbox"]

    # Validate bboxes (must have 4 elements)
    if len(img_bbox) != 4 or len(chunk_bbox) != 4:
        return 0.0

    # Check for zero bboxes
    if (img_bbox[2] - img_bbox[0] == 0) or (img_bbox[3] - img_bbox[1] == 0):
        return 0.0
    if (chunk_bbox[2] - chunk_bbox[0] == 0) or (chunk_bbox[3] - chunk_bbox[1] == 0):
        return 0.0

    # Compute IoU (Intersection over Union) for better alignment score
    x1_i = max(img_bbox[0], chunk_bbox[0])
    y1_i = max(img_bbox[1], chunk_bbox[1])
    x2_i = min(img_bbox[2], chunk_bbox[2])
    y2_i = min(img_bbox[3], chunk_bbox[3])

    if x2_i <= x1_i or y2_i <= y1_i:
        # No intersection - compute distance-based score
        img_center = [(img_bbox[0] + img_bbox[2]) / 2, (img_bbox[1] + img_bbox[3]) / 2]
        chunk_center = [
            (chunk_bbox[0] + chunk_bbox[2]) / 2,
            (chunk_bbox[1] + chunk_bbox[3]) / 2,
        ]

        distance = np.sqrt(
            (img_center[0] - chunk_center[0]) ** 2
            + (img_center[1] - chunk_center[1]) ** 2
        )
        # Normalize (heuristic - adjust based on typical page dimensions)
        max_distance = 1000.0  # Adjust based on your page sizes
        score = max(0.0, 1.0 - (distance / max_distance))
        return score

    # Compute IoU
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    img_area = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
    chunk_area = (chunk_bbox[2] - chunk_bbox[0]) * (chunk_bbox[3] - chunk_bbox[1])
    union = img_area + chunk_area - intersection

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def compute_global_proximity(image: Dict, chunk: Dict) -> float:
    """
    Compute weak supervision score based on global proximity (page distance).

    Scores based on how close pages are in the document. Same page = 1.0,
    decreases with page distance.
    """
    img_page = image.get("page")
    chunk_page = chunk.get("page")

    if img_page is None or chunk_page is None:
        return 0.0

    # Same page
    if img_page == chunk_page:
        return 1.0

    # Different pages - compute distance-based score
    page_distance = abs(img_page - chunk_page)

    # Exponential decay: score decreases with distance
    # Adjust decay rate based on typical document size
    # For manual with ~100 pages, distance of 10 pages ≈ 0.37 score
    decay_rate = 0.1  # Adjust this to control how quickly score decreases
    score = np.exp(-decay_rate * page_distance)

    return float(score)


def insert_embeddings(schema: str, use_local: bool = False, use_global: bool = False):
    """Insert CLIP embeddings into specified schema."""
    # Load data
    if not IMAGE_METADATA_FILE.exists():
        print(f"Error: {IMAGE_METADATA_FILE} not found. Run pdf_processor.py first.")
        return

    if not TEXT_CHUNKS_FILE.exists():
        print(f"Error: {TEXT_CHUNKS_FILE} not found. Run pdf_processor.py first.")
        return

    with open(IMAGE_METADATA_FILE, "r") as f:
        images = json.load(f)

    with open(TEXT_CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    # Load OpenCLIP model
    try:
        model, preprocess, tokenizer, device = load_clip_model()
    except ImportError as e:
        print(f"Warning: {e}")
        print("Install with: pip install open-clip-torch")
        print("Using placeholder embeddings.")
        model = None
        preprocess = None
        tokenizer = None
        device = None
    except Exception as e:
        print(f"Warning: Error loading CLIP model: {e}")
        print("Using placeholder embeddings.")
        model = None
        preprocess = None
        tokenizer = None
        device = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Insert images
        image_records = []
        for img in images:
            image_path = IMAGES_DIR / img.get("filename", "")

            if model is not None and image_path.exists():
                try:
                    embedding = compute_image_embedding(
                        image_path, model, preprocess, device
                    )
                except Exception as e:
                    print(
                        f"Warning: Error computing embedding for {img.get('filename')}: {e}"
                    )
                    # Fallback to placeholder
                    embedding = np.random.rand(CLIP_DIM).astype(np.float32)
            else:
                # Placeholder embedding
                embedding = np.random.rand(CLIP_DIM).astype(np.float32)

            image_records.append(
                (
                    img["image_id"],
                    img["manual_id"],
                    img.get("page"),
                    img.get("bbox"),
                    img.get("bbox_source"),
                    img.get("caption"),
                    img.get("filename"),
                    img.get("image_type"),
                    embedding.tolist(),  # Convert to list for pgvector
                )
            )

        execute_values(
            cur,
            sql.SQL("""
            INSERT INTO {}.images 
            (image_id, manual_id, page, bbox, bbox_source, caption, filename, image_type, clip_embedding)
            VALUES %s
            ON CONFLICT (image_id) DO UPDATE SET
                clip_embedding = EXCLUDED.clip_embedding
            """).format(sql.Identifier(schema)),
            image_records,
        )
        print(f"Inserted {len(image_records)} images into {schema}")

        # Insert text chunks
        chunk_records = []
        for chunk in chunks:
            if model is not None and tokenizer is not None:
                try:
                    embedding = compute_text_embedding(
                        chunk["text"], model, tokenizer, device
                    )
                except Exception as e:
                    print(
                        f"Warning: Error computing embedding for chunk {chunk.get('chunk_id')}: {e}"
                    )
                    # Fallback to placeholder
                    embedding = np.random.rand(CLIP_DIM).astype(np.float32)
            else:
                # Placeholder embedding
                embedding = np.random.rand(CLIP_DIM).astype(np.float32)

            chunk_records.append(
                (
                    chunk["chunk_id"],
                    chunk["manual_id"],
                    chunk.get("page"),
                    chunk.get("bbox"),
                    chunk["text"],
                    embedding.tolist(),
                )
            )

        execute_values(
            cur,
            sql.SQL("""
            INSERT INTO {}.text_chunks
            (chunk_id, manual_id, page, bbox, text, clip_embedding)
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                clip_embedding = EXCLUDED.clip_embedding
            """).format(sql.Identifier(schema)),
            chunk_records,
        )
        print(f"Inserted {len(chunk_records)} text chunks into {schema}")

        # Compute and insert alignments if needed
        if use_local or use_global:
            alignment_records = []

            for img in images:
                img_manual = img["manual_id"]

                for chunk in chunks:
                    if chunk["manual_id"] != img_manual:
                        continue

                    scores = []
                    alignment_types = []

                    if use_local:
                        # Local proximity: only check same-page pairs for bounding box proximity
                        img_page = img.get("page")
                        chunk_page = chunk.get("page")
                        if img_page == chunk_page:
                            local_score = compute_local_proximity(img, chunk)
                            if local_score > 0.05:  # Threshold to avoid noise
                                scores.append(local_score)
                                alignment_types.append("local")

                    if use_global:
                        # Global proximity: check all pairs, score by page distance
                        global_score = compute_global_proximity(img, chunk)
                        if global_score > 0.05:  # Threshold to avoid noise
                            scores.append(global_score)
                            alignment_types.append("global")

                    # Combined score
                    if use_local and use_global and len(scores) == 2:
                        combined_score = (scores[0] + scores[1]) / 2  # Average
                        if combined_score > 0.1:  # Combined threshold
                            alignment_records.append(
                                (
                                    img["image_id"],
                                    chunk["chunk_id"],
                                    combined_score,
                                    "combined",
                                )
                            )
                    else:
                        # Individual alignment types
                        for score, a_type in zip(scores, alignment_types):
                            alignment_records.append(
                                (img["image_id"], chunk["chunk_id"], score, a_type)
                            )

            if alignment_records:
                execute_values(
                    cur,
                    sql.SQL("""
                    INSERT INTO {}.alignments
                    (image_id, chunk_id, weak_score, alignment_type)
                    VALUES %s
                    ON CONFLICT (image_id, chunk_id, alignment_type) DO UPDATE
                    SET weak_score = EXCLUDED.weak_score
                    """).format(sql.Identifier(schema)),
                    alignment_records,
                )
                print(f"Inserted {len(alignment_records)} alignments into {schema}")

        conn.commit()
        cur.close()
        conn.close()
        print(f"\nSuccessfully populated {schema} schema")

    except Exception as e:
        print(f"Error inserting embeddings: {e}")
        raise


if __name__ == "__main__":
    # Allow specifying which schema to populate
    if len(sys.argv) > 1:
        schema = sys.argv[1]
        if schema == "vanilla_clip":
            insert_embeddings("vanilla_clip", use_local=False, use_global=False)
        elif schema == "clip_local":
            insert_embeddings("clip_local", use_local=True, use_global=False)
        elif schema == "clip_global":
            insert_embeddings("clip_global", use_local=False, use_global=True)
        elif schema == "clip_combined":
            insert_embeddings("clip_combined", use_local=True, use_global=True)
        else:
            print(f"Unknown schema: {schema}")
            print(
                "Available schemas: vanilla_clip, clip_local, clip_global, clip_combined"
            )
    else:
        # Populate all schemas
        print("Inserting into vanilla_clip...")
        insert_embeddings("vanilla_clip", use_local=False, use_global=False)

        print("\nInserting into clip_local...")
        insert_embeddings("clip_local", use_local=True, use_global=False)

        print("\nInserting into clip_global...")
        insert_embeddings("clip_global", use_local=False, use_global=True)

        print("\nInserting into clip_combined...")
        insert_embeddings("clip_combined", use_local=True, use_global=True)
