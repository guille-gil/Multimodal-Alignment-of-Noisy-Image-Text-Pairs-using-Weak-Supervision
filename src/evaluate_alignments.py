"""
Evaluation script for multimodal alignment system.

Computes metrics and generates visualizations to assess the quality of
CLIP embeddings and weak supervision alignments across different schemas.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from psycopg2 import sql

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import get_db_connection  # noqa: E402

# Schemas to evaluate
SCHEMAS = ["vanilla_clip", "clip_local", "clip_global", "clip_combined"]

# Output directory for charts (relative to project root)
OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_image_text_pairs(schema: str) -> List[Tuple[str, str, str, str]]:
    """
    Get all image-text pairs from the same manual and page.

    Returns: List of (image_id, chunk_id, manual_id, page) tuples
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        sql.SQL("""
        SELECT DISTINCT i.image_id, t.chunk_id, i.manual_id, i.page
        FROM {}.images i
        JOIN {}.text_chunks t ON i.manual_id = t.manual_id AND i.page = t.page
        """).format(
            sql.Identifier(schema),
            sql.Identifier(schema),
        )
    )

    pairs = cur.fetchall()
    cur.close()
    conn.close()

    return pairs


def compute_similarity(image_id: str, chunk_id: str, schema: str) -> float:
    """Compute cosine similarity between image and text chunk embeddings."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get embeddings
    cur.execute(
        sql.SQL("""
        SELECT clip_embedding FROM {}.images WHERE image_id = %s
        """).format(sql.Identifier(schema)),
        (image_id,),
    )
    img_embedding = cur.fetchone()[0]

    cur.execute(
        sql.SQL("""
        SELECT clip_embedding FROM {}.text_chunks WHERE chunk_id = %s
        """).format(sql.Identifier(schema)),
        (chunk_id,),
    )
    chunk_embedding = cur.fetchone()[0]

    # Compute cosine similarity (1 - distance since embeddings are normalized)
    cur.execute(
        """
        SELECT 1 - (%s::vector <=> %s::vector) AS similarity
        """,
        (img_embedding, chunk_embedding),
    )
    similarity = cur.fetchone()[0]

    cur.close()
    conn.close()

    return float(similarity)


def get_top_k_similar_chunks(
    image_id: str, schema: str, k: int = 10
) -> List[Tuple[str, float]]:
    """Get top K most similar text chunks for an image."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get image embedding
    cur.execute(
        sql.SQL("""
        SELECT clip_embedding FROM {}.images WHERE image_id = %s
        """).format(sql.Identifier(schema)),
        (image_id,),
    )
    img_embedding = cur.fetchone()[0]

    # Get top K similar chunks (same manual and page)
    cur.execute(
        sql.SQL("""
        SELECT chunk_id, 1 - (clip_embedding <=> %s::vector) AS similarity
        FROM {}.text_chunks t
        JOIN {}.images i ON t.manual_id = i.manual_id AND t.page = i.page
        WHERE i.image_id = %s
        ORDER BY similarity DESC
        LIMIT %s
        """).format(
            sql.Identifier(schema),
            sql.Identifier(schema),
        ),
        (img_embedding, image_id, k),
    )

    results = [(row[0], float(row[1])) for row in cur.fetchall()]

    cur.close()
    conn.close()

    return results


def get_weak_supervision_scores(schema: str) -> Dict[str, List[float]]:
    """Get weak supervision alignment scores by type."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        sql.SQL("""
        SELECT alignment_type, weak_score
        FROM {}.alignments
        ORDER BY alignment_type
        """).format(sql.Identifier(schema))
    )

    scores_by_type = defaultdict(list)
    for alignment_type, score in cur.fetchall():
        scores_by_type[alignment_type].append(float(score))

    cur.close()
    conn.close()

    return dict(scores_by_type)


def compute_top_k_accuracy(
    schema: str, k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute Top-K accuracy for image-text matching.

    For each image, check if the true paired chunk appears in top K results.
    """
    pairs = get_image_text_pairs(schema)

    if not pairs:
        return {k: 0.0 for k in k_values}

    correct_counts = {k: 0 for k in k_values}

    for image_id, true_chunk_id, _, _ in pairs:
        top_k_chunks = get_top_k_similar_chunks(image_id, schema, k=max(k_values))
        chunk_ids = [chunk_id for chunk_id, _ in top_k_chunks]

        for k in k_values:
            if true_chunk_id in chunk_ids[:k]:
                correct_counts[k] += 1

    accuracy = {k: correct_counts[k] / len(pairs) for k in k_values}
    return accuracy


def compute_mrr(schema: str) -> float:
    """Compute Mean Reciprocal Rank for image-text matching."""
    pairs = get_image_text_pairs(schema)

    if not pairs:
        return 0.0

    reciprocal_ranks = []

    for image_id, true_chunk_id, _, _ in pairs:
        top_k_chunks = get_top_k_similar_chunks(image_id, schema, k=100)

        for rank, (chunk_id, _) in enumerate(top_k_chunks, 1):
            if chunk_id == true_chunk_id:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # Not found in top 100
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def compute_average_similarity(schema: str) -> float:
    """Compute average similarity score for true pairs."""
    pairs = get_image_text_pairs(schema)

    if not pairs:
        return 0.0

    similarities = []
    for image_id, chunk_id, _, _ in pairs:
        sim = compute_similarity(image_id, chunk_id, schema)
        similarities.append(sim)

    return np.mean(similarities)


def plot_similarity_distributions(schemas: List[str]):
    """Plot distribution of similarity scores across schemas."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, schema in enumerate(schemas):
        pairs = get_image_text_pairs(schema)
        similarities = []

        for image_id, chunk_id, _, _ in pairs[:500]:  # Sample for performance
            try:
                sim = compute_similarity(image_id, chunk_id, schema)
                similarities.append(sim)
            except Exception:
                continue

        if similarities:
            axes[idx].hist(similarities, bins=50, alpha=0.7, edgecolor="black")
            axes[idx].set_title(f"Similarity Distribution: {schema}")
            axes[idx].set_xlabel("Cosine Similarity")
            axes[idx].set_ylabel("Frequency")
            axes[idx].axvline(
                np.mean(similarities),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(similarities):.3f}",
            )
            axes[idx].legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "similarity_distributions.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"Saved similarity distributions to {OUTPUT_DIR / 'similarity_distributions.png'}"
    )
    plt.close()


def plot_top_k_comparison():
    """Plot Top-K accuracy comparison across schemas."""
    k_values = [1, 5, 10, 20]
    schema_accuracies = {}

    for schema in SCHEMAS:
        try:
            accuracies = compute_top_k_accuracy(schema, k_values)
            schema_accuracies[schema] = accuracies
        except Exception as e:
            print(f"Warning: Error evaluating {schema}: {e}")
            continue

    if not schema_accuracies:
        print("Warning: No schemas available for comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(k_values))
    width = 0.2

    for idx, (schema, accuracies) in enumerate(schema_accuracies.items()):
        values = [accuracies[k] for k in k_values]
        ax.bar(
            x + idx * width,
            values,
            width,
            label=schema.replace("_", " ").title(),
        )

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K Accuracy Comparison Across Schemas")
    ax.set_xticks(x + width * (len(schema_accuracies) - 1) / 2)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_k_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved Top-K comparison to {OUTPUT_DIR / 'top_k_comparison.png'}")
    plt.close()


def plot_weak_supervision_scores():
    """Plot distribution of weak supervision alignment scores."""
    schemas_with_alignments = ["clip_local", "clip_global", "clip_combined"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, schema in enumerate(schemas_with_alignments):
        try:
            scores_by_type = get_weak_supervision_scores(schema)

            for alignment_type, scores in scores_by_type.items():
                axes[idx].hist(
                    scores,
                    bins=30,
                    alpha=0.6,
                    label=alignment_type,
                    edgecolor="black",
                )

            axes[idx].set_title(f"Weak Supervision Scores: {schema}")
            axes[idx].set_xlabel("Alignment Score")
            axes[idx].set_ylabel("Frequency")
            axes[idx].legend()
            axes[idx].grid(axis="y", alpha=0.3)
        except Exception as e:
            print(f"Warning: Error plotting weak supervision for {schema}: {e}")
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "weak_supervision_scores.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"Saved weak supervision scores to {OUTPUT_DIR / 'weak_supervision_scores.png'}"
    )
    plt.close()


def print_metrics_report():
    """Print comprehensive metrics report."""
    print("\n" + "=" * 80)
    print("MULTIMODAL ALIGNMENT EVALUATION REPORT")
    print("=" * 80 + "\n")

    all_metrics = {}

    for schema in SCHEMAS:
        print(f"\nSchema: {schema.upper().replace('_', ' ')}")
        print("-" * 80)

        try:
            # Check if schema exists
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = 'images'
                """,
                (schema,),
            )
            exists = cur.fetchone()[0] > 0
            cur.close()
            conn.close()

            if not exists:
                print("  Warning: Schema not found in database")
                continue

            # Compute metrics
            top_k_acc = compute_top_k_accuracy(schema, [1, 5, 10])
            mrr = compute_mrr(schema)
            avg_sim = compute_average_similarity(schema)

            # Get counts
            pairs = get_image_text_pairs(schema)

            print(f"  Total Image-Text Pairs: {len(pairs)}")
            print(f"  Average Similarity: {avg_sim:.4f}")
            print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
            print(f"  Top-1 Accuracy: {top_k_acc[1]:.4f} ({top_k_acc[1] * 100:.2f}%)")
            print(f"  Top-5 Accuracy: {top_k_acc[5]:.4f} ({top_k_acc[5] * 100:.2f}%)")
            print(
                f"  Top-10 Accuracy: {top_k_acc[10]:.4f} ({top_k_acc[10] * 100:.2f}%)"
            )

            # Weak supervision stats (if applicable)
            if schema in ["clip_local", "clip_global", "clip_combined"]:
                try:
                    scores_by_type = get_weak_supervision_scores(schema)
                    if scores_by_type:
                        print("  Weak Supervision Alignments:")
                        for align_type, scores in scores_by_type.items():
                            print(
                                f"    - {align_type}: {len(scores)} pairs, "
                                f"avg score: {np.mean(scores):.4f}"
                            )
                except Exception:
                    pass

            all_metrics[schema] = {
                "top_k": top_k_acc,
                "mrr": mrr,
                "avg_similarity": avg_sim,
                "num_pairs": len(pairs),
            }

        except Exception as e:
            print(f"  Error evaluating schema: {e}")
            continue

    # Save metrics to JSON
    metrics_file = OUTPUT_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")
    print("\n" + "=" * 80)


def main():
    """Run complete evaluation pipeline."""
    print("🔍 Starting evaluation...")

    try:
        # Print metrics report
        print_metrics_report()

        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_similarity_distributions(SCHEMAS)
        plot_top_k_comparison()
        plot_weak_supervision_scores()

        print(f"\nEvaluation complete! Results saved to {OUTPUT_DIR}/")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
