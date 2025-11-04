"""
Main pipeline orchestrator for the multimodal alignment system.

Executes the complete pipeline with smart step skipping and operator-in-the-loop
support for lexical component filtering.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import get_db_connection

# Paths (relative to project root)
RAW_DIR = BASE_DIR / "data/raw/manuals"
PROCESSED_DIR = BASE_DIR / "data/processed"
IMAGES_DIR = PROCESSED_DIR / "images"
IMAGE_METADATA_FILE = PROCESSED_DIR / "image_metadata.json"
TEXT_CHUNKS_FILE = PROCESSED_DIR / "text_chunks.json"
LEXICAL_COMPONENTS_FILE = PROCESSED_DIR / "lexical_components.json"
FILTERED_LEXICAL_FILE = PROCESSED_DIR / "filtered_lexical_components.json"


class PipelineOrchestrator:
    """Orchestrates the complete multimodal alignment pipeline."""

    def __init__(self):
        self.steps_completed = {
            "pdf_processing": False,
            "lexical_filtering": False,
            "db_setup": False,
            "embeddings_inserted": False,
        }

    def check_pdf_processing(self) -> bool:
        """Check if PDF processing has been completed."""
        return (
            IMAGE_METADATA_FILE.exists()
            and TEXT_CHUNKS_FILE.exists()
            and LEXICAL_COMPONENTS_FILE.exists()
            and IMAGES_DIR.exists()
            and len(list(IMAGES_DIR.glob("*"))) > 0
        )

    def check_lexical_filtering(self) -> bool:
        """Check if lexical components have been filtered."""
        return FILTERED_LEXICAL_FILE.exists()

    def check_db_setup(self) -> bool:
        """Check if database schemas are set up."""
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Check if all required schemas exist
            cur.execute(
                """
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name IN ('vanilla_clip', 'clip_lexical', 'clip_positional', 'clip_combined')
                """
            )
            schemas = [row[0] for row in cur.fetchall()]
            required_schemas = {
                "vanilla_clip",
                "clip_lexical",
                "clip_positional",
                "clip_combined",
            }

            cur.close()
            conn.close()

            return required_schemas.issubset(set(schemas))
        except Exception as e:
            print(f"  Could not check database setup: {e}")
            return False

    def check_embeddings_inserted(self, schema: str) -> bool:
        """Check if embeddings have been inserted into a schema."""
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute(
                f"""
                SELECT COUNT(*) FROM {schema}.images
                """
            )
            image_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT COUNT(*) FROM {schema}.text_chunks
                """
            )
            chunk_count = cur.fetchone()[0]

            cur.close()
            conn.close()

            return image_count > 0 and chunk_count > 0
        except Exception:
            return False

    def step_pdf_processing(self, force: bool = False):
        """Step 1: Process PDF documents."""
        if not force and self.check_pdf_processing():
            print(" PDF processing already completed. Skipping...")
            print(f"   Found {len(json.load(open(IMAGE_METADATA_FILE)))} images")
            print(f"   Found {len(json.load(open(TEXT_CHUNKS_FILE)))} text chunks")
            return

        print("\n" + "=" * 80)
        print("STEP 1: PDF Processing")
        print("=" * 80)

        if not RAW_DIR.exists() or not any(RAW_DIR.glob("*")):
            print(f" No files found in {RAW_DIR}")
            print("   Please add PDF or Word documents to process.")
            sys.exit(1)

        print(f"📄 Processing documents from {RAW_DIR}...")
        result = subprocess.run(
            [sys.executable, "src/pdf_processor.py"], capture_output=False
        )

        if result.returncode != 0:
            print(" PDF processing failed!")
            sys.exit(1)

        print(" PDF processing completed!")

    def step_lexical_filtering(self, force: bool = False, skip_prompt: bool = False):
        """Step 2: Filter lexical components (operator in the loop)."""
        if not force and self.check_lexical_filtering():
            print(" Lexical components already filtered. Skipping...")
            return

        print("\n" + "=" * 80)
        print("STEP 2: Lexical Component Filtering (Operator in the Loop)")
        print("=" * 80)

        if not LEXICAL_COMPONENTS_FILE.exists():
            print(" Lexical components not found. Run PDF processing first.")
            sys.exit(1)

        # Load and display current lexical components
        with open(LEXICAL_COMPONENTS_FILE, "r") as f:
            lexical_data = json.load(f)

        print(f"\n Found {lexical_data['total_components']} unique lexical components")
        print("\nTop 20 most frequent terms:")
        for i, comp in enumerate(lexical_data["components"][:20], 1):
            print(f"  {i:2d}. {comp['term']:30s} (count: {comp['count']})")

        if not skip_prompt:
            print("\n" + "-" * 80)
            print("To filter non-relevant terms, edit src/filter_lexical_components.py")
            print("and add terms to EXCLUDE_TERMS set, then run:")
            print("  python3 src/filter_lexical_components.py")
            print(
                "\nPress Enter to continue with filtering, or 'skip' to skip this step..."
            )
            response = input().strip().lower()

            if response == "skip":
                print("⚠️  Skipping lexical filtering. Using all components.")
                return

        # Run filtering script
        print("\n Running lexical component filter...")
        result = subprocess.run(
            [sys.executable, "src/filter_lexical_components.py"],
            capture_output=False,
        )

        if result.returncode != 0:
            print("  Lexical filtering failed or was skipped.")
        else:
            print(" Lexical filtering completed!")

    def step_db_setup(self, force: bool = False):
        """Step 3: Set up database schemas."""
        if not force and self.check_db_setup():
            print(" Database schemas already set up. Skipping...")
            return

        print("\n" + "=" * 80)
        print("STEP 3: Database Setup")
        print("=" * 80)

        print("  Setting up PostgreSQL schemas and tables...")
        result = subprocess.run(
            [sys.executable, "src/setup_vector_db.py"], capture_output=False
        )

        if result.returncode != 0:
            print(" Database setup failed!")
            sys.exit(1)

        print(" Database setup completed!")

    def step_insert_embeddings(
        self, force: bool = False, schemas: Optional[list] = None
    ):
        """Step 4: Insert CLIP embeddings."""
        if schemas is None:
            schemas = [
                "vanilla_clip",
                "clip_lexical",
                "clip_positional",
                "clip_combined",
            ]

        print("\n" + "=" * 80)
        print("STEP 4: Insert CLIP Embeddings")
        print("=" * 80)

        # Check which schemas need embeddings
        schemas_to_process = []
        for schema in schemas:
            if force or not self.check_embeddings_inserted(schema):
                schemas_to_process.append(schema)
            else:
                print(f" {schema} already has embeddings. Skipping...")

        if not schemas_to_process:
            print(" All schemas already have embeddings. Skipping...")
            return

        print(f" Inserting embeddings into: {', '.join(schemas_to_process)}")
        print("   (This may take a while - first run will download CLIP model)")

        for schema in schemas_to_process:
            print(f"\n   Processing {schema}...")
            result = subprocess.run(
                [sys.executable, "src/insert_clip_embeddings.py", schema],
                capture_output=False,
            )

            if result.returncode != 0:
                print(f" Failed to insert embeddings into {schema}")
                continue

            print(f"    {schema} completed")

        print("\n Embedding insertion completed!")

    def step_evaluation(self):
        """Step 5: Run evaluation."""
        print("\n" + "=" * 80)
        print("STEP 5: Evaluation")
        print("=" * 80)

        print(" Computing metrics and generating visualizations...")
        result = subprocess.run(
            [sys.executable, "src/evaluate_alignments.py"], capture_output=False
        )

        if result.returncode != 0:
            print("  Evaluation completed with warnings")
        else:
            print(" Evaluation completed!")
            print("   Check evaluation_results/ directory for metrics and charts")

    def run(
        self,
        skip_pdf: bool = False,
        skip_lexical: bool = False,
        skip_db: bool = False,
        skip_embeddings: bool = False,
        skip_eval: bool = False,
        force: bool = False,
    ):
        """Run complete pipeline with optional step skipping."""
        print("\n" + "🚀 " + "=" * 78)
        print("MULTIMODAL ALIGNMENT PIPELINE")
        print("=" * 80 + "\n")

        try:
            # Step 1: PDF Processing
            if not skip_pdf:
                self.step_pdf_processing(force=force)
            else:
                print("⏭  Skipping PDF processing")

            # Step 2: Lexical Filtering
            if not skip_lexical:
                self.step_lexical_filtering(force=force, skip_prompt=False)
            else:
                print("⏭  Skipping lexical filtering")

            # Step 3: Database Setup
            if not skip_db:
                self.step_db_setup(force=force)
            else:
                print("⏭  Skipping database setup")

            # Step 4: Embedding Insertion
            if not skip_embeddings:
                self.step_insert_embeddings(force=force)
            else:
                print("⏭  Skipping embedding insertion")

            # Step 5: Evaluation
            if not skip_eval:
                self.step_evaluation()
            else:
                print("⏭  Skipping evaluation")

            print("\n" + "=" * 80)
            print(" PIPELINE COMPLETE!")
            print("=" * 80)

        except KeyboardInterrupt:
            print("\n\n  Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n Pipeline failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete multimodal alignment pipeline"
    )
    parser.add_argument(
        "--skip-pdf", action="store_true", help="Skip PDF processing step"
    )
    parser.add_argument(
        "--skip-lexical",
        action="store_true",
        help="Skip lexical component filtering",
    )
    parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding insertion",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all steps (ignore existing results)",
    )

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()
    orchestrator.run(
        skip_pdf=args.skip_pdf,
        skip_lexical=args.skip_lexical,
        skip_db=args.skip_db,
        skip_embeddings=args.skip_embeddings,
        skip_eval=args.skip_eval,
        force=args.force,
    )


if __name__ == "__main__":
    main()
