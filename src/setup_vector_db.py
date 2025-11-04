"""
Database setup script for pgvector-based CLIP embedding storage.

Creates 4 schemas:
- vanilla_clip: Pure CLIP embeddings (no weak supervision)
- clip_local: CLIP + local proximity (bounding box proximity on same page)
- clip_global: CLIP + global proximity (page distance across document)
- clip_combined: CLIP + both local and global proximity
"""

import sys
from pathlib import Path

from psycopg2 import OperationalError, sql

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import CLIP_DIM, get_db_connection


def setup_database():
    """Create schemas and tables for all 4 CLIP alignment strategies."""
    try:
        # Connect to PostgreSQL
        conn = get_db_connection()
        print(" Connection to PostgreSQL successful!")

        cur = conn.cursor()

        # Enable pgvector extension
        try:
            # First check if extension is available
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions 
                    WHERE name = 'vector'
                );
            """)
            is_available = cur.fetchone()[0]

            if not is_available:
                print(
                    " Error: pgvector extension is not installed on the PostgreSQL server"
                )
                print("   The extension must be installed by a database administrator.")
                print("   For PostgreSQL 15, installation typically requires:")
                print("   1. Installing pgvector package on the server")
                print(
                    "   2. Or compiling from source: https://github.com/pgvector/pgvector"
                )
                print("   Contact your database administrator to install pgvector.")
                raise Exception("pgvector extension not available on server")

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print(" pgvector extension enabled")
        except Exception as e:
            if (
                "not available" in str(e).lower()
                or "could not open extension" in str(e).lower()
            ):
                print(
                    "\n Error: pgvector extension is not installed on the PostgreSQL server"
                )
                print("   DETAIL: The extension files are missing from the server.")
                print(
                    "\n   SOLUTION: Contact your database administrator to install pgvector."
                )
                print("   Installation guide: https://github.com/pgvector/pgvector")
                print("\n   You can check extension status with:")
                print("   python3 utils/check_db_connection.py")
                raise
            raise

        # Define schemas
        schemas = ["vanilla_clip", "clip_local", "clip_global", "clip_combined"]

        for schema_name in schemas:
            # Create schema
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
            )

            # Create images table
            cur.execute(
                sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.images (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255) UNIQUE NOT NULL,
                    manual_id VARCHAR(255) NOT NULL,
                    page INTEGER,
                    bbox REAL[],
                    bbox_source VARCHAR(50),
                    caption TEXT,
                    filename VARCHAR(255),
                    image_type VARCHAR(50),
                    clip_embedding vector({}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """).format(sql.Identifier(schema_name), sql.Literal(CLIP_DIM))
            )

            # Create text_chunks table
            cur.execute(
                sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.text_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    manual_id VARCHAR(255) NOT NULL,
                    page INTEGER,
                    bbox REAL[],
                    text TEXT NOT NULL,
                    clip_embedding vector({}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """).format(sql.Identifier(schema_name), sql.Literal(CLIP_DIM))
            )

            # Create alignment table (for weak supervision pairs)
            cur.execute(
                sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.alignments (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255) REFERENCES {}.images(image_id),
                    chunk_id VARCHAR(255) REFERENCES {}.text_chunks(chunk_id),
                    weak_score REAL,  -- Confidence score from weak supervision
                    alignment_type VARCHAR(50),  -- 'local', 'global', 'combined'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_id, chunk_id, alignment_type)
                );
            """).format(
                    sql.Identifier(schema_name),
                    sql.Identifier(schema_name),
                    sql.Identifier(schema_name),
                )
            )

            # Create indexes for efficient similarity search
            # HNSW index for vector similarity (if pgvector version supports it)
            try:
                cur.execute(
                    sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_images_embedding_idx 
                    ON {schema}.images 
                    USING hnsw (clip_embedding vector_cosine_ops);
                """).format(schema=sql.Identifier(schema_name))
                )

                cur.execute(
                    sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_chunks_embedding_idx 
                    ON {schema}.text_chunks 
                    USING hnsw (clip_embedding vector_cosine_ops);
                """).format(schema=sql.Identifier(schema_name))
                )
            except Exception as e:
                # Fallback to ivfflat if HNSW not available
                print(f"  HNSW not available, using ivfflat: {e}")
                cur.execute(
                    sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_images_embedding_idx 
                    ON {schema}.images 
                    USING ivfflat (clip_embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """).format(schema=sql.Identifier(schema_name))
                )

                cur.execute(
                    sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_chunks_embedding_idx 
                    ON {schema}.text_chunks 
                    USING ivfflat (clip_embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """).format(schema=sql.Identifier(schema_name))
                )

            # Indexes for metadata queries
            cur.execute(
                sql.SQL("""
                CREATE INDEX IF NOT EXISTS {schema}_images_manual_idx 
                ON {schema}.images(manual_id);
            """).format(schema=sql.Identifier(schema_name))
            )

            cur.execute(
                sql.SQL("""
                CREATE INDEX IF NOT EXISTS {schema}_chunks_manual_idx 
                ON {schema}.text_chunks(manual_id);
            """).format(schema=sql.Identifier(schema_name))
            )

            print(f" Schema '{schema_name}' created with tables and indexes")

        conn.commit()
        cur.close()
        conn.close()
        print("\n Database setup complete!")

    except OperationalError as e:
        print(" Could not connect to PostgreSQL:")
        print(e)
    except Exception as e:
        print(f" Error setting up database: {e}")
        raise


if __name__ == "__main__":
    setup_database()
