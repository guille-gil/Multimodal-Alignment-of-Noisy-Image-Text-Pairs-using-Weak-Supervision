"""
Database setup script for pgvector-based CLIP embedding storage.

Creates 4 schemas:
- vanilla_clip: Standard CLIP embeddings
- clip_lexical: CLIP + lexical component weak supervision
- clip_positional: CLIP + positional (bbox) weak supervision  
- clip_combined: CLIP + both lexical and positional weak supervision
"""

import psycopg2
from psycopg2 import sql, OperationalError
from dotenv import load_dotenv
import os

load_dotenv()

# DB parameters from environment
DB_HOST = os.getenv("DB_HOST", "bachata.service.rug.nl")
DB_NAME = os.getenv("DB_NAME", "aixpert")
DB_USER = os.getenv("DB_USER", "pnumber")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# CLIP embedding dimension (CLIP ViT-B/32 = 512, ViT-L/14 = 768)
CLIP_DIM = int(os.getenv("CLIP_DIM", "512"))


def setup_database():
    """Create schemas and tables for all 4 CLIP alignment strategies."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("✅ Connection to PostgreSQL successful!")
        
        cur = conn.cursor()
        
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ pgvector extension enabled")
        
        # Define schemas
        schemas = [
            "vanilla_clip",
            "clip_lexical", 
            "clip_positional",
            "clip_combined"
        ]
        
        for schema_name in schemas:
            # Create schema
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(schema_name)
            ))
            
            # Create images table
            cur.execute(sql.SQL("""
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
            """).format(
                sql.Identifier(schema_name),
                sql.Literal(CLIP_DIM)
            ))
            
            # Create text_chunks table
            cur.execute(sql.SQL("""
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
            """).format(
                sql.Identifier(schema_name),
                sql.Literal(CLIP_DIM)
            ))
            
            # Create alignment table (for weak supervision pairs)
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.alignments (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255) REFERENCES {}.images(image_id),
                    chunk_id VARCHAR(255) REFERENCES {}.text_chunks(chunk_id),
                    weak_score REAL,  -- Confidence score from weak supervision
                    alignment_type VARCHAR(50),  -- 'lexical', 'positional', 'combined'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_id, chunk_id, alignment_type)
                );
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(schema_name),
                sql.Identifier(schema_name)
            ))
            
            # Create indexes for efficient similarity search
            # HNSW index for vector similarity (if pgvector version supports it)
            try:
                cur.execute(sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_images_embedding_idx 
                    ON {schema}.images 
                    USING hnsw (clip_embedding vector_cosine_ops);
                """).format(schema=sql.Identifier(schema_name)))
                
                cur.execute(sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_chunks_embedding_idx 
                    ON {schema}.text_chunks 
                    USING hnsw (clip_embedding vector_cosine_ops);
                """).format(schema=sql.Identifier(schema_name)))
            except Exception as e:
                # Fallback to ivfflat if HNSW not available
                print(f"⚠️  HNSW not available, using ivfflat: {e}")
                cur.execute(sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_images_embedding_idx 
                    ON {schema}.images 
                    USING ivfflat (clip_embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """).format(schema=sql.Identifier(schema_name)))
                
                cur.execute(sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {schema}_chunks_embedding_idx 
                    ON {schema}.text_chunks 
                    USING ivfflat (clip_embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """).format(schema=sql.Identifier(schema_name)))
            
            # Indexes for metadata queries
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS {schema}_images_manual_idx 
                ON {schema}.images(manual_id);
            """).format(schema=sql.Identifier(schema_name)))
            
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS {schema}_chunks_manual_idx 
                ON {schema}.text_chunks(manual_id);
            """).format(schema=sql.Identifier(schema_name)))
            
            print(f"✅ Schema '{schema_name}' created with tables and indexes")
        
        conn.commit()
        cur.close()
        conn.close()
        print("\n🔒 Database setup complete!")
        
    except OperationalError as e:
        print("❌ Could not connect to PostgreSQL:")
        print(e)
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise


if __name__ == "__main__":
    setup_database()

