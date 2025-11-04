"""
PostgreSQL database connection checker.

Tests database connectivity, credentials, and configuration without modifying
any data or schema. Useful for troubleshooting connection issues.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "bachata.service.rug.nl")
DB_NAME = os.getenv("DB_NAME", "aixpert")
DB_USER = os.getenv("DB_USER", "pnumber")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))


def check_connection():
    """Check PostgreSQL database connection and configuration."""
    print("=" * 80)
    print("PostgreSQL Database Connection Check")
    print("=" * 80)
    print()

    # Check environment variables
    print("📋 Configuration:")
    print(f"   Host: {DB_HOST}")
    print(f"   Port: {DB_PORT}")
    print(f"   Database: {DB_NAME}")
    print(f"   User: {DB_USER}")
    print(f"   Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'NOT SET'}")
    print()

    if not DB_PASSWORD:
        print("Error: DB_PASSWORD not set in .env file")
        return False

    # Test connection
    print("🔌 Testing connection...")
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2-binary not installed")
        print("   Install with: pip install psycopg2-binary")
        return False

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10,
        )
        print("Connection successful!")
        print()

        cur = conn.cursor()

        # Check PostgreSQL version
        print("Database Information:")
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"   PostgreSQL: {version.split(',')[0]}")
        print()

        # Check pgvector extension
        print("🔍 Checking pgvector extension...")
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_available_extensions 
                WHERE name = 'vector'
            ) as available;
            """
        )
        available = cur.fetchone()[0]

        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_extension 
                WHERE extname = 'vector'
            ) as installed;
            """
        )
        installed = cur.fetchone()[0]

        if installed:
            print(" pgvector extension is installed")
            # Get version if available
            try:
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                )
                ext_version = cur.fetchone()
                if ext_version:
                    print(f"   Version: {ext_version[0]}")
            except Exception:
                pass
        elif available:
            print(" pgvector extension is available but not installed")
            print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")
        else:
            print("pgvector extension is not available")
        print()

        # List existing schemas (read-only)
        print("📁 Existing Schemas:")
        cur.execute(
            """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            ORDER BY schema_name;
            """
        )
        schemas = [row[0] for row in cur.fetchall()]
        if schemas:
            for schema in schemas:
                print(f"   • {schema}")
        else:
            print("   (no custom schemas found)")
        print()

        # Check required schemas for this project
        print("Project Schema Status:")
        required_schemas = {
            "vanilla_clip",
            "clip_lexical",
            "clip_positional",
            "clip_combined",
        }
        cur.execute(
            """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('vanilla_clip', 'clip_lexical', 'clip_positional', 'clip_combined')
            ORDER BY schema_name;
            """
        )
        existing_schemas = {row[0] for row in cur.fetchall()}

        for schema in sorted(required_schemas):
            if schema in existing_schemas:
                # Check if tables exist
                cur.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s 
                    ORDER BY table_name;
                    """,
                    (schema,),
                )
                tables = [row[0] for row in cur.fetchall()]
                status = "✅" if tables else "⚠️  (empty)"
                print(f"   {status} {schema}", end="")
                if tables:
                    print(f" - Tables: {', '.join(tables)}")
                else:
                    print(" - No tables")
            else:
                print(f" {schema} - Not found")
        print()

        # Test basic query (read-only)
        print("Testing read-only query...")
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        if result and result[0] == 1:
            print("Read operations working")
        print()

        # Check connection properties
        print("Connection Properties:")
        print(f"Server version: {conn.server_version}")
        print(f"Protocol version: {conn.protocol_version}")
        print()

        cur.close()
        conn.close()

        print("=" * 80)
        print("All checks completed successfully!")
        print("=" * 80)
        return True

    except psycopg2.OperationalError as e:
        print(f"Connection failed: {e}")
        return False

    except psycopg2.ProgrammingError as e:
        print(f"Authentication failed: {e}")
        return False

    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"   Type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    success = check_connection()
    sys.exit(0 if success else 1)
