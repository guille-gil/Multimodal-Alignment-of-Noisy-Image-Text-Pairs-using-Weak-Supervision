"""
Manage local PostgreSQL database via Docker.

This script provides commands to start, stop, and check the status of
the local PostgreSQL database container running in Docker.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DOCKER_COMPOSE_FILE = BASE_DIR / "docker-compose.yml"
CONTAINER_NAME = "multimodal_align_db"


def check_docker_installed():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_container_running():
    """Check if the database container is running."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name={CONTAINER_NAME}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return CONTAINER_NAME in result.stdout
    except subprocess.CalledProcessError:
        return False


def start_local_db():
    """Start the local PostgreSQL database container."""
    if not check_docker_installed():
        print(" Docker is not installed or not running.")
        print(
            "   Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        )
        return False

    if is_container_running():
        print(" Local database container is already running")
        return True

    print(" Starting local PostgreSQL database...")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d"],
            check=True,
            cwd=BASE_DIR,
        )

        # Wait for database to be ready
        print(" Waiting for database to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if is_container_running():
                time.sleep(1)
                try:
                    # Try to connect
                    import psycopg2

                    conn = psycopg2.connect(
                        host="localhost",
                        port=5433,
                        dbname="postgres",
                        user="postgres",
                        password="postgres",
                        connect_timeout=2,
                    )
                    conn.close()
                    print(" Local database is ready!")
                    return True
                except Exception:
                    continue
        print("  Database container started but may not be ready yet")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to start database: {e}")
        return False


def stop_local_db():
    """Stop the local PostgreSQL database container."""
    print(" Stopping local PostgreSQL database...")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down"],
            check=True,
            cwd=BASE_DIR,
        )
        print(" Local database stopped")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to stop database: {e}")
        return False


def remove_local_db():
    """Stop and remove the local PostgreSQL database container and volumes."""
    print("  Removing local PostgreSQL database (this will delete all data)...")
    response = input(
        "   Are you sure? This will delete all data in the local database. (yes/no): "
    )
    if response.lower() != "yes":
        print("   Cancelled.")
        return False

    try:
        subprocess.run(
            ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down", "-v"],
            check=True,
            cwd=BASE_DIR,
        )
        print(" Local database and volumes removed")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to remove database: {e}")
        return False


def status_local_db():
    """Check the status of the local database container."""
    if not check_docker_installed():
        print(" Docker is not installed or not running.")
        return False

    if is_container_running():
        print(" Local database container is running")
        try:
            # Try to connect
            import psycopg2

            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                dbname="postgres",
                user="postgres",
                password="postgres",
                connect_timeout=2,
            )
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            cur.close()
            conn.close()
            print(f"   PostgreSQL version: {version.split(',')[0]}")
            print("   Connection:  Ready")
        except Exception as e:
            print(f"   Connection:  Not ready ({e})")
        return True
    else:
        print(" Local database container is not running")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage local PostgreSQL database")
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "remove"],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "start":
        start_local_db()
    elif args.action == "stop":
        stop_local_db()
    elif args.action == "status":
        status_local_db()
    elif args.action == "remove":
        remove_local_db()
