"""
Test script to verify the PDF processing pipeline installation and functionality.
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")

    try:
        import fitz

        print("✓ PyMuPDF (fitz) imported successfully")
    except ImportError as e:
        print(f"✗ PyMuPDF import failed: {e}")
        return False

    try:
        import pdfplumber

        print("✓ pdfplumber imported successfully")
    except ImportError as e:
        print(f"✗ pdfplumber import failed: {e}")
        return False

    try:
        import spacy

        print("✓ spaCy imported successfully")
    except ImportError as e:
        print(f"✗ spaCy import failed: {e}")
        return False

    try:
        import numpy as np

        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        from PIL import Image

        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False

    try:
        from tqdm import tqdm

        print("✓ tqdm imported successfully")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False

    return True


def test_spacy_model():
    """Test if spaCy English model is available."""
    print("\nTesting spaCy English model...")

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        print("✓ English spaCy model loaded successfully")

        # Test basic functionality
        doc = nlp("This is a test sentence.")
        tokens = [token.text for token in doc]
        print(f"✓ Tokenization test: {tokens}")

        return True
    except OSError:
        print("✗ English spaCy model not found")
        print("  Install with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"✗ spaCy model test failed: {e}")
        return False


def test_directory_structure():
    """Test if the required directory structure exists."""
    print("\nTesting directory structure...")

    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "data",
        base_dir / "data" / "raw",
        base_dir / "data" / "raw" / "manuals",
        base_dir / "data" / "processed",
        base_dir / "data" / "processed" / "images",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False

    return all_exist


def test_pdf_processor():
    """Test if the PDF processor can be imported."""
    print("\nTesting PDF processor import...")

    try:
        from src.pdf_processor import PDFProcessor

        print("✓ PDFProcessor imported successfully")

        # Test initialization
        processor = PDFProcessor("data/raw/manuals", "data/processed")
        print("✓ PDFProcessor initialized successfully")

        return True
    except Exception as e:
        print(f"✗ PDFProcessor test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("PDF Processing Pipeline - Installation Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_spacy_model,
        test_directory_structure,
        test_pdf_processor,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Summary:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed! The pipeline is ready to use.")
        print("\nTo process PDFs:")
        print("1. Place PDF files in data/raw/manuals/")
        print("2. Run: python src/pdf_processor.py")
    else:
        print(f"✗ {total - passed} out of {total} tests failed.")
        print("Please fix the issues above before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
