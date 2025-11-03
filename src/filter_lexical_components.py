"""
Filter lexical components to remove non-relevant terms.

This script loads the lexical components with frequencies and allows operators
to filter out non-relevant terms by maintaining an exclusion list.

Usage:
    1. Run pdf_processor.py to generate lexical_components.json
    2. Edit the EXCLUDE_TERMS list below with terms to filter out
    3. Run: python3 filter_lexical_components.py
    4. Check filtered_lexical_components.json for the final list
"""

import json
from pathlib import Path

# Paths (relative to project root)
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/processed/lexical_components.json"
OUTPUT_FILE = BASE_DIR / "data/processed/filtered_lexical_components.json"

# Terms to exclude (manually maintained by operators)
# Add terms that are not relevant for CLIP alignment
# These might be truncations, OCR errors, or non-visual concepts
EXCLUDE_TERMS = {
    # Example truncations/artifacts (uncomment as needed):
    # "proce",      # Likely truncation of "proces" (process)
    # "visionplaa", # Likely truncation of "visionplaat" (vision plate)
    # Add other non-relevant terms here:
    # "example_term",
    # "another_term",
}


def filter_lexical_components():
    """Load lexical components, filter out excluded terms, and save filtered version."""
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run pdf_processor.py first.")
        return

    # Load original components
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = data.get("total_components", 0)
    original_components = data.get("components", [])

    # Filter out excluded terms
    filtered_components = [
        comp for comp in original_components if comp["term"] not in EXCLUDE_TERMS
    ]

    # Calculate statistics
    excluded_count = len(original_components) - len(filtered_components)
    total_occurrences_remaining = sum(comp["count"] for comp in filtered_components)

    # Create filtered data structure
    filtered_data = {
        "total_components": len(filtered_components),
        "total_occurrences": total_occurrences_remaining,
        "excluded_count": excluded_count,
        "excluded_terms": sorted(list(EXCLUDE_TERMS)),
        "components": filtered_components,
    }

    # Save filtered components
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered lexical components:")
    print(f"  Original: {original_count} terms")
    print(f"  Excluded: {excluded_count} terms")
    print(f"  Remaining: {len(filtered_components)} terms")
    print(f"  Saved to: {OUTPUT_FILE}")

    # Print excluded terms for review
    if EXCLUDE_TERMS:
        print(f"\nExcluded terms: {', '.join(sorted(EXCLUDE_TERMS))}")

    # Print top 10 remaining terms by frequency
    if filtered_components:
        print(f"\nTop 10 remaining terms by frequency:")
        for i, comp in enumerate(filtered_components[:10], 1):
            print(f"  {i}. {comp['term']}: {comp['count']}")


if __name__ == "__main__":
    filter_lexical_components()
