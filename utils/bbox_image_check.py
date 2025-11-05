import json
from pathlib import Path

# Load metadata
metadata_path = Path("data/processed/image_metadata.json")
images_dir = Path("data/processed/images")

if not metadata_path.exists():
    print(f"Error: {metadata_path} not found")
    exit(1)

with open(metadata_path, "r") as f:
    data = json.load(f)

# Count images in JSON
total_in_json = len(data)
with_filename = [d for d in data if d.get("filename")]
without_filename = [d for d in data if not d.get("filename")]

# Count image files on disk
if images_dir.exists():
    files_on_disk = set(f.name for f in images_dir.iterdir() if f.is_file())
    total_files_on_disk = len(files_on_disk)
else:
    files_on_disk = set()
    total_files_on_disk = 0

# Files tracked in JSON
files_in_json = {d["filename"] for d in data if d.get("filename")}

# Find mismatches
files_only_on_disk = files_on_disk - files_in_json
files_only_in_json = files_in_json - files_on_disk

# Count valid/invalid bboxes
nonzero = sum(
    1 for d in data if d.get("bbox") and any(coord != 0 for coord in d["bbox"])
)
zero = total_in_json - nonzero

print("=" * 80)
print("IMAGE METADATA ANALYSIS")
print("=" * 80)
print(f"\nImages in JSON metadata: {total_in_json}")
print(f"  - With filename (raster): {len(with_filename)}")
print(f"  - Without filename (vector): {len(without_filename)}")
print(f"\nImage files on disk: {total_files_on_disk}")

if total_files_on_disk != len(files_in_json):
    print(f"\n⚠️  MISMATCH DETECTED:")
    print(f"  - Files on disk not in JSON: {len(files_only_on_disk)}")
    print(f"  - Files in JSON not on disk: {len(files_only_in_json)}")
    if len(files_only_on_disk) > 0:
        print(f"\n  Sample files not in JSON (first 5):")
        for fname in list(files_only_on_disk)[:5]:
            print(f"    - {fname}")

print(f"\nBounding Box Analysis (from JSON):")
print(f"  - Valid bounding boxes: {nonzero} ({nonzero / total_in_json:.2%})" if total_in_json > 0 else "  - Valid bounding boxes: 0")
print(f"  - Zero bounding boxes: {zero} ({zero / total_in_json:.2%})" if total_in_json > 0 else "  - Zero bounding boxes: 0")

# Count bbox sources
sources = {}
for d in data:
    src = d.get("bbox_source", "unknown")
    sources[src] = sources.get(src, 0) + 1

print("\nBounding box sources (from JSON):")
for src, count in sorted(sources.items()):
    print(f"  {src}: {count} ({count / total_in_json:.2%})" if total_in_json > 0 else f"  {src}: {count}")

print("\n" + "=" * 80)
if total_files_on_disk != len(files_in_json):
    print("⚠️  WARNING: Image files on disk don't match JSON metadata!")
    print("   This suggests the JSON was overwritten or processing was incomplete.")
    print("   Consider re-running PDF processing to regenerate metadata.")
print("=" * 80)
