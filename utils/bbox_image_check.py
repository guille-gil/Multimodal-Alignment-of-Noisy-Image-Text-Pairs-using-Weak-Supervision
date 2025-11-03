import json
from pathlib import Path

path = Path("data/processed/image_metadata.json")
with open(path, "r") as f:
    data = json.load(f)

total = len(data)
nonzero = sum(
    1 for d in data if d.get("bbox") and any(coord != 0 for coord in d["bbox"])
)
zero = total - nonzero

print(f"Total images: {total}")
print(f"Images with valid bounding boxes: {nonzero} ({nonzero / total:.2%})")
print(f"Images with zero bounding boxes: {zero} ({zero / total:.2%})")

# Optional: count how many came from OCR approximation if you added bbox_source
sources = {}
for d in data:
    src = d.get("bbox_source", "unknown")
    sources[src] = sources.get(src, 0) + 1

print("\nBounding box sources:")
for src, count in sources.items():
    print(f"  {src}: {count} ({count / total:.2%})")
