# save_reference_files.py
from pathlib import Path

LIST_PATH = Path("/storage/yoavmp/ai_in_mri/splits/fastmri_run1/test_internal_list.txt")
N = 5

paths = []
with LIST_PATH.open("r") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        paths.append(s)
        if len(paths) >= N:
            break

if len(paths) < N:
    raise RuntimeError(f"Only found {len(paths)} valid lines in {LIST_PATH}")

print("REFERENCE_FILES = [")
for p in paths:
    print(f"    {p!r},")
print("]")

# optional: save to a file next to the list
out_path = LIST_PATH.parent / f"reference_files_{N}.txt"
out_path.write_text("\n".join(paths) + "\n")
print(f"\nWrote: {out_path}")
