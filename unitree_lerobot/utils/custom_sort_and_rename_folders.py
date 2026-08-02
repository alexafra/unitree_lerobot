

"""
Combine all episodes from pick_toothepaste_3107_{1,2,3}
into pick_toothepaste_3107_a by *copying*, renumbering from 0000.
"""

from pathlib import Path
import shutil

# --- config ---
BASE = Path.cwd()                     # run from pick_toothepaste_3107_all
SOURCES = [
    BASE / "pick_toothepaste_3107_1",
    BASE / "pick_toothepaste_3107_2",
    BASE / "pick_toothepaste_3107_3",
]
TARGET = BASE / "pick_toothepaste_3107_a"

def main():
    TARGET.mkdir(exist_ok=True)

    # Collect every episode folder (preserve source order)
    episodes = []
    for src in SOURCES:
        if not src.exists():
            print(f"Warning: {src} does not exist, skipping")
            continue
        found = sorted(
            [p for p in src.iterdir() if p.is_dir() and p.name.startswith("episode_")],
            key=lambda p: p.name
        )
        episodes.extend(found)
        print(f"Found {len(found)} episodes in {src.name}")

    if not episodes:
        print("No episodes found. Exiting.")
        return

    print(f"\nTotal episodes to copy: {len(episodes)}")
    print(f"Target: {TARGET}\n")

    for i, src_ep in enumerate(episodes):
        new_name = f"episode_{i:04d}"
        dst = TARGET / new_name

        if dst.exists():
            print(f"  Skipping {new_name} (already exists)")
            continue

        print(f"  {src_ep.parent.name}/{src_ep.name}  →  {new_name}")
        shutil.copytree(src_ep, dst)

    print(f"\nDone. {len(list(TARGET.glob('episode_*')))} episodes now in {TARGET.name}")

if __name__ == "__main__":
    main()
