"""
pick_test_scans.py — Randomly select 60 test scans from the DINOv2-eligible
file list and let the dentist review / swap them via an interactive CLI.

Commands:
    list [page]       — Show selected scans (paginated, 20 per page)
    replace N [N2 ..] — Replace scan(s) at position N with random from pool
    reshuffle         — Re-randomize all 60 selections
    save              — Write JSON files and exit
    quit              — Exit without saving

Outputs:
    validation/validation_test_scans.json  — 60 selected test filenames
    validation/validation_base_scans.json  — remaining ~581 base filenames
"""

import json
import random
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DINOV2_FILENAMES = PROJECT_ROOT / "train" / "dinov2_filenames.json"
OUTPUT_TEST = SCRIPT_DIR / "validation_test_scans.json"
OUTPUT_BASE = SCRIPT_DIR / "validation_base_scans.json"

NUM_TEST = 60
PAGE_SIZE = 20


# ─── Core logic ──────────────────────────────────────────────────────

def load_all_filenames():
    with open(DINOV2_FILENAMES, "r") as f:
        return json.load(f)


def print_header():
    print("\n" + "=" * 65)
    print(f"  OASIS — Pick {NUM_TEST} Validation Scans")
    print("=" * 65)


def print_help():
    print("""
  Commands:
    list [page]         Show selected scans (page 1, 2, 3)
    replace N [N2 ..]   Replace scan(s) at position N with random from pool
    reshuffle           Re-randomize all 60 selections
    save                Write JSON files and exit
    quit / q            Exit without saving
    help / h            Show this help
""")


def print_scans(selected, pool, page=1):
    """Show a paginated list of selected scans."""
    total_pages = (len(selected) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(selected))

    print(f"\n  Selected scans — page {page}/{total_pages}  "
          f"({len(selected)} selected | {len(pool)} in pool)")
    print("-" * 65)
    print(f"  {'#':<5} {'Filename':<45} {'Patient UUID'}")
    print("-" * 65)
    for i in range(start, end):
        fname = selected[i]
        uuid = fname.split("_")[0]
        print(f"  {i + 1:<5} {fname:<45} {uuid}")
    print("-" * 65)


def confirm_save(selected, pool):
    """Write the two JSON files."""
    test_set = sorted(selected)
    base_set = sorted(pool)

    with open(OUTPUT_TEST, "w") as f:
        json.dump(test_set, f, indent=2)
    with open(OUTPUT_BASE, "w") as f:
        json.dump(base_set, f, indent=2)

    print(f"\n  ✅  Files saved!")
    print(f"      Test scans:  {len(test_set):>4}  →  {OUTPUT_TEST.name}")
    print(f"      Base scans:  {len(base_set):>4}  →  {OUTPUT_BASE.name}")


# ─── Interactive loop ────────────────────────────────────────────────

def run_picker(all_filenames):
    """Interactive CLI for picking / swapping test scans."""
    pool = set(all_filenames)
    selected = random.sample(all_filenames, NUM_TEST)
    pool -= set(selected)

    print_header()
    print_scans(selected, pool)
    print_help()

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting without saving.")
            return

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # ── list ─────────────────────────────────────────────────────
        if cmd == "list":
            page = 1
            if len(parts) > 1:
                try:
                    page = int(parts[1])
                except ValueError:
                    print("  Usage: list [page_number]")
                    continue
            print_scans(selected, pool, page)

        # ── replace ──────────────────────────────────────────────────
        elif cmd == "replace":
            if len(parts) < 2:
                print("  Usage: replace N [N2 N3 ...]  (1-based positions)")
                continue
            indices = []
            bad = False
            for tok in parts[1:]:
                try:
                    idx = int(tok) - 1   # convert to 0-based
                    if idx < 0 or idx >= len(selected):
                        print(f"  ❌  Position {tok} out of range (1–{len(selected)})")
                        bad = True
                        break
                    indices.append(idx)
                except ValueError:
                    print(f"  ❌  '{tok}' is not a valid number")
                    bad = True
                    break
            if bad:
                continue

            for idx in indices:
                if not pool:
                    print("  ⚠  Pool is empty — no more files to swap in.")
                    break
                old = selected[idx]
                new = random.choice(list(pool))
                pool.discard(new)
                pool.add(old)
                selected[idx] = new
                print(f"  #{idx + 1}: {old}  →  {new}")
            print(f"  ({len(selected)} selected | {len(pool)} in pool)")

        # ── reshuffle ────────────────────────────────────────────────
        elif cmd == "reshuffle":
            confirm = input("  Reshuffle all 60 selections? [y/N] ").strip().lower()
            if confirm in ("y", "yes"):
                full = set(all_filenames)
                selected = random.sample(list(full), NUM_TEST)
                pool = full - set(selected)
                print("  🎲  Reshuffled all selections.")
                print_scans(selected, pool)
            else:
                print("  Cancelled.")

        # ── save ─────────────────────────────────────────────────────
        elif cmd == "save":
            confirm_save(selected, pool)
            return

        # ── quit ─────────────────────────────────────────────────────
        elif cmd in ("quit", "q", "exit"):
            print("  Exiting without saving.")
            return

        # ── help ─────────────────────────────────────────────────────
        elif cmd in ("help", "h"):
            print_help()

        else:
            print(f"  Unknown command: '{cmd}'. Type 'help' for commands.")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    all_fnames = load_all_filenames()
    print(f"Loaded {len(all_fnames)} DINOv2-eligible filenames.")

    if len(all_fnames) < NUM_TEST:
        print(f"❌  Need at least {NUM_TEST} files, only found {len(all_fnames)}.")
        return

    run_picker(all_fnames)


if __name__ == "__main__":
    main()
