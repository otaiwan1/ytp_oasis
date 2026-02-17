"""
pick_test_scans.py — Randomly select 60 test scans from the DINOv2-eligible
file list and let the dentist review / swap them via a tkinter GUI.

Outputs:
    validation/validation_test_scans.json  — 60 selected test filenames
    validation/validation_base_scans.json  — remaining ~581 base filenames
"""

import json
import random
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DINOV2_FILENAMES = PROJECT_ROOT / "train" / "dinov2_filenames.json"
OUTPUT_TEST = SCRIPT_DIR / "validation_test_scans.json"
OUTPUT_BASE = SCRIPT_DIR / "validation_base_scans.json"

NUM_TEST = 60


# ─── Core logic ──────────────────────────────────────────────────────

def load_all_filenames():
    with open(DINOV2_FILENAMES, "r") as f:
        return json.load(f)


class ScanPicker:
    """Tkinter application for picking / swapping test scans."""

    def __init__(self, master: tk.Tk, all_filenames: list[str]):
        self.master = master
        self.all_filenames = all_filenames
        self.pool = set(all_filenames)         # remaining available files
        self.selected: list[str] = []          # currently chosen 60 files

        # Initial random pick
        self.selected = random.sample(all_filenames, NUM_TEST)
        self.pool -= set(self.selected)

        self._build_ui()
        self._refresh_list()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        self.master.title(f"OASIS — Pick {NUM_TEST} Validation Scans")
        self.master.geometry("820x700")
        self.master.configure(bg="#f5f5f5")

        # Header
        hdr = tk.Frame(self.master, bg="#f5f5f5")
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(hdr, text=f"Select {NUM_TEST} Test Scans",
                 font=("Helvetica", 16, "bold"), bg="#f5f5f5").pack(side="left")
        self.count_label = tk.Label(hdr, text="", font=("Helvetica", 12),
                                    bg="#f5f5f5", fg="#555")
        self.count_label.pack(side="right")

        # Treeview (table)
        cols = ("#", "Filename", "Patient UUID")
        frame = tk.Frame(self.master)
        frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 selectmode="extended", height=22)
        self.tree.heading("#", text="#", anchor="center")
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Patient UUID", text="Patient UUID")
        self.tree.column("#", width=40, anchor="center", stretch=False)
        self.tree.column("Filename", width=480)
        self.tree.column("Patient UUID", width=270)

        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        btn_frame = tk.Frame(self.master, bg="#f5f5f5")
        btn_frame.pack(fill="x", padx=16, pady=(0, 12))

        tk.Button(btn_frame, text="🔄  Replace Selected",
                  command=self._replace_selected,
                  font=("Helvetica", 11), padx=12, pady=6,
                  bg="#ff9800", fg="white", activebackground="#e68900",
                  relief="flat", cursor="hand2").pack(side="left", padx=(0, 8))

        tk.Button(btn_frame, text="🎲  Reshuffle All",
                  command=self._reshuffle_all,
                  font=("Helvetica", 11), padx=12, pady=6,
                  bg="#2196f3", fg="white", activebackground="#1976d2",
                  relief="flat", cursor="hand2").pack(side="left", padx=(0, 8))

        tk.Button(btn_frame, text="✅  Confirm & Save",
                  command=self._confirm_save,
                  font=("Helvetica", 11, "bold"), padx=16, pady=6,
                  bg="#4caf50", fg="white", activebackground="#388e3c",
                  relief="flat", cursor="hand2").pack(side="right")

    # ── Actions ──────────────────────────────────────────────────────

    def _refresh_list(self):
        """Redraw the treeview from self.selected."""
        self.tree.delete(*self.tree.get_children())
        for i, fname in enumerate(self.selected, 1):
            uuid = fname.split("_")[0]
            self.tree.insert("", "end", iid=str(i - 1),
                             values=(i, fname, uuid))
        self.count_label.config(
            text=f"{len(self.selected)} selected  |  "
                 f"{len(self.pool)} remaining in pool")

    def _replace_selected(self):
        """Delete every selected row and replace with a random file."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select one or more rows first.")
            return
        for iid in sel:
            idx = int(iid)
            old = self.selected[idx]
            if not self.pool:
                messagebox.showwarning("Pool empty",
                                       "No more files available to swap in.")
                return
            new = random.choice(list(self.pool))
            self.pool.discard(new)
            self.pool.add(old)
            self.selected[idx] = new
        self._refresh_list()

    def _reshuffle_all(self):
        """Re-pick all 60 files randomly."""
        if not messagebox.askyesno("Reshuffle",
                                   f"Replace all {NUM_TEST} selections?"):
            return
        full = set(self.all_filenames)
        self.selected = random.sample(list(full), NUM_TEST)
        self.pool = full - set(self.selected)
        self._refresh_list()

    def _confirm_save(self):
        """Write the two JSON files and quit."""
        test_set = sorted(self.selected)
        base_set = sorted(self.pool)

        with open(OUTPUT_TEST, "w") as f:
            json.dump(test_set, f, indent=2)
        with open(OUTPUT_BASE, "w") as f:
            json.dump(base_set, f, indent=2)

        messagebox.showinfo(
            "Saved",
            f"✅  Files saved!\n\n"
            f"  Test scans:  {len(test_set)}  →  {OUTPUT_TEST.name}\n"
            f"  Base scans:  {len(base_set)}  →  {OUTPUT_BASE.name}")
        self.master.destroy()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    all_fnames = load_all_filenames()
    print(f"Loaded {len(all_fnames)} DINOv2-eligible filenames.")

    if len(all_fnames) < NUM_TEST:
        print(f"❌  Need at least {NUM_TEST} files, only found {len(all_fnames)}.")
        return

    root = tk.Tk()
    ScanPicker(root, all_fnames)
    root.mainloop()


if __name__ == "__main__":
    main()
