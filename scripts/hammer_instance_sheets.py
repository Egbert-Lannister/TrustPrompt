"""Contact sheets + CSV template for manual HAMMER instance->material labelling.

  python -m scripts.hammer_instance_sheets --root D:/Python_Script/TrustPrompt/data/source_cache/hammer_subset_v1 \
      --output results/hammer_instance_sheets_v1

HAMMER ships instance IDs but no material labels (docs/hammer_audit.md). For each
train/val scene (2-11) this samples 3 frames from every object sequence, draws
each instance's outline and ID on the RGB, and writes one PNG per scene plus a
CSV with one row per (scene, instance_id) to fill in. Test scenes 12-14 are not
opened here; label them only after method freeze (labels are method-independent,
but this keeps the test data untouched during development).
Allowed material values: transparent, reflective_metal, partial_reflective,
translucent, diffuse, background, mixed, unknown.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tp.data.hammer import SEQ_RE, scene_split
from tp.provenance import start_run, write_json

MATERIALS = ["transparent", "reflective_metal", "partial_reflective", "translucent", "diffuse", "background", "mixed", "unknown"]


def outline(mask):
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    edge[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return edge


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames-per-sequence", type=int, default=3)
    args = ap.parse_args()
    receipt = start_run(args.output, dict(root=str(args.root), frames_per_sequence=args.frames_per_sequence),
                        [sys.executable, "-m", "scripts.hammer_instance_sheets", *sys.argv[1:]], 0)
    by_scene = defaultdict(list)
    for seq in sorted(p for p in args.root.iterdir() if p.is_dir()):
        m = SEQ_RE.match(seq.name)
        if m and scene_split(int(m.group(1))) in ("train", "val"):
            frames = sorted(int(f.stem) for f in (seq / "polarization" / "_instance").glob("[0-9]*.png") if re.fullmatch(r"\d{6}", f.stem))
            pick = [frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, args.frames_per_sequence)] if frames else []
            by_scene[int(m.group(1))] += [(seq.name, i) for i in pick]
    rows, cmap = [], plt.get_cmap("tab20")
    for scene, picks in sorted(by_scene.items()):
        n = len(picks); cols = 3; nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(6 * cols, 4.8 * nrows), squeeze=False)
        area = defaultdict(int); seen_in = defaultdict(set)
        for ax, (seq, idx) in zip(axes.ravel(), picks):
            pol = args.root / seq / "polarization"
            rgb = np.array(Image.open(pol / "rgb" / f"{idx:06d}.png").convert("RGB"))
            inst = np.array(Image.open(pol / "_instance" / f"{idx:06d}.png"))
            over = rgb.copy()
            for k, iid in enumerate(np.unique(inst)):
                mask = inst == iid
                area[int(iid)] += int(mask.sum()); seen_in[int(iid)].add(f"{seq}/{idx:06d}")
                over[outline(mask)] = (np.array(cmap(k % 20)[:3]) * 255).astype(np.uint8)
            ax.imshow(over); ax.set_title(f"{seq} / {idx:06d}", fontsize=10); ax.set_axis_off()
            for iid in np.unique(inst):
                ys, xs = np.nonzero(inst == iid)
                cy, cx = int(np.median(ys)), int(np.median(xs))
                ax.text(cx, cy, str(int(iid)), color="white", fontsize=11, ha="center", va="center",
                        bbox=dict(facecolor="black", alpha=.6, pad=1, edgecolor="none"))
        for ax in axes.ravel()[n:]:
            ax.set_axis_off()
        fig.suptitle(f"HAMMER scene{scene} ({scene_split(scene)}): instance IDs for material labelling", fontsize=14)
        fig.tight_layout(); fig.savefig(args.output / f"scene{scene:02d}_instances.png", dpi=80); plt.close(fig)
        for iid in sorted(area):
            rows.append(dict(scene=f"scene{scene}", split=scene_split(scene), instance_id=iid,
                             pixels_in_sheet_frames=area[iid], frames=len(seen_in[iid]), material="", object_name="", notes=""))
    with (args.output / "hammer_materials_template.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output / "LABELLING.md").write_text(
        "Fill `material` for every row with one of: " + ", ".join(MATERIALS) + ".\n"
        "Use the scene PNG: each instance outline is labelled with its ID at its median pixel.\n"
        "`object_name` is free text (e.g. wine glass, fork). Two people label independently; disagreements go in notes.\n"
        "Save the result as configs/hammer_materials_v1.csv (versioned). Test scenes 12-14 are labelled after method freeze.\n",
        encoding="utf-8")
    receipt.update(stage="complete", scenes=len(by_scene), instance_rows=len(rows))
    write_json(args.output / "receipt.json", receipt)
    print(f"{len(by_scene)} scene sheets, {len(rows)} instance rows -> {args.output}")


if __name__ == "__main__":
    main()
