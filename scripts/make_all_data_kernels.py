"""Generate the all-data (train + validation images) copies of the Kaggle detector-training kernels from the
originals, so the two stay in sync:

    python scripts/make_all_data_kernels.py

Each copy: trains on train + val images (val stays as the ultralytics validation set, now seen in training, so its
metrics are meaningless and early stopping is disabled), saves weights under a *_all name, and the YOLO11l copy skips
its in-kernel candidate building (the v10 rebuild kernel does that with the final refiner).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MERGE = "train_entries = train_entries + val_entries  # ALL DATA: validation images join training\n"

JOBS = {
    "yolo11m": dict(src="kaggle_kernel_train_cls/train_yolo_cls.py", dst="kaggle_kernel_all_yolo11m",
                    patches=[("train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)\n",
                              None),
                             ("patience=15", "patience=100"),
                             ("yolo11m_cls_best.pt", "yolo11m_cls_all_best.pt")]),
    "rtdetr": dict(src="kaggle_kernel_rtdetr_cls/train_rtdetr_cls.py", dst="kaggle_kernel_all_rtdetr",
                   patches=[("train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)\n",
                             None),
                            ("patience=15", "patience=100"),
                            ("rtdetr_cls_best.pt", "rtdetr_cls_all_best.pt")]),
    "yolo11l": dict(src="kaggle_kernel_yolo11l/train_yolo11l_kaggle.py", dst="kaggle_kernel_all_yolo11l",
                    patches=[("train_entries, val_entries, per_image = train_val_split()\n", None),
                             ("patience=15", "patience=100"),
                             ('shutil.copy(best, WORK / f"{TAG}_best.pt")', 'shutil.copy(best, WORK / f"{TAG}_all_best.pt")')],
                    cut_after='    print(f"training done ({(time.time() - t0) / 3600:.2f}h)", flush=True)\n'),
}


def main():
    for name, j in JOBS.items():
        text = (ROOT / j["src"]).read_text(encoding="utf-8")
        # merge line must be indented like the line it follows
        fixed = []
        for old, new in j["patches"]:
            if new is None:
                indent = text[: text.index(old)].rsplit("\n", 1)[-1]
                new = old + indent + MERGE
            fixed.append((old, new))
        text = patch_exact(text, fixed)
        if "cut_after" in j:
            marker = j["cut_after"]
            assert text.count(marker) == 1
            head, _ = text.split(marker)
            text = head + marker + '    print("weights saved; candidates are built by the v10 rebuild kernel", flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'
        dst = ROOT / j["dst"]
        dst.mkdir(exist_ok=True)
        code = Path(j["src"]).name
        (dst / code).write_text(text, encoding="utf-8")
        meta = json.loads((ROOT / Path(j["src"]).parent / "kernel-metadata.json").read_text())
        slug = f"filament-all-{name}"
        meta.update(id=f"trishanthmellimi/{slug}", title=slug, code_file=code)
        (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", dst)


def patch_exact(text, patches):
    for old, new in patches:
        assert text.count(old) == 1, f"pattern not found exactly once: {old!r} ({text.count(old)})"
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
