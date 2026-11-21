"""Download and partition TIGER-Lab/MMLU-Pro into a 20-20-60 Train/Val/Test split."""

import json
from pathlib import Path
from datasets import load_dataset

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "mmlu_pro"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    shuffled = ds.shuffle(seed=SEED)
    n = len(shuffled)
    train_end = int(0.20 * n)
    val_end = int(0.40 * n)

    splits = {
        "train": shuffled.select(range(0, train_end)),
        "validation_dcal": shuffled.select(range(train_end, val_end)),
        "test": shuffled.select(range(val_end, n)),
    }
    for name, subset in splits.items():
        out_path = OUT_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(subset)} records to {out_path}")


if __name__ == "__main__":
    main()
