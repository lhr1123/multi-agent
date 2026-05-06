"""Dataset loading helpers."""

import glob
import json
import os
from typing import Dict, List, Optional


def load_gsmhard_dataset(path: str, limit: Optional[int] = 3, offset: int = 0):
    """Load GSM-hard samples from a JSONL file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < max(0, offset):
                continue
            if limit is not None and len(tasks) >= limit:
                break
            data = json.loads(line)
            tasks.append(
                {
                    "index": i,
                    "input": data.get("input", ""),
                    "target": data.get("target"),
                }
            )
    return tasks


def load_mmlu_pro_dataset(
    path: str,
    split: str = "test",
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, object]]:
    """Load MMLU-Pro samples from a local dataset directory or parquet file."""
    split_data = load_mmlu_pro_splits(path).get(split, [])
    start = max(0, offset)
    if limit is None:
        return split_data[start:]
    return split_data[start : start + max(0, limit)]


def load_mmlu_pro_splits(path: str) -> Dict[str, List[Dict[str, object]]]:
    """
    Load MMLU-Pro test / validation splits from a local dataset directory.

    Supported inputs:
    - dataset root directory like ``dataset/MMLU-Pro``
    - direct parquet file path for test-only loading
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MMLU-Pro dataset path not found: {path}")

    if os.path.isfile(path):
        data_files = {"test": path}
    else:
        data_dir = os.path.join(path, "data")
        test_files = sorted(glob.glob(os.path.join(data_dir, "test-*.parquet")))
        validation_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))

        if not test_files and not validation_files:
            raise FileNotFoundError(
                f"No MMLU-Pro parquet files found under: {data_dir}"
            )

        data_files = {}
        if test_files:
            data_files["test"] = test_files
        if validation_files:
            data_files["validation"] = validation_files

    import datasets

    loaded = datasets.load_dataset("parquet", data_files=data_files)
    splits: Dict[str, List[Dict[str, object]]] = {}
    for split_name in loaded.keys():
        rows = []
        for item in loaded[split_name]:
            rows.append(
                {
                    "question_id": item.get("question_id"),
                    "question": item.get("question", ""),
                    "options": list(item.get("options", []) or []),
                    "answer": item.get("answer"),
                    "answer_index": item.get("answer_index"),
                    "cot_content": item.get("cot_content", ""),
                    "category": item.get("category", "unknown"),
                    "src": item.get("src", ""),
                }
            )
        splits[split_name] = rows
    return splits
