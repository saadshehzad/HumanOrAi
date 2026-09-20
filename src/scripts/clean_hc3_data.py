from pathlib import Path

from datasets import ClassLabel, Dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = ROOT / "src" / "data" / "raw" / "hc3"
PROCESSED_DATASET_PATH = ROOT / "src" / "data" / "processed" / "hc3"

dataset = load_from_disk(str(RAW_DATASET_PATH))["train"]

data = []
seen = set()

for row in dataset:
    for answer in row["human_answers"]:
        text = answer.strip()

        if text and text not in seen:
            data.append({
                "text": text,
                "label": 0
            })
            seen.add(text)

    for answer in row["chatgpt_answers"]:
        text = answer.strip()

        if text and text not in seen:
            data.append({
                "text": text,
                "label": 1
            })
            seen.add(text)

clean_dataset = Dataset.from_list(data).cast_column(
    "label",
    ClassLabel(names=["human", "chatgpt"]),
)

clean_dataset.save_to_disk(str(PROCESSED_DATASET_PATH))

print("Clean dataset:", len(clean_dataset))