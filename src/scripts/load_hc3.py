from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = ROOT / "src" / "data" / "raw" / "hc3"

dataset = load_dataset(
	"Hello-SimpleAI/HC3",
	"all",
	trust_remote_code=True,
)

dataset.save_to_disk(str(RAW_DATASET_PATH))

print(dataset)