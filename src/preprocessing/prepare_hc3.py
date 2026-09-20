from pathlib import Path

from datasets import load_from_disk

DATA_DIR = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = DATA_DIR / "data" / "raw" / "hc3"
PROCESSED_DATASET_PATH = DATA_DIR / "data" / "processed" / "hc3"


try:
    processed_dataset = load_from_disk(str(PROCESSED_DATASET_PATH))
except FileNotFoundError:
    processed_dataset = None

try:
    train = load_from_disk(str(PROCESSED_DATASET_PATH / "train"))
except FileNotFoundError:
    train = None

try:
    validation_test = load_from_disk(str(PROCESSED_DATASET_PATH / "validation"))
except FileNotFoundError:
    validation_test = None
