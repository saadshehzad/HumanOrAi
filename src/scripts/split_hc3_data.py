from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = ROOT / "src" / "data" / "processed" / "hc3"


def split_hc3_data():
    dataset = load_from_disk(str(PROCESSED_DATA_PATH))
    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label",
    )

    validation_test = train_test["test"].train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="label",
    )

    train = train_test["train"]
    validation = validation_test["train"]
    test = validation_test["test"]

    train.save_to_disk(str(PROCESSED_DATA_PATH / "train"))
    validation.save_to_disk(str(PROCESSED_DATA_PATH / "validation"))
    test.save_to_disk(str(PROCESSED_DATA_PATH / "test"))
    return train, validation, test


if __name__ == "__main__":
    train, validation, test = split_hc3_data()
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))