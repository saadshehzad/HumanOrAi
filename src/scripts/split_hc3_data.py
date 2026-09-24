from pathlib import Path
from datasets import load_from_disk


# Find the root folder of the project.
ROOT = Path(__file__).resolve().parents[2]

# Location of our cleaned HC3 dataset.
PROCESSED_DATA_PATH = ROOT / "src" / "data" / "processed" / "hc3"


def split_hc3_data():
    dataset = load_from_disk(str(PROCESSED_DATA_PATH))

    # This means keep 80% data for training and 20% for testing.
    # Remember, train_test is not one dataset. It contains 
    # two datasets: (train → 80%, test  → 20%)
    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label",
    )

    # This means split the 20% test data into two equal parts:
    # 50% becomes validation data (10% of the total dataset).
    # 50% becomes test data (10% of the total dataset).
    validation_test = train_test["test"].train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="label",
    )

    # Get the 80% training data.
    train = train_test["train"]
    
    # Get half of the temporary 20% data as validation data.
    validation = validation_test["train"]
    
    # Get the other half of the temporary 20% data as test data.
    test = validation_test["test"]

    # Save the training dataset.
    train.save_to_disk(str(PROCESSED_DATA_PATH / "train"))

    # Save the validation dataset.
    validation.save_to_disk(str(PROCESSED_DATA_PATH / "validation"))

    # Save the test dataset.
    test.save_to_disk(str(PROCESSED_DATA_PATH / "test"))

    # Return all three datasets.
    return train, validation, test


# Run the function when this file is executed directly.
if __name__ == "__main__":
    train, validation, test = split_hc3_data()

    # Show the number of samples in each dataset.
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))