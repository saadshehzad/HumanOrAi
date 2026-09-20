from datasets import load_from_disk

dataset = load_from_disk("data/processed/hc3")

train_test = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

validation_test = train_test["test"].train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column="label"
)

train = train_test["train"]
validation = validation_test["train"]
test = validation_test["test"]

train.save_to_disk("data/processed/hc3/train")
validation.save_to_disk("data/processed/hc3/validation")
test.save_to_disk("data/processed/hc3/test")

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))