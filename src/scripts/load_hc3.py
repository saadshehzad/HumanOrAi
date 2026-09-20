from datasets import load_dataset

dataset = load_dataset("Hello-SimpleAI/HC3")

dataset.save_to_disk("data/raw/hc3")

print(dataset)