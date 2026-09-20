from datasets import load_dataset

dataset = load_dataset(
	"Hello-SimpleAI/HC3",
	"all",
	trust_remote_code=True,
)

dataset.save_to_disk("data/raw/hc3")

print(dataset)