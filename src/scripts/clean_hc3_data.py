from datasets import load_from_disk, Dataset

dataset = load_from_disk("data/raw/hc3")["train"]

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

clean_dataset = Dataset.from_list(data)

clean_dataset.save_to_disk("data/processed/hc3")

print("Clean dataset:", len(clean_dataset))