import os
import json
from datasets import load_dataset


# Preprocess the SQuAD dataset for question generation and reasoning.
def preprocess_squad_data(dataset, include_reasoning=False):
    processed_data = []

    for item in dataset:
        context = item["context"]
        question = item["question"]
        answers = item["answers"]["text"] if "answers" in item else []

        if not context or not question:
            continue

        answer = answers[0] if answers else "?"
        reasoning = (
            "Think step by step and explain your reasoning before answering in JSON format."
            if include_reasoning
            else ""
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {question}\n{reasoning}",
            },
            {
                "role": "assistant",
                "content": f"```json\n{{\n  \"answer\": \"{answer}\"\n}}\n```",
            },
        ]

        processed_data.append({"messages": messages})

    return processed_data


# Load the SQuAD dataset from Hugging Face
print("Loading SQuAD dataset...")
dataset = load_dataset("rajpurkar/squad")

# Preprocess the training and validation datasets
print("Preprocessing training data with reasoning...")
train_data = preprocess_squad_data(dataset["train"], include_reasoning=True)
print("Preprocessing validation data with reasoning...")
validation_data = preprocess_squad_data(dataset["validation"], include_reasoning=True)

# Save preprocessed data
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

train_file_path = os.path.join(data_folder, "train_data_chat.json")
validation_file_path = os.path.join(data_folder, "validation_data_chat.json")

print("Saving preprocessed data to the `data/` folder...")
with open(train_file_path, "w", encoding="utf-8") as train_file:
    json.dump(train_data, train_file, indent=2)
with open(validation_file_path, "w", encoding="utf-8") as val_file:
    json.dump(validation_data, val_file, indent=2)

print(f"Preprocessing completed. Files saved to:\n- {train_file_path}\n- {validation_file_path}")
